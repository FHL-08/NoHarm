"""CTRL-2.3@1 The executor source contains no code path that snaps an
object between the palms.

The contract under test is a property of the product source itself:
the files the requirement touches — harmless/executor.py and
harmless/control/** — must not contain any path that writes an object's
simulated pose. A "snap" is a kinematic state edit: writing qpos/qvel
(or the derived xpos/xquat buffers) of live MjData repositions a body
without dynamics. The current grasp() does exactly that — a hold_at
helper pins the prop's free joint to a snap_to point centred between
the palms — and this audit is what fails on it.

Checked two ways:

- statically, by AST: any assignment into <x>.data.qpos /
  <x>.data.qvel / <x>.data.xpos / <x>.data.xquat (through self, a
  local alias like `d = self.data`, or an attribute chain), any bulk
  copy into those buffers, and any identifier spelling snap/teleport.
  Whole-array `[:] =` restores (a scene reset, not a palm snap) and
  the authored mocap person driver / ctrl actuator targets are not
  snap paths and are allowed.
- dynamically, belt and suspenders for a snap hidden where the AST
  cannot see it: a real grasp('bowl') is run with the bowl's world
  position recorded every physics step — any single-step jump far
  beyond physical speed is a teleport.

On a tree without the snap the audit finds nothing and the bowl's
per-step motion stays continuous.
"""

import ast
import math
import os
import re
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]
SCENE = REPO / "kitchen.xml"
SRC_FILES = [REPO / "harmless" / "executor.py"]
CTRL_DIR = REPO / "harmless" / "control"

# writing these live-state arrays IS the snap: qpos/qvel reposition a
# body outright; xpos/xquat are solver outputs and only deception could
# write them. ctrl (actuator targets) and mocap_* (the authored person
# driver) are legitimate channels and are not audited.
STATE_ATTRS = ("qpos", "qvel", "xpos", "xquat")
# "snapshot"/_snap_every are the frame recorder, not object teleports
SNAP_WORD = re.compile(r"snap(?!shot|_every)|teleport", re.I)
BULK_COPY = re.compile(r"copyto|put_along_axis|place", re.I)

# at dt=0.002 s a physical move is < 1 cm; a between-palms snap is cm-scale
JUMP_M = 0.025


def _mujoco():
    import mujoco
    return mujoco


def _chain(node):
    """Dotted path of a Name/Attribute expression, e.g.
    self.data.qpos -> ('self', 'data', 'qpos')."""
    out = []
    while isinstance(node, ast.Attribute):
        out.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        out.append(node.id)
    return tuple(reversed(out))


def _aliases(tree):
    """Names bound to a `*.data` or `*.model` object anywhere in the
    module (e.g. `d = self.data`, `m, d = self.model, self.data`)."""
    data, model = set(), set()

    def bind(t, v):
        if not isinstance(t, ast.Name) or v is None:
            return
        ch = _chain(v)
        if not ch:
            return
        if ch[-1] == "data":
            data.add(t.id)
        elif ch[-1] == "model":
            model.add(t.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Tuple) and \
                        isinstance(node.value, ast.Tuple):
                    for tt, vv in zip(t.elts, node.value.elts):
                        bind(tt, vv)
                else:
                    bind(t, node.value)
        elif isinstance(node, ast.AnnAssign):
            bind(node.target, node.value)
    return data, model


def _audit(path):
    """Snap/teleport code paths in one product source file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    data_names, model_names = _aliases(tree)
    hits = []

    for node in ast.walk(tree):
        # identifier vocabulary: a thing called snap/teleport exists to snap
        for tok in (getattr(node, "name", None),
                    getattr(node, "attr", None),
                    getattr(node, "id", None)):
            if isinstance(tok, str) and SNAP_WORD.search(tok):
                hits.append(f"{path.name}:{node.lineno}: identifier "
                            f"{tok!r} — snap/teleport machinery")
                break

        # writes into live simulation state
        tgts, value = [], None
        if isinstance(node, ast.Assign):
            tgts, value = node.targets, node.value
        elif isinstance(node, ast.AugAssign):
            tgts, value = [node.target], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            tgts, value = [node.target], node.value
        for t in tgts:
            whole = False
            if isinstance(t, ast.Subscript):
                ch = _chain(t.value)
                s = t.slice
                whole = isinstance(s, ast.Slice) and s.lower is None \
                    and s.upper is None and s.step is None
            else:
                ch = _chain(t)
            if len(ch) < 2:
                continue
            owner, attr = ch[-2], ch[-1]
            if attr in STATE_ATTRS and not whole and \
                    (owner == "data" or owner in data_names):
                hits.append(
                    f"{path.name}:{node.lineno}: writes live sim state "
                    f"`{'.'.join(ch)}` — kinematic repositioning of a "
                    "body is the snap between the palms")

        # bulk-copy calls that smuggle the same write past `=`
        if isinstance(node, ast.Call):
            fn = _chain(node.func)
            if fn and BULK_COPY.search(fn[-1]) and node.args:
                ch = _chain(node.args[0].value
                            if isinstance(node.args[0], ast.Subscript)
                            else node.args[0])
                if len(ch) >= 2 and ch[-1] in STATE_ATTRS and \
                        (ch[-2] == "data" or ch[-2] in data_names):
                    hits.append(
                        f"{path.name}:{node.lineno}: bulk-writes live "
                        f"sim state `{'.'.join(ch)}`")

    return hits


def _executor(tmp_path):
    try:
        from harmless.executor import Executor
    except Exception as exc:
        pytest.fail(
            f"harmless.executor cannot be imported "
            f"({type(exc).__name__}: {exc})")
    try:
        return Executor(str(SCENE), record=False,
                        out_dir=str(tmp_path / "episode"),
                        skills_path=None,
                        ws_dir=str(tmp_path / "ws"))
    except Exception as exc:
        pytest.fail(
            f"the kitchen scene failed to load in the Executor "
            f"({type(exc).__name__}: {exc})")


def _teleport(ex, name, xyz, yaw=None):
    """Arrange the trial's initial condition: place a free body at rest.
    Test setup only — every later motion is the physics engine's."""
    mujoco = _mujoco()
    m, d = ex.model, ex.data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    jnt = m.body_jntadr[bid]
    qa, va = m.jnt_qposadr[jnt], m.jnt_dofadr[jnt]
    d.qpos[qa:qa + 3] = xyz
    if yaw is not None:
        d.qpos[qa + 3:qa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    d.qvel[va:va + 6] = 0
    mujoco.mj_forward(m, d)


def _park_person(ex):
    if hasattr(ex, "_fall"):
        ex._fall = (float(ex.data.time), np.array([1.0, 0.0, 0.0]))


def _palm(ex, side):
    return np.asarray(ex.pos(f"{side}_palm"), float)


def _place_palms(ex, name, standoff=0.04, timeout=6.0):
    """A caller-built pregrasp: both palms to the object's left and right
    sides a few cm off the skin via the public hands_path primitive."""
    info = ex.geom_info(name)
    lo, hi = np.array(info["aabb_min"]), np.array(info["aabb_max"])
    c = (lo + hi) / 2
    R = ex.data.xmat[ex.pelvis_body].reshape(3, 3)
    left = R[:, 1].copy()
    left[2] = 0.0
    left /= max(np.linalg.norm(left), 1e-9)
    half = float(np.abs(left[:2]) @ ((hi - lo)[:2] / 2))
    z = float(hi[2]) - 0.03
    off = half + standoff
    lift = z + 0.10
    tgts = {"left": np.r_[c[:2] + left[:2] * off, z],
            "right": np.r_[c[:2] - left[:2] * off, z]}
    paths = {}
    for s, p in tgts.items():
        cur = _palm(ex, s)
        paths[s] = [np.r_[cur[:2], max(lift, cur[2])],
                    np.r_[p[:2], lift], p]
    ex.hands_path(paths["right"], paths["left"])
    t0 = float(ex.data.time)
    errs = {"right": 9.0, "left": 9.0}
    while float(ex.data.time) - t0 < timeout:
        ex.run_seconds(0.1)
        st = ex.arm_status()
        errs = {s: st.get(s, {}).get("palm_error_m", 9.0) for s in tgts}
        if all(st.get(s, {}).get("via_points_left") == 0 and e < 0.03
               for s, e in errs.items()):
            break
    return errs


def _watch(ex, name):
    """Wrap ex.step: record, per physics step, the object's world position
    and the activation state of every equality constraint that involves
    its body."""
    mujoco = _mujoco()
    m, d = ex.model, ex.data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    prop_eqs = [i for i in range(m.neq)
                if int(m.eq_objtype[i]) == int(mujoco.mjtObj.mjOBJ_BODY)
                and bid in (int(m.eq_obj1id[i]), int(m.eq_obj2id[i]))]
    log = {"step": ex.step, "jump": 0.0, "prop_eq": set(),
           "last": d.xpos[bid].copy()}
    orig = ex.step

    def step(n=1):
        orig(n)
        p = d.xpos[bid]
        log["jump"] = max(log["jump"],
                          float(np.linalg.norm(p - log["last"])))
        log["last"] = p.copy()
        for i in prop_eqs:
            if d.eq_active[i]:
                log["prop_eq"].add(i)

    ex.step = step
    return log


def test_ctrl_2_3_no_code_path_snaps_object_between_palms(tmp_path):
    # --- static: the product source itself
    files = [p for p in SRC_FILES if p.is_file()]
    if CTRL_DIR.is_dir():
        files += sorted(CTRL_DIR.rglob("*.py"))
    hits = []
    for p in files:
        hits += _audit(p)
    assert not hits, (
        "the executor source still contains code that snaps an object "
        "between the palms:\n  " + "\n  ".join(sorted(set(hits))[:12]))

    # --- dynamic: a real grasp must move the bowl continuously, never
    # teleport it (catches any snap the AST could not see)
    ex = _executor(tmp_path)
    _park_person(ex)
    _teleport(ex, "bowl", [0.0, -0.93, 1.10])
    _teleport(ex, "pelvis", [0.0, -1.28, 0.793], yaw=math.pi / 2)
    ex.run_seconds(0.5)
    errs = _place_palms(ex, "bowl")
    if max(errs.values()) >= 0.10:
        pytest.fail(
            f"pregrasp setup failed to place the palms beside the bowl "
            f"(errors {errs}) — the trial cannot start")
    log = _watch(ex, "bowl")
    res = ex.grasp("bowl")
    if not (isinstance(res, dict) and "error" in res):
        ex.run_seconds(0.4)
        assert log["jump"] < JUMP_M, (
            f"the bowl jumped {log['jump']:.3f} m inside one physics "
            "step during grasp() — a snap between the palms is a "
            "kinematic teleport, whatever the source looks like")
        assert not log["prop_eq"], (
            "an equality constraint on the bowl was activated to pin it "
            "between the palms — the weld side of the weld-snap path")
        ex.step = log["step"]
