"""POL-7.1@1 With only the post-trained checkpoint's joint_pos chunks
driving the G1 at 50 Hz and no scripted reach in the loop, a held-out
start ends in a measured palm-bowl contact.

The contract under test is the artifact POL-6 produces: the GR00T N
checkpoint post-trained on the G1 bowl-touch demonstrations, served
through the harmless/policies/gr00t.py adapter this requirement
touches.

  the post-trained checkpoint: found through the finetune module's
    declared output, the gr00t adapter's resolution, or the artifact
    directories the training pipeline writes to (outside git — the HF
    cache, ~/.cache, the gitignored demo/ and runs/ trees). It must be a
    GR00T action-head checkpoint (action_head / state_encoder /
    action_decoder tensors) that differs from the released nvidia
    GR00T-N weights — a released checkpoint alone was never trained on
    the G1 bowl demos.

  joint_pos chunks driving the G1 at 50 Hz: the test's eval loop calls
    the adapter's act(observation), maps each chunk step onto the
    declared G1 joint_pos joints, and writes them to the joint
    actuators once per 0.02 s control tick — nothing else moves the
    robot. act() is asserted to return an action CHUNK (more than one
    future step) of finite targets inside declared bounds, and asserted
    never to advance sim time itself.

  no scripted reach in the loop: every scripted-reach entry point on the
    executor (arm_reach, arm_path, hands_path, reach, reach_xyz,
    pick_up, put_down, grasp, release, move_joints, set_controller,
    walk) is taped for the whole eval — any call fails the line, since
    the bowl touch must come from the trained network, not a hand
    trajectory.

  a held-out start: the eval does not start from the authored keyframe
    — the pelvis is placed at a fixed offset stance south-east of the
    island facing the bowl and the arm joints are perturbed — a start
    that is not the recorded-demo start pose.

  ends in a measured palm-bowl contact: physics-measured — a MuJoCo
    contact pair between a geom anywhere in the bowl body's subtree
    (the authored 'bowl' free body carries no geoms of its own; its
    collision geoms sit on jointless child bodies like 'bowl_1_0') and
    a geom of a hand/palm/wrist body in the robot tree, inside a 45 s
    sim horizon. A distance is not a contact: Executor.palm_gaps() is
    a surface distance, so a palm hovering centimetres off the bowl
    does not count, and no predicate from the code under test is
    consulted.

There is no post-trained checkpoint yet — checkpoint discovery fails on
the base tree.
"""

import importlib
import inspect
import json
import math
import os
import re
import struct
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]
SCENE = REPO / "kitchen.xml"
CTRL_DT = 0.02            # 50 Hz
HORIZON_S = 45.0
MIN_CHUNK = 2
BOUNDS_TOL = 1e-2
SUBGOAL = "approach and touch the bowl"

GR00T_N = re.compile(r"gr00t[._\- ]?n|gr00tn|isaac[._\- ]?gr00t", re.I)
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
PATH_ATTR = re.compile(
    r"ckpt|checkpoint|out|save|export|dir|path|weights|model", re.I)
HEAD_TENSOR = re.compile(
    r"action_head|state_encoder|action_encoder|action_decoder|"
    r"vlln|vl_self|future_tokens|gr00t", re.I)
HAND_BODY = re.compile(r"wrist|hand|palm", re.I)
SCRIPTED = (
    "arm_reach", "arm_path", "hands_path", "reach", "reach_xyz",
    "pick_up", "put_down", "grasp", "release", "move_joints",
    "set_controller", "walk", "walker_off", "program",
)


def _import(name, required=True):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        if not required:
            return None
        pytest.fail(
            f"{name} cannot be imported ({type(exc).__name__}: {exc})")


# ---------------- checkpoint discovery ---------------------------------


def _safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 100_000_000:
            return {}, 0
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def _st_tensor(path, meta, base):
    dt = meta["dtype"]
    s, e = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + s)
        raw = f.read(e - s)
    if dt == "F32":
        arr = np.frombuffer(raw, dtype=np.float32)
    elif dt == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dt == "BF16":
        arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
               << 16).view(np.float32)
    elif dt == "F64":
        arr = np.frombuffer(raw, dtype=np.float64).astype(np.float32)
    elif dt in ("I64", "I32", "I16", "I8", "U8", "U16", "U32", "U64"):
        arr = np.frombuffer(
            raw, dtype=getattr(np, dt.lower())).astype(np.float32)
    else:
        return None
    return arr.reshape(meta["shape"]).astype(np.float32)


def _tensor_index(artifact):
    idx = {}
    files = []
    p = Path(artifact)
    if p.is_file():
        files = [p]
    elif p.is_dir():
        try:
            files = [f for f in p.rglob("*") if f.is_file()
                     and f.suffix.lower() in WEIGHT_EXT]
        except Exception:
            files = []
    for f in files:
        if f.suffix.lower() == ".safetensors":
            try:
                hdr, base = _safetensors_header(f)
            except Exception:
                continue
            for name, meta in hdr.items():
                if name == "__metadata__" or not isinstance(meta, dict):
                    continue
                idx[name] = (
                    tuple(meta.get("shape") or []),
                    lambda m=meta, b=base, pp=f: _st_tensor(pp, m, b))
        elif f.suffix.lower() in (".bin", ".pt", ".pth", ".ckpt") \
                and f.stat().st_size < 8_000_000_000:
            try:
                import torch
                try:
                    obj = torch.load(str(f), map_location="cpu",
                                     weights_only=True)
                except Exception:
                    obj = torch.load(str(f), map_location="cpu")
            except Exception:
                continue
            stack = [("", obj)]
            while stack:
                pre, cur = stack.pop()
                if isinstance(cur, Mapping):
                    for k, v in cur.items():
                        stack.append(
                            (f"{pre}.{k}" if pre else str(k), v))
                elif hasattr(cur, "state_dict") and callable(
                        cur.state_dict):
                    try:
                        stack.append((pre, cur.state_dict()))
                    except Exception:
                        pass
                else:
                    try:
                        import torch
                        if isinstance(cur, torch.Tensor) and cur.numel():
                            idx[pre] = (
                                tuple(cur.shape),
                                lambda t=cur: t.detach().float()
                                .cpu().numpy())
                    except Exception:
                        pass
    return idx


def _same(a, b, tol=1e-6):
    if a is None or b is None or a.shape != b.shape:
        return False
    n = min(a.size, 1 << 17)
    x = np.asarray(a, dtype=np.float32).reshape(-1)[:n]
    y = np.asarray(b, dtype=np.float32).reshape(-1)[:n]
    return bool(np.allclose(x, y, rtol=1e-4, atol=tol))


def _name_variants(name):
    yield name
    if name.startswith("action_head."):
        yield name[len("action_head."):]
    else:
        yield f"action_head.{name}"
    yield name.split(".")[-1]


def _released_value(idx, name):
    for cand in _name_variants(name):
        if cand in idx:
            try:
                return idx[cand][1]()
            except Exception:
                return None
    return None


def _hf_gr00t_snapshot():
    roots = []
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(env):
            roots.append(Path(os.environ[env]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    for root in roots:
        if not root.is_dir():
            continue
        for d in root.glob("models--*GR00T*/snapshots/*/"):
            if any(d.glob("*.safetensors")) or any(d.glob("*.bin")):
                return d
    return None


def _module_declared_paths(mods):
    out = []
    for mod in mods:
        if mod is None:
            continue
        for name, v in vars(mod).items():
            if not PATH_ATTR.search(name):
                continue
            for p in v if isinstance(v, (list, tuple)) else [v]:
                if isinstance(p, (str, os.PathLike)) \
                        and os.path.exists(p):
                    out.append(Path(p))
        for _n, f in vars(mod).items():
            if not callable(f) or not inspect.isfunction(f):
                continue
            try:
                sig = inspect.signature(f)
            except Exception:
                continue
            for prm in sig.parameters.values():
                nml = prm.name.lower()
                if prm.default is inspect.Parameter.empty:
                    continue
                if re.search(r"ckpt|checkpoint|out|save|export|weights",
                             nml) and isinstance(
                        prm.default, (str, os.PathLike)) \
                        and os.path.exists(prm.default):
                    out.append(Path(prm.default))
    return out


def _scan_artifact_dirs():
    """Directories under the usual artifact roots that hold weight
    files and name like a checkpoint run."""
    roots = [REPO / "demo", REPO / "runs",
             Path.home() / ".cache",
             Path.home() / ".local" / "share",
             Path("/tmp")]
    out = []
    pat = re.compile(r"gr00t|ckpt|checkpoint|finetune|posttrain|"
                     r"post_train|bowl|noharm|harmless|policy", re.I)
    for root in roots:
        if not root.is_dir():
            continue
        for depth in range(4):
            for d in root.glob("/".join(["*"] * (depth + 1))):
                try:
                    if not d.is_dir() or not pat.search(d.name):
                        continue
                    spots = [d] + list(d.glob("snapshots/*/"))
                    for s in spots:
                        if any(f.is_file()
                               and f.suffix.lower() in WEIGHT_EXT
                               and f.stat().st_size > 100_000
                               for f in s.iterdir() if f.is_file()):
                            out.append(s)
                except Exception:
                    continue
            if len(out) > 40:
                return out
    return out


def _find_posttrained_ckpt(rel_idx):
    finetune = _import("harmless.policies.finetune", required=False)
    gr00t = _import("harmless.policies.gr00t", required=False)
    demos = _import("harmless.policies.demos", required=False)
    cands = _module_declared_paths([finetune, gr00t, demos]) \
        + _scan_artifact_dirs()
    seen, uniq = set(), []
    for c in cands:
        try:
            key = str(c.resolve())
        except Exception:
            key = str(c)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    hf_root = Path.home() / ".cache" / "huggingface" / "hub"
    scored = []
    for c in uniq:
        try:
            res = c.resolve()
        except Exception:
            res = c
        if _under(res, hf_root) and not re.search(
                r"finetune|post|g1|bowl|noharm|harmless|local",
                str(res), re.I):
            continue   # an unmodified released snapshot is not post-trained
        idx = _tensor_index(c)
        if not idx:
            continue
        headish = [n for n in idx if HEAD_TENSOR.search(n)]
        if not headish:
            continue
        diff = 0
        overlap = 0
        for name in headish[:12]:
            rel = _released_value(rel_idx, name) if rel_idx else None
            if rel is None:
                continue
            overlap += 1
            try:
                cur = idx[name][1]()
            except Exception:
                continue
            if cur is not None and not _same(cur, rel):
                diff += 1
        scored.append((diff, overlap, c))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return scored


def _under(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except Exception:
        return False


def _git_tracked(path):
    try:
        subprocess.check_output(
            ["git", "-C", str(REPO), "ls-files", "--error-unmatch",
             str(path)], stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _git_ignored(path):
    r = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q", str(path)])
    return r.returncode == 0


# ---------------- measured palm-bowl contact --------------------------


def _under_body(model, bid, root):
    """bid is root or descends from it (walk body_parentid)."""
    while bid > 0:
        if bid == root:
            return True
        bid = int(model.body_parentid[bid])
    return False


def _in_robot(ex, bid):
    return _under_body(ex.model, bid, ex.pelvis_body)


def _palm_bowl_contact(ex):
    """Measured palm-bowl contact on live sim state, read straight off
    data.contact: a MuJoCo contact pair between a geom anywhere in the
    bowl body's subtree (the authored 'bowl' free body carries no geoms
    of its own — they live on jointless children like 'bowl_1_0') and a
    geom of a hand/palm/wrist body in the robot tree. A distance
    (palm_gaps) is not a contact, and no predicate from the code under
    test is consulted."""
    import mujoco
    m, d = ex.model, ex.data
    bowl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bowl")
    if bowl < 0:
        return False
    for c in range(int(d.ncon)):
        con = d.contact[c]
        b1 = int(m.geom_bodyid[con.geom1])
        b2 = int(m.geom_bodyid[con.geom2])
        if _under_body(m, b2, bowl):
            b1, b2 = b2, b1
        if not _under_body(m, b1, bowl):
            continue
        name = mujoco.mj_id2name(
            m, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
        if HAND_BODY.search(name) and _in_robot(ex, b2):
            return True
    return False


# ---------------- the eval loop ----------------------------------------


def _with_instruction(obs, text):
    obs = dict(obs)
    for key in ("instruction", "subgoal", "task", "prompt", "goal",
                "language_instruction", "command"):
        obs[key] = text
    return obs


def _scalar(v):
    try:
        float(v)
        return True
    except Exception:
        return False


def _steps_from(out, joints, order, depth=0):
    """Normalize any plausible chunk spelling into a list of per-step
    {joint: target} dicts."""
    if out is None or depth > 4:
        return None
    joints = list(joints)
    jset = set(joints)
    order = list(order) if order else joints
    if isinstance(out, Mapping):
        js = set(out) & jset
        if js and all(_scalar(out[k]) for k in js):
            return [{k: float(out[k]) for k in js}]
        seq = [k for k in js
               if isinstance(out[k], (list, tuple, np.ndarray))
               and np.asarray(out[k], dtype=object).ndim == 1
               and len(out[k]) > 1]
        if seq:
            h = min(len(out[k]) for k in seq)
            try:
                return [{k: float(out[k][i]) for k in seq}
                        for i in range(h)]
            except Exception:
                pass
        for k in ("action_chunk", "actions", "chunk", "joint_pos",
                  "targets", "trajectory", "traj", "horizon", "action",
                  "steps", "plan"):
            if k in out:
                r = _steps_from(out[k], joints, order, depth + 1)
                if r:
                    return r
        for v in out.values():
            r = _steps_from(v, joints, order, depth + 1)
            if r:
                return r
        return None
    if isinstance(out, (list, tuple)) and out \
            and all(isinstance(s, Mapping) for s in out):
        steps = []
        for s in out:
            js = set(s) & jset
            if not js:
                return None
            steps.append({k: float(s[k]) for k in js if _scalar(s[k])})
        return steps
    try:
        arr = np.asarray(out, dtype=float)
    except Exception:
        arr = None
    if arr is not None and arr.ndim == 2 and min(arr.shape) >= 1:
        if arr.shape[1] == len(order):
            return [dict(zip(order, row)) for row in arr]
        if arr.shape[0] == len(order) and arr.shape[1] > 1:
            return [dict(zip(order, col)) for col in arr.T]
        n = min(len(order), arr.shape[1])
        return [dict(zip(order[:n], row[:n])) for row in arr]
    if arr is not None and arr.ndim == 1 and arr.size == len(joints):
        return [dict(zip(joints, arr))]
    return None


def _call_act(pol, obs):
    act = getattr(pol, "act", None)
    if not callable(act):
        if callable(pol):
            return pol(obs)
        pytest.fail("the policy exposes no act(observation) call")
    for args, kw in (((obs,), {}),
                     ((obs,), {"instruction": SUBGOAL}),
                     ((obs,), {"subgoal": SUBGOAL}),
                     ((obs, SUBGOAL), {})):
        try:
            return act(*args, **kw)
        except TypeError:
            continue
    return act(obs)


def _make_policy(ckpt, emb, space):
    gr00t = _import("harmless.policies.gr00t")
    cls = getattr(gr00t, "Gr00tVLAPolicy", None)
    if cls is None:
        try:
            from harmless import registry
            for n in registry.list_policies():
                if re.search(r"gr00t|vla|isaac", str(n), re.I):
                    cls = registry.get_policy(n)
                    break
        except Exception:
            pass
    if cls is None:
        for _n, obj in sorted(vars(gr00t).items()):
            if inspect.isclass(obj) \
                    and obj.__module__ == gr00t.__name__:
                cls = obj
                break
    assert cls is not None, "no gr00t adapter class found"
    errs = []
    for args, kw in [
            ((), {"embodiment": emb, "checkpoint": str(ckpt)}),
            ((), {"embodiment": emb, "checkpoint_path": str(ckpt)}),
            ((), {"embodiment": emb, "ckpt": str(ckpt)}),
            ((), {"embodiment": emb, "weights": str(ckpt)}),
            ((), {"embodiment": emb, "model_path": str(ckpt)}),
            ((), {"embodiment": emb}),
            ((), {"action_space": space, "checkpoint": str(ckpt)}),
            ((), {"action_space": space}),
            ((emb, str(ckpt)), {}),
            ((str(ckpt),), {}),
            ((emb,), {}),
            ((), {}),
    ]:
        try:
            pol = cls(*args, **kw)
            break
        except Exception as exc:
            errs.append(f"{kw or args}: {type(exc).__name__}: {exc}")
    else:
        pytest.fail(
            "the gr00t adapter could not be constructed on the "
            "post-trained checkpoint:\n" + "\n".join(errs[:8]))
    if getattr(pol, "action_space", None) is None:
        bind = getattr(pol, "bind", None)
        if callable(bind):
            for target in (emb, space):
                try:
                    bind(target)
                    break
                except Exception:
                    continue
    # if the adapter never got told about the checkpoint, point it now
    for attr in ("checkpoint", "checkpoint_path", "ckpt"):
        if getattr(pol, attr, None) in (None, ""):
            try:
                setattr(pol, attr, str(ckpt))
            except Exception:
                pass
    return pol


def _apply_step(ex, emb, step):
    import mujoco
    m, d = ex.model, ex.data
    for j, v in step.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jid < 0:
            continue
        a = ex.ctrl_for_jid.get(jid)
        if a is None:
            continue
        d.ctrl[a] = float(v)


def _held_out_start(ex):
    """A fixed non-keyframe stance: south-east of the island facing the
    bowl, arm joints perturbed — a start the demo set was not recorded
    from."""
    import mujoco
    m, d = ex.model, ex.data
    bowl = ex.pos("bowl")
    px, py = float(bowl[0]) + 0.55, float(bowl[1]) - 0.45
    yaw = math.atan2(float(bowl[1]) - py, float(bowl[0]) - px)
    d.qpos[0:3] = [px, py, 0.79]
    d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    d.qvel[:] = 0.0
    for name, dv in (("right_shoulder_pitch_joint", 0.15),
                     ("left_elbow_joint", -0.12),
                     ("right_wrist_roll_joint", 0.08)):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] += dv
    mujoco.mj_forward(m, d)


def _check_chunk(steps, space):
    assert steps is not None, (
        "act() returned nothing shaped like an action chunk — no "
        "per-step joint targets found")
    assert len(steps) >= MIN_CHUNK, (
        f"act() emitted {len(steps)} control step(s) — a VLA emits an "
        "action CHUNK of more than one future step")
    declared = set(space.joints)
    for i, st in enumerate(steps):
        assert st, f"chunk step {i} commands no joints"
        bad = sorted(set(st) - declared)
        assert not bad, (
            f"chunk step {i} commands joints outside the G1 joint_pos "
            f"contract: {bad}")
        for j, v in st.items():
            lo, hi = space.bounds[j]
            assert np.isfinite(v), (
                f"chunk step {i} joint '{j}' target is {v} — not "
                "finite")
            assert lo - BOUNDS_TOL <= v <= hi + BOUNDS_TOL, (
                f"chunk step {i} joint '{j}' target {v} is outside "
                f"its declared joint_pos bounds [{lo}, {hi}]")
    return steps


def test_pol_7_1_posttrained_chunks_drive_g1_to_measured_bowl_contact(
        tmp_path):
    import mujoco

    # ---- a post-trained GR00T checkpoint exists -------------------------
    released = _hf_gr00t_snapshot()
    rel_idx = _tensor_index(released) if released else {}
    scored = _find_posttrained_ckpt(rel_idx)
    assert scored, (
        "no post-trained GR00T checkpoint found — nothing under the "
        "finetune/gr00t declared paths or the artifact dirs carries "
        "action-head tensors")
    diff, overlap, ckpt = scored[0]
    if overlap:
        assert diff > 0, (
            f"the checkpoint at {ckpt} is bit-identical to the released"
            " GR00T N weights on every shared tensor — it was never "
            "post-trained")
    ckpt = Path(ckpt).resolve()

    # ---- outside git -----------------------------------------------------
    assert not _under(ckpt, REPO / ".git"), (
        f"the checkpoint {ckpt} resolves inside .git")
    if _under(ckpt, REPO):
        assert _git_ignored(ckpt), (
            f"the checkpoint {ckpt} sits inside the worktree on a "
            "tracked path — it must live outside git")
    tracked = [str(p) for p in ckpt.rglob("*")
               if p.is_file() and p.suffix.lower() in WEIGHT_EXT
               and _git_tracked(p)]
    assert not tracked, (
        f"checkpoint weight files are git-tracked: {tracked[:4]}")

    # ---- the held-out eval: chunks only, no scripted reach ---------------
    try:
        from harmless.executor import Executor
        from harmless.embodiments.g1 import G1Embodiment
    except Exception as exc:
        pytest.fail(f"executor/embodiment import failed: {exc}")
    ex = Executor(str(SCENE), record=False,
                  out_dir=str(tmp_path / "eval"), skills_path=None,
                  ws_dir=str(tmp_path / "ws"))
    emb = G1Embodiment(ex.model, ex.data)
    space = emb.action_space
    scripted_calls = []

    patched = []
    import harmless.executor as exmod
    for name in SCRIPTED:
        fn = getattr(exmod.Executor, name, None)
        if not callable(fn):
            continue
        def wrap(fn, name):
            def w(*a, **k):
                scripted_calls.append(name)
                return fn(*a, **k)
            return w
        patched.append((exmod.Executor, name, fn))
        setattr(exmod.Executor, name, wrap(fn, name))

    try:
        _held_out_start(ex)
        pol = _make_policy(ckpt, emb, space)
        order = list(getattr(pol, "action_joints", None)
                     or space.joints)
        steps_per_tick = max(1, int(round(CTRL_DT / ex.model.opt.timestep)))
        chunk, i = [], 0
        contact_at = None
        ticks = int(HORIZON_S / CTRL_DT)
        for tick in range(ticks):
            if i >= len(chunk):
                obs = _with_instruction(emb.observation(), SUBGOAL)
                t0 = float(ex.data.time)
                out = _call_act(pol, obs)
                assert float(ex.data.time) == t0, (
                    "act() advanced sim time — the checkpoint's "
                    "inference is stepping the sim itself")
                chunk = _steps_from(out, space.joints, order)
                if tick == 0:
                    _check_chunk(chunk, space)
                if not chunk:
                    break
                i = 0
            _apply_step(ex, emb, chunk[i])
            i += 1
            ex.step(steps_per_tick)
            if _palm_bowl_contact(ex):
                contact_at = float(ex.data.time)
                break
        assert not scripted_calls, (
            "a scripted reach was used in the eval loop — the contact "
            "must come from the post-trained network, not hand "
            f"trajectories: {scripted_calls[:6]}")
        assert contact_at is not None, (
            f"no measured palm-bowl contact after {HORIZON_S} s of "
            "sim — the post-trained checkpoint's joint_pos chunks "
            "never got a palm onto the bowl "
            f"(palm gaps now: {ex.palm_gaps('bowl')})")
    finally:
        for obj, attr, old in reversed(patched):
            try:
                setattr(obj, attr, old)
            except Exception:
                pass
        try:
            ex.close()
        except Exception:
            pass
