"""POL-2.2@1 A scripted subgoal (approach and touch the bowl) executes
end to end on the G1 adapter through the VLA.

The contract under test is the whole loop the line names: a SCRIPTED
subgoal — the fixed instruction "approach and touch the bowl" handed to
the served GR00T policy by this test — drives the G1 embodiment in the
real kitchen scene, and every actuator target reaching MuJoCo comes
from the VLA's emitted joint_pos chunks. Nothing is scripted except the
subgoal: no canned reach trajectory, no teleport, no weld.

The seam mirrors the eval pipeline (harmless.eval): the gr00t adapter
is resolved from harmless.policies.gr00t / the policy registry, bound
to the G1 adapter (harmless.embodiments.g1) on the real MjModel/MjData
of kitchen.xml, and stepped at the embodiment's declared 50 Hz control
rate — observation() in, action chunk out, each step applied through
Executor.set() onto the position actuators, physics stepped forward.
The subgoal is delivered through whichever input the adapter exposes:
a set_subgoal/set_instruction/instruct-style method, a pre-existing
instruction/subgoal attribute, an act(obs, instruction) parameter, or
the observation's instruction keys (all are tried; the instruction is
also injected into every observation).

Measured from simulation state only:

  through the VLA — act() ran and produced chunk steps; at least one
    emitted joint target was written to an actuator (targets that name
    no G1 joint count as rejected);
  approach — the measured palm-to-bowl-surface gap fell at least 0.20 m
    below its spawn value, or the pelvis closed at least 0.15 m on the
    bowl (the spawn pose has the nearest palm ~0.95 m from the bowl);
  touch — a real robot↔bowl contact was recorded in MjData.contact
    while the pelvis was above 0.5 m (an executed approach, not a
    collapse onto the worktop).

harmless/policies/ does not exist yet — failing to import it is the
expected pre-implementation failure.
"""

import importlib
import inspect
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]
SCENE = str(REPO / "kitchen.xml")
SUBGOAL = "approach and touch the bowl"
TARGET = "bowl"
SIM_BUDGET_S = 30.0
WALL_BUDGET_S = 900.0
MAX_ACT_CALLS = 200
APPROACH_MARGIN_M = 0.20
PELVIS_APPROACH_M = 0.15
CONTACT_PELVIS_MIN_Z = 0.5

POLICY_NAME = re.compile(r"gr00t|groot|isaac|vla|n1[._-]?5", re.I)
CHUNK_KEY = ("action_chunk", "actions", "chunk", "joint_pos", "targets",
             "trajectory", "traj", "horizon", "action", "steps", "plan",
             "position_targets", "joint_targets")
SUBGOAL_METHODS = ("set_subgoal", "set_instruction", "instruct",
                   "set_task", "set_goal", "command", "plan",
                   "submit_subgoal", "submit")
SUBGOAL_ATTRS = ("instruction", "subgoal", "task", "goal", "command",
                 "prompt", "lang_instruction", "language_instruction")
OBS_KEYS = ("instruction", "subgoal", "task", "prompt", "goal",
            "language_instruction", "command")


def _import(name):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        pytest.fail(
            f"{name} cannot be imported ({type(exc).__name__}: {exc}) "
            "— the gr00t policy adapter does not exist yet")


def _adapter_cls(gr00t_mod):
    try:
        from harmless import registry
    except Exception:
        registry = None
    if registry is not None:
        names = []
        try:
            names = list(registry.list_policies())
        except Exception:
            for attr in ("POLICIES", "policies", "POLICY_REGISTRY"):
                reg = getattr(registry, attr, None)
                if isinstance(reg, dict):
                    names = list(reg)
                    break
        for n in names:
            if not POLICY_NAME.search(str(n)):
                continue
            try:
                return registry.get_policy(n)
            except Exception:
                for attr in ("POLICIES", "policies", "POLICY_REGISTRY"):
                    reg = getattr(registry, attr, None)
                    if isinstance(reg, dict) and reg.get(n) is not None:
                        return reg[n]
    try:
        from harmless.policy import Policy
        base = Policy
    except Exception:
        base = None
    for name, obj in sorted(vars(gr00t_mod).items()):
        if inspect.isclass(obj) and obj.__module__ == gr00t_mod.__name__:
            if base is None or issubclass(obj, base):
                return obj
    for name, obj in sorted(vars(gr00t_mod).items()):
        if inspect.isclass(obj) and callable(getattr(obj, "act", None)):
            return obj
    pytest.fail(
        "no gr00t policy adapter found — harmless.registry has no "
        "gr00t/vla-named policy and harmless.policies.gr00t declares "
        "no Policy subclass")


def _executor(tmp_path):
    try:
        from harmless.executor import Executor
    except Exception as exc:
        pytest.fail(
            f"harmless.executor cannot be imported "
            f"({type(exc).__name__}: {exc})")
    try:
        return Executor(SCENE, record=False,
                        out_dir=str(tmp_path / "episode"),
                        skills_path=None,
                        ws_dir=str(tmp_path / "ws"))
    except Exception as exc:
        pytest.fail(
            f"the kitchen scene failed to load in the Executor "
            f"({type(exc).__name__}: {exc})")


def _instantiate(cls, emb, space):
    tries = [
        ((), {"embodiment": emb}),
        ((), {"embodiment": emb, "action_space": space}),
        ((), {"action_space": space}),
        ((), {"embodiment": emb, "instruction": SUBGOAL}),
        ((), {"instruction": SUBGOAL}),
        ((), {"subgoal": SUBGOAL}),
        ((), {"task": SUBGOAL}),
        ((emb,), {}),
        ((space,), {}),
        ((), {}),
    ]
    errs = []
    for args, kw in tries:
        try:
            pol = cls(*args, **kw)
            break
        except Exception as exc:
            errs.append(f"{args} {kw}: {type(exc).__name__}: {exc}")
    else:
        pytest.fail(
            "the gr00t adapter could not be constructed/bound to the G1 "
            "contract:\n" + "\n".join(errs[:6]))
    if getattr(pol, "action_space", None) is None:
        bind = getattr(pol, "bind", None)
        if callable(bind):
            for target in (emb, space):
                try:
                    bind(target)
                    break
                except Exception:
                    continue
    return pol


def _deliver_subgoal(pol):
    """Script the subgoal into the adapter through whatever input it
    exposes. Returns the seams used."""
    used = []
    for meth in SUBGOAL_METHODS:
        f = getattr(pol, meth, None)
        if not callable(f):
            continue
        for arg in (SUBGOAL, {"instruction": SUBGOAL},
                    {"subgoal": SUBGOAL}, (TARGET,)):
            try:
                f(arg)
                used.append(f"{meth}({arg!r})")
                break
            except Exception:
                continue
    for attr in SUBGOAL_ATTRS:
        if hasattr(pol, attr) or hasattr(type(pol), attr):
            try:
                setattr(pol, attr, SUBGOAL)
                used.append(f"attr:{attr}")
            except Exception:
                pass
    try:
        sig = inspect.signature(getattr(pol, "act"))
        for p in sig.parameters.values():
            if re.search(r"instruct|subgoal|task|goal|command|prompt",
                         p.name, re.I):
                used.append(f"act kwarg:{p.name}")
                break
    except (TypeError, ValueError):
        pass
    return used


def _call_act(pol, obs):
    act = getattr(pol, "act", None)
    if not callable(act):
        if callable(pol):
            return pol(obs)
        pytest.fail("the gr00t adapter exposes no act(observation) call")
    for args, kw in (((obs,), {}),
                     ((obs,), {"instruction": SUBGOAL}),
                     ((obs,), {"subgoal": SUBGOAL}),
                     ((obs, SUBGOAL), {})):
        try:
            return act(*args, **kw)
        except TypeError:
            continue
    return act(obs)


def _scalar(v):
    try:
        float(v)
        return True
    except Exception:
        return False


def _steps_from(out, joints, joint_order=None, depth=0):
    if out is None or depth > 4:
        return None
    joints = list(joints)
    jset = set(joints)
    order = list(joint_order) if joint_order else joints
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
        for k in CHUNK_KEY:
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
    if arr is not None and arr.ndim == 1 and arr.size == len(joints):
        return [dict(zip(joints, arr))]
    for attr in CHUNK_KEY:
        v = getattr(out, attr, None)
        if v is not None:
            r = _steps_from(v, joints, order, depth + 1)
            if r:
                return r
    return None


def _joint_order(pol, space):
    for attr in ("action_joints", "joints", "joint_names", "action_names",
                 "action_dims", "controlled_joints"):
        v = getattr(pol, attr, None)
        if isinstance(v, (list, tuple)) and v \
                and all(isinstance(x, str) for x in v):
            return list(v)
    return list(getattr(space, "joints", []) or [])


def _bowl_contacts(ex):
    out = []
    for c in ex.contacts():
        fields = f"{c.get('other', '')} {c.get('other_body', '')}".lower()
        if "bowl" in fields:
            out.append(c)
    return out


def _palm_gap(ex):
    try:
        return min(ex.palm_gaps(TARGET).values())
    except Exception:
        pass
    import mujoco
    lo = np.array(ex.geom_info(TARGET)["aabb_min"])
    hi = np.array(ex.geom_info(TARGET)["aabb_max"])
    gaps = []
    for s in ("left", "right"):
        sid = mujoco.mj_name2id(ex.model, mujoco.mjtObj.mjOBJ_SITE,
                                f"{s}_palm")
        if sid >= 0:
            p = ex.data.site_xpos[sid]
            gaps.append(float(np.linalg.norm(p - np.clip(p, lo, hi))))
    return min(gaps)


def test_pol_2_2_scripted_subgoal_approach_touch_bowl_e2e(tmp_path):
    gr00t_mod = _import("harmless.policies.gr00t")
    _import("harmless.policies.serve")
    cls = _adapter_cls(gr00t_mod)

    ex = _executor(tmp_path)
    try:
        g1_mod = _import("harmless.embodiments.g1")
        emb = g1_mod.G1Embodiment(ex.model, ex.data)
        space = emb.action_space
        assert space is not None and space.joints, (
            "the G1 adapter declares no joint_pos action space")
        pol = _instantiate(cls, emb, space)
        _deliver_subgoal(pol)
        order = _joint_order(pol, space)
        rate = float(getattr(space, "rate_hz", 0) or 50.0)
        ctrl = max(1, int(round(1.0 / (rate * ex.model.opt.timestep))))

        bowl_pos = ex.pos(TARGET).copy()
        gap0 = _palm_gap(ex)
        pel0 = float(np.linalg.norm(
            (ex.data.xpos[ex.pelvis_body] - bowl_pos)[:2]))
        min_gap, min_pel = gap0, pel0
        act_calls = applied = rejected = 0
        queue, contacts = [], []
        contact_z = None
        t0 = time.monotonic()
        sim_end = float(ex.data.time) + SIM_BUDGET_S

        while ex.data.time < sim_end and contact_z is None:
            if time.monotonic() - t0 > WALL_BUDGET_S \
                    or act_calls > MAX_ACT_CALLS:
                break
            if not queue:
                obs = emb.observation()
                for k in OBS_KEYS:
                    obs[k] = SUBGOAL
                try:
                    out = _call_act(pol, obs)
                except Exception as exc:
                    pytest.fail(
                        f"the VLA's act() raised {type(exc).__name__}: "
                        f"{exc} mid-episode")
                act_calls += 1
                steps = _steps_from(out, space.joints, order)
                assert steps, (
                    "the VLA returned no joint_pos chunk for the "
                    "scripted subgoal — act() produced nothing "
                    "applicable to the G1 contract")
                queue.extend(steps)
            step = queue.pop(0)
            for j, v in step.items():
                try:
                    ex.set(j, float(v))
                    applied += 1
                except Exception:
                    rejected += 1
            ex.step(ctrl)
            min_gap = min(min_gap, _palm_gap(ex))
            min_pel = min(min_pel, float(np.linalg.norm(
                (ex.data.xpos[ex.pelvis_body] - bowl_pos)[:2])))
            hits = _bowl_contacts(ex)
            if hits:
                contacts.extend(hits)
                contact_z = float(ex.data.xpos[ex.pelvis_body][2])
    finally:
        ex.close()

    assert act_calls >= 1 and applied > 0, (
        f"no VLA action reached the robot: {act_calls} act() calls, "
        f"{applied} applied and {rejected} rejected joint targets — "
        "the subgoal did not execute THROUGH the VLA")
    approached = (min_gap <= gap0 - APPROACH_MARGIN_M
                  or min_pel <= pel0 - PELVIS_APPROACH_M)
    assert approached, (
        "no approach was measured: palm-to-bowl gap went from "
        f"{gap0:.3f} m to {min_gap:.3f} m and pelvis-to-bowl from "
        f"{pel0:.3f} m to {min_pel:.3f} m over "
        f"{SIM_BUDGET_S:.0f} s of VLA control")
    assert contacts, (
        "the robot never touched the bowl — no robot↔bowl contact "
        f"appeared in MjData over {SIM_BUDGET_S:.0f} s "
        f"(approach did happen: gap {gap0:.3f} -> {min_gap:.3f} m)")
    assert contact_z >= CONTACT_PELVIS_MIN_Z, (
        f"the bowl contact happened with the pelvis at z={contact_z:.2f}"
        " m — the robot collapsed onto the worktop instead of "
        "executing an approach-and-touch")
