"""EMB-2.2@1 The scripted policy completes a tabletop pick on the Franka
adapter in under 60 simulated seconds.

The contract under test is a scripted pick driven through the embodiment
slot: a Franka tabletop scene under scenes/ (the requirement's touched
assets), the Franka adapter bound to it, and the scripted policy —
resolved from the policy registry or from policy-shaped callables the
Franka module ships — run in the same loop the eval harness uses:
`observation()` in, joint targets out, each target clamped into the
declared bounds and written to its actuator, physics stepped at the
adapter's declared rate_hz.

Success is measured from simulation state only (K-1): a scene object
body — a free-joint body that is not part of the robot's actuated chain —
must be lifted at least 4 cm above its resting height while in contact
with the robot's actuated bodies (a real grasp: contacts are solver
state, so weld-snaps and teleports do not count, K-2), or brought to
rest at least 4 cm above its start after such a hold, or brought to rest
at a site/body/geom named like a goal/target once a hold or lift was
seen. The pick must first register while the simulation clock is still
under 60.0 s.

`harmless/embodiments/franka.py` and `scenes/` do not exist yet —
failing to import the module is the expected pre-implementation failure.
The rollout uses raw mujoco because the kitchen Executor is wired to the
G1 scene; what is exercised here is the adapter/policy contract. Policy
candidates are tried in order — pick/franka/script-named policies from
`harmless.registry` first, then Policy subclasses and pick-named
factories defined in the Franka module, then every other registered
policy — and the rollout stops at the first measured pick.
"""

import importlib
import inspect
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[2]

SIM_BUDGET_S = 60.0
LIFT_M = 0.04
REST_SPEED = 0.05
GOAL_TOL_M = 0.15
WALL_BUDGET_S = 240.0
MAX_SCENES = 3
MAX_POLICIES = 8

FRANKA_NAME = re.compile(r"franka|panda|emika", re.I)
PICK_NAME = re.compile(r"pick|grasp|lift|tabletop|script|franka|panda",
                       re.I)
GOAL_NAME = re.compile(r"goal|target|destin|place|drop|bin|pad|zone|"
                       r"marker|dock", re.I)
ROBOT_NAME = re.compile(r"franka|panda|link|joint|finger|hand|grip|arm",
                        re.I)
APPLY_NAMES = ("apply", "apply_action", "apply_targets", "set_joints",
               "set_targets", "send", "command", "actuate", "write_ctrl",
               "write", "drive")


def _mujoco():
    import mujoco
    return mujoco


def _franka_module():
    try:
        return importlib.import_module("harmless.embodiments.franka")
    except Exception as exc:
        pytest.fail(
            f"harmless.embodiments.franka cannot be imported "
            f"({type(exc).__name__}: {exc}) — the Franka adapter does "
            f"not exist yet")


def _adapter_entries(mod):
    """(label, class-or-instance) for every Franka adapter reachable."""
    out = []
    try:
        from harmless import registry
        try:
            names = list(registry.list_embodiments())
        except Exception:
            names = list(getattr(registry, "EMBODIMENTS", {}) or {})
        for n in names:
            if not FRANKA_NAME.search(str(n)):
                continue
            try:
                out.append((f"registry:{n}", registry.get_embodiment(n)))
            except Exception:
                reg = getattr(registry, "EMBODIMENTS", {})
                if isinstance(reg, dict) and reg.get(n) is not None:
                    out.append((f"registry:{n}", reg[n]))
    except Exception:
        pass
    try:
        from harmless.embodiment import Embodiment
        base = Embodiment
    except Exception:
        base = None
    for name, obj in sorted(vars(mod).items()):
        if not inspect.isclass(obj):
            continue
        if base is not None and issubclass(obj, base) and obj is not base:
            out.append((f"franka.{name}", obj))
    seen, dedup = set(), []
    for label, cls in out:
        if id(cls) not in seen:
            seen.add(id(cls))
            dedup.append((label, cls))
    return dedup


def _scene_paths():
    out = []
    for d in (REPO / "scenes", REPO / "assets" / "franka"):
        if d.is_dir():
            out += sorted(d.rglob("*.xml"))
    for p in sorted(REPO.rglob("*.xml")):
        if FRANKA_NAME.search(p.name) or FRANKA_NAME.search(p.parent.name):
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _bind_adapter(cls, model, data):
    if not inspect.isclass(cls) and not callable(cls):
        emb = cls
        try:
            emb.model, emb.data = model, data
        except Exception:
            pass
        return emb
    for call in (lambda: cls(model, data),
                 lambda: cls(model=model, data=data),
                 lambda: cls(model)):
        try:
            emb = call()
            for attr, v in (("model", model), ("data", data)):
                if getattr(emb, attr, None) is None:
                    try:
                        setattr(emb, attr, v)
                    except Exception:
                        pass
            return emb
        except Exception:
            continue
    raise TypeError(f"{cls!r} would not bind to a MjModel/MjData pair")


def _space_of(emb):
    space = getattr(emb, "action_space", None)
    if space is None and getattr(emb, "joints", None) is not None:
        space = emb
    return space


def _robot_bodies(model):
    act = set()
    for a in range(model.nu):
        jid = int(model.actuator_trnid[a, 0])
        if jid >= 0:
            act.add(int(model.jnt_bodyid[jid]))
    robot = set()
    for b in range(1, model.nbody):
        bb = b
        while bb > 0:
            if bb in act:
                robot.add(b)
                break
            bb = int(model.body_parentid[bb])
    return robot


def _object_bodies(model, robot):
    mujoco = _mujoco()
    out = []
    for b in range(1, model.nbody):
        if b in robot:
            continue
        for k in range(int(model.body_jntnum[b])):
            j = int(model.body_jntadr[b]) + k
            if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                out.append(b)
                break
    return out


def _object_names(model, bodies):
    mujoco = _mujoco()
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
            for b in bodies]


def _goal_points(model, data, robot):
    mujoco = _mujoco()
    pts = []
    kinds = ((mujoco.mjtObj.mjOBJ_SITE, model.nsite, data.site_xpos),
             (mujoco.mjtObj.mjOBJ_GEOM, model.ngeom, data.geom_xpos),
             (mujoco.mjtObj.mjOBJ_BODY, model.nbody, data.xpos))
    for kind, count, pos in kinds:
        for i in range(count):
            name = mujoco.mj_id2name(model, kind, i)
            if not name or not GOAL_NAME.search(name):
                continue
            if ROBOT_NAME.search(name):
                continue
            if kind == mujoco.mjtObj.mjOBJ_GEOM \
                    and int(model.geom_bodyid[i]) in robot:
                continue
            if kind == mujoco.mjtObj.mjOBJ_BODY and i in robot:
                continue
            pts.append(pos[i].copy())
    return pts


def _held(model, data, obj, robot):
    for c in data.contact[: int(data.ncon)]:
        b1 = int(model.geom_bodyid[int(c.geom1)])
        b2 = int(model.geom_bodyid[int(c.geom2)])
        if (b1 == obj and b2 in robot) or (b2 == obj and b1 in robot):
            return True
    return False


def _actuator_for(model, name):
    mujoco = _mujoco()
    name = str(name)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid >= 0:
        for a in range(model.nu):
            if int(model.actuator_trnid[a, 0]) == jid:
                return a
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid >= 0:
        return aid
    for a in range(model.nu):
        an = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        if an and (an in name or name in an):
            return a
    if re.search(r"grip|finger|hand|grasp|claw", name, re.I):
        for a in range(model.nu):
            an = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
            jid = int(model.actuator_trnid[a, 0])
            jn = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, jid) or "" if jid >= 0 \
                else ""
            if re.search(r"grip|finger|hand|grasp|claw", an + " " + jn,
                         re.I):
                return a
    return -1


def _apply(model, data, emb, space, action):
    import numpy as np
    joints = list(getattr(space, "joints", None) or [])
    if not isinstance(action, Mapping):
        try:
            vals = list(action)
        except TypeError:
            return 0
        if joints and len(vals) == len(joints):
            action = dict(zip(joints, vals))
        else:
            return 0
    for name in APPLY_NAMES:
        f = getattr(emb, name, None)
        if callable(f):
            try:
                f(dict(action))
                return len(action)
            except Exception:
                break
    bounds = getattr(space, "bounds", None) or {}
    applied = 0
    for j, v in dict(action).items():
        aid = _actuator_for(model, j)
        if aid < 0:
            continue
        try:
            b = bounds.get(j)
            lo, hi = (float(b[0]), float(b[1])) if b is not None \
                else (-np.inf, np.inf)
            data.ctrl[aid] = float(np.clip(float(v), lo, hi))
            applied += 1
        except Exception:
            continue
    return applied


def _act(pol, obs, t):
    act = getattr(pol, "act", None)
    cands = []
    if callable(act):
        cands = [lambda: act(obs), lambda: act(obs, t), lambda: act()]
    elif callable(pol):
        cands = [lambda: pol(obs), lambda: pol(obs, t), lambda: pol()]
    elif hasattr(pol, "__next__"):
        cands = [lambda: next(pol)]
    err = None
    for call in cands:
        try:
            return call()
        except TypeError as e:
            err = e
    if err is not None:
        raise err
    raise TypeError(f"{pol!r} is not callable and has no act()")


def _bind_policy(cls, emb, obj_names):
    if not inspect.isclass(cls):
        pol = cls
    else:
        pol = None
        attempts = [lambda: cls(emb), lambda: cls(embodiment=emb),
                    lambda: cls()]
        for obj in obj_names[:1]:
            attempts += [
                lambda o=obj: cls(emb, o),
                lambda o=obj: cls(emb, obj=o),
                lambda o=obj: cls(emb, target=o),
                lambda o=obj: cls(emb, object=o)]
        for call in attempts:
            try:
                pol = call()
                break
            except Exception:
                continue
        if pol is None:
            raise TypeError(f"{cls!r} could not be constructed")
    bind = getattr(pol, "bind", None)
    if callable(bind):
        try:
            bind(emb)
        except Exception:
            pass
    return pol


def _policy_candidates(mod, emb, obj_names):
    out, seen = [], set()

    def add(label, cls):
        if id(cls) in seen:
            return
        seen.add(id(cls))
        out.append((label, cls))

    try:
        from harmless import registry
        try:
            pnames = list(registry.list_policies())
        except Exception:
            pnames = list(getattr(registry, "POLICIES", {}) or {})
        ranked = []
        for n in pnames:
            try:
                cls = registry.get_policy(n)
            except Exception:
                reg = getattr(registry, "POLICIES", {})
                cls = reg.get(n) if isinstance(reg, dict) else None
            if cls is None or "agent" in str(n).lower():
                continue
            ranked.append((0 if PICK_NAME.search(str(n)) else 1,
                           f"policy '{n}'", cls))
        for _, label, cls in sorted(ranked, key=lambda x: x[0]):
            add(f"registry {label}", cls)
    except Exception:
        pass

    try:
        from harmless.policy import Policy
        pbase = Policy
    except Exception:
        pbase = None
    try:
        from harmless.embodiment import Embodiment
        ebase = Embodiment
    except Exception:
        ebase = None
    for name, obj in sorted(vars(mod).items()):
        if inspect.isclass(obj) and obj.__module__ == mod.__name__:
            if ebase is not None and issubclass(obj, ebase):
                continue
            if (pbase is not None and issubclass(obj, pbase)
                    and obj is not pbase) or PICK_NAME.search(name) \
                    or callable(getattr(obj, "act", None)):
                add(f"franka.{name}", obj)
        elif callable(obj) and PICK_NAME.search(name) \
                and not name.startswith("_"):
            add(f"franka.{name}", obj)
            for call in (lambda f=obj: f(emb), lambda f=obj: f()):
                try:
                    pol = call()
                except Exception:
                    continue
                if pol is not None and pol is not obj:
                    add(f"franka.{name}()", pol)
                    break
    return out


def _rollout(model, data, emb, pol, space):
    import numpy as np
    mujoco = _mujoco()
    rate = float(getattr(space, "rate_hz", 0) or 0) or 50.0
    sub = max(1, int(round(1.0 / (rate * float(model.opt.timestep)))))
    robot = _robot_bodies(model)
    objects = _object_bodies(model, robot)
    if not objects:
        return {"picked": False, "reason": "no free-joint object body "
                    "found in the scene — nothing to pick"}
    z0 = {b: float(data.xpos[b][2]) for b in objects}
    goals = _goal_points(model, data, robot)
    max_lift, ever_held, armed = 0.0, False, False
    applied_any = False
    wall0 = time.monotonic()
    while float(data.time) < SIM_BUDGET_S:
        if time.monotonic() - wall0 > WALL_BUDGET_S:
            return {"picked": False,
                    "reason": "rollout exceeded wall budget",
                    "max_lift": round(max_lift, 4)}
        try:
            obs = emb.observation() if callable(
                getattr(emb, "observation", None)) else {}
        except Exception:
            obs = {}
        try:
            action = _act(pol, obs, float(data.time))
        except Exception as e:
            return {"picked": False, "reason": f"policy act() raised "
                    f"{type(e).__name__}: {e} at "
                    f"t={float(data.time):.2f}",
                    "max_lift": round(max_lift, 4),
                    "ever_held": ever_held}
        applied_any = applied_any or bool(
            _apply(model, data, emb, space, action))
        if not applied_any and float(data.time) > 5.0:
            return {"picked": False,
                    "reason": "policy produced no applicable joint "
                              "targets in the first 5 sim seconds",
                    "max_lift": round(max_lift, 4)}
        mujoco.mj_step(model, data, int(sub))
        for b in objects:
            lift = float(data.xpos[b][2]) - z0[b]
            max_lift = max(max_lift, lift)
            held = _held(model, data, b, robot)
            ever_held = ever_held or held
            armed = armed or held or lift >= LIFT_M
            resting = float(np.linalg.norm(data.cvel[b][3:])) < REST_SPEED
            t = float(data.time)
            if lift >= LIFT_M and (held or (resting and ever_held)):
                return {"picked": True, "t": t, "how": "lifted",
                        "lift": round(lift, 4), "held": held,
                        "obj": int(b)}
            if armed and resting and any(
                    float(np.linalg.norm(data.xpos[b] - g)) < GOAL_TOL_M
                    for g in goals):
                return {"picked": True, "t": t, "how": "at-goal",
                        "lift": round(lift, 4), "obj": int(b)}
    return {"picked": False,
            "reason": "no measured pick inside the 60 s budget",
            "max_lift": round(max_lift, 4), "ever_held": ever_held}


def _scenes():
    mujoco = _mujoco()
    out = []
    for path in _scene_paths():
        try:
            model = mujoco.MjModel.from_xml_path(str(path))
        except Exception:
            continue
        out.append((path, model))
    return out


def test_emb_2_2_scripted_policy_tabletop_pick_under_60_sim_seconds():
    mod = _franka_module()
    adapters = _adapter_entries(mod)
    assert adapters, (
        "no Franka adapter found — neither a registered embodiment named "
        "like franka/panda nor an Embodiment subclass in "
        "harmless/embodiments/franka.py")
    scenes = _scenes()
    assert scenes, (
        "no loadable MJCF scene under scenes/ or assets/franka/ — the "
        "tabletop pick scene does not exist yet")

    mujoco = _mujoco()
    report = []
    deadline = time.monotonic() + 900.0
    for path, model in scenes[:MAX_SCENES]:
        for alabel, cls in adapters:
            data0 = mujoco.MjData(model)
            try:
                emb0 = _bind_adapter(cls, model, data0)
            except Exception as e:
                report.append(f"{path.name} {alabel}: adapter would not "
                              f"bind ({type(e).__name__}: {e})")
                continue
            space = _space_of(emb0)
            if space is None or not list(
                    getattr(space, "joints", None) or []):
                report.append(f"{path.name} {alabel}: no declared "
                              "action_space joints")
                continue
            try:
                mujoco.mj_forward(model, data0)
            except Exception:
                pass
            robot = _robot_bodies(model)
            obj_names = _object_names(
                model, _object_bodies(model, robot))
            try:
                pols = _policy_candidates(mod, emb0, obj_names)
            except Exception as e:
                pols = []
                report.append(f"policy discovery failed: "
                              f"{type(e).__name__}: {e}")
            if not pols:
                report.append("no scripted pick policy found in the "
                              "registry or the Franka module")
            for plabel, pcls in pols[:MAX_POLICIES]:
                data = mujoco.MjData(model)
                try:
                    emb = _bind_adapter(cls, model, data)
                    pol = _bind_policy(pcls, emb, obj_names)
                except Exception as e:
                    report.append(f"{path.name} {plabel}: could not "
                                  f"construct/bind "
                                  f"({type(e).__name__}: {e})")
                    continue
                res = _rollout(model, data, emb, pol,
                               _space_of(emb) or space)
                if res.get("picked") and res.get("t", 1e9) < SIM_BUDGET_S:
                    return
                report.append(f"{path.name} {plabel}: {res}")
                if time.monotonic() > deadline:
                    report.append("stopped at the overall wall budget")
                    pytest.fail(
                        "the scripted policy never completed a measured "
                        "tabletop pick on the Franka adapter inside 60 "
                        "simulated seconds:\n" + "\n".join(report[:16]))
    pytest.fail(
        "the scripted policy never completed a measured tabletop pick on "
        "the Franka adapter inside 60 simulated seconds:\n"
        + "\n".join(report[:16]))
