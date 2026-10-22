"""CORE-2.1@1 A policy class exposes `act(observation)` returning actions
matching the paired embodiment's action_space.

The contract under test is the policy slot itself: a class reachable
through `harmless.registry` (a name -> class mapping, a callable
returning one, or a name list resolvable through a lookup function) or
declared in `harmless/policy.py` (or a `harmless/policies/` package) must
expose a callable `act(observation)`. Paired with an embodiment — an
adapter resolved through the same registry and bound to the kitchen
scene's MjModel/MjData pair — the policy must answer a real
`observation()` with an action that is a member of the embodiment's
declared `action_space`: joint targets addressed to declared joint
names, every one numeric and inside its declared (lo, hi) bound.

`harmless/policy.py` does not exist yet and the registry exposes no
policy slot — finding no candidate class is the expected
pre-implementation failure. Construction is probed as Cls(embodiment),
Cls(action_space), Cls(model, data), the keyword equivalents
(embodiment=/action_space=), and Cls(), followed by an optional
bind/attach/pair(embodiment) hook; some construction must succeed for
`act` to be exercised. The action may be a joint -> target mapping
(partial coverage counts: `ActionSpace.clamp` accepts subsets), a
sequence of length len(joints) parallel to the declared joint order, a
list of (joint, value) pairs, or any of those wrapped once under a
target-like field (targets/joint_pos/joints/values/action/positions/
qpos). A real `embodiment.observation()` is fed to `act`; if no bound
embodiment can produce one, a synthetic observation carrying the
declared joints is used instead.
"""

import importlib
import inspect
import os
import pkgutil
import signal
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[2]

NAME_KEYS = ("joints", "joint_names", "names", "actuators", "dof_names")
BOUND_KEYS = ("bounds", "limits", "joint_bounds", "ranges")
TARGET_KEYS = (
    "targets", "joint_pos", "joints", "values", "action",
    "positions", "qpos", "commands",
)
POLICY_REGISTRY_ATTRS = (
    "POLICIES", "policies", "POLICY_REGISTRY", "policy_registry",
    "list_policies", "registry",
)
POLICY_LOOKUP_FNS = ("get_policy", "policy", "get", "resolve", "lookup")
EMBODIMENT_REGISTRY_ATTRS = (
    "EMBODIMENTS", "embodiments", "EMBODIMENT_REGISTRY",
    "embodiment_registry", "list_embodiments",
)
EMBODIMENT_LOOKUP_FNS = (
    "get_embodiment", "embodiment", "get", "resolve", "lookup",
)
BIND_FNS = ("bind", "attach", "pair", "set_embodiment")

ACT_TIMEOUT_S = 30


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout("act() did not return within the time budget")


def _field(space, keys):
    names = list(keys) + [k.upper() for k in keys]
    if isinstance(space, Mapping):
        for k in names:
            if k in space:
                return space[k]
    for k in names:
        if hasattr(space, k):
            return getattr(space, k)
    return None


def _joint_names(space):
    names = _field(space, NAME_KEYS)
    if names is None:
        return None
    if isinstance(names, Mapping):
        names = list(names)
    try:
        names = list(names)
    except TypeError:
        return None
    if not names or not all(isinstance(n, str) and n for n in names):
        return None
    return names


def _bounds_map(space, joints):
    bounds = _field(space, BOUND_KEYS)
    if bounds is None or isinstance(bounds, (str, bytes)):
        return None
    if isinstance(bounds, Mapping):
        if not all(j in bounds for j in joints):
            return None
        pairs = [bounds[j] for j in joints]
    else:
        try:
            pairs = list(bounds)
        except TypeError:
            return None
        if len(pairs) != len(joints):
            return None
    out = {}
    for j, p in zip(joints, pairs):
        try:
            lo, hi = float(p[0]), float(p[1])
        except Exception:
            return None
        if not lo < hi:
            return None
        out[j] = (lo, hi)
    return out


def _numeric_scalar(value):
    if isinstance(value, (bool, str, bytes)):
        return False
    try:
        a = np.asarray(value)
    except Exception:
        return False
    if a.size != 1 or not np.issubdtype(a.dtype, np.number):
        return False
    try:
        float(value)
    except Exception:
        return False
    return True


def _unwrap_targets(action, joint_set, depth=0):
    if action is None or depth > 2:
        return None
    if hasattr(action, "_asdict"):
        return _unwrap_targets(action._asdict(), joint_set, depth + 1)
    if isinstance(action, Mapping):
        if any(k in joint_set for k in action):
            return action
        for key in TARGET_KEYS:
            if key in action:
                inner = _unwrap_targets(action[key], joint_set, depth + 1)
                if inner is not None:
                    return inner
        return action
    if isinstance(action, (list, tuple)):
        if action and all(
            isinstance(p, (list, tuple)) and len(p) == 2
            and isinstance(p[0], str)
            for p in action
        ):
            return dict(action)
        return action
    for key in TARGET_KEYS:
        if hasattr(action, key):
            v = getattr(action, key)
            if not callable(v):
                inner = _unwrap_targets(v, joint_set, depth + 1)
                if inner is not None:
                    return inner
    try:
        a = np.asarray(action)
    except Exception:
        return None
    if a.ndim == 1 and a.size > 0 and np.issubdtype(a.dtype, np.number):
        return a
    return None


def _action_matches(action, space, joints):
    pairs = _bounds_map(space, joints)
    if pairs is None:
        return "the paired action_space's bounds cannot be read as joint -> (lo, hi)"
    joint_set = set(joints)
    tgt = _unwrap_targets(action, joint_set)
    if tgt is None:
        return (
            f"act() returned {type(action).__name__} carrying no joint "
            f"targets"
        )
    if isinstance(tgt, Mapping):
        if not tgt:
            return "act() returned an empty target mapping (no joint addressed)"
        unknown = [k for k in tgt if k not in joint_set]
        if unknown:
            return f"act() targets undeclared joints: {unknown[:4]}"
        bad = [
            k for k, v in tgt.items()
            if not _numeric_scalar(v)
            or not (pairs[k][0] - 1e-9 <= float(v) <= pairs[k][1] + 1e-9)
        ]
        if bad:
            return (
                f"act() targets missing/outside declared bounds for "
                f"joints: {bad[:4]}"
            )
        return None
    try:
        arr = np.asarray(tgt, dtype=float).ravel()
    except Exception:
        return "act() returned a non-numeric action"
    if arr.size != len(joints):
        return (
            f"act() returned {arr.size} target(s); the space declares "
            f"{len(joints)} joints"
        )
    for j, v in zip(joints, arr):
        lo, hi = pairs[j]
        if not (lo - 1e-9 <= float(v) <= hi + 1e-9):
            return f"act() target {j}={float(v)} outside [{lo}, {hi}]"
    return None


def _lookup(reg, name, fns):
    for fn in fns:
        f = getattr(reg, fn, None)
        if callable(f):
            try:
                c = f(name)
            except Exception:
                continue
            return c if inspect.isclass(c) else type(c)
    return None


def _resolve_registry_value(reg, obj, fns):
    if inspect.isclass(obj):
        return [obj]
    val = obj
    if callable(obj):
        try:
            val = obj()
        except Exception:
            return []
    out = []
    if isinstance(val, Mapping):
        entries = val.values()
    elif isinstance(val, (list, tuple, set)):
        entries = val
    else:
        return out
    for item in entries:
        if inspect.isclass(item):
            out.append(item)
        elif isinstance(item, str):
            cls = _lookup(reg, item, fns)
            if cls is not None:
                out.append(cls)
        elif callable(item):
            try:
                produced = item()
            except Exception:
                continue
            out.append(
                produced if inspect.isclass(produced) else type(produced))
        else:
            out.append(type(item))
    return out


def _registry_classes(attrs, fns):
    try:
        reg = importlib.import_module("harmless.registry")
    except Exception:
        return []
    classes = []
    for attr in attrs:
        obj = getattr(reg, attr, None)
        if obj is not None:
            classes += _resolve_registry_value(reg, obj, fns)
    return classes


def _scan_modules(names):
    classes = []
    for modname in names:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        classes += [o for o in vars(mod).values() if inspect.isclass(o)]
    return classes


def _policy_modules():
    names = ["harmless.policy", "harmless.registry"]
    try:
        pkg = importlib.import_module("harmless.policies")
    except Exception:
        return names
    names.append("harmless.policies")
    for m in pkgutil.iter_modules(pkg.__path__):
        names.append(f"harmless.policies.{m.name}")
    return names


def _dedupe(classes):
    seen, out = set(), []
    for c in classes:
        if id(c) not in seen:
            seen.add(id(c))
            out.append(c)
    return out


def _policy_classes():
    classes = _registry_classes(POLICY_REGISTRY_ATTRS, POLICY_LOOKUP_FNS)
    classes += [
        c for c in _scan_modules(_policy_modules())
        if callable(getattr(c, "act", None))
    ]
    return _dedupe(classes)


def _embodiment_modules():
    names = ["harmless.embodiment", "harmless.registry"]
    try:
        pkg = importlib.import_module("harmless.embodiments")
    except Exception:
        return names
    names.append("harmless.embodiments")
    for m in pkgutil.iter_modules(pkg.__path__):
        names.append(f"harmless.embodiments.{m.name}")
    return names


def _embodiment_classes():
    classes = _registry_classes(
        EMBODIMENT_REGISTRY_ATTRS, EMBODIMENT_LOOKUP_FNS)
    classes += _scan_modules(_embodiment_modules())
    return _dedupe(classes)


def _construct(cls, *argsets):
    errors = []
    for args in argsets:
        try:
            return cls(*args)
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
    raise TypeError("; ".join(errors))


def _space_of(cls, emb):
    space = getattr(cls, "action_space", None)
    if space is None and emb is not None:
        space = getattr(emb, "action_space", None)
    return space


def _bound_embodiment(model, data):
    errors = []
    for cls in _embodiment_classes():
        try:
            emb = _construct(cls, (model, data), (model,), ())
        except Exception as e:
            errors.append(f"{cls.__name__}: {e}")
            continue
        space = _space_of(cls, emb)
        joints = _joint_names(space) if space is not None else None
        if joints:
            return emb, space, joints
        errors.append(f"{cls.__name__}: no readable action_space")
    raise TypeError(
        "no embodiment class binds to the scene with a readable "
        "action_space: " + "; ".join(errors[:6])
    )


def _observation(emb, space, joints):
    try:
        obs = emb.observation()
        if obs is not None:
            return obs
    except Exception:
        pass
    return {
        "joints": {j: 0.0 for j in joints},
        "joint_pos": np.zeros(len(joints)),
        "joint_vel": np.zeros(len(joints)),
        "cameras": {"head_cam": np.zeros((8, 8, 3), dtype=np.uint8)},
        "time": 0.0,
    }


def _construct_policy(cls, emb, space, model, data):
    attempts = [
        ((emb,), {}),
        ((space,), {}),
        ((model, data), {}),
        ((), {"embodiment": emb}),
        ((), {"action_space": space}),
        ((), {}),
    ]
    errors = []
    for args, kwargs in attempts:
        try:
            return cls(*args, **kwargs)
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
    raise TypeError("; ".join(errors))


def _call_act(inst, cls, obs):
    act = getattr(inst, "act", None) if inst is not None else None
    if not callable(act):
        act = getattr(cls, "act", None)
    try:
        return act(obs)
    except TypeError:
        return act(observation=obs)


def _check_policy(cls, emb, space, joints, obs, model, data):
    if not inspect.isclass(cls):
        return "not a class"
    if not callable(getattr(cls, "act", None)):
        return "no callable act(observation)"
    inst, inst_err = None, None
    try:
        inst = _construct_policy(cls, emb, space, model, data)
    except Exception as e:
        inst_err = str(e)
    if inst is not None:
        for fn in BIND_FNS:
            f = getattr(inst, fn, None)
            if callable(f):
                try:
                    f(emb)
                except Exception:
                    pass
    if inst is None and inst_err:
        try:
            action = cls.act(obs)
        except Exception as e:
            return (
                f"cannot construct ({inst_err}); cls.act(observation) "
                f"also raised {type(e).__name__}: {e}"
            )
    else:
        try:
            old = signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(ACT_TIMEOUT_S)
        except (ValueError, OSError):
            old = None
        try:
            action = _call_act(inst, cls, obs)
        except _Timeout as e:
            return str(e)
        except Exception as e:
            return f"act(observation) raised {type(e).__name__}: {e}"
        finally:
            if old is not None:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
    if action is None:
        return "act(observation) returned None"
    return _action_matches(action, space, joints)


def test_core_2_1_policy_act_returns_actions_matching_action_space():
    classes = _policy_classes()
    assert classes, (
        "no policy class found: harmless.registry exposes no policy "
        "mapping and no class under harmless/policy.py or "
        "harmless/policies/ exists to carry act(observation)"
    )
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(REPO / "kitchen.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    emb, space, joints = _bound_embodiment(model, data)
    obs = _observation(emb, space, joints)

    failures = []
    for cls in classes:
        err = _check_policy(cls, emb, space, joints, obs, model, data)
        if err is None:
            return
        failures.append(f"{cls.__module__}.{cls.__name__}: {err}")
    pytest.fail(
        "no policy class satisfies the act(observation) contract:\n"
        + "\n".join(failures[:12])
    )
