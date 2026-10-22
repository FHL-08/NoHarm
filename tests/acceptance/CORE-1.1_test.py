"""CORE-1.1@1 An embodiment class declares `action_space` (joint names,
bounds, control rate) and `observation()` returning joint state and camera
frames.

The contract under test is the embodiment slot itself: a class reachable
through the embodiments package or the registry must declare which joints
it drives, the bounds each joint target may take, and the rate at which it
accepts new targets; and its instances must produce observations bundling
the measured joint state with camera frames.

The test discovers candidate classes two ways: through `harmless.registry`
(a name -> class mapping, or a callable returning one, or a name list
resolvable through a lookup function) and by scanning `harmless/embodiment`
and `harmless/embodiments/*` for classes carrying `action_space` and
`observation`. Neither exists yet — finding no class is the expected
pre-implementation failure. An embodiment binds to the running physics as
Cls(model, data) (a mujoco.MjModel/MjData pair from the kitchen scene);
Cls(model) and Cls() are also accepted, but some construction must succeed
for `observation()` to be exercised. `action_space` may be an object or a
mapping; its joint names, bounds and control rate may sit under any of the
usual field names, but all three must be present and consistent (a bound
with lo < hi for every declared joint, a positive rate).
"""

import importlib
import inspect
import os
import pkgutil
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[2]

NAME_KEYS = ("joints", "joint_names", "names", "actuators", "dof_names")
BOUND_KEYS = ("bounds", "limits", "joint_bounds", "ranges")
RATE_KEYS = ("rate_hz", "control_rate", "control_hz", "rate", "hz", "frequency")


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


def _bounds_ok(space, joints):
    bounds = _field(space, BOUND_KEYS)
    if bounds is None or isinstance(bounds, (str, bytes)):
        return False
    if isinstance(bounds, Mapping):
        if not all(j in bounds for j in joints):
            return False
        pairs = [bounds[j] for j in joints]
    else:
        try:
            pairs = list(bounds)
        except TypeError:
            return False
        if len(pairs) != len(joints):
            return False
    for p in pairs:
        try:
            lo, hi = float(p[0]), float(p[1])
        except Exception:
            return False
        if not lo < hi:
            return False
    return True


def _rate_ok(space):
    try:
        return float(_field(space, RATE_KEYS)) > 0
    except (TypeError, ValueError):
        return False


def _numeric(value):
    if isinstance(value, Mapping):
        return any(_numeric(v) for v in value.values())
    if isinstance(value, (str, bytes, bool)):
        return False
    try:
        a = np.asarray(value)
    except Exception:
        return False
    return a.size >= 1 and np.issubdtype(a.dtype, np.number)


def _covers_joints(value, joints):
    return (
        isinstance(value, Mapping)
        and all(j in value for j in joints)
        and all(_numeric(value[j]) for j in joints)
    )


def _arr(value):
    try:
        return np.asarray(value)
    except Exception:
        return None


def _obs_items(obs):
    if isinstance(obs, Mapping):
        return list(obs.items())
    if hasattr(obs, "_asdict"):
        return list(obs._asdict().items())
    items = []
    for k in dir(obs):
        if k.startswith("_"):
            continue
        try:
            v = getattr(obs, k)
        except Exception:
            continue
        if not callable(v):
            items.append((k, v))
    return items


def _joint_state_ok(obs, items, joints):
    if _covers_joints(obs, joints):
        return True
    for k, v in items:
        if _covers_joints(v, joints):
            return True
        if isinstance(k, str) and "joint" in k.lower() or k in ("qpos", "positions"):
            a = _arr(v)
            if a is not None and a.ndim >= 1 and a.size == len(joints) and _numeric(v):
                return True
    return False


def _is_image(value):
    a = _arr(value)
    return a is not None and a.ndim >= 2 and a.size > 0 and _numeric(value)


def _camera_frames_ok(items):
    for _, v in items:
        if _is_image(v):
            return True
        if isinstance(v, Mapping) and any(_is_image(x) for x in v.values()):
            return True
        if isinstance(v, (list, tuple)) and any(_is_image(x) for x in v):
            return True
    return False


def _construct(cls, model, data):
    errors = []
    for args in ((model, data), (model,), ()):
        try:
            return cls(*args)
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
    raise TypeError("; ".join(errors))


def _lookup(reg, name):
    for fn in ("get_embodiment", "embodiment", "get", "resolve", "lookup"):
        f = getattr(reg, fn, None)
        if callable(f):
            try:
                c = f(name)
            except Exception:
                continue
            return c if inspect.isclass(c) else type(c)
    return None


def _resolve_registry_value(reg, obj):
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
            cls = _lookup(reg, item)
            if cls is not None:
                out.append(cls)
        elif callable(item):
            try:
                produced = item()
            except Exception:
                continue
            out.append(produced if inspect.isclass(produced) else type(produced))
        else:
            out.append(type(item))
    return out


def _from_registry():
    try:
        reg = importlib.import_module("harmless.registry")
    except Exception:
        return []
    classes = []
    for attr in (
        "EMBODIMENTS", "embodiments", "EMBODIMENT_REGISTRY",
        "embodiment_registry", "list_embodiments", "registry",
    ):
        obj = getattr(reg, attr, None)
        if obj is not None:
            classes += _resolve_registry_value(reg, obj)
    return classes


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


def _candidate_classes():
    classes = _from_registry()
    for modname in _embodiment_modules():
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        classes += [o for o in vars(mod).values() if inspect.isclass(o)]
    seen, out = set(), []
    for c in classes:
        if id(c) not in seen:
            seen.add(id(c))
            out.append(c)
    return out


def _check_contract(cls, model, data):
    if not inspect.isclass(cls):
        return "not a class"
    space = getattr(cls, "action_space", None)
    emb, emb_err = None, None
    try:
        emb = _construct(cls, model, data)
    except Exception as e:
        emb_err = str(e)
    if space is None and emb is not None:
        space = getattr(emb, "action_space", None)
    if space is None:
        suffix = f" (construction also failed: {emb_err})" if emb_err else ""
        return "no action_space declared" + suffix
    joints = _joint_names(space)
    if joints is None:
        return "action_space has no non-empty joint-name list (joints/joint_names)"
    if not _bounds_ok(space, joints):
        return "action_space bounds missing, or not a lo < hi pair for every declared joint"
    if not _rate_ok(space):
        return "action_space control rate missing or not a positive number"
    if not callable(getattr(cls, "observation", None)):
        return "no callable observation()"
    if emb is None:
        return f"cannot construct an instance to call observation(): {emb_err}"
    try:
        obs = emb.observation()
    except Exception as e:
        return f"observation() raised {type(e).__name__}: {e}"
    items = _obs_items(obs)
    if not items:
        return f"observation() returned {type(obs).__name__} with no readable members"
    if not _joint_state_ok(obs, items, joints):
        return "observation() has no joint state covering every declared joint"
    if not _camera_frames_ok(items):
        return "observation() has no camera frames (no 2D+ image arrays)"
    return None


def test_core_1_1_embodiment_class_declares_action_space_and_observation():
    classes = _candidate_classes()
    assert classes, (
        "no embodiment class found: harmless.registry exposes no embodiment "
        "mapping and no class under harmless/embodiment.py or "
        "harmless/embodiments/ exists to carry action_space/observation"
    )
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(REPO / "kitchen.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    failures = []
    for cls in classes:
        err = _check_contract(cls, model, data)
        if err is None:
            return
        failures.append(f"{cls.__module__}.{cls.__name__}: {err}")
    pytest.fail(
        "no embodiment class satisfies the action/observation contract:\n"
        + "\n".join(failures[:12])
    )
