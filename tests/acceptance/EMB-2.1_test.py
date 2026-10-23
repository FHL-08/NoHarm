"""EMB-2.1@1 A Franka Panda adapter registers an 8-D joint_pos action
space (7 joints plus gripper).

The contract under test is the second registered embodiment: importing
`harmless.embodiments.franka` (the module the requirement touches) must
make a Franka/Panda-named adapter resolvable through `harmless.registry`
— the slot every embodiment fills — and an instance of that adapter must
declare an `action_space` of kind "joint_pos" spanning exactly 8
dimensions: the Panda's seven arm joints plus one gripper dimension.

`harmless/embodiments/franka.py` does not exist yet — failing to import
it is the expected pre-implementation failure. The adapter is
instantiated unbound first (the G1 adapter declares its space without a
model) and, when that yields no space, bound to each loadable MJCF under
scenes/ and assets/franka/ — the directories the requirement touches —
so the check does not depend on the scene's file name. The gripper
dimension is any declared joint whose name carries a gripper token
(grip/finger/hand/grasp/claw); the other seven dims are the arm. Every
declared joint must carry a real bound (lo < hi) and the declared
command rate must be positive.
"""

import importlib
import inspect
import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[2]

FRANKA_NAME = re.compile(r"franka|panda|emika", re.I)
GRIPPER_NAME = re.compile(r"grip|finger|hand|grasp|claw|squeeze", re.I)
REGISTER_NAMES = ("franka", "panda", "franka_panda", "emika_panda",
                  "franka_emika", "panda_arm")


def _franka_module():
    try:
        return importlib.import_module("harmless.embodiments.franka")
    except Exception as exc:
        pytest.fail(
            f"harmless.embodiments.franka cannot be imported "
            f"({type(exc).__name__}: {exc}) — the Franka adapter does "
            f"not exist yet")


def _registry():
    try:
        from harmless import registry
        return registry
    except Exception as exc:
        pytest.fail(
            f"harmless.registry cannot be imported "
            f"({type(exc).__name__}: {exc}) — the embodiment slot is "
            f"broken")


def _registered_names(registry):
    names = []
    try:
        names = list(registry.list_embodiments())
    except Exception:
        for attr in ("EMBODIMENTS", "embodiments", "EMBODIMENT_REGISTRY"):
            reg = getattr(registry, attr, None)
            if isinstance(reg, dict):
                names = list(reg)
                break
    return names


def _registered_franka(registry):
    """Resolve the registered Franka adapter class/factory, or None."""
    names = _registered_names(registry)
    hits = [n for n in names if FRANKA_NAME.search(str(n))]
    for n in hits + [n for n in REGISTER_NAMES if n not in hits]:
        try:
            return n, registry.get_embodiment(n)
        except Exception:
            for attr in ("EMBODIMENTS", "embodiments", "EMBODIMENT_REGISTRY"):
                reg = getattr(registry, attr, None)
                if isinstance(reg, dict) and reg.get(n) is not None:
                    return n, reg[n]
    return None


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


def _ctor_attempts(cls, model, data):
    yield "cls(model, data)", lambda: cls(model, data)
    yield "cls(model=model, data=data)", \
        lambda: cls(model=model, data=data)
    yield "cls(model)", lambda: cls(model)


def _instances(entry):
    """Yield (instance, how-built) pairs for the registered entry."""
    if not inspect.isclass(entry) and not callable(entry):
        yield entry, "registered instance"
        return
    try:
        yield entry(), "cls()"
    except Exception:
        pass
    try:
        import mujoco
    except Exception:
        return
    for path in _scene_paths():
        try:
            model = mujoco.MjModel.from_xml_path(str(path))
        except Exception:
            continue
        data = mujoco.MjData(model)
        try:
            mujoco.mj_forward(model, data)
        except Exception:
            pass
        for label, call in _ctor_attempts(entry, model, data):
            try:
                yield call(), f"{label} on {path.name}"
            except Exception:
                continue


def _space_of(emb):
    space = getattr(emb, "action_space", None)
    if space is None:
        for attr in ("space", "actions", "action_spec"):
            space = getattr(emb, attr, None)
            if space is not None:
                break
    if space is None and emb is not None \
            and getattr(emb, "joints", None) is not None:
        space = emb
    return space


def _field(space, keys):
    for k in keys:
        v = getattr(space, k, None)
        if v is not None:
            return v
    return None


def test_emb_2_1_franka_adapter_registers_8d_joint_pos_space():
    _franka_module()
    registry = _registry()
    resolved = _registered_franka(registry)
    assert resolved is not None, (
        "importing harmless.embodiments.franka did not register a "
        "Franka/Panda adapter — harmless.registry knows only: "
        f"{_registered_names(registry) or 'nothing'}")
    name, entry = resolved

    space = None
    tried = []
    for emb, how in _instances(entry):
        s = _space_of(emb)
        if s is None:
            tried.append(f"{how}: no action_space")
            continue
        space = s
        break
    assert space is not None, (
        f"the registered adapter {name!r} never produced an action_space "
        f"({'; '.join(tried[:4]) or 'could not be instantiated'})")

    kind = _field(space, ("kind", "type", "action_kind", "family"))
    assert isinstance(kind, str) and "joint" in kind.lower() \
        and "pos" in kind.lower(), (
            f"the Franka action space kind is {kind!r}, not joint_pos")

    joints = _field(space, ("joints", "joint_names", "names", "dofs",
                            "axes"))
    joints = list(joints) if joints is not None else []
    if not joints:
        bounds0 = _field(space, ("bounds", "ranges", "limits"))
        if isinstance(bounds0, dict):
            joints = list(bounds0)
    assert len(joints) == 8, (
        f"the Franka joint_pos space declares {len(joints)} dims "
        f"{joints[:12]}, not the Panda's 8 (7 joints plus gripper)")

    bounds = _field(space, ("bounds", "ranges", "limits")) or {}
    unbounded, bad = [], []
    for j in joints:
        b = bounds.get(j) if isinstance(bounds, dict) else None
        if b is None:
            unbounded.append(j)
            continue
        try:
            lo, hi = float(b[0]), float(b[1])
        except Exception:
            bad.append(j)
            continue
        if not lo < hi:
            bad.append(j)
    assert not unbounded and not bad, (
        f"the Franka action space bounds are incomplete: no bound for "
        f"{unbounded}, malformed or empty bound for {bad}")

    grip = [j for j in joints if GRIPPER_NAME.search(str(j))]
    assert grip and len(joints) - len(grip) == 7, (
        f"the 8 declared dims {joints} do not decompose as 7 arm joints "
        f"plus a gripper dimension (gripper-like dims found: {grip})")

    rate = _field(space, ("rate_hz", "rate", "hz", "frequency"))
    assert rate is not None and float(rate) > 0, (
        f"the Franka command rate is {rate!r}, not a positive rate_hz")
