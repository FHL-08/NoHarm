"""CORE-5.1@1 Pairing a policy whose action space differs from the
embodiment's raises a compatibility error before the first simulation
step.

The contract under test is the pairing gate that must live in
`harmless/compat.py`: when a policy's declared action space — joint
names, per-joint bounds, or command rate — differs from the paired
embodiment's, the act of pairing must raise a compatibility error, and
it must do so from the declared contracts alone, before any physics
step has run.

`harmless/compat.py` does not exist yet — failing to import it is the
expected pre-implementation failure. The check's name is not fixed by
the line, so the test probes the module's public callables: common
spellings (pair/check/assert_compatible/ensure_compatible/validate...)
first, then every public callable the module defines. Each is driven
with the argument shapes a caller could use — (policy, embodiment),
(policy, action_space), (policy_space, embodiment_space), both orders,
and the keyword equivalents — and, for entry points that return a
pairing object, the zero-argument confirm methods such an object could
carry. Every object passed is dual-readable as a space carrier and as
a space (`action_space` plus the joints/bounds/rate_hz fields
themselves) so attribute conventions do not decide the result.

A call counts as evidence only when the mismatched pair is rejected —
an exception or a nonzero SystemExit that is not a call-shape complaint
(TypeError/AttributeError from wrong arguments, NotImplementedError
from a stub) — AND the identical call on a matching space is not
rejected: a gate that refuses every pairing is not a compatibility
check. After the probes the bound embodiment's MjData must still read
time == 0: the mismatch was caught before the first mj_step.
"""

import importlib
import inspect
import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[2]

CHECK_NAMES = (
    "pair", "pair_policy", "pairing", "check", "check_compat",
    "check_compatibility", "check_pairing", "compatibility_check",
    "assert_compat", "assert_compatible", "assert_pairable",
    "ensure_compat", "ensure_compatible", "require_compat",
    "require_compatible", "verify", "verify_compat",
    "verify_compatible", "validate", "validate_compat",
    "validate_compatibility", "compatible", "is_compatible",
    "bind", "guard", "match", "compare", "connect", "attach",
    "compat",
)

FOLLOWUP_NAMES = (
    "check", "validate", "verify", "assert_compatible",
    "ensure_compatible", "raise_if_incompatible", "bind", "pair",
    "commit", "apply", "confirm", "run",
)

_COMPAT_HINT = re.compile(
    r"compat|mismatch|incompat|differ|conflict|action.?space|"
    r"joint|bound|rate|hz", re.I)

_SHAPES = (
    ("fn(policy, embodiment)",
     lambda p, e, ps, es: ((p, e), {})),
    ("fn(policy, action_space)",
     lambda p, e, ps, es: ((p, es), {})),
    ("fn(policy_space, embodiment_space)",
     lambda p, e, ps, es: ((ps, es), {})),
    ("fn(embodiment, policy)",
     lambda p, e, ps, es: ((e, p), {})),
    ("fn(embodiment_space, policy_space)",
     lambda p, e, ps, es: ((es, ps), {})),
    ("fn(policy, embodiment=embodiment)",
     lambda p, e, ps, es: ((p,), {"embodiment": e})),
    ("fn(policy, action_space=space)",
     lambda p, e, ps, es: ((p,), {"action_space": es})),
    ("fn(policy=policy, embodiment=embodiment)",
     lambda p, e, ps, es: ((), {"policy": p, "embodiment": e})),
    ("fn(policy=policy, action_space=space)",
     lambda p, e, ps, es: ((), {"policy": p, "action_space": es})),
    ("fn(policy_space=space, embodiment_space=space)",
     lambda p, e, ps, es: ((), {"policy_space": ps,
                               "embodiment_space": es})),
    ("fn(policy_space=space, embodiment=embodiment)",
     lambda p, e, ps, es: ((), {"policy_space": ps, "embodiment": e})),
    ("fn(policy=policy_space, embodiment=embodiment_space)",
     lambda p, e, ps, es: ((), {"policy": ps, "embodiment": es})),
)


def _compat_module():
    try:
        return importlib.import_module("harmless.compat")
    except Exception as exc:
        pytest.fail(
            f"harmless.compat cannot be imported "
            f"({type(exc).__name__}: {exc}) — the compatibility check "
            f"does not exist yet")


def _candidates(mod):
    out, seen = [], set()

    def add(name, fn):
        if callable(fn) and id(fn) not in seen:
            seen.add(id(fn))
            out.append((name, fn))

    for name in CHECK_NAMES:
        add(name, getattr(mod, name, None))
    for name, obj in sorted(vars(mod).items()):
        if name.startswith("_"):
            continue
        mod_name = getattr(obj, "__module__", None)
        if mod_name is not None and mod_name != mod.__name__:
            continue
        add(name, obj)
    return out


def _embodiment_class():
    try:
        from harmless import registry
        for name in ("g1",):
            try:
                return registry.get_embodiment(name)
            except Exception:
                pass
        for name in registry.list_embodiments():
            try:
                return registry.get_embodiment(name)
            except Exception:
                continue
    except Exception:
        pass
    import pkgutil

    from harmless.embodiment import Embodiment
    try:
        pkg = importlib.import_module("harmless.embodiments")
    except Exception:
        return None
    for m in pkgutil.iter_modules(pkg.__path__):
        try:
            sub = importlib.import_module(f"harmless.embodiments.{m.name}")
        except Exception:
            continue
        for obj in vars(sub).values():
            if (inspect.isclass(obj) and issubclass(obj, Embodiment)
                    and obj is not Embodiment):
                return obj
    return None


def _bound_embodiment():
    cls = _embodiment_class()
    emb = model = data = None
    if cls is not None:
        try:
            import mujoco
            model = mujoco.MjModel.from_xml_path(str(REPO / "kitchen.xml"))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            emb = cls(model, data)
        except Exception:
            model = data = None
            try:
                emb = cls()
            except Exception:
                emb = None
    space = getattr(emb, "action_space", None) if emb is not None else None
    if space is None:
        pytest.fail(
            "no embodiment with a declared action_space could be built — "
            "the CORE-1 contract is broken and pairing cannot be "
            "exercised")
    for f in ("kind", "joints", "bounds", "rate_hz"):
        if not hasattr(emb, f) and hasattr(space, f):
            setattr(emb, f, getattr(space, f))
    return emb, model, data, space


from harmless.embodiment import ActionSpace
from harmless.policy import ScriptedPolicy


class _SpaceView(ActionSpace):
    """An ActionSpace that also answers .action_space -> itself."""

    @property
    def action_space(self):
        return self


class _PairablePolicy(ScriptedPolicy):
    """A Policy that also reads as an ActionSpace directly."""

    def __init__(self, space):
        self.embodiment = None
        self.action_space = space
        self.kind = space.kind
        self.joints = list(space.joints)
        self.bounds = dict(space.bounds)
        self.rate_hz = space.rate_hz


def _like(space):
    return _SpaceView(space.kind, space.joints, space.bounds,
                      space.rate_hz)


def _joint_mismatch(space):
    dropped = space.joints[-1]
    added = "phantom_wrist_servo"
    joints = list(space.joints[:-1]) + [added]
    bounds = dict(space.bounds)
    bounds[added] = bounds[dropped]
    return (_SpaceView(space.kind, joints, bounds, space.rate_hz),
            (added, dropped))


def _bounds_mismatch(space):
    joint = space.joints[len(space.joints) // 2]
    lo, hi = space.bounds[joint]
    bounds = dict(space.bounds)
    bounds[joint] = (lo - 0.25, hi + 0.25)
    return (_SpaceView(space.kind, list(space.joints), bounds,
                       space.rate_hz), joint)


def _rate_mismatch(space):
    rate = space.rate_hz / 2.0
    if rate == space.rate_hz:
        rate = space.rate_hz + 1.0
    return (_SpaceView(space.kind, list(space.joints),
                       dict(space.bounds), rate),
            (space.rate_hz, rate))


def _invoke(fn, args, kwargs):
    try:
        return "ok", fn(*args, **kwargs)
    except SystemExit as exc:
        return "exit", exc
    except Exception as exc:
        return "raised", exc


def _usage_error(exc):
    if isinstance(exc, (NotImplementedError, AttributeError)):
        return True
    if isinstance(exc, TypeError):
        return not _COMPAT_HINT.search(str(exc))
    return False


def _rejects(outcome):
    kind, payload = outcome
    if kind == "exit":
        return bool(payload.code)
    if kind == "raised":
        return not _usage_error(payload)
    return False


def _outcome(fn, args, kwargs):
    kind, payload = _invoke(fn, args, kwargs)
    if kind != "ok":
        return (kind, payload), ""
    for attr in FOLLOWUP_NAMES:
        method = getattr(payload, attr, None)
        if callable(method):
            follow = _invoke(method, (), {})
            if follow[0] != "ok":
                return follow, f".{attr}()"
    return ("ok", payload), ""


def _rejections(mod, bad_policy, bad_space, emb, espace_view):
    good_space = _like(espace_view)
    good_policy = _PairablePolicy(good_space)
    found, silent, unconditional = [], [], []
    for name, fn in _candidates(mod):
        for label, mk in _SHAPES:
            args, kwargs = mk(bad_policy, emb, bad_space, espace_view)
            bad, suffix = _outcome(fn, args, kwargs)
            if not _rejects(bad):
                if bad[0] == "ok":
                    silent.append(f"{name} {label}{suffix}")
                continue
            gargs, gkwargs = mk(good_policy, emb, good_space,
                                espace_view)
            good, _ = _outcome(fn, gargs, gkwargs)
            if _rejects(good):
                unconditional.append(f"{name} {label}{suffix}")
                continue
            found.append((f"{name} {label}{suffix}", bad[1]))
    return found, silent, unconditional


def _mismatch_cases(espace):
    joint_space, names = _joint_mismatch(espace)
    bounds_space, _ = _bounds_mismatch(espace)
    rate_space, _ = _rate_mismatch(espace)
    return {
        "joint names": joint_space,
        "bounds": bounds_space,
        "rate": rate_space,
    }


def test_core_5_1_mismatched_space_rejected_before_first_step():
    mod = _compat_module()
    emb, model, data, espace = _bound_embodiment()
    espace_view = _like(espace)

    failures = []
    for field, bad_space in _mismatch_cases(espace).items():
        found, silent, unconditional = _rejections(
            mod, _PairablePolicy(bad_space), bad_space, emb,
            espace_view)
        if found:
            continue
        detail = f"the {field} mismatch was never rejected"
        if silent:
            detail += ("; it was accepted silently via "
                       + ", ".join(silent[:3]))
        if unconditional:
            detail += ("; the only rejecting calls also rejected a "
                       "matching pair via "
                       + ", ".join(unconditional[:3]))
        failures.append(detail)

    assert not failures, (
        "no callable in harmless.compat raises a compatibility error "
        "when pairing a policy whose action space differs:\n"
        + "\n".join(failures))

    if data is not None:
        assert data.time == 0.0, (
            f"the simulation clock advanced to {data.time} during "
            f"pairing — the mismatch was not caught before the first "
            f"simulation step")
