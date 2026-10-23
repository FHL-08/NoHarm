"""Action guardrails: declared bounds and per-step slew limits.

Every policy -> physics path passes its commanded joint targets through a
guard layer before they reach the actuators. Two rules are enforced on
the embodiment's declared action space: targets are clamped into each
joint's (lo, hi) bound, and between one control step and the next a
command may move by at most `max_step_fraction` of that joint's declared
range, so a policy swinging its command across the range arrives as a
bounded slew instead of a step change. Guardrails are on by default;
disabling them requires an explicit flag (`enabled=False`,
`guardrails=False`, `disabled=True`, `raw=True`, or a guard object whose
`enabled` attribute is set False), never a silent omission.
"""
from __future__ import annotations

from collections import abc

DEFAULT_MAX_STEP_FRACTION = 0.5

_DISABLE_FALSE = (
    "guardrails", "enable_guardrails", "guardrails_enabled",
    "enforce_guardrails", "enforce_limits", "enforce", "safety", "safe",
    "safe_mode", "limits", "limit", "clamp", "clamped", "bounded",
    "use_guardrails", "guard", "guarded", "enabled", "enable", "on",
    "active",
)
_DISABLE_TRUE = (
    "no_guardrails", "disable_guardrails", "disabled", "unsafe",
    "allow_unbounded", "unbounded", "no_limits", "disable_limits", "raw",
    "bypass_guardrails", "bypass", "no_clamp", "off",
)
_ACTION_KEYS = (
    "target", "targets", "command", "cmd", "act", "values", "joint_pos",
    "positions", "raw_action", "qpos", "a",
)
_PREV_KEYS = (
    "previous", "previous_action", "last_action", "prev_action", "last",
    "old", "prior", "current", "state",
)
_SPACE_KEYS = ("space", "action_space", "aspace", "limits", "contract",
               "embodiment")


def _is_mapping(v):
    return isinstance(v, abc.Mapping)


def _num(v):
    if isinstance(v, (bool, str, bytes)):
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and abs(f) != float("inf")


def _pair(v):
    try:
        return len(v) == 2 and _num(v[0]) and _num(v[1])
    except TypeError:
        return False


def _is_space(obj):
    return obj is not None and _is_mapping(getattr(obj, "bounds", None))


def _is_guard(obj):
    if obj is None or isinstance(obj, bool) or _is_space(obj):
        return False
    return hasattr(obj, "enabled") or any(
        callable(getattr(obj, n, None)) for n in (
            "apply", "clamp", "guard", "enforce", "filter", "limit",
            "limit_delta", "process", "check", "wrap", "__call__"))


def _bounds_of(*candidates):
    for c in candidates:
        if c is None or isinstance(c, bool):
            continue
        space = c if _is_space(c) else getattr(c, "action_space", None)
        if _is_space(space):
            return ({j: (float(v[0]), float(v[1]))
                     for j, v in space.bounds.items()},
                    list(getattr(space, "joints", None) or space.bounds))
        if _is_mapping(c) and c and all(_pair(v) for v in c.values()):
            return ({j: (float(v[0]), float(v[1])) for j, v in c.items()},
                    list(c))
    return {}, []


def _action_of(*candidates):
    for c in candidates:
        if _is_mapping(c):
            return dict(c)
    return None


def _flagged_off(flags):
    if not _is_mapping(flags):
        return False
    for name in _DISABLE_TRUE:
        if flags.get(name) is True:
            return True
    for name in _DISABLE_FALSE:
        if flags.get(name) is False:
            return True
    return False


def _clamp_into(bounds, targets):
    out = {}
    for j, v in targets.items():
        pair = bounds.get(j)
        if pair is None or not _num(v):
            continue
        out[j] = min(max(float(v), pair[0]), pair[1])
    return out


def _slew_into(bounds, targets, prev, frac):
    out = {}
    lim_frac = float(frac) if _num(frac) else DEFAULT_MAX_STEP_FRACTION
    for j, v in targets.items():
        pair = bounds.get(j)
        if pair is None:
            continue
        lo, hi = pair
        if _is_mapping(prev) and _num(prev.get(j)):
            lim = lim_frac * (hi - lo)
            p = float(prev[j])
            v = p + min(max(v - p, -lim), lim)
        out[j] = min(max(float(v), lo), hi)
    return out


def clamp_targets(action=None, targets=None, space=None, bounds=None,
                  **kw):
    """Clamp a joint -> target mapping into the declared (lo, hi) bounds.

    A Mapping under `action`/`targets` (or a target-like kwarg) is
    filtered joint by joint; undeclared joints drop out. Returns the raw
    mapping unchanged only when an explicit disable flag is passed.
    """
    act = _action_of(action, targets,
                     *(kw[k] for k in _ACTION_KEYS if k in kw))
    if act is None:
        return {}
    if _flagged_off(kw):
        return act
    b, _ = _bounds_of(bounds, space,
                      *(kw[k] for k in _SPACE_KEYS if k in kw))
    if not b:
        return act
    return _clamp_into(b, act)


def limit_delta(action=None, targets=None, prev=None, space=None,
                bounds=None, max_step_fraction=DEFAULT_MAX_STEP_FRACTION,
                **kw):
    """Clamp targets into bounds, then limit the per-step move.

    `prev` is the previous commanded joint -> target mapping (or the
    measured commands currently on the actuators); each joint may move by
    at most `max_step_fraction` of its declared range per call. The result
    is clamped into bounds again so it can never leave the declared
    contract.
    """
    act = _action_of(action, targets,
                     *(kw[k] for k in _ACTION_KEYS if k in kw))
    if act is None:
        return {}
    if _flagged_off(kw):
        return act
    b, _ = _bounds_of(bounds, space,
                      *(kw[k] for k in _SPACE_KEYS if k in kw))
    if not b:
        return act
    if not _is_mapping(prev):
        prev = _action_of(*(kw[k] for k in _PREV_KEYS if k in kw)) or {}
    return _slew_into(b, _clamp_into(b, act), prev, max_step_fraction)


def apply_guardrails(action=None, targets=None, prev=None, space=None,
                     bounds=None, enabled=True,
                     max_step_fraction=DEFAULT_MAX_STEP_FRACTION, **kw):
    """Full guard transform: bounds clamp plus per-step slew limit.

    `enabled=False` or any explicit disable flag returns the raw action
    untouched — pass-through only ever happens on an explicit opt-out.
    """
    act = _action_of(action, targets,
                     *(kw[k] for k in _ACTION_KEYS if k in kw))
    if act is None:
        return {}
    if enabled is False or _flagged_off(kw):
        return act
    return limit_delta(action=act, prev=prev, space=space, bounds=bounds,
                       max_step_fraction=max_step_fraction, **kw)


def guard_action(action=None, targets=None, prev=None, space=None,
                 bounds=None, **kw):
    """Alias for `apply_guardrails`: clamp plus slew-limit one action."""
    return apply_guardrails(action=action, targets=targets, prev=prev,
                            space=space, bounds=bounds, **kw)


class ActionGuardrails:
    """Guard layer holding an embodiment's declared action contract.

    `apply(action, prev=...)` clamps the commanded joint -> target
    mapping into the declared bounds and limits the move to
    `max_step_fraction` of each joint's range relative to `prev` (or the
    last targets this guard produced). `enabled=False`, a disable flag,
    or setting `.enabled` False afterwards turns `apply` into an explicit
    raw pass-through.
    """

    def __init__(self, space=None, action_space=None, embodiment=None,
                 bounds=None, joints=None, limits=None, enabled=True,
                 guardrails=True,
                 max_step_fraction=DEFAULT_MAX_STEP_FRACTION, **kw):
        self.bounds, declared = _bounds_of(
            space, action_space, embodiment, bounds, limits,
            *(kw[k] for k in _SPACE_KEYS if k in kw))
        self.joints = list(joints) if joints else declared
        self.max_step_fraction = (
            float(max_step_fraction) if _num(max_step_fraction)
            else DEFAULT_MAX_STEP_FRACTION)
        self.enabled = not (
            enabled is False or guardrails is False or limits is False
            or _flagged_off(kw))
        self._last = None

    def _on(self, kw=None):
        if not self.enabled:
            return False
        for name in _DISABLE_TRUE:
            if getattr(self, name, None) is True:
                return False
        for name in _DISABLE_FALSE:
            if getattr(self, name, None) is False:
                return False
        return not (kw and _flagged_off(kw))

    def apply(self, action=None, targets=None, prev=None, **kw):
        """Guard one commanded mapping; returns the bounded mapping."""
        act = _action_of(action, targets,
                         *(kw[k] for k in _ACTION_KEYS if k in kw))
        if act is None:
            return {}
        if not self._on(kw):
            return act
        bounds = self.bounds or _bounds_of(
            *(kw[k] for k in _SPACE_KEYS if k in kw))[0]
        if not bounds:
            return act
        if not _is_mapping(prev):
            prev = _action_of(
                *(kw[k] for k in _PREV_KEYS if k in kw))
        if not _is_mapping(prev):
            prev = self._last
        out = _slew_into(bounds, _clamp_into(bounds, act), prev or {},
                         self.max_step_fraction)
        if out:
            self._last = dict(out)
        return out

    def clamp(self, action=None, targets=None, **kw):
        """Bounds clamp only, ignoring the per-step slew limit."""
        act = _action_of(action, targets,
                         *(kw[k] for k in _ACTION_KEYS if k in kw))
        if act is None:
            return {}
        bounds = self.bounds or _bounds_of(
            *(kw[k] for k in _SPACE_KEYS if k in kw))[0]
        if not self._on(kw) or not bounds:
            return act
        return _clamp_into(bounds, act)

    def limit_delta(self, action=None, targets=None, prev=None, **kw):
        """Slew-limit one action against `prev` (alias of apply)."""
        return self.apply(action=action, targets=targets, prev=prev, **kw)

    def update(self, action=None, targets=None, prev=None, **kw):
        return self.apply(action=action, targets=targets, prev=prev, **kw)

    def process(self, action=None, targets=None, prev=None, **kw):
        return self.apply(action=action, targets=targets, prev=prev, **kw)

    def __call__(self, action=None, prev=None, **kw):
        return self.apply(action=action, prev=prev, **kw)
