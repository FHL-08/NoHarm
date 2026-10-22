"""Policy-embodiment compatibility: the pairing gate run before physics.

A policy declares the action space it intends to drive — joint names,
per-joint bounds, command rate — and an embodiment declares the space it
accepts. `check_compatibility` compares the two declared contracts and
raises `CompatibilityError` naming every mismatched field, so a
mismatched pair fails at pairing time, before the first simulation
step, instead of mid-rollout. `pair` is the same gate plus the bind.
"""
from __future__ import annotations

import sys
from collections.abc import Mapping

from harmless.embodiment import ActionSpace


class CompatibilityError(ValueError):
    """Raised when a policy's declared action space does not match the
    embodiment's declared action space."""


class Pairing:
    """Record of a validated policy/embodiment pair."""

    def __init__(self, policy=None, embodiment=None, policy_space=None,
                 embodiment_space=None):
        self.policy = policy
        self.embodiment = embodiment
        self.policy_space = policy_space
        self.embodiment_space = embodiment_space

    def __repr__(self):
        return (
            f"Pairing(policy={self.policy!r}, "
            f"embodiment={self.embodiment!r})"
        )


def _is_space(obj):
    return isinstance(obj, ActionSpace) or (
        isinstance(getattr(obj, "bounds", None), Mapping)
        and getattr(obj, "joints", None) is not None
    )


def _space_of(obj):
    if obj is None:
        return None
    space = getattr(obj, "action_space", None)
    if _is_space(space):
        return space
    if _is_space(obj):
        return obj
    return None


def _resolve(policy=None, embodiment=None, action_space=None,
             policy_space=None, embodiment_space=None):
    pspace = _space_of(policy_space)
    if pspace is None:
        pspace = _space_of(policy)
    espace = _space_of(action_space)
    if espace is None:
        espace = _space_of(embodiment_space)
    if espace is None:
        espace = _space_of(embodiment)
    if pspace is None or espace is None:
        raise TypeError("expected a policy and an embodiment")
    return pspace, espace


def _bounds_differ(a, b):
    if a is None or b is None:
        return a != b
    try:
        return float(a[0]) != float(b[0]) or float(a[1]) != float(b[1])
    except Exception:
        return a != b


def _diff_spaces(pspace, espace):
    diffs = []
    pkind = getattr(pspace, "kind", None)
    ekind = getattr(espace, "kind", None)
    if pkind != ekind:
        diffs.append(
            f"action space kind differs: policy {pkind!r} "
            f"vs embodiment {ekind!r}")
    pjoints = list(getattr(pspace, "joints", None) or [])
    ejoints = list(getattr(espace, "joints", None) or [])
    eset = set(ejoints)
    pset = set(pjoints)
    if pjoints != ejoints:
        diffs.append(
            "joint names differ: only in policy "
            f"{[j for j in pjoints if j not in eset]}, "
            f"only in embodiment {[j for j in ejoints if j not in pset]}")
    pbounds = dict(getattr(pspace, "bounds", None) or {})
    ebounds = dict(getattr(espace, "bounds", None) or {})
    for joint in pjoints:
        if joint not in eset:
            continue
        if _bounds_differ(pbounds.get(joint), ebounds.get(joint)):
            diffs.append(
                f"bounds differ for joint {joint!r}: policy "
                f"{pbounds.get(joint)} vs embodiment "
                f"{ebounds.get(joint)}")
    prate = getattr(pspace, "rate_hz", None)
    erate = getattr(espace, "rate_hz", None)
    if prate != erate:
        diffs.append(
            f"rate_hz differs: policy {prate} Hz vs embodiment "
            f"{erate} Hz")
    return diffs


def compatibility_diffs(policy=None, embodiment=None, action_space=None,
                        policy_space=None, embodiment_space=None,
                        **_unused):
    """Return the list of mismatched action-space fields, empty when the
    policy's declared space matches the embodiment's."""
    pspace, espace = _resolve(policy, embodiment, action_space,
                              policy_space, embodiment_space)
    return _diff_spaces(pspace, espace)


def check_compatibility(policy=None, embodiment=None, action_space=None,
                        policy_space=None, embodiment_space=None,
                        **unused):
    """Raise `CompatibilityError` naming each mismatched field when the
    policy's declared action space differs from the embodiment's. Run
    before the first simulation step. Returns the policy on a match."""
    pspace, espace = _resolve(policy, embodiment, action_space,
                              policy_space, embodiment_space)
    diffs = _diff_spaces(pspace, espace)
    if diffs:
        raise CompatibilityError(
            "policy action space is incompatible with the "
            "embodiment's: " + "; ".join(diffs))
    return policy if policy is not None else policy_space


def pair(policy=None, embodiment=None, action_space=None,
         policy_space=None, embodiment_space=None, **unused):
    """Validate compatibility, then bind the policy to the embodiment
    when it has a `bind` method. Raises `CompatibilityError` on a
    mismatched action space, before any physics runs."""
    pspace, espace = _resolve(policy, embodiment, action_space,
                              policy_space, embodiment_space)
    diffs = _diff_spaces(pspace, espace)
    if diffs:
        raise CompatibilityError(
            "policy action space is incompatible with the "
            "embodiment's: " + "; ".join(diffs))
    target = embodiment
    if target is None:
        target = action_space if action_space is not None else embodiment_space
    binder = getattr(policy, "bind", None)
    if target is not None and callable(binder) and not _is_space(policy):
        binder(target)
    return Pairing(policy, target, pspace, espace)


def compatible(policy=None, embodiment=None, action_space=None,
               policy_space=None, embodiment_space=None, **_unused):
    """True when the policy's declared action space matches the
    embodiment's; False otherwise."""
    pspace, espace = _resolve(policy, embodiment, action_space,
                              policy_space, embodiment_space)
    return not _diff_spaces(pspace, espace)


is_compatible = compatible
assert_compatible = check_compatibility
assert_pairable = check_compatibility
ensure_compatible = check_compatibility
require_compatible = check_compatibility
validate_compatibility = check_compatibility
check = check_compatibility
verify = check_compatibility
bind = pair
connect = pair
attach = pair


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print(
            "usage: python -m harmless.compat <policy> <embodiment>",
            file=sys.stderr)
        return 2
    from harmless.registry import get_embodiment, get_policy

    embodiment = get_embodiment(args[1])()
    policy = get_policy(args[0])(embodiment)
    check_compatibility(policy, embodiment)
    print(f"{args[0]} is compatible with {args[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
