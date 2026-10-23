"""Embodiment contract: the action/observation slot every robot adapter fills.

An embodiment declares its `action_space` — the joints it drives, the bounds
each joint target may take, and the rate at which it accepts new targets —
and produces `observation()` bundles of measured joint state and camera
frames. Concrete adapters live under `harmless/embodiments/` and are looked
up by name through `harmless.registry`.
"""
from __future__ import annotations

import sys
from collections.abc import Mapping


class ActionSpace:
    """Declared action contract for one embodiment.

    `kind` names the action family (e.g. "joint_pos"); `joints` is the
    ordered list of driven joint names; `bounds` maps every declared joint
    to a (lo, hi) target range with lo < hi; `rate_hz` is the positive rate
    at which the embodiment accepts new targets.
    """

    def __init__(self, kind, joints, bounds, rate_hz):
        if not isinstance(bounds, Mapping):
            raise TypeError("bounds must map joint name -> (lo, hi)")
        missing = [j for j in joints if j not in bounds]
        if missing:
            raise ValueError(f"no bound declared for joints: {missing}")
        self.kind = kind
        self.joints = list(joints)
        self.bounds = {j: (float(lo), float(hi)) for j, (lo, hi) in bounds.items()}
        self.rate_hz = float(rate_hz)

    def clamp(self, targets):
        return {
            j: min(max(float(v), self.bounds[j][0]), self.bounds[j][1])
            for j, v in targets.items()
            if j in self.bounds
        }

    def __repr__(self):
        return (
            f"ActionSpace(kind={self.kind!r}, joints={len(self.joints)}, "
            f"rate_hz={self.rate_hz})"
        )


class Embodiment:
    """Base class for robot adapters bound to a MjModel/MjData pair.

    Subclasses declare `action_space` (usually in __init__, where the real
    model supplies joint names and ranges) and implement `observation()`,
    which must return measured joint state covering every declared joint
    plus camera frames.
    """

    name = "embodiment"
    description = ""
    action_space = None

    def __init__(self, model=None, data=None):
        self.model = model
        self.data = data

    def observation(self):
        raise NotImplementedError

    def close(self):
        pass


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] == "embodiments":
        from harmless.registry import list_embodiments

        for name in list_embodiments():
            print(name)
        return 0
    print("usage: python -m harmless.embodiment embodiments", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
