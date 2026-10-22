"""Policy contract: the act(observation) slot every controller fills.

A policy pairs with an embodiment adapter (or just its action_space) and
turns each observation into joint targets inside that declared space, so
LLM agents, served VLAs, and scripted controllers compete in one slot.
Concrete policies register by name in `harmless.registry` and are looked
up the same way embodiments are.
"""
from __future__ import annotations

import sys
from collections.abc import Mapping

from harmless.embodiment import ActionSpace


def _is_space(obj):
    return isinstance(obj, ActionSpace) or (
        isinstance(getattr(obj, "bounds", None), Mapping)
        and getattr(obj, "joints", None) is not None
    )


class Policy:
    """Base class for controllers paired with an embodiment.

    Construction accepts the embodiment itself, its `action_space`, or a
    MjModel/MjData pair; `bind(embodiment)` re-pairs after construction.
    Subclasses implement `act(observation)`, returning joint -> target
    inside the paired `action_space`.
    """

    name = "policy"
    description = ""
    action_space = None

    def __init__(self, embodiment=None, action_space=None, **_unused):
        self.embodiment = None
        self.action_space = None
        if action_space is None and _is_space(embodiment):
            embodiment, action_space = None, embodiment
        if _is_space(action_space):
            self.action_space = action_space
        if embodiment is not None and not _is_space(embodiment):
            self.bind(embodiment)

    def bind(self, embodiment):
        self.embodiment = embodiment
        space = getattr(embodiment, "action_space", None)
        if _is_space(space):
            self.action_space = space
        return self

    def act(self, observation):
        raise NotImplementedError

    def close(self):
        pass


class AgentPolicy(Policy):
    """The GPT-6-astra agent: a model that writes its own control code.

    The agent drives the embodiment through the episode loop
    (`harmless.agent.run_episode`) — frames in, one `program(code)` call
    out, measured receipt back — not through single-step `act()` calls.
    """

    name = "agent"
    description = (
        "GPT-6-astra LLM agent driving the embodiment through the "
        "episode loop"
    )

    def act(self, observation):
        raise RuntimeError(
            "the agent policy acts through the episode loop "
            "(harmless.agent.run_episode), not single-step act()")


class ScriptedPolicy(Policy):
    """Scripted baseline: holds the measured pose.

    `act` echoes each declared joint's measured position (the bounds
    midpoint when the observation lacks one), clamped into the paired
    `action_space` — a deterministic hold-position controller filling the
    same contract the model policies fill.
    """

    name = "scripted"
    description = (
        "scripted baseline holding the measured pose inside declared "
        "bounds"
    )

    def act(self, observation):
        measured = {}
        if isinstance(observation, Mapping):
            joints = observation.get("joints")
            if isinstance(joints, Mapping):
                measured = joints
        space = self.action_space
        if space is None:
            return {j: float(v) for j, v in measured.items()}
        targets = {}
        for joint in space.joints:
            lo, hi = space.bounds[joint]
            try:
                value = float(measured[joint])
            except Exception:
                value = (lo + hi) / 2.0
            targets[joint] = min(max(value, lo), hi)
        return targets


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] == "policies":
        from harmless.registry import list_policies

        for name in list_policies():
            print(name)
        return 0
    print("usage: python -m harmless.policy policies", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
