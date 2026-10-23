"""Registry: name -> adapter class for the contract slots.

Embodiments are looked up by name so tasks and policies can bind to a
robot through the shared action/observation contract instead of hard-wired
G1 calls. Policies are looked up the same way so LLM agents, served VLAs,
and scripted controllers compete in one slot. More slots (tasks) register
alongside as they land.
"""
from __future__ import annotations

import sys

from harmless.embodiments.g1 import G1Embodiment
from harmless.policy import AgentPolicy, ScriptedPolicy

EMBODIMENTS = {"g1": G1Embodiment}
POLICIES = {"agent": AgentPolicy, "scripted": ScriptedPolicy}

embodiments = EMBODIMENTS
EMBODIMENT_REGISTRY = EMBODIMENTS
policies = POLICIES
POLICY_REGISTRY = POLICIES


def list_embodiments():
    return list(EMBODIMENTS)


def get_embodiment(name):
    return EMBODIMENTS[name]


def register_embodiment(name, cls):
    EMBODIMENTS[name] = cls
    return cls


def list_policies():
    return list(POLICIES)


def get_policy(name):
    return POLICIES[name]


def register_policy(name, cls):
    POLICIES[name] = cls
    return cls


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else "embodiments"
    if cmd == "embodiments":
        for name in list_embodiments():
            print(name)
        return 0
    if cmd == "policies":
        for name in list_policies():
            print(name)
        return 0
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
