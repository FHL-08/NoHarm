"""Registry: name -> adapter class for the contract slots.

Embodiments are looked up by name so tasks and policies can bind to a
robot through the shared action/observation contract instead of hard-wired
G1 calls. More slots (policies, tasks) register alongside as they land.
"""
from __future__ import annotations

import sys

from harmless.embodiments.g1 import G1Embodiment

EMBODIMENTS = {"g1": G1Embodiment}

embodiments = EMBODIMENTS
EMBODIMENT_REGISTRY = EMBODIMENTS


def list_embodiments():
    return list(EMBODIMENTS)


def get_embodiment(name):
    return EMBODIMENTS[name]


def register_embodiment(name, cls):
    EMBODIMENTS[name] = cls
    return cls


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else "embodiments"
    if cmd == "embodiments":
        for name in list_embodiments():
            print(name)
        return 0
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
