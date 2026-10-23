"""CORE-3.2@1 `harmless tasks` lists the six household tasks defined in
the repo (transport, vase_to_island, knife_to_counter, apple_to_bowl,
plate_to_island, handover).

The `harmless tasks` command does not exist yet — no working spelling of
it is the expected pre-implementation failure. Because the touched files
allow several entry points (a `harmless` executable on PATH, a package
`__main__.py` giving `python -m harmless tasks`, or a `__main__`/`main()`
inside the touched `harmless/tasks/` package or `harmless/task` module),
the test tries each spelling. The first that produces a real command —
exit 0 with output — is held to the contract: it must list all six task
names, each as a whole word. A spelling that only errors is still
evidence the command exists and fails the line.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SIX = ("transport", "vase_to_island", "knife_to_counter",
       "apple_to_bowl", "plate_to_island", "handover")

CANDIDATES = [
    ["harmless", "tasks"],
    [sys.executable, "-m", "harmless", "tasks"],
    [sys.executable, "-m", "harmless.tasks"],
    [sys.executable, "-m", "harmless.cli", "tasks"],
    [sys.executable, "-m", "harmless.task", "tasks"],
    [sys.executable, "-m", "harmless.registry", "tasks"],
]

MISSING_MARKERS = ("No module named", "cannot be directly executed")


def _run(cmd):
    try:
        return subprocess.run(
            cmd, cwd=REPO, capture_output=True, text=True, timeout=60
        )
    except FileNotFoundError:
        return None


def _missing(result):
    if result is None:
        return True
    return any(m in result.stderr for m in MISSING_MARKERS)


def _listed(name, text):
    return re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
        text,
    )


def test_core_3_2_harmless_tasks_lists_the_six_household_tasks():
    results = [(cmd, _run(cmd)) for cmd in CANDIDATES]
    present = [(cmd, r) for cmd, r in results if not _missing(r)]

    for cmd, r in present:
        if r.returncode == 0 and all(_listed(n, r.stdout) for n in SIX):
            return

    for cmd, r in present:
        if r.returncode == 0 and r.stdout.strip():
            missing = [n for n in SIX if not _listed(n, r.stdout)]
            raise AssertionError(
                f"`{' '.join(cmd)}` ran but did not list all six household "
                f"tasks; missing {missing}; stdout: {r.stdout.strip()[:400]!r}"
            )

    errored = [(cmd, r) for cmd, r in present if r.returncode != 0]
    if errored:
        cmd, r = errored[0]
    elif present:
        cmd, r = present[0]
    else:
        raise AssertionError(
            "no `harmless tasks` command exists under any spelling "
            "(harmless executable, python -m harmless tasks, "
            "python -m harmless.tasks, python -m harmless.cli tasks, "
            "python -m harmless.task tasks, "
            "python -m harmless.registry tasks)"
        )
    raise AssertionError(
        f"`{' '.join(cmd)}` exists but exited {r.returncode} without "
        f"listing the six household tasks; stderr: {r.stderr.strip()[:300]}"
    )
