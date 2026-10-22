"""CORE-1.2@1 `harmless embodiments` lists at least the G1 adapter by name.

The `harmless` command does not exist yet — no working spelling of it is
the expected pre-implementation failure. Because the touched files allow
several entry points (a `harmless` executable on PATH, a package
`__main__.py` giving `python -m harmless embodiments`, or a `__main__`
inside the touched `harmless/embodiments/` package or `harmless/registry`
module), the test tries each spelling. The first that produces a real
command — exit 0 with output — is held to the contract: it must list a
name identifying the G1 adapter (matching `g1` or `unitree`,
case-insensitive). A spelling that only errors is still evidence the
command exists and fails the line.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

CANDIDATES = [
    ["harmless", "embodiments"],
    [sys.executable, "-m", "harmless", "embodiments"],
    [sys.executable, "-m", "harmless.embodiments"],
    [sys.executable, "-m", "harmless.cli", "embodiments"],
    [sys.executable, "-m", "harmless.registry", "embodiments"],
    [sys.executable, "-m", "harmless.embodiment", "embodiments"],
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


def test_core_1_2_harmless_embodiments_lists_g1():
    results = [(cmd, _run(cmd)) for cmd in CANDIDATES]
    present = [(cmd, r) for cmd, r in results if not _missing(r)]

    for cmd, r in present:
        if r.returncode == 0 and re.search(r"g1|unitree", r.stdout, re.I):
            return

    for cmd, r in present:
        if r.returncode == 0 and r.stdout.strip():
            raise AssertionError(
                f"`{' '.join(cmd)}` ran but did not list the G1 adapter "
                f"by name; stdout: {r.stdout.strip()[:300]!r}"
            )

    errored = [(cmd, r) for cmd, r in present if r.returncode != 0]
    if errored:
        cmd, r = errored[0]
    elif present:
        cmd, r = present[0]
    else:
        raise AssertionError(
            "no `harmless embodiments` command exists under any spelling "
            "(harmless executable, python -m harmless embodiments, "
            "python -m harmless.embodiments, python -m harmless.cli "
            "embodiments, python -m harmless.registry embodiments, "
            "python -m harmless.embodiment embodiments)"
        )
    raise AssertionError(
        f"`{' '.join(cmd)}` exists but exited {r.returncode} without "
        f"listing the G1 adapter; stderr: {r.stderr.strip()[:300]}"
    )
