"""CORE-2.2@1 `harmless policies` lists at least `agent` and `scripted`.

No spelling of the `harmless policies` command works yet — the registry's
`main` answers `policies` with "unknown command" — so failing to find a
working spelling is the expected pre-implementation failure. Because the
touched files allow several entry points (a `harmless` executable on
PATH, a package `__main__.py` giving `python -m harmless policies`, or a
`__main__`/`main` inside the touched `harmless/policy` module or
`harmless/registry` module, or a `harmless/policies/` package), the test
tries each spelling. The first that produces a real command — exit 0
with output — is held to the contract: it must list both the names
`agent` and `scripted` (case-insensitive). A spelling that only errors
is still evidence the command exists and fails the line.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

CANDIDATES = [
    ["harmless", "policies"],
    [sys.executable, "-m", "harmless", "policies"],
    [sys.executable, "-m", "harmless.policy", "policies"],
    [sys.executable, "-m", "harmless.policies", "policies"],
    [sys.executable, "-m", "harmless.cli", "policies"],
    [sys.executable, "-m", "harmless.registry", "policies"],
]

MISSING_MARKERS = ("No module named", "cannot be directly executed")

REQUIRED = ("agent", "scripted")


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


def _lists_all(stdout):
    return all(
        re.search(rf"\b{name}\b", stdout, re.I) for name in REQUIRED
    )


def test_core_2_2_harmless_policies_lists_agent_and_scripted():
    results = [(cmd, _run(cmd)) for cmd in CANDIDATES]
    present = [(cmd, r) for cmd, r in results if not _missing(r)]

    for cmd, r in present:
        if r.returncode == 0 and _lists_all(r.stdout):
            return

    for cmd, r in present:
        if r.returncode == 0 and r.stdout.strip():
            raise AssertionError(
                f"`{' '.join(cmd)}` ran but did not list both `agent` "
                f"and `scripted`; stdout: {r.stdout.strip()[:300]!r}"
            )

    errored = [(cmd, r) for cmd, r in present if r.returncode != 0]
    if errored:
        cmd, r = errored[0]
    elif present:
        cmd, r = present[0]
    else:
        raise AssertionError(
            "no `harmless policies` command exists under any spelling "
            "(harmless executable, python -m harmless policies, "
            "python -m harmless.policy policies, python -m "
            "harmless.policies policies, python -m harmless.cli "
            "policies, python -m harmless.registry policies)"
        )
    raise AssertionError(
        f"`{' '.join(cmd)}` exists but exited {r.returncode} without "
        f"listing `agent` and `scripted`; stderr: "
        f"{r.stderr.strip()[:300]}"
    )
