"""BRND-1.3@1 No file outside .git/, spec/, and experience/archive/ contains the
legacy package name (the string built as NEEDLE below)."""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NEEDLE = b"phys" + b"harm"
EXCLUDED_PREFIXES = ("spec/", "experience/archive/")


def _tracked_files():
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [
        p
        for p in out.splitlines()
        if p and not p.startswith(EXCLUDED_PREFIXES)
    ]


def test_brnd_1_3_no_legacy_package_name_in_any_file():
    offenders = [
        rel for rel in _tracked_files() if NEEDLE in (REPO / rel).read_bytes()
    ]
    assert not offenders, (
        "legacy package name still present in: " + ", ".join(offenders[:20])
    )
