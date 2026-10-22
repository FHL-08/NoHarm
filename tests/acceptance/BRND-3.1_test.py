"""BRND-3.1@1 README.md names inspect-robots by Robocurve (Jay Chooi and
Aris Zhu) as architectural inspiration."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
README = REPO / "README.md"
WINDOW = 1000
NAMES = ("Robocurve", "Jay Chooi", "Aris Zhu")
ROLE_MARKERS = ("architectur", "inspir")


def _name_re(name):
    return re.compile(
        r"[^a-z0-9]+".join(re.escape(part) for part in name.split()),
        re.IGNORECASE,
    )


def _windows_around(text, needle):
    for match in re.finditer(re.escape(needle), text, re.IGNORECASE):
        yield text[max(0, match.start() - WINDOW): match.end() + WINDOW]


def test_brnd_3_1_readme_names_inspect_robots_as_architectural_inspiration():
    text = re.sub(r"\s+", " ", README.read_text(encoding="utf-8"))

    windows = list(_windows_around(text, "inspect-robots"))
    assert windows, "README.md never mentions inspect-robots"

    best_missing = None
    for window in windows:
        missing = [n for n in NAMES if not _name_re(n).search(window)]
        missing += [m for m in ROLE_MARKERS if m not in window.lower()]
        if not missing:
            return
        if best_missing is None or len(missing) < len(best_missing):
            best_missing = missing

    raise AssertionError(
        "README.md mentions inspect-robots but never names it as "
        "architectural inspiration by Robocurve (Jay Chooi and Aris Zhu); "
        "the closest mention is still missing: " + ", ".join(best_missing)
    )
