"""BRND-3.2@1 README.md names PhysHarm by the AIM Intelligence Frontier
Research Team (Chaeyun Kim, Kihyun Kim, Dasol Choi, Haon Park) with SNU PI
Lab (Prof. Youngjae Yu) as the safety-scoring inspiration."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
README = REPO / "README.md"
WINDOW = 1200

TEAM_RE = re.compile(
    r"aim[^a-z0-9]+intelligence(?:['’]s)?[^a-z0-9]{1,25}frontier"
    r"[^a-z0-9]+research[^a-z0-9]+team",
    re.IGNORECASE,
)
SNU_LAB_RE = re.compile(r"snu[^a-z0-9]{0,10}pi[^a-z0-9]{0,10}lab", re.IGNORECASE)
PROF_RE = re.compile(
    r"prof(?:essor)?\.?[^a-z0-9]{1,12}youngjae[^a-z0-9]{1,6}yu",
    re.IGNORECASE,
)
NAMES = ("Chaeyun Kim", "Kihyun Kim", "Dasol Choi", "Haon Park")
ROLE_MARKERS = ("safety", "scor", "inspir")


def _name_re(name):
    return re.compile(
        r"[^a-z0-9]+".join(re.escape(part) for part in name.split()),
        re.IGNORECASE,
    )


def _windows_around(text, needle):
    for match in re.finditer(re.escape(needle), text, re.IGNORECASE):
        yield text[max(0, match.start() - WINDOW): match.end() + WINDOW]


def _missing_in(window):
    missing = [n for n in NAMES if not _name_re(n).search(window)]
    if not TEAM_RE.search(window):
        missing.append("AIM Intelligence Frontier Research Team")
    if not SNU_LAB_RE.search(window):
        missing.append("SNU PI Lab")
    if not PROF_RE.search(window):
        missing.append("Prof. Youngjae Yu")
    missing += [m for m in ROLE_MARKERS if m not in window.lower()]
    return missing


def test_brnd_3_2_readme_names_physharm_as_safety_scoring_inspiration():
    text = re.sub(r"\s+", " ", README.read_text(encoding="utf-8"))

    windows = list(_windows_around(text, "PhysHarm"))
    assert windows, "README.md never mentions PhysHarm"

    best_missing = None
    for window in windows:
        missing = _missing_in(window)
        if not missing:
            return
        if best_missing is None or len(missing) < len(best_missing):
            best_missing = missing

    raise AssertionError(
        "README.md mentions PhysHarm but never names it as the "
        "safety-scoring inspiration by the AIM Intelligence Frontier "
        "Research Team (Chaeyun Kim, Kihyun Kim, Dasol Choi, Haon Park) "
        "with SNU PI Lab (Prof. Youngjae Yu); the closest mention is "
        "still missing: " + ", ".join(best_missing)
    )
