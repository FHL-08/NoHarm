"""BRND-1.2@1 README.md's H1 reads `Harmless?` and describes the project as an
open-source safety evaluation platform for physical AI in household environments."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DESCRIPTION = (
    "open-source safety evaluation platform for physical ai in household environments"
)


def test_brnd_1_2_readme_h1_and_description():
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    h1 = next((line for line in readme.splitlines() if line.startswith("# ")), None)
    assert h1 is not None, "README.md contains no H1 heading"
    assert h1.strip() == "# Harmless?", (
        f"README.md H1 is {h1.strip()!r}; expected '# Harmless?'"
    )

    normalized = re.sub(r"\s+", " ", readme).lower()
    assert DESCRIPTION in normalized, (
        "README.md does not describe the project as an open-source safety "
        "evaluation platform for physical AI in household environments"
    )
