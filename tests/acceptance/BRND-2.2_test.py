"""BRND-2.2@1 README.md renders the logo in its header block at a stated pixel width."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
SRC_RE = re.compile(
    r"""src\s*=\s*["']\.?/?assets/brand/logo\.svg["']""", re.IGNORECASE
)
WIDTH_RE = re.compile(
    r"""width\s*=\s*["']?\d+(?:px)?["']?|width\s*:\s*\d+(?:\.\d+)?px""",
    re.IGNORECASE,
)


def _header_block(readme: str) -> str:
    lines = readme.splitlines()
    end = next(
        (i for i, line in enumerate(lines) if line.strip() == "---"),
        len(lines),
    )
    return "\n".join(lines[:end])


def test_brnd_2_2_readme_header_renders_logo_at_stated_pixel_width():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    header = _header_block(readme)

    imgs = IMG_RE.findall(header)
    assert imgs, (
        "README.md header block (everything before the first '---' rule) "
        "contains no <img> tag; the logo must be rendered there so a pixel "
        "width can be stated"
    )

    logo_imgs = [tag for tag in imgs if SRC_RE.search(tag)]
    assert logo_imgs, (
        "README.md header block has <img> tag(s) but none references "
        "assets/brand/logo.svg: " + " | ".join(imgs)
    )

    assert any(WIDTH_RE.search(tag) for tag in logo_imgs), (
        "the logo <img> in README.md's header block does not state a pixel "
        "width (expected width=\"N\" or style width:Npx): "
        + " | ".join(logo_imgs)
    )
