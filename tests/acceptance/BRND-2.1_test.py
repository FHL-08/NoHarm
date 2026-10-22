"""BRND-2.1@1 assets/brand/logo.svg parses as SVG with a viewBox and no raster elements."""

import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOGO = REPO / "assets" / "brand" / "logo.svg"
RASTER_MARKERS = (
    "data:image/png",
    "data:image/jpeg",
    "data:image/jpg",
    "data:image/gif",
    "data:image/bmp",
    "data:image/webp",
    "data:image/tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".tiff",
)


def test_brnd_2_1_logo_svg_parses_with_viewbox_and_no_raster():
    assert LOGO.is_file(), "assets/brand/logo.svg does not exist"

    text = LOGO.read_text(encoding="utf-8")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise AssertionError(
            f"assets/brand/logo.svg does not parse as XML/SVG: {exc}"
        ) from exc

    root_tag = root.tag.rsplit("}", 1)[-1].lower()
    assert root_tag == "svg", (
        f"root element of assets/brand/logo.svg is <{root_tag}>, expected <svg>"
    )

    assert root.get("viewBox"), (
        "assets/brand/logo.svg has no viewBox attribute; "
        "a vector logo needs one to scale from favicon to poster"
    )

    image_elements = [
        el
        for el in root.iter()
        if el.tag.rsplit("}", 1)[-1].lower() == "image"
    ]
    assert not image_elements, (
        "assets/brand/logo.svg contains raster <image> element(s); "
        "the logo must be pure vector"
    )

    hits = [m for m in RASTER_MARKERS if m in text.lower()]
    assert not hits, (
        "assets/brand/logo.svg references or embeds raster data: "
        + ", ".join(hits)
    )
