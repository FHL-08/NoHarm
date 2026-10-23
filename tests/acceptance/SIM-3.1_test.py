"""SIM-3.1@1 kitchen.xml includes the generated assets/ithor_kitchen.xml, the
Menagerie G1, and the person without redefining their meshes.

kitchen.xml only composes: its scene comes from included assets — the
generated iTHOR kitchen (assets/ithor_kitchen.xml, produced by
env/kitchen/make_room.py), the MuJoCo Menagerie Unitree G1
(assets/unitree_g1/), and the person — and it declares no meshes of its own
for them. The test parses kitchen.xml and follows its <include> tree:

  - the generated iTHOR file is included: an include resolves to
    assets/ithor_kitchen.xml, that file carries its generator marker
    (authored by env/kitchen/make_room.py, not by hand), and a generator
    script under env/kitchen/ names it as its output;
  - the Menagerie G1 is included: an include resolves to a g1*.xml file
    under assets/unitree_g1/ that defines the pelvis-rooted robot;
  - the person is included too: a body named "person" (the name the scene's
    welds and the executor contract use) is declared inside one of the
    included files, not authored inline in kitchen.xml's own worldbody —
    or, equivalently, a dedicated included person file provides the rigged
    person subtree;
  - no redefinition: kitchen.xml declares no <mesh> asset that re-uses a
    mesh name an included file already declares, and no <mesh> whose file
    lives inside the person asset directory (assets/person/) — the
    composed assets' meshes come from their own files.

The current file fails: it includes the kitchen and the G1 but authors the
person inline, declaring her baked OBJ meshes in its own <asset> block.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
KITCHEN = REPO / "kitchen.xml"
ITHOR = REPO / "assets" / "ithor_kitchen.xml"
G1_DIR = REPO / "assets" / "unitree_g1"
PERSON_DIR = REPO / "assets" / "person"

BODY_RE = re.compile(r"<body\b[^>]*\bname\s*=\s*\"([^\"]+)\"")
MESH_RE = re.compile(r"<mesh\b[^>]*>")


def _parse(path):
    return ET.parse(path).getroot()


def _included_files(path, seen=None):
    """Every file reached through <include> elements, transitively, resolved
    to absolute paths (the entry file itself first)."""
    seen = seen if seen is not None else {path.resolve()}
    files = [path.resolve()]
    for inc in _parse(path).iter("include"):
        f = inc.get("file")
        if not f:
            continue
        p = (path.parent / f).resolve()
        if p in seen or not p.is_file() or p.suffix != ".xml":
            continue
        seen.add(p)
        files.extend(_included_files(p, seen))
    return files


def _bodies(path):
    """Body names a file declares in its own worldbody (inclusion merges
    these into the composed scene)."""
    return set(BODY_RE.findall(path.read_text(encoding="utf-8",
                                                errors="replace")))


def _declared_meshes(path):
    """(name, file) pairs of <mesh> assets a file declares."""
    out = []
    root = _parse(path)
    for mesh in root.iter("mesh"):
        out.append((mesh.get("name"), mesh.get("file")))
    return out


def test_sim_3_1_kitchen_xml_includes_assets_without_redefining_meshes():
    assert KITCHEN.is_file(), "kitchen.xml does not exist"
    includes = _included_files(KITCHEN)

    # the generated iTHOR kitchen is included
    assert ITHOR in includes, (
        "kitchen.xml does not include assets/ithor_kitchen.xml — "
        f"includes found: {[str(p.relative_to(REPO)) for p in includes]}")
    head = ITHOR.read_text(encoding="utf-8", errors="replace")[:800]
    assert re.search(r"generated", head, re.IGNORECASE), (
        "assets/ithor_kitchen.xml carries no generated-artifact marker — "
        "the line requires the *generated* file (env/kitchen/make_room.py "
        "output), not a hand-authored one")
    generators = [
        f for f in sorted((REPO / "env" / "kitchen").glob("*.py"))
        if "ithor" in f.read_text(encoding="utf-8", errors="replace").lower()
    ]
    assert generators, (
        "no generator under env/kitchen/ produces the iTHOR file — "
        "the line requires assets/ithor_kitchen.xml to be generated, "
        "not hand-edited")

    # the Menagerie G1 is included
    g1_includes = [
        p for p in includes
        if G1_DIR in p.parents and p.name.startswith("g1")
    ]
    assert g1_includes, (
        "kitchen.xml does not include the Menagerie G1 — no include "
        "resolves to assets/unitree_g1/g1*.xml; includes found: "
        f"{[str(p.relative_to(REPO)) for p in includes]}")
    assert any("pelvis" in _bodies(p) for p in g1_includes), (
        "the included assets/unitree_g1 file defines no pelvis-rooted "
        "robot — the Menagerie G1's root body is 'pelvis'")

    # the person is included: a body named "person" declared by an
    # included file, not by kitchen.xml's own worldbody (the scene's welds
    # and the executor contract address her as 'person')
    included_only = [p for p in includes if p != KITCHEN]
    person_in_include = any("person" in _bodies(p) for p in included_only)
    assert person_in_include, (
        "no included file declares a body named 'person' — the person is "
        "authored inside kitchen.xml itself, but the line requires "
        "kitchen.xml to *include* her like the other assets")
    if "person" in _bodies(KITCHEN):
        pytest.fail(
            "kitchen.xml declares the person body inline — under the "
            "line she is one of the included assets, not authored in "
            "the composer file")

    # no mesh redefinition: kitchen.xml must not re-declare meshes the
    # included assets own, nor reach into the person asset's files
    kitchen_meshes = _declared_meshes(KITCHEN)
    included_mesh_names = {
        n for p in includes if p != KITCHEN
        for n, _f in _declared_meshes(p) if n
    }
    bad = []
    for name, file in kitchen_meshes:
        if name and name in included_mesh_names:
            bad.append(f"mesh '{name}' redeclares an included asset's mesh")
        if file:
            resolved = (KITCHEN.parent / file).resolve()
            if PERSON_DIR in resolved.parents or resolved.parent == PERSON_DIR:
                bad.append(f"mesh '{name}' file {file} reaches into "
                           "assets/person/ — the person asset owns its "
                           "meshes")
    assert not bad, "kitchen.xml redefines composed assets' meshes: " + \
        "; ".join(bad)

    # sanity: the composed scene still compiles and carries all three actors
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("NOHARM_HEADLESS", "1")
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(KITCHEN))
    names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    }
    assert "pelvis" in names, "composed model has no G1 (no pelvis body)"
    assert "person" in names, "composed model has no person body"
