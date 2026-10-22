"""NPC-1.1@1 A Blender headless script rig-loads the upstream Rocketbox FBX and
exports a skinned glTF with the Bip01 skeleton.

The line is judged on committed files, since the pipeline's product — the
exported person asset — lives in the repo under assets/person/:

  - the script: a Blender headless (bpy) python file under env/person/ or
    assets/person/ that brings in an FBX (bpy.ops.import_scene.fbx or the
    newer wm.fbx_import importer), writes glTF (export_scene.gltf), and names
    the upstream Rocketbox FBX as its source. A bpy script is runnable via
    `blender --background --python`, which is what 'headless' means here.
  - the export: a glTF 2.0 document under assets/person/ that is *skinned* —
    at least one skin whose joints are bound to a mesh through
    JOINTS_0/WEIGHTS_0 vertex attributes — and whose joints carry the Rocketbox
    biped's Bip01 bone names, not a hand-made or baked-statue rig.
"""

import json
import re
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIRS = (REPO / "env" / "person", REPO / "assets" / "person")
ASSET_DIR = REPO / "assets" / "person"

HAS_BPY = re.compile(r"\bbpy\b")
FBX_IMPORT = re.compile(r"import_scene\s*\.\s*fbx|fbx_import|import_fbx", re.IGNORECASE)
GLTF_EXPORT = re.compile(r"export_scene\s*\.\s*gltf|export_gltf|gltf_export", re.IGNORECASE)
ROCKETBOX_FBX = re.compile(r"rocketbox|female[_ ]?adult|male[_ ]?adult", re.IGNORECASE)
FBX_NAME = re.compile(r"[\w./\\-]+\.fbx", re.IGNORECASE)


def _gltf_json(path):
    """Return the glTF JSON document of a .gltf or .glb file."""
    raw = path.read_bytes()
    if path.suffix.lower() == ".glb":
        magic, _version, _length = struct.unpack_from("<4sII", raw, 0)
        if magic != b"glTF":
            raise ValueError(f"{path.name}: not a GLB container")
        clen, ctype = struct.unpack_from("<I4s", raw, 12)
        if ctype.rstrip(b" \x00") != b"JSON":
            raise ValueError(f"{path.name}: first GLB chunk is not JSON")
        return json.loads(raw[20:20 + clen])
    return json.loads(raw.decode("utf-8-sig"))


def _gltf_docs():
    """Every parseable glTF 2.0 document under assets/person/, as (path, doc)."""
    docs = []
    for pat in ("*.gltf", "*.glb"):
        for f in sorted(ASSET_DIR.rglob(pat)):
            try:
                doc = _gltf_json(f)
            except Exception:
                continue
            if str(doc.get("asset", {}).get("version", "")).startswith("2"):
                docs.append((f, doc))
    return docs


def _node_name(doc, i):
    return doc.get("nodes", [])[i].get("name") or ""


def _is_skinned_bip01(doc):
    """True when the document has a skin bound to a mesh whose joints are all
    named for the Bip01 biped (the Rocketbox skeleton)."""
    nodes = doc.get("nodes", [])
    for skin in doc.get("skins") or []:
        joints = skin.get("joints") or []
        if not joints or "inverseBindMatrices" not in skin:
            continue
        if not all(_node_name(doc, j).startswith("Bip01") for j in joints):
            continue
        for node in nodes:
            mesh = doc.get("meshes", [])[node["mesh"]] if "mesh" in node else None
            if node.get("skin") is None or mesh is None:
                continue
            for prim in mesh.get("primitives", []):
                attrs = prim.get("attributes", {})
                if "JOINTS_0" in attrs and "WEIGHTS_0" in attrs:
                    return True
    return False


def test_npc_1_1_blender_script_rig_loads_rocketbox_fbx_exports_bip01_gltf():
    candidates = [
        f for d in SCRIPT_DIRS if d.is_dir() for f in sorted(d.rglob("*.py"))
    ]
    scripts = []
    for f in candidates:
        text = f.read_text(encoding="utf-8", errors="replace")
        if (
            HAS_BPY.search(text)
            and FBX_IMPORT.search(text)
            and GLTF_EXPORT.search(text)
            and FBX_NAME.search(text)
            and ROCKETBOX_FBX.search(text)
        ):
            scripts.append(f)
    assert scripts, (
        "no Blender headless script found under env/person/ or assets/person/ "
        "that imports an FBX (bpy.ops.import_scene.fbx / wm.fbx_import) and "
        "exports glTF (export_scene.gltf) from the upstream Rocketbox source; "
        "checked: " + (", ".join(str(p.relative_to(REPO)) for p in candidates)
                       or "no .py files at all")
    )

    docs = _gltf_docs()
    assert docs, (
        "no glTF 2.0 person asset found under assets/person/ — the Blender "
        "script's export must be committed for the line to be true"
    )
    good = [str(f.relative_to(REPO)) for f, doc in docs if _is_skinned_bip01(doc)]
    assert good, (
        "no exported person asset under assets/person/ is a skinned glTF with "
        "the Bip01 skeleton: expected a skin bound to a mesh via "
        "JOINTS_0/WEIGHTS_0 whose every joint name starts with 'Bip01'; "
        "checked: " + ", ".join(str(f.relative_to(REPO)) for f, _ in docs)
    )
