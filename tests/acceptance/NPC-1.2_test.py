"""NPC-1.2@1 The exported person asset loads with an armature of 20 or more
bones.

'The exported person asset' is the skinned glTF NPC-1.1's pipeline writes under
assets/person/. 'Loads' is judged by a conforming glTF read of the committed
file: it parses as glTF 2.0 and every external resource it declares (buffers,
images) resolves — a file a loader would reject is not a loadable asset. Its
'armature' is the skeleton a loader reconstructs from the skin's joints: the
joint nodes connected through the node hierarchy. Twenty or more bones means
one connected joint subtree of at least twenty nodes — an actual skeleton, not
a baked statue or a handful of root markers.
"""

import json
import struct
import urllib.parse
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO / "assets" / "person"
MIN_BONES = 20


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


def _missing_resources(path, doc):
    """External URIs the document declares that do not resolve next to it."""
    missing = []
    for kind in ("buffers", "images"):
        for entry in doc.get(kind) or []:
            uri = entry.get("uri")
            if uri and not uri.startswith("data:"):
                rel = urllib.parse.unquote(uri)
                if not (path.parent / rel).is_file():
                    missing.append(rel)
    return missing


def _largest_armature(doc):
    """Size of the biggest connected joint subtree across the doc's skins —
    the bone count of the armature a loader rebuilds from them."""
    nodes = doc.get("nodes", [])
    parent = {}
    for i, node in enumerate(nodes):
        for c in node.get("children") or []:
            parent[c] = i
    best = 0
    for skin in doc.get("skins") or []:
        joints = {j for j in (skin.get("joints") or []) if 0 <= j < len(nodes)}
        groups = {}
        for j in joints:
            root = j
            while parent.get(root) in joints:
                root = parent[root]
            groups.setdefault(root, set()).add(j)
        best = max(best, max((len(g) for g in groups.values()), default=0))
    return best


def test_npc_1_2_exported_person_asset_loads_with_armature_of_20_bones():
    docs = _gltf_docs()
    assert docs, (
        "no glTF 2.0 person asset found under assets/person/ — the exported "
        "asset must be committed for it to load at all"
    )

    seen = []
    for f, doc in docs:
        rel = str(f.relative_to(REPO))
        missing = _missing_resources(f, doc)
        if missing:
            seen.append(f"{rel}: does not load, missing resources "
                        + ", ".join(missing))
            continue
        bones = _largest_armature(doc)
        if bones >= MIN_BONES:
            return
        seen.append(f"{rel}: armature of {bones} bone(s)")
    raise AssertionError(
        "no exported person asset under assets/person/ loads with an "
        f"armature of {MIN_BONES} or more bones — "
        + "; ".join(seen)
    )
