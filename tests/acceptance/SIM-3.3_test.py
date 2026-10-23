"""SIM-3.3@1 kitchen.xml loads the rigged person asset produced by the NPC-1
pipeline, not the baked static OBJ.

The NPC-1 pipeline (env/person/rig_person.py) exports
assets/person/female_adult_01.glb — a skinned glTF on the Rocketbox Bip01
skeleton — while assets/person/{body,head,hair}.obj are baked static statue
meshes produced later for the old inline person body.

"The rigged person asset" in the composed MuJoCo scene means the person is
an articulated skeleton: a person subtree carrying bodies/joints named for
the Bip01 bones of that glb's skin (a real rig, at least 20 bone-named
members — NPC-1.2's own armature floor), not a single rigid body wearing
baked meshes. The test reads the glb's skin joints, finds the person
subtree in the model compiled from kitchen.xml (the body named "person",
or the subtree holding the Bip01-named bodies), and counts members whose
names match skeleton bones.

"Not the baked static OBJ": no file in kitchen.xml's include tree may
reference the baked assets/person/*.obj statue meshes.

Today the person is one rigid body with three baked OBJ visual geoms —
no Bip01 members and the OBJs are referenced directly in kitchen.xml.
"""

import json
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KITCHEN = REPO / "kitchen.xml"
PERSON_DIR = REPO / "assets" / "person"
MIN_BONES = 20

_norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())


def _glb_json(path):
    raw = path.read_bytes()
    magic, _v, _l = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF":
        raise ValueError(f"{path.name}: not a GLB")
    clen, ctype = struct.unpack_from("<I4s", raw, 12)
    if ctype.rstrip(b" \x00") != b"JSON":
        raise ValueError(f"{path.name}: first chunk is not JSON")
    return json.loads(raw[20:20 + clen])


def _npc1_skeleton():
    """Bip01 joint names of the NPC-1 skinned glTF, normalized."""
    docs = []
    for f in sorted(PERSON_DIR.rglob("*.glb")) + sorted(
            PERSON_DIR.rglob("*.gltf")):
        try:
            doc = _glb_json(f) if f.suffix == ".glb" else json.loads(
                f.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if str(doc.get("asset", {}).get("version", "")).startswith("2"):
            docs.append(doc)
    names = set()
    for doc in docs:
        nodes = doc.get("nodes", [])
        for skin in doc.get("skins") or []:
            for j in skin.get("joints") or []:
                if 0 <= j < len(nodes):
                    names.add(_norm(nodes[j].get("name") or ""))
    return names


def _include_tree(path, seen=None):
    """(kitchen.xml plus every resolved include) as raw text and paths."""
    seen = seen if seen is not None else {path.resolve()}
    files = [path.resolve()]
    for inc in ET.parse(path).iter("include"):
        f = inc.get("file")
        if not f:
            continue
        p = (path.parent / f).resolve()
        if p in seen or not p.is_file() or p.suffix != ".xml":
            continue
        seen.add(p)
        files.extend(_include_tree(p, seen))
    return files


def _descendant_bodies(model, root):
    out, stack = set(), [root]
    for b in range(model.nbody):
        if model.body_parentid[b] == root:
            stack.append(b)
    while stack:
        b = stack.pop()
        if b in out:
            continue
        out.add(b)
        for c in range(model.nbody):
            if model.body_parentid[c] == b:
                stack.append(c)
    return out


def test_sim_3_3_person_is_the_rigged_npc1_asset_not_baked_obj():
    skeleton = _npc1_skeleton()
    assert skeleton, (
        "no skinned glTF under assets/person/ — the NPC-1 pipeline's "
        "rigged asset must exist for kitchen.xml to load it")

    # not the baked static OBJ: no file in the include tree references the
    # statue meshes
    baked = {f.name for f in PERSON_DIR.glob("*.obj")}
    assert baked, "no baked OBJs found under assets/person/ to check against"
    refs = []
    for f in _include_tree(KITCHEN):
        text = f.read_text(encoding="utf-8", errors="replace")
        for b in baked:
            if re.search(r"[\w./\\-]*" + re.escape(b) + r"\b", text):
                refs.append(f"{f.relative_to(REPO)} references {b}")
    assert not refs, (
        "the scene still uses the baked static person OBJ(s): "
        + "; ".join(refs))

    # the rigged person: the person subtree must carry >= MIN_BONES members
    # named for the glb's Bip01 skeleton
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("NOHARM_HEADLESS", "1")
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(KITCHEN))

    def bname(i):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""

    def jname(i):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""

    def matches(name):
        n = _norm(name)
        if n in skeleton:
            return True
        # a converter may drop the common "Bip01"/"Bip01_" bone prefix
        return any(n and n == j[5:] for j in skeleton
                   if j.startswith("bip01"))

    candidates = [
        i for i in range(model.nbody) if bname(i) == "person"
    ] or list(range(1, model.nbody))
    best = (0, None)
    for root in candidates:
        sub = _descendant_bodies(model, root) | {root}
        names = {bname(b) for b in sub}
        for b in sub:
            for j in range(model.body_jntadr[b],
                           model.body_jntadr[b] + model.body_jntnum[b]):
                names.add(jname(j))
        hits = sum(1 for n in names if matches(n))
        if hits > best[0]:
            best = (hits, root)
    hits, root = best
    assert root is not None and hits >= MIN_BONES, (
        f"the person in the composed scene is not rigged: at most {hits} "
        f"member names match the NPC-1 Bip01 skeleton, need "
        f"{MIN_BONES}+ — she is the baked statue, not the rigged asset")
