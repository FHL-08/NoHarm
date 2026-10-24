"""Bake the NPC-1 rigged person into assets/ithor_kitchen.xml as an MJCF subtree.

Source: assets/person/female_adult_01.glb — the skinned Rocketbox
Female_Adult_01 (Bip01 skeleton) exported by env/person/rig_person.py.
MuJoCo cannot load a skinned glTF, so this bakes it the way the scene
consumes it: one MJCF body per skin joint (named after the Bip01 bone),
placed at its bind-pose transform inside the real Bip01 hierarchy, each
carrying the skin faces that bone dominates (dominant-weight assignment,
baked through the same LBS path as bake_glb.py). Meshes/textures land in
assets/ithor_kitchen/person/ and the marked person block is spliced into
assets/ithor_kitchen.xml — the only file the scene's include tree places
inside the meshdir, so its asset paths resolve for both relative and
absolute model loads (and for Newton's per-file resolver). make_room.py
calls inject() after writing the room file so regeneration keeps her.

Usage: .venv/bin/python env/kitchen/make_person.py
"""
import io
import json
import re
import struct
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GLB = REPO / "assets" / "person" / "female_adult_01.glb"
ITHOR_XML = REPO / "assets" / "ithor_kitchen.xml"
MESH_DIR = REPO / "assets" / "ithor_kitchen" / "person"

MARK = "<!-- GENERATED person block: env/kitchen/make_person.py -->"
MARK_END = "<!-- /GENERATED person block -->"

CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
      5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

BONE_MASS_KG = 45.0        # skeleton segment masses, total
COLLIDER_MASS_KG = 16.0    # kept on the root capsules (was 61 kg pre-rig)
MIN_BONE_MASS = 0.02
PERSON_POS = "1.75 1.5 0"

Y2Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # glTF y-up -> z-up


def _load_glb(path):
    raw = path.read_bytes()
    magic, _v, _l = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF":
        raise ValueError(f"{path}: not a GLB")
    clen, ctype = struct.unpack_from("<I4s", raw, 12)
    if ctype.rstrip(b" \x00") != b"JSON":
        raise ValueError(f"{path}: first chunk is not JSON")
    doc = json.loads(raw[20:20 + clen])
    off = 20 + clen
    blen, _bt = struct.unpack_from("<I4s", raw, off)
    return doc, raw[off + 8:off + 8 + blen]


def _quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _orthonormal(m3):
    return np.stack([m3[:, k] / max(np.linalg.norm(m3[:, k]), 1e-12)
                     for k in range(3)], axis=1)


def _quat_of(r):
    m00, m01, m02 = r[0]
    m10, m11, m12 = r[1]
    m20, m21, m22 = r[2]
    t = m00 + m11 + m22
    if t > 0:
        s = 0.5 / np.sqrt(t + 1)
        q = [0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s]
    elif m00 > m11 and m00 > m22:
        s = 2 * np.sqrt(1 + m00 - m11 - m22)
        q = [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s]
    elif m11 > m22:
        s = 2 * np.sqrt(1 + m11 - m00 - m22)
        q = [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s]
    else:
        s = 2 * np.sqrt(1 + m22 - m00 - m11)
        q = [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s]
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def _clean(name):
    return re.sub(r"[^A-Za-z0-9]", "_", name).strip("_")


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def bake():
    """Returns (asset_lines, body_lines) and writes meshes/textures into
    assets/ithor_kitchen/person/."""
    doc, blob = _load_glb(GLB)
    nodes = doc["nodes"]

    def acc(i):
        a = doc["accessors"][i]
        bv = doc["bufferViews"][a["bufferView"]]
        dt = np.dtype(CT[a["componentType"]])
        n = NC[a["type"]]
        off = (bv.get("byteOffset") or 0) + (a.get("byteOffset") or 0)
        stride = bv.get("byteStride") or dt.itemsize * n
        rr = np.frombuffer(blob, np.uint8,
                           count=stride * (a["count"] - 1) + dt.itemsize * n,
                           offset=off)
        arr = np.lib.stride_tricks.as_strided(
            rr, (a["count"], dt.itemsize * n), (stride, 1)).copy()
        out = arr.view(dt).reshape(a["count"], n)
        if a.get("normalized"):
            out = out.astype(np.float32) / np.iinfo(dt).max
        return out

    parent = {c: i for i, n in enumerate(nodes)
              for c in (n.get("children") or [])}
    local = {}
    for i, n in enumerate(nodes):
        if n.get("matrix"):
            local[i] = np.array(n["matrix"], float).reshape(4, 4).T
        else:
            t = np.array(n.get("translation") or [0, 0, 0], float)
            r = np.array(n.get("rotation") or [0, 0, 0, 1], float)
            s = np.array(n.get("scale") or [1, 1, 1], float)
            m = np.eye(4)
            m[:3, :3] = _quat_mat(r / np.linalg.norm(r)) * s
            m[:3, 3] = t
            local[i] = m
    cache = {}

    def glob(i):
        if i not in cache:
            cache[i] = (glob(parent[i]) @ local[i]) if i in parent else local[i]
        return cache[i]
    G = [glob(i) for i in range(len(nodes))]

    skin = doc["skins"][0]
    ibm = acc(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    J = np.stack([G[j] @ ibm[k] for k, j in enumerate(skin["joints"])])

    # skinned verts per primitive at bind pose (y-up world)
    prims = []
    mesh = doc["meshes"][next(n["mesh"] for n in nodes
                             if n.get("mesh") is not None)]
    for prim in mesh["primitives"]:
        p = acc(prim["attributes"]["POSITION"]).astype(float)
        jo = acc(prim["attributes"]["JOINTS_0"]).astype(int)
        w = acc(prim["attributes"]["WEIGHTS_0"]).astype(float)
        w = w / w.sum(1, keepdims=True)
        pw = np.einsum("vk,vkij,vj->vi", w, J[jo],
                       np.c_[p, np.ones(len(p))])[:, :3]
        uv = (acc(prim["attributes"]["TEXCOORD_0"]).astype(float)
              if prim["attributes"].get("TEXCOORD_0") is not None else None)
        prims.append({"pw": pw, "uv": uv,
                      "idx": acc(prim["indices"]).reshape(-1, 3),
                      "dom": jo[np.arange(len(jo)), w.argmax(1)],
                      "mat": prim.get("material")})

    # person frame: z-up, feet on z=0; the glb already faces -y
    floor = min(p["pw"][:, 1].min() for p in prims)
    shift = np.array([0.0, 0.0, -floor])
    F = {}   # joint node -> (R, t) rigid frame in person space
    for j in skin["joints"]:
        F[j] = (_orthonormal(Y2Z @ G[j][:3, :3]), Y2Z @ G[j][:3, 3] + shift)

    # face -> dominant skin joint
    bone_faces = {}
    for p in prims:
        for f in p["idx"]:
            counts = {}
            for v in f:
                counts[p["dom"][v]] = counts.get(p["dom"][v], 0) + 1
            slot = max(counts, key=counts.get)
            bone_faces.setdefault((skin["joints"][slot], p["mat"]),
                                  []).append((p, f))

    MESH_DIR.mkdir(parents=True, exist_ok=True)
    for old in MESH_DIR.glob("*"):
        old.unlink()

    # per-(bone, material) OBJs, verts baked into the bone frame
    meshes = {}
    bone_verts = {j: [] for j in skin["joints"]}
    for (j, mat), faces in sorted(bone_faces.items()):
        r, t = F[j]
        verts, uvs, tris = [], [], []
        remap = {}
        for p, f in faces:
            tri = []
            for vi in f:
                k = (id(p), int(vi))
                if k not in remap:
                    loc = r.T @ ((p["pw"][vi] @ Y2Z.T) + shift - t)
                    verts.append(loc)
                    uvs.append(p["uv"][vi] if p["uv"] is not None
                               else np.zeros(2))
                    remap[k] = len(verts)
                tri.append(remap[k])
            tris.append(tri)
            bone_verts[j].extend(p["pw"][v] for v in f)
        fn = f"p{j}_m{mat}.obj"
        with open(MESH_DIR / fn, "w") as fh:
            for v in verts:
                fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for uv in uvs:
                fh.write(f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}\n")
            for tri in tris:
                fh.write("f " + " ".join(f"{v}/{v}" for v in tri) + "\n")
        meshes[(j, mat)] = fn

    # embedded textures next to the meshes
    tex_files = {}
    for mi, mat in enumerate(doc.get("materials") or []):
        pbr = mat.get("pbrMetallicRoughness") or {}
        ti = (pbr.get("baseColorTexture") or {}).get("index")
        if ti is None:
            continue
        img = doc["images"][doc["textures"][ti]["source"]]
        bv = doc["bufferViews"][img["bufferView"]]
        start = bv.get("byteOffset") or 0
        Image.open(io.BytesIO(blob[start:start + bv["byteLength"]])).save(
            MESH_DIR / f"tex_{mi}.png")
        tex_files[mi] = f"tex_{mi}.png"

    # segment masses from vertex counts, normalized to BONE_MASS_KG
    counts = np.array([max(len(bone_verts[j]), 1) for j in skin["joints"]],
                      float)
    masses = np.clip(counts / counts.sum() * BONE_MASS_KG, MIN_BONE_MASS, None)
    masses *= BONE_MASS_KG / masses.sum()
    mass_of = {j: masses[k] for k, j in enumerate(skin["joints"])}

    # MJCF tree: mirror the skin hierarchy under body "person"
    joints_set = set(skin["joints"])
    root_j = next(j for j in skin["joints"]
                  if parent.get(j) not in joints_set)
    bone_meshes = {}
    for (j, mat), fn in meshes.items():
        bone_meshes.setdefault(j, []).append((mat, fn))
    body_lines, head_top = [], None

    def emit(j, depth):
        nonlocal head_top
        pad = "  " * depth
        r, t = F[j]
        pj = parent.get(j)
        if pj in joints_set:
            pr, pt = F[pj]
            rl, tl = pr.T @ r, pr.T @ (t - pt)
        else:
            rl, tl = r, t
        vs = np.array(bone_verts[j]) if bone_verts[j] else None
        if vs is not None and len(vs):
            vs_z = vs @ Y2Z.T + shift
            com = r.T @ (vs_z.mean(0) - t)
            d = (vs_z - vs_z.mean(0)) @ r   # vert offsets in the bone frame
            i3 = np.maximum(
                mass_of[j] * np.array([(d[:, 1] ** 2 + d[:, 2] ** 2).mean(),
                                       (d[:, 0] ** 2 + d[:, 2] ** 2).mean(),
                                       (d[:, 0] ** 2 + d[:, 1] ** 2).mean()]),
                1e-6)
            if _clean(nodes[j].get("name")) == "Bip01_Head":
                head_top = vs_z[vs_z[:, 2].argmax()]
        else:
            com, i3 = np.zeros(3), np.full(3, 1e-5)
        body_lines.append(
            f'{pad}<body name="{_clean(nodes[j].get("name"))}" '
            f'pos="{_fmt(tl)}" quat="{_fmt(_quat_of(rl))}">')
        body_lines.append(f'{pad}  <inertial pos="{_fmt(com)}" '
                          f'mass="{mass_of[j]:.4f}" diaginertia="{_fmt(i3)}"/>')
        for mat, fn in sorted(bone_meshes.get(j, [])):
            body_lines.append(
                f'{pad}  <geom name="pg_{j}_{mat}" type="mesh" '
                f'mesh="pm_{j}_{mat}" material="pmat_{mat}" '
                f'contype="0" conaffinity="0" group="1"/>')
        for c in nodes[j].get("children") or []:
            if c in joints_set:
                emit(c, depth + 1)
        body_lines.append(f"{pad}</body>")

    emit(root_j, 2)

    site_pos = _fmt(head_top) if head_top is not None else "0 0.02 1.6"
    cm = COLLIDER_MASS_KG
    body = [
        f'  <body name="person" pos="{PERSON_POS}" quat="0 0 0 1">',
        '    <freejoint name="person_free"/>',
        '    <geom name="person_legs" type="capsule" fromto="0 0 0.14 0 0 0.85"'
        f' size="0.14" rgba="0 0 0 0" group="4" mass="{cm * 0.4:.3f}" friction="0.8"/>',
        '    <geom name="person_torso" type="capsule" fromto="0 0 0.95 0 0 1.38"'
        f' size="0.17" rgba="0 0 0 0" group="4" mass="{cm * 0.4:.3f}"/>',
        '    <geom name="person_arms" type="capsule" fromto="-0.16 0 0.95 0.16 0 0.95"'
        f' size="0.06" rgba="0 0 0 0" group="4" mass="{cm * 0.1:.3f}"/>',
        '    <geom name="person_head" type="sphere" pos="0 0.02 1.6" size="0.11"'
        f' rgba="0 0 0 0" group="4" mass="{cm * 0.1:.3f}"/>',
        f'    <site name="person_head_site" pos="{site_pos}" size="0.01" rgba="0 0 0 0"/>',
        *body_lines,
        '  </body>',
    ]
    assets = []
    for mi, fn in sorted(tex_files.items()):
        assets.append(f'<texture name="ptex_{mi}" type="2d" '
                      f'file="ithor_kitchen/person/{fn}"/>')
        assets.append(f'<material name="pmat_{mi}" texture="ptex_{mi}" '
                      f'specular="0.1"/>')
    for (j, mat), fn in sorted(meshes.items()):
        assets.append(f'<mesh name="pm_{j}_{mat}" '
                      f'file="ithor_kitchen/person/{fn}"/>')
    return assets, body, len(meshes), sum(mass_of.values()) + cm


def inject(path=ITHOR_XML):
    """Write the person meshes and splice the marked person block into
    assets/ithor_kitchen.xml (idempotent)."""
    assets, body, nmesh, total = bake()
    text = Path(path).read_text()
    text = re.sub(
        re.escape(MARK) + r".*?" + re.escape(MARK_END) + r"\n?",
        "", text, flags=re.DOTALL)
    asset_block = (f"{MARK}\n" + "\n".join(f"    {ln}" for ln in assets)
                   + f"\n    {MARK_END}\n")
    body_block = (f"{MARK}\n" + "\n".join(f"  {ln}" for ln in body)
                  + f"\n  {MARK_END}\n")
    text = text.replace("  </asset>", asset_block + "  </asset>", 1)
    text = text.replace("  </worldbody>", body_block + "  </worldbody>", 1)
    Path(path).write_text(text)
    print(f"make_person: {nmesh} bone meshes, person mass {total:.1f} kg "
          f"-> {Path(path).relative_to(REPO)}")


if __name__ == "__main__":
    inject()
