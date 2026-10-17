"""Bake a skinned glTF character at animation frame(s) into static OBJ meshes (z-up, metres)."""
import sys, os, struct, numpy as np, pygltflib

src, anim_name, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
times = [float(t) for t in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0.0]
g = pygltflib.GLTF2().load(src)
blob = g.binary_blob()
CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def acc(i):
    a = g.accessors[i]
    bv = g.bufferViews[a.bufferView]
    dt = np.dtype(CT[a.componentType])
    n = NC[a.type]
    off = (bv.byteOffset or 0) + (a.byteOffset or 0)
    stride = bv.byteStride or dt.itemsize * n
    raw = np.frombuffer(blob, np.uint8, count=stride * (a.count - 1) + dt.itemsize * n, offset=off)
    arr = np.lib.stride_tricks.as_strided(raw, (a.count, dt.itemsize * n), (stride, 1)).copy()
    out = arr.view(dt).reshape(a.count, n)
    if a.normalized:
        out = out.astype(np.float32) / np.iinfo(dt).max
    return out


def quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


anim = next((a for a in g.animations if a.name == anim_name), None)
parent = {c: i for i, n in enumerate(g.nodes) for c in (n.children or [])}


def local_trs(t):
    T = {i: np.array(n.translation or [0, 0, 0], float) for i, n in enumerate(g.nodes)}
    R = {i: np.array(n.rotation or [0, 0, 0, 1], float) for i, n in enumerate(g.nodes)}
    S = {i: np.array(n.scale or [1, 1, 1], float) for i, n in enumerate(g.nodes)}
    if anim:
        for ch in anim.channels:
            s = anim.samplers[ch.sampler]
            ts, vs = acc(s.input)[:, 0], acc(s.output)
            k = int(np.clip(np.searchsorted(ts, t % ts[-1]), 0, len(ts) - 1))
            v = vs[k]
            {"translation": T, "rotation": R, "scale": S}[ch.target.path][ch.target.node] = v.astype(float)
    M = {}
    for i, n in enumerate(g.nodes):
        if n.matrix:
            M[i] = np.array(n.matrix, float).reshape(4, 4).T
        else:
            m = np.eye(4)
            m[:3, :3] = quat_mat(R[i] / np.linalg.norm(R[i])) * S[i]
            m[:3, 3] = T[i]
            M[i] = m
    G = {}
    def glob(i):
        if i not in G:
            G[i] = (glob(parent[i]) @ M[i]) if i in parent else M[i]
        return G[i]
    out = [glob(i) for i in range(len(g.nodes))]
    # optional rest-pose tweak: POSE="Bone Name:deg,..." rotates that bone's
    # subtree about the glTF forward (z) axis through the bone origin
    for item in filter(None, os.environ.get("POSE", "").split(",")):
        bn, deg = item.rsplit(":", 1)
        bi = next(i for i, n in enumerate(g.nodes) if n.name == bn)
        a = np.radians(float(deg)); c, s_ = np.cos(a), np.sin(a)
        Rz = np.eye(4); Rz[:2, :2] = [[c, -s_], [s_, c]]
        p = out[bi][:3, 3].copy()
        T1, T2 = np.eye(4), np.eye(4); T1[:3, 3] = p; T2[:3, 3] = -p
        W = T1 @ Rz @ T2
        sub, stack = set(), [bi]
        while stack:
            k = stack.pop(); sub.add(k); stack += g.nodes[k].children or []
        for k in sub:
            out[k] = W @ out[k]
    return out


os.makedirs(out_dir, exist_ok=True)
# save textures
tex_files = {}
for mi, mat in enumerate(g.materials):
    pbr = mat.pbrMetallicRoughness
    if pbr and pbr.baseColorTexture is not None and g.images[g.textures[pbr.baseColorTexture.index].source].bufferView is not None:
        img = g.images[g.textures[pbr.baseColorTexture.index].source]
        bv = g.bufferViews[img.bufferView]
        fn = f"tex_{mi}.png"
        import io
        from PIL import Image
        Image.open(io.BytesIO(blob[bv.byteOffset or 0:(bv.byteOffset or 0) + bv.byteLength])).convert("RGB").save(os.path.join(out_dir, fn))
        tex_files[mi] = fn
Y2Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # glTF y-up -> z-up
for fi, t in enumerate(times):
    Gs = local_trs(t)
    for ni, node in enumerate(g.nodes):
        if node.mesh is None:
            continue
        skin = g.skins[node.skin] if node.skin is not None else None
        if skin:
            ibm = acc(skin.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1)
            J = np.stack([Gs[j] @ ibm[k] for k, j in enumerate(skin.joints)])
        for pi, prim in enumerate(g.meshes[node.mesh].primitives):
            P = acc(prim.attributes.POSITION).astype(float)
            if skin:
                jo, w = acc(prim.attributes.JOINTS_0).astype(int), acc(prim.attributes.WEIGHTS_0).astype(float)
                w /= w.sum(1, keepdims=True)
                Ph = np.c_[P, np.ones(len(P))]
                P = np.einsum("vk,vkij,vj->vi", w, J[jo], Ph)[:, :3]
            else:
                P = (Gs[ni] @ np.c_[P, np.ones(len(P))].T).T[:, :3]
            P = P @ Y2Z.T
            uv = acc(prim.attributes.TEXCOORD_0) if prim.attributes.TEXCOORD_0 is not None else None
            idx = acc(prim.indices).reshape(-1, 3)
            name = f"f{fi}_n{ni}_p{pi}_m{prim.material}"
            with open(os.path.join(out_dir, name + ".obj"), "w") as f:
                f.write("".join(f"v {a:.5f} {b:.5f} {c:.5f}\n" for a, b, c in P))
                if uv is not None:
                    f.write("".join(f"vt {u:.5f} {1 - v:.5f}\n" for u, v in uv))
                    f.write("".join(f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}\n" for a, b, c in idx))
                else:
                    f.write("".join(f"f {a+1} {b+1} {c+1}\n" for a, b, c in idx))
            print(name, P.min(0).round(3), P.max(0).round(3), tex_files.get(prim.material))
