"""POL-2.1@2 The gr00t policy adapter loads released Isaac GR00T N
weights from the Hugging Face cache and emits joint_pos action chunks
on the G1 contract.

The contract under test has three clauses, all checked against the real
G1 contract (harmless.embodiments.g1's declared joint_pos action space
and the harmless.policy.Policy act(observation) slot):

  adapter: harmless/policies/gr00t.py (a file this requirement touches)
    provides a policy adapter that pairs with the G1 like every other
    policy — resolved through harmless.registry's policy slot under a
    gr00t/vla name, or as a Policy subclass defined in the module.

  serves Isaac GR00T N weights: a real GR00T-N checkpoint artifact is
    present — a file or HF-snapshot-style directory identifiable as
    Isaac GR00T N by name or by a config.json mentioning gr00t, found
    under assets/, the repo, the HF cache, or at a path the touched
    modules publish — and the weights the adapter actually infers with
    are THOSE released weights: at least one parameter tensor carried
    by the adapter's model (reached through its attributes, its
    module's globals, or the serve module's globals) matches a tensor
    in the checkpoint (same shape; equal values after float cast, or
    near-perfect correlation if the served copy is quantized or
    scaled). A GR00T-shaped model with random or foreign weights fails
    here, as does a stub emitting canned actions. The weights are
    *served*: harmless/policies/serve.py (the other touched file)
    carries the serving layer — during adapter construction or act()
    at least one function/method defined in serve.py must run (a
    monkeypatch spy counts them) or the adapter must hold an object
    whose class is defined there.

  joint_pos action chunks: given a real G1 observation (joint
    state, IMU, touch, pelvis pose, camera frames) plus a scripted
    instruction, the adapter's act() (or a chunk entry point it
    exposes) emits an action CHUNK — more than one future control
    step — where every step maps declared G1 joint_pos joints to
    finite scalar targets inside the declared per-joint bounds. Two
    different observations must produce different chunks: the output
    is computed by the served network, not a constant.

harmless/policies/ does not exist yet — failing to import it is the
expected pre-implementation failure.
"""

import importlib
import inspect
import json
import os
import re
import struct
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]
SCENE = REPO / "kitchen.xml"
SUBGOAL = "approach and touch the bowl"
BOUNDS_TOL = 1e-2
MIN_CHUNK = 2
MAX_TENSOR_ELEMS = 20_000_000

POLICY_NAME = re.compile(r"gr00t|groot|isaac|vla|n1[._-]?5", re.I)
GR00T_N = re.compile(r"gr00t[._\- ]?n|gr00tn", re.I)
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx")
PATH_ATTR = re.compile(
    r"weight|ckpt|checkpoint|model|artifact|snapshot|hub|repo|path|dir",
    re.I)
CHUNK_KEY = ("action_chunk", "actions", "chunk", "joint_pos", "targets",
             "trajectory", "traj", "horizon", "action", "steps", "plan",
             "position_targets", "joint_targets")


def _import(name):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        pytest.fail(
            f"{name} cannot be imported ({type(exc).__name__}: {exc}) "
            "— the gr00t policy adapter does not exist yet")


def _adapter_cls(gr00t_mod):
    try:
        from harmless import registry
    except Exception:
        registry = None
    if registry is not None:
        names = []
        try:
            names = list(registry.list_policies())
        except Exception:
            for attr in ("POLICIES", "policies", "POLICY_REGISTRY"):
                reg = getattr(registry, attr, None)
                if isinstance(reg, dict):
                    names = list(reg)
                    break
        for n in names:
            if not POLICY_NAME.search(str(n)):
                continue
            try:
                return registry.get_policy(n)
            except Exception:
                for attr in ("POLICIES", "policies", "POLICY_REGISTRY"):
                    reg = getattr(registry, attr, None)
                    if isinstance(reg, dict) and reg.get(n) is not None:
                        return reg[n]
    try:
        from harmless.policy import Policy
        base = Policy
    except Exception:
        base = None
    for name, obj in sorted(vars(gr00t_mod).items()):
        if inspect.isclass(obj) and obj.__module__ == gr00t_mod.__name__:
            if base is None or issubclass(obj, base):
                return obj
    for name, obj in sorted(vars(gr00t_mod).items()):
        if inspect.isclass(obj) and callable(getattr(obj, "act", None)):
            return obj
    pytest.fail(
        "no gr00t policy adapter found — harmless.registry has no "
        "gr00t/vla-named policy and harmless.policies.gr00t declares "
        "no Policy subclass")


def _g1_space():
    mod = _import("harmless.embodiments.g1")
    emb = mod.G1Embodiment()
    space = getattr(emb, "action_space", None)
    assert space is not None, "G1Embodiment declares no action_space"
    return space


def _g1_observation():
    import mujoco
    mod = _import("harmless.embodiments.g1")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "kitchen")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
    mujoco.mj_forward(model, data)
    emb = mod.G1Embodiment(model, data)
    obs = emb.observation()
    return emb, obs


def _with_instruction(obs, text):
    obs = dict(obs)
    for key in ("instruction", "subgoal", "task", "prompt", "goal",
                "language_instruction", "command"):
        obs[key] = text
    return obs


def _instantiate(cls, emb, space):
    tries = [
        ((), {"embodiment": emb}),
        ((), {"embodiment": emb, "action_space": space}),
        ((), {"action_space": space}),
        ((), {"embodiment": emb, "instruction": SUBGOAL}),
        ((), {"instruction": SUBGOAL}),
        ((emb,), {}),
        ((space,), {}),
        ((), {}),
    ]
    errs = []
    for args, kw in tries:
        try:
            pol = cls(*args, **kw)
            break
        except Exception as exc:
            errs.append(f"{args} {kw}: {type(exc).__name__}: {exc}")
    else:
        pytest.fail(
            "the gr00t adapter could not be constructed/bound to the G1 "
            "contract:\n" + "\n".join(errs[:6]))
    if getattr(pol, "action_space", None) is None:
        bind = getattr(pol, "bind", None)
        if callable(bind):
            for target in (emb, space):
                try:
                    bind(target)
                    break
                except Exception:
                    continue
    return pol


def _call_act(pol, obs):
    act = getattr(pol, "act", None)
    if not callable(act):
        if callable(pol):
            return pol(obs)
        pytest.fail("the gr00t adapter exposes no act(observation) call")
    for args, kw in (((obs,), {}),
                     ((obs,), {"instruction": SUBGOAL}),
                     ((obs,), {"subgoal": SUBGOAL}),
                     ((obs, SUBGOAL), {})):
        try:
            return act(*args, **kw)
        except TypeError:
            continue
    return act(obs)


def _scalar(v):
    try:
        float(v)
        return True
    except Exception:
        return False


def _steps_from(out, joints, joint_order=None, depth=0):
    """Normalize any plausible chunk spelling into a list of per-step
    {joint: target} dicts. Returns None when nothing chunk-shaped is
    found."""
    if out is None or depth > 4:
        return None
    joints = list(joints)
    jset = set(joints)
    order = list(joint_order) if joint_order else joints
    if isinstance(out, Mapping):
        js = set(out) & jset
        if js and all(_scalar(out[k]) for k in js):
            return [{k: float(out[k]) for k in js}]
        seq = [k for k in js
               if isinstance(out[k], (list, tuple, np.ndarray))
               and np.asarray(out[k], dtype=object).ndim == 1
               and len(out[k]) > 1]
        if len(seq) >= 1:
            h = min(len(out[k]) for k in seq)
            try:
                return [{k: float(out[k][i]) for k in seq}
                        for i in range(h)]
            except Exception:
                pass
        for k in CHUNK_KEY:
            if k in out:
                r = _steps_from(out[k], joints, order, depth + 1)
                if r:
                    return r
        for v in out.values():
            r = _steps_from(v, joints, order, depth + 1)
            if r:
                return r
        return None
    if isinstance(out, (list, tuple)) and out \
            and all(isinstance(s, Mapping) for s in out):
        steps = []
        for s in out:
            js = set(s) & jset
            if not js:
                return None
            steps.append({k: float(s[k]) for k in js if _scalar(s[k])})
        return steps
    try:
        arr = np.asarray(out, dtype=float)
    except Exception:
        arr = None
    if arr is not None and arr.ndim == 2 and min(arr.shape) >= 1:
        if arr.shape[1] == len(order):
            return [dict(zip(order, row)) for row in arr]
        if arr.shape[0] == len(order) and arr.shape[1] > 1:
            return [dict(zip(order, col)) for col in arr.T]
    if arr is not None and arr.ndim == 1 and arr.size == len(joints):
        return [dict(zip(joints, arr))]
    for attr in CHUNK_KEY:
        v = getattr(out, attr, None)
        if v is not None:
            r = _steps_from(v, joints, order, depth + 1)
            if r:
                return r
    return None


def _joint_order(pol, space):
    for attr in ("action_joints", "joints", "joint_names", "action_names",
                 "action_dims", "controlled_joints"):
        v = getattr(pol, attr, None)
        if isinstance(v, (list, tuple)) and v \
                and all(isinstance(x, str) for x in v):
            return list(v)
    return list(getattr(space, "joints", []) or [])


def _chunk_of(pol, obs, space):
    order = _joint_order(pol, space)
    out = _call_act(pol, obs)
    steps = _steps_from(out, space.joints, order)
    if steps is None and isinstance(out, (list, tuple)):
        for v in out:
            steps = _steps_from(v, space.joints, order)
            if steps:
                break
    if steps is None:
        for meth in ("predict", "infer", "get_action_chunk",
                     "action_chunk", "chunk", "plan", "forward"):
            f = getattr(pol, meth, None)
            if not callable(f):
                continue
            try:
                steps = _steps_from(f(obs), space.joints, order)
            except Exception:
                continue
            if steps:
                break
    return steps


def _check_chunk(steps, space):
    assert steps is not None, (
        "the adapter's act() returned nothing shaped like an action "
        "chunk (no per-step joint targets found)")
    assert len(steps) >= MIN_CHUNK, (
        f"the adapter emitted {len(steps)} control step(s) — a VLA "
        "emits an action CHUNK of more than one future step")
    declared = set(space.joints)
    for i, step in enumerate(steps):
        assert step, f"chunk step {i} commands no joints"
        bad = sorted(set(step) - declared)
        assert not bad, (
            f"chunk step {i} commands joints outside the G1 joint_pos "
            f"contract: {bad}")
        for j, v in step.items():
            lo, hi = space.bounds[j]
            assert np.isfinite(v), (
                f"chunk step {i} joint '{j}' target is {v} — not finite")
            assert lo - BOUNDS_TOL <= v <= hi + BOUNDS_TOL, (
                f"chunk step {i} joint '{j}' target {v} is outside its "
                f"declared joint_pos bounds [{lo}, {hi}]")
    return steps


def _flatten_steps(steps):
    if not steps:
        return np.zeros(0)
    keys = sorted(set().union(*[set(s) for s in steps]))
    rows = [[s.get(k, np.nan) for k in keys] for s in steps]
    return np.asarray(rows, dtype=float).reshape(-1)


def _serve_spy(serve_mod, gr00t_mod):
    """Count invocations of every callable defined in serve.py — module
    functions, class methods, and the same functions re-exported through
    gr00t.py's globals."""
    hits = []
    patched = []

    def wrap(fn, label):
        def w(*a, **k):
            hits.append(label)
            return fn(*a, **k)
        return w

    def patch(obj, attr, value):
        try:
            old = getattr(obj, attr)
        except Exception:
            return
        patched.append((obj, attr, old))
        setattr(obj, attr, value)

    for name, v in list(vars(serve_mod).items()):
        if name.startswith("_"):
            continue
        if inspect.isfunction(v):
            patch(serve_mod, name, wrap(v, f"serve.{name}"))
        elif inspect.isclass(v) and v.__module__ == serve_mod.__name__:
            for mn, mv in list(vars(v).items()):
                if mn.startswith("__"):
                    continue
                if inspect.isfunction(mv):
                    patch(v, mn, wrap(mv, f"serve.{v.__name__}.{mn}"))
                elif isinstance(mv, staticmethod):
                    patch(v, mn, staticmethod(
                        wrap(mv.__func__, f"serve.{v.__name__}.{mn}")))
                elif isinstance(mv, classmethod):
                    patch(v, mn, classmethod(
                        wrap(mv.__func__, f"serve.{v.__name__}.{mn}")))
    for name, v in list(vars(gr00t_mod).items()):
        if inspect.isfunction(v) and v.__module__ == serve_mod.__name__:
            patch(gr00t_mod, name, wrap(v, f"gr00t.{name}"))
    return hits, patched


def _restore(patched):
    for obj, attr, old in patched:
        setattr(obj, attr, old)


def _serve_entry_points(serve_mod):
    out = []
    for name, v in vars(serve_mod).items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(v) or inspect.isclass(v):
            out.append(name)
    return out


def _holds_serve_object(pol, serve_name, depth=0, seen=None):
    if seen is None:
        seen = set()
    if pol is None or id(pol) in seen or depth > 4:
        return None
    seen.add(id(pol))
    if type(pol).__module__ == serve_name:
        return pol
    try:
        items = list(vars(pol).items())
    except TypeError:
        return None
    for name, v in items:
        if name.startswith("__"):
            continue
        hit = _holds_serve_object(v, serve_name, depth + 1, seen)
        if hit is not None:
            return hit
    return None


def _candidate_paths(gr00t_mod, serve_mod):
    out = []
    for mod in (gr00t_mod, serve_mod):
        for name, v in vars(mod).items():
            if not PATH_ATTR.search(name):
                continue
            for p in v if isinstance(v, (list, tuple)) else [v]:
                if isinstance(p, (str, os.PathLike)) and os.path.exists(p):
                    out.append(Path(p))
    for root in (REPO / "assets", REPO):
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in (".venv", ".git", "node_modules",
                                        "__pycache__")]
            for n in dirnames + filenames:
                if "gr00t" in n.lower():
                    out.append(Path(dirpath) / n)
    hub_cache = Path.home() / ".cache" / "huggingface"
    if hub_cache.is_dir():
        for dirpath, dirnames, _f in os.walk(hub_cache):
            for d in dirnames:
                if "gr00t" in d.lower():
                    out.append(Path(dirpath) / d)
            if len(out) > 40:
                break
    seen, uniq = set(), []
    for p in out:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _weight_files(path):
    if path.is_file():
        return [path] if path.suffix.lower() in WEIGHT_EXT else []
    out = []
    try:
        for f in path.rglob("*"):
            if f.is_file() and f.suffix.lower() in WEIGHT_EXT \
                    and f.stat().st_size > 1_000_000:
                out.append(f)
    except Exception:
        pass
    return out


def _config_says_gr00t(path):
    files = [path / "config.json"] if path.is_dir() else []
    if path.is_dir():
        try:
            files += list(path.glob("**/config.json"))[:4]
        except Exception:
            pass
    for f in files:
        try:
            txt = f.read_text(errors="ignore")
        except Exception:
            continue
        if "gr00t" in txt.lower():
            return True
    return False


def _artifact_is_gr00t_n(path):
    if not _weight_files(path):
        return False
    if GR00T_N.search(path.name):
        return True
    if _config_says_gr00t(path):
        return True
    if path.is_dir() and any(
            GR00T_N.search(p) for p in
            [part for part in path.parts] + [path.parent.name]):
        return True
    return False


def _safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 100_000_000:
            return {}
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def _st_tensor(path, meta, base):
    dt = meta["dtype"]
    s, e = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + s)
        raw = f.read(e - s)
    if dt == "F32":
        arr = np.frombuffer(raw, dtype=np.float32)
    elif dt == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dt == "BF16":
        arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
               << 16).view(np.float32)
    elif dt == "F64":
        arr = np.frombuffer(raw, dtype=np.float64).astype(np.float32)
    elif dt in ("I64", "I32", "I16", "I8", "U8", "U16", "U32", "U64"):
        arr = np.frombuffer(
            raw, dtype=getattr(np, dt.lower())).astype(np.float32)
    else:
        return None
    return arr.reshape(meta["shape"]).astype(np.float32)


def _tensor_index(artifact):
    """{name: (shape, loader)} for every tensor in the checkpoint,
    reading only safetensors headers up front."""
    idx = {}
    for f in _weight_files(artifact):
        if f.suffix.lower() == ".safetensors":
            try:
                hdr, base = _safetensors_header(f)
            except Exception:
                continue
            for name, meta in hdr.items():
                if name == "__metadata__" or not isinstance(meta, dict):
                    continue
                shape = tuple(meta.get("shape") or [])
                idx[name] = (shape, lambda m=meta, b=base, p=f:
                             _st_tensor(p, m, b))
        elif f.suffix.lower() in (".bin", ".pt", ".pth", ".ckpt") \
                and f.stat().st_size < 6_000_000_000:
            try:
                import torch
                try:
                    obj = torch.load(str(f), map_location="cpu",
                                     weights_only=True)
                except Exception:
                    obj = torch.load(str(f), map_location="cpu")
            except Exception:
                try:
                    import torch
                    mod = torch.jit.load(str(f), map_location="cpu")
                    obj = dict(mod.named_parameters())
                except Exception:
                    continue
            stack = [("", obj)]
            while stack:
                pre, cur = stack.pop()
                if isinstance(cur, Mapping):
                    for k, v in cur.items():
                        stack.append((f"{pre}.{k}" if pre else str(k), v))
                elif hasattr(cur, "state_dict") and callable(cur.state_dict):
                    try:
                        stack.append((pre, cur.state_dict()))
                    except Exception:
                        pass
                else:
                    try:
                        import torch
                        if isinstance(cur, torch.Tensor) and cur.numel():
                            idx[pre] = (tuple(cur.shape),
                                        lambda t=cur:
                                        t.detach().float().cpu().numpy())
                    except Exception:
                        pass
    return idx


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def _adapter_tensors(*roots):
    """name -> float32 numpy array for every parameter-ish tensor
    reachable from the adapter, its module globals, and serve.py."""
    torch = _torch()
    out, seen = {}, set()

    def put(name, arr):
        if name not in out:
            out[name] = arr

    def from_tensor(name, t):
        try:
            if torch is not None and isinstance(t, torch.Tensor):
                if t.numel() < 16 or t.numel() > MAX_TENSOR_ELEMS:
                    return
                put(name, t.detach().float().cpu().numpy())
            elif isinstance(t, np.ndarray) and t.size >= 16 \
                    and t.size <= MAX_TENSOR_ELEMS \
                    and np.issubdtype(t.dtype, np.number):
                put(name, t.astype(np.float32))
        except Exception:
            pass

    def harvest(obj, name, depth):
        if obj is None or depth > 5 or id(obj) in seen:
            return
        seen.add(id(obj))
        if torch is not None and (
                isinstance(obj, torch.Tensor)
                or isinstance(obj, (torch.nn.Module,
                                    torch.jit.ScriptModule))):
            if isinstance(obj, torch.Tensor):
                from_tensor(name, obj)
                return
            try:
                for pn, t in obj.named_parameters():
                    from_tensor(f"{name}.{pn}", t)
            except Exception:
                pass
            try:
                sd = obj.state_dict()
                for k, t in sd.items():
                    from_tensor(f"{name}.{k}", t)
            except Exception:
                pass
            return
        if isinstance(obj, np.ndarray):
            from_tensor(name, obj)
            return
        if isinstance(obj, Mapping):
            for k, v in list(obj.items())[:400]:
                harvest(v, f"{name}.{k}", depth + 1)
            return
        if isinstance(obj, (list, tuple)) and len(obj) <= 64:
            for i, v in enumerate(obj):
                harvest(v, f"{name}[{i}]", depth + 1)
            return
        if inspect.ismodule(obj) or inspect.isclass(obj) \
                or inspect.isfunction(obj) or inspect.ismethod(obj):
            return
        sd = getattr(obj, "state_dict", None)
        if callable(sd):
            try:
                for k, t in sd().items():
                    from_tensor(f"{name}.{k}", t)
            except Exception:
                pass
        try:
            items = list(vars(obj).items())
        except TypeError:
            return
        for k, v in items:
            if k.startswith("__"):
                continue
            harvest(v, f"{name}.{k}", depth + 1)

    for i, r in enumerate(roots):
        harvest(r, f"r{i}", 0)
    return out


def _match(a, b):
    if a.shape != b.shape:
        return False
    n = min(a.size, 1 << 17)
    x = np.asarray(a, dtype=np.float32).reshape(-1)[:n]
    y = np.asarray(b, dtype=np.float32).reshape(-1)[:n]
    if np.allclose(x, y, rtol=5e-2, atol=1e-3):
        return True
    sx, sy = float(x.std()), float(y.std())
    if sx < 1e-12 or sy < 1e-12:
        return False
    r = float(np.corrcoef(x, y)[0, 1])
    return r > 0.999


def _name_tail(name):
    parts = [p for p in re.split(r"[./]", name) if p]
    drop = {"module", "model", "policy", "net", "network", "adapter"}
    return ".".join(p for p in parts if p.lower() not in drop) or name


def test_pol_2_1_gr00t_adapter_serves_weights_emits_joint_pos_chunks(
        tmp_path):
    gr00t_mod = _import("harmless.policies.gr00t")
    serve_mod = _import("harmless.policies.serve")
    cls = _adapter_cls(gr00t_mod)
    space = _g1_space()

    entries = _serve_entry_points(serve_mod)
    assert entries, (
        "harmless.policies.serve exposes no serving entry point — the "
        "serving layer for the GR00T weights is empty")

    artifacts = [p for p in _candidate_paths(gr00t_mod, serve_mod)
                 if _artifact_is_gr00t_n(p)]
    assert artifacts, (
        "no Isaac GR00T N weights artifact found — nothing named for "
        "GR00T N under assets/, the repo, the HF cache, or a weights "
        "path published by harmless.policies.gr00t/serve")
    artifact = artifacts[0]

    hits, patched = _serve_spy(serve_mod, gr00t_mod)
    pol = None
    try:
        emb, obs_a = _g1_observation()
        pol = _instantiate(cls, emb, space)
        obs_a = _with_instruction(obs_a, SUBGOAL)
        steps_a = _check_chunk(
            _chunk_of(pol, obs_a, space), space)

        import mujoco
        for jname in ("right_shoulder_pitch_joint",
                      "right_elbow_joint", "waist_yaw_joint"):
            jid = mujoco.mj_name2id(
                emb.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                emb.data.qpos[emb.model.jnt_qposadr[jid]] += 0.35
        mujoco.mj_forward(emb.model, emb.data)
        obs_b = _with_instruction(emb.observation(),
                                  "pick up the mug with the left hand")
        steps_b = _check_chunk(
            _chunk_of(pol, obs_b, space), space)

        va, vb = _flatten_steps(steps_a), _flatten_steps(steps_b)
        differ = va.shape != vb.shape or not np.allclose(
            np.nan_to_num(va), np.nan_to_num(vb), atol=1e-4)
        assert differ, (
            "identical action chunks for different observations — the "
            "output is constant/canned, not computed by served GR00T "
            "weights")
    finally:
        _restore(patched)

    served = bool(hits) or _holds_serve_object(pol, serve_mod.__name__) \
        is not None
    assert served, (
        "the adapter never touches harmless.policies.serve — no serve.py "
        "function/method ran while binding and inferring, and the "
        "adapter holds no client/server object defined there. The "
        "weights are not 'served' through the serving layer")

    idx = _tensor_index(artifact)
    assert idx, (
        f"the GR00T N artifact at {artifact} has no readable weight "
        "tensors")
    tensors = _adapter_tensors(pol, vars(gr00t_mod), vars(serve_mod))
    assert tensors, (
        "the adapter carries no parameter tensors — no served model "
        "found on the policy, in gr00t.py, or in serve.py")
    by_shape = {}
    for name, (shape, _load) in idx.items():
        by_shape.setdefault(shape, []).append(name)
    tails = {}
    for name in idx:
        tails.setdefault(_name_tail(name), []).append(name)
    matches = []
    for name, arr in list(tensors.items())[:160]:
        cands = list(idx.get(name, None) and [name] or [])
        cands += tails.get(_name_tail(name), [])[:4]
        cands += by_shape.get(tuple(arr.shape), [])[:6]
        seen_c = set()
        for c in cands:
            if c in seen_c:
                continue
            seen_c.add(c)
            shape, loader = idx[c]
            if tuple(shape) != tuple(arr.shape):
                continue
            try:
                other = loader()
            except Exception:
                continue
            if other is not None and _match(arr, other):
                matches.append((name, c))
                break
        if len(matches) >= 3:
            break
    assert matches, (
        f"no parameter tensor on the adapter matches any of the "
        f"{len(idx)} checkpoint tensors in {artifact} — the adapter "
        "serves different/random weights, not the released Isaac GR00T "
        "N weights")
