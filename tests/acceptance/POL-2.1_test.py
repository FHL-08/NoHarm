"""POL-2.1@3 The gr00t policy adapter downloads released Isaac GR00T N
weights from Hugging Face into the local HF cache (not git), loads
them, and emits joint_pos action chunks on the G1 contract.

The contract under test has four clauses, all checked against the real
G1 contract (harmless.embodiments.g1's declared joint_pos action space
and the harmless.policy.Policy act(observation) slot):

  adapter: harmless/policies/gr00t.py (a file this requirement touches)
    provides a policy adapter that pairs with the G1 like every other
    policy — resolved through harmless.registry's policy slot under a
    gr00t/vla name, or as a Policy subclass defined in the module.

  downloads from Hugging Face into the local HF cache: while the
    adapter acquires its weights, HF download machinery must run —
    huggingface_hub snapshot_download / hf_hub_download (spied at the
    package, on the HfApi class, and on every from-imported copy bound
    inside any loaded module), an hf/huggingface-cli download
    subprocess, or an HTTP fetch of huggingface.co — naming a released
    Isaac GR00T N repo (nvidia/GR00T-N*, Isaac-GR00T*). The checkpoint
    the adapter serves must resolve under the local HF cache root
    (HF_HUB_CACHE/HF_HOME or the default ~/.cache/huggingface/hub).

  not git: the served checkpoint does not live inside the repository —
    no git clone/lfs/pull fetching the weights is observed, no
    GR00T-identifiable weight artifact sits in the worktree or in the
    git index, and the loaded artifact resolves outside the repo.

  loads them + emits chunks: the adapter carries parameter tensors that
    match the checkpoint (same shape; equal values after float cast, or
    near-perfect correlation if the served copy is quantized) — or,
    when the weights live in a spawned serve process, that process's
    command line ties back to the HF-cache checkpoint — and act() on a
    real G1 observation emits an action CHUNK: more than one future
    control step, every step mapping declared G1 joint_pos joints to
    finite targets inside declared per-joint bounds. Two different
    observations must produce different chunks.

harmless/policies/ does not exist yet — failing to import it is the
expected pre-implementation failure.
"""

import importlib
import inspect
import json
import os
import re
import struct
import sys
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
GR00T_N = re.compile(r"gr00t[._\- ]?n|gr00tn|isaac[._\- ]?gr00t", re.I)
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx")
PATH_ATTR = re.compile(
    r"weight|ckpt|checkpoint|model|artifact|snapshot|hub|repo|path|dir",
    re.I)
CHUNK_KEY = ("action_chunk", "actions", "chunk", "joint_pos", "targets",
             "trajectory", "traj", "horizon", "action", "steps", "plan",
             "position_targets", "joint_targets")
HF_SUBPROCESS = re.compile(
    r"huggingface-cli|hf\s+download|hfd|hf_transfer|huggingface\.co|"
    r"snapshot_download|hf_hub_download", re.I)
GIT_FETCH = re.compile(
    r"\bgit\b.*\b(clone|lfs|submodule|pull|fetch)\b|git-lfs|GIT_LFS",
    re.I)
GR00T_N_REPO = re.compile(r"gr00t[._\- ]?n|gr00tn", re.I)
WEIGHT_WORD = re.compile(
    r"weights|checkpoint|ckpt|safetensors|\.bin\b|\.pt\b|\.ckpt\b",
    re.I)
GR00T_REPO_IDS = ("nvidia/GR00T-N1.5-3B", "nvidia/GR00T-N1-2B")


def _import(name, required=True):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        if not required:
            return None
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
        ((), {"model_id": GR00T_REPO_IDS[0]}),
        ((), {"repo_id": GR00T_REPO_IDS[0]}),
        ((), {"checkpoint": GR00T_REPO_IDS[0]}),
        ((), {"embodiment": emb, "model_id": GR00T_REPO_IDS[0]}),
        ((emb,), {}),
        ((space,), {}),
        ((GR00T_REPO_IDS[0],), {}),
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
            "contract:\n" + "\n".join(errs[:8]))
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


# ---------------- weights-acquisition spies ----------------------------


def _repo_id_of(args, kwargs):
    v = kwargs.get("repo_id") or kwargs.get("model_id")
    if isinstance(v, str):
        return v
    for a in args:
        if isinstance(a, str) and "/" in a \
                and not a.startswith(("/", ".", "http")):
            return a
    return None


def _cmd_str(cmd):
    if isinstance(cmd, str):
        return cmd
    try:
        return " ".join(str(c) for c in cmd)
    except Exception:
        return str(cmd)


def _install_spies(state):
    """Record every weights-acquisition route while the adapter loads:
    huggingface_hub download calls, spawned subprocesses, and HTTP
    fetches. Wrappers pass through to the real implementations."""
    calls = state["calls"]
    patched = state["patched"]
    originals = state["originals"]

    def patch(obj, attr, value):
        try:
            old = getattr(obj, attr)
        except Exception:
            return
        patched.append((obj, attr, old))
        setattr(obj, attr, value)

    def wrap_hub(fn, label):
        def w(*a, **k):
            result = fn(*a, **k)
            calls.append({"kind": label, "repo_id": _repo_id_of(a, k),
                          "kwargs": dict(k), "result": result})
            return result
        return w
    state["wrap_hub"] = wrap_hub

    try:
        import huggingface_hub as hf
    except Exception:
        hf = None
    if hf is not None:
        for fname in ("snapshot_download", "hf_hub_download"):
            fn = getattr(hf, fname, None)
            if callable(fn):
                originals[fname] = fn
                patch(hf, fname, wrap_hub(fn, fname))
        api = getattr(hf, "HfApi", None)
        if api is not None:
            for fname in ("snapshot_download", "hf_hub_download"):
                fn = getattr(api, fname, None)
                if callable(fn):
                    originals[f"HfApi.{fname}"] = fn
                    patch(api, fname, wrap_hub(fn, f"HfApi.{fname}"))

    import subprocess
    orig_popen_init = subprocess.Popen.__init__
    orig_system = os.system
    orig_popen = os.popen

    def popen_init(self, args, *a, **k):
        calls.append({"kind": "subprocess", "cmd": _cmd_str(args)})
        return orig_popen_init(self, args, *a, **k)

    def system(cmd):
        calls.append({"kind": "subprocess", "cmd": _cmd_str(cmd)})
        return orig_system(cmd)

    def popen(cmd, *a, **k):
        calls.append({"kind": "subprocess", "cmd": _cmd_str(cmd)})
        return orig_popen(cmd, *a, **k)

    patch(subprocess.Popen, "__init__", popen_init)
    patch(os, "system", system)
    patch(os, "popen", popen)

    try:
        import requests
        orig_request = requests.sessions.Session.request

        def request(self, method, url, *a, **k):
            calls.append({"kind": "http", "url": str(url)})
            return orig_request(self, method, url, *a, **k)

        patch(requests.sessions.Session, "request", request)
    except Exception:
        pass

    try:
        import urllib.request
        orig_open = urllib.request.urlopen
        originals["urllib.request.urlopen"] = orig_open

        def urlopen(url, *a, **k):
            calls.append({"kind": "http", "url": str(url)})
            return orig_open(url, *a, **k)

        patch(urllib.request, "urlopen", urlopen)
    except Exception:
        pass


def _sweep_bindings(state):
    """Patch by-identity copies of the original HF download functions
    bound inside any already-loaded module (from-imports inside
    transformers / the gr00t package / harmless.policies), so calls
    that bypass the canonical package attribute are still observed."""
    originals = state["originals"]
    patched = state["patched"]
    wrap_hub = state["wrap_hub"]
    names = {id(fn): name for name, fn in originals.items()}
    for mod in list(sys.modules.values()):
        try:
            d = vars(mod)
        except Exception:
            continue
        for attr, v in list(d.items()):
            if id(v) in names:
                try:
                    patched.append((mod, attr, v))
                    setattr(mod, attr, wrap_hub(v, names[id(v)]))
                except Exception:
                    pass


def _restore(patched):
    for obj, attr, old in reversed(patched):
        try:
            setattr(obj, attr, old)
        except Exception:
            pass


def _hf_download_hits(calls):
    """Observed HF downloads naming a released Isaac GR00T N repo."""
    hits, any_hub = [], []
    for c in calls:
        kind = str(c.get("kind", ""))
        if kind.endswith("download"):
            any_hub.append(c)
            if GR00T_N.search(str(c.get("repo_id") or "")):
                hits.append(c)
        elif kind == "subprocess":
            s = str(c.get("cmd") or "")
            if GR00T_N.search(s) and HF_SUBPROCESS.search(s):
                hits.append(c)
        elif kind == "http":
            u = str(c.get("url") or "")
            if "huggingface.co" in u and GR00T_N.search(u):
                hits.append(c)
    return hits, any_hub


def _git_violations(calls):
    bad = []
    for c in calls:
        if c.get("kind") != "subprocess":
            continue
        s = str(c.get("cmd") or "")
        if not GIT_FETCH.search(s):
            continue
        if "huggingface.co" in s or GR00T_N_REPO.search(s) \
                or WEIGHT_WORD.search(s):
            bad.append(s)
    return bad


def _hf_cache_roots():
    roots = []

    def add(p):
        if p is None:
            return
        try:
            r = Path(p).expanduser().resolve()
        except Exception:
            r = Path(p).expanduser()
        if str(r) not in [str(x) for x in roots]:
            roots.append(r)

    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        add(os.environ.get(env))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        add(Path(hf_home) / "hub")
    try:
        from huggingface_hub import constants as _c
        add(getattr(_c, "HF_HUB_CACHE", None))
        hh = getattr(_c, "HF_HOME", None)
        if hh:
            add(Path(hh) / "hub")
    except Exception:
        pass
    add(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def _under(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except Exception:
        return False


def _candidate_paths(gr00t_mod, serve_mod):
    out = []
    for mod in (gr00t_mod, serve_mod):
        if mod is None:
            continue
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
                                        "__pycache__", "runs")]
            for n in dirnames + filenames:
                if "gr00t" in n.lower():
                    out.append(Path(dirpath) / n)
    for cache in _hf_cache_roots():
        if not cache.is_dir():
            continue
        for dirpath, dirnames, _f in os.walk(cache):
            for d in dirnames:
                if "gr00t" in d.lower():
                    out.append(Path(dirpath) / d)
            if len(out) > 60:
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
    if path.is_file() and path.name == "config.json":
        files.append(path)
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


def _snapshot_root(path):
    """Lift a file/dir inside an HF snapshot layout to the snapshot
    directory (.../snapshots/<sha>) when one exists."""
    p = Path(path)
    try:
        parts = p.resolve().parts
    except Exception:
        parts = p.parts
    if "snapshots" in parts:
        i = len(parts) - 1 - list(reversed(parts)).index("snapshots")
        if i + 1 < len(parts):
            return Path(*parts[:i + 2])
    return p if p.is_dir() else p.parent


def _downloaded_artifacts(calls):
    """Checkpoint paths returned by observed HF download calls."""
    out = []
    for c in calls:
        r = c.get("result")
        if isinstance(r, (str, os.PathLike)):
            out.append(_snapshot_root(Path(r)))
        elif isinstance(r, (list, tuple)):
            for v in r:
                if isinstance(v, (str, os.PathLike)):
                    out.append(_snapshot_root(Path(v)))
    return out


def _repo_gr00t_artifacts():
    """GR00T-identifiable weight artifacts inside the worktree."""
    bad = []
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = [d for d in dirnames
                       if d not in (".venv", ".git", "node_modules",
                                    "__pycache__", "runs")]
        for n in filenames:
            f = Path(dirpath) / n
            if f.suffix.lower() not in WEIGHT_EXT:
                continue
            try:
                if f.stat().st_size <= 1_000_000:
                    continue
            except Exception:
                continue
            if GR00T_N.search(f.name) or GR00T_N.search(
                    str(f.parent)) or _config_says_gr00t(f.parent):
                bad.append(f)
    return bad


def _git_tracked_gr00t():
    """GR00T-identifiable weight files tracked by git."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "ls-files", "-z"],
            stderr=subprocess.DEVNULL)
    except Exception:
        return []
    bad = []
    for raw in out.split(b"\x00"):
        name = raw.decode(errors="ignore")
        if not name or Path(name).suffix.lower() not in WEIGHT_EXT:
            continue
        if GR00T_N.search(name):
            bad.append(name)
    return bad


# ---------------- checkpoint tensors ----------------------------------


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
                elif hasattr(cur, "state_dict") and callable(
                        cur.state_dict):
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


def _spawn_references_artifact(calls, artifact):
    """A spawned serve process whose command line names the HF-cache
    checkpoint or a GR00T-N repo id."""
    needle = str(artifact)
    for c in calls:
        if c.get("kind") != "subprocess":
            continue
        s = str(c.get("cmd") or "")
        if needle in s or GR00T_N.search(s):
            return True
    return False


def _match_checkpoint(tensors, idx):
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
    return matches


def _prime_lazy_download(pol):
    """Give a lazily-loading adapter one more chance to acquire its
    weights before the download clause is judged."""
    for meth in ("load", "load_weights", "ensure_weights", "download",
                 "prepare", "warmup", "start", "connect"):
        f = getattr(pol, meth, None)
        if callable(f):
            try:
                f()
            except TypeError:
                continue
            except Exception:
                pass


def test_pol_2_1_gr00t_downloads_hf_cache_loads_emits_joint_pos_chunks(
        tmp_path):
    state = {"calls": [], "patched": [], "originals": {}, "hf": None}
    _install_spies(state)
    calls = state["calls"]
    pol = emb = None
    try:
        gr00t_mod = _import("harmless.policies.gr00t")
        serve_mod = _import("harmless.policies.serve", required=False)
        _sweep_bindings(state)
        cls = _adapter_cls(gr00t_mod)
        space = _g1_space()

        emb, obs_a = _g1_observation()
        pol = _instantiate(cls, emb, space)
        obs_a = _with_instruction(obs_a, SUBGOAL)
        steps_a = _check_chunk(_chunk_of(pol, obs_a, space), space)

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
        steps_b = _check_chunk(_chunk_of(pol, obs_b, space), space)

        va, vb = _flatten_steps(steps_a), _flatten_steps(steps_b)
        differ = va.shape != vb.shape or not np.allclose(
            np.nan_to_num(va), np.nan_to_num(vb), atol=1e-4)
        assert differ, (
            "identical action chunks for different observations — the "
            "output is constant/canned, not computed by served GR00T "
            "weights")

        hits, any_hub = _hf_download_hits(calls)
        if not hits and not any_hub:
            _prime_lazy_download(pol)
            _sweep_bindings(state)
            hits, any_hub = _hf_download_hits(calls)

        roots = _hf_cache_roots()
        artifacts = []
        for p in _downloaded_artifacts(calls) + _candidate_paths(
                gr00t_mod, serve_mod):
            if _artifact_is_gr00t_n(p) and any(
                    _under(p, r) for r in roots):
                artifacts.append(p)
        artifacts = [
            Path(a) for a in dict.fromkeys(str(a) for a in artifacts)]
        spawn_tie = any(
            c.get("kind") == "subprocess" and (
                GR00T_N.search(str(c.get("cmd") or "")) or any(
                    str(a) in str(c.get("cmd") or "")
                    for a in artifacts))
            for c in calls)

        assert hits or (artifacts and (any_hub or spawn_tie)), (
            "the adapter never downloaded released Isaac GR00T N "
            "weights from Hugging Face — no snapshot_download / "
            "hf_hub_download call naming a GR00T-N repo (observed "
            f"repo ids: {[c.get('repo_id') for c in any_hub][:8]}), "
            "no hf/huggingface-cli download subprocess, no "
            "huggingface.co fetch, and no spawned serve process "
            "reached a GR00T-N checkpoint in the HF cache "
            f"({len(calls)} recorded call(s), cache roots: "
            f"{[str(r) for r in roots]})")

        bad_git = _git_violations(calls)
        assert not bad_git, (
            "the weights were fetched through git — '(not git)' "
            f"forbids git clone/lfs/pull for the checkpoint: {bad_git}")

        assert artifacts, (
            "no Isaac GR00T N checkpoint found under the local HF "
            f"cache ({[str(r) for r in roots]}) — the download must "
            "land in the HF cache, not elsewhere")
        artifact = artifacts[0]

        assert not _under(artifact, REPO), (
            f"the served GR00T N checkpoint {artifact} resolves inside "
            "the repository — the weights must live in the local HF "
            "cache, not in git")

        in_tree = _repo_gr00t_artifacts()
        assert not in_tree, (
            "GR00T weight artifacts vendored inside the worktree "
            f"(not git): {[str(p) for p in in_tree[:6]]}")
        tracked = _git_tracked_gr00t()
        assert not tracked, (
            "GR00T weight artifacts tracked by git: "
            f"{tracked[:6]}")

        idx = _tensor_index(artifact)
        assert idx, (
            f"the GR00T N artifact at {artifact} has no readable "
            "weight tensors")
        tensors = _adapter_tensors(
            pol, vars(gr00t_mod),
            vars(serve_mod) if serve_mod is not None else {})
        if tensors:
            matches = _match_checkpoint(tensors, idx)
            assert matches, (
                f"no parameter tensor on the adapter matches any of "
                f"the {len(idx)} checkpoint tensors in {artifact} — "
                "the adapter serves different/random weights, not "
                "the released Isaac GR00T N weights")
        else:
            assert _spawn_references_artifact(calls, artifact), (
                "the adapter carries no parameter tensors and no "
                "spawned serve process references the HF-cache "
                "checkpoint — nothing shows the released GR00T N "
                "weights were loaded")
    finally:
        _restore(state["patched"])
        try:
            if pol is not None:
                pol.close()
        except Exception:
            pass
        try:
            if emb is not None:
                emb.close()
        except Exception:
            pass
