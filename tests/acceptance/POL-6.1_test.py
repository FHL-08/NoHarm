"""POL-6.1@1 finetune.py post-trains the released GR00T N checkpoint on
the POL-5 episodes on one 8 GB GPU with the vision-language backbone frozen,
writing the checkpoint outside git.

The contract under test is harmless/policies/finetune.py (a file this
requirement touches) — the post-training step that turns the released
Isaac GR00T N checkpoint and the demos.py episode dataset into a G1
bowl-touch checkpoint:

  post-trains the released GR00T N checkpoint: while the entry point
    runs, the test tapes every weight-file read (safetensors/torch.load)
    and every Hugging Face download call. At least one read must land
    inside the released GR00T-N snapshot in the local HF cache
    (models--nvidia--GR00T*) or a download call must name a released
    GR00T-N repo — and the frozen tensors of the trained artifact must
    equal the released values.

  on the POL-5 episodes: the entry is driven with a real episode
    dataset recorded by demos.py (the POL-5 recorder, landed on the
    base)
    under a bounded run — one recorded episode is enough to prove the
    training loop consumes the LeRobot episodes.

  on one 8 GB GPU: the run's accelerator memory stays inside one card's
    budget — torch.cuda peak allocation <= ~8.6 GB on a single device
    index — or, for a CPU-only run, the process's peak RSS delta stays
    within the same budget.

  with the vision-language backbone frozen: every trainable-relevant
    tensor named like the vision-language stack (backbone/vlln/
    vl_self_attention/vision/language/eagle/siglip/llm/text/image/vit)
    must be excluded from the optimizer's gradients or bit-identical to
    its released value afterwards, while at least one non-VL tensor
    measurably changes — proving gradients moved the action head but
    not the backbone. A backbone present only as fixed feature
    projectors (no trainable tensors at all) also satisfies 'frozen'.

  writing the checkpoint outside git: the produced checkpoint resolves
    to a path that is not git-tracked — outside the worktree or under a
    gitignored subtree — and no new weight artifact appears inside the
    repository's tracked set during the run.

harmless/policies/finetune.py does not exist yet — failing to import it
is the expected pre-implementation failure.
"""

import importlib
import inspect
import json
import os
import re
import signal
import struct
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]
SCENE = REPO / "kitchen.xml"
GPU_BUDGET_B = 8.6e9          # ~8 GB card (the box ships an 8151 MiB GPU)
RSS_BUDGET_B = 9.5e9          # host-side proxy when no CUDA is used
RUN_DEADLINE_S = 900

GR00T_N = re.compile(r"gr00t[._\- ]?n|gr00tn|isaac[._\- ]?gr00t", re.I)
GR00T_REPO = re.compile(r"nvidia/gr00t", re.I)
VL_NAME = re.compile(
    r"vlln|vl_self|backbone|vision|language|eagle|siglip|llm|"
    r"text|image|visual|vit\b|vlm|clip", re.I)
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")

ENTRY_NAMES = (
    "finetune", "fine_tune", "post_train", "posttrain", "train",
    "train_gr00t", "finetune_gr00t", "run", "main",
)
DATA_NAMES = (
    "dataset", "dataset_dir", "dataset_path", "data", "data_dir",
    "data_path", "demos", "demos_dir", "demo_dir", "lerobot",
    "lerobot_dataset", "episodes", "episodes_dir", "source", "input",
    "input_dir", "train_data", "root", "records", "dataset_root",
)
OUT_NAMES = (
    "out", "out_dir", "output", "output_dir", "output_path", "out_path",
    "checkpoint", "checkpoint_dir", "ckpt", "ckpt_dir", "save_dir",
    "save_to", "dest", "destination", "export", "weights_out",
    "model_out", "save_path", "write_to",
)
STEP_NAMES = (
    "steps", "max_steps", "n_steps", "train_steps", "num_steps",
    "epochs", "max_epochs", "n_epochs", "iters", "iterations",
    "num_iters", "updates", "n_iters", "num_updates",
)
BATCH_NAMES = ("batch_size", "batch", "bs", "bsz", "per_device_batch")
BASE_NAMES = (
    "base", "base_ckpt", "base_checkpoint", "pretrained", "init",
    "init_checkpoint", "from_checkpoint", "repo_id", "model_id",
    "model", "weights", "source_ckpt", "checkpoint_in",
)


def _import(name, required=True):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        if not required:
            return None
        pytest.fail(
            f"{name} cannot be imported ({type(exc).__name__}: {exc}) "
            "— the post-training entry point does not exist yet")


# ---------------- released GR00T-N snapshot ---------------------------


def _hf_gr00t_snapshot():
    roots = []
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(env):
            roots.append(Path(os.environ[env]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    for root in roots:
        if not root.is_dir():
            continue
        for d in root.glob("models--*GR00T*/snapshots/*/"):
            if any(d.glob("*.safetensors")) or any(
                    d.glob("*.bin")) or any(d.glob("*.pt")):
                return d
    return None


def _safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 100_000_000:
            return {}, 0
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
    """{name: (shape, loader)} for every tensor under a checkpoint
    path, reading only safetensors headers up front."""
    idx = {}
    files = []
    p = Path(artifact)
    if p.is_file():
        files = [p]
    elif p.is_dir():
        try:
            files = [f for f in p.rglob("*") if f.is_file()
                     and f.suffix.lower() in WEIGHT_EXT]
        except Exception:
            files = []
    for f in files:
        if f.suffix.lower() == ".safetensors":
            try:
                hdr, base = _safetensors_header(f)
            except Exception:
                continue
            for name, meta in hdr.items():
                if name == "__metadata__" or not isinstance(meta, dict):
                    continue
                idx[name] = (
                    tuple(meta.get("shape") or []),
                    lambda m=meta, b=base, pp=f: _st_tensor(pp, m, b))
        elif f.suffix.lower() in (".bin", ".pt", ".pth", ".ckpt") \
                and f.stat().st_size < 8_000_000_000:
            try:
                import torch
                try:
                    obj = torch.load(str(f), map_location="cpu",
                                     weights_only=True)
                except Exception:
                    obj = torch.load(str(f), map_location="cpu")
            except Exception:
                continue
            stack = [("", obj)]
            while stack:
                pre, cur = stack.pop()
                if isinstance(cur, Mapping):
                    for k, v in cur.items():
                        stack.append(
                            (f"{pre}.{k}" if pre else str(k), v))
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
                            idx[pre] = (
                                tuple(cur.shape),
                                lambda t=cur: t.detach().float()
                                .cpu().numpy())
                    except Exception:
                        pass
    return idx


def _same(a, b, tol=1e-6):
    if a is None or b is None or a.shape != b.shape:
        return False
    n = min(a.size, 1 << 17)
    x = np.asarray(a, dtype=np.float32).reshape(-1)[:n]
    y = np.asarray(b, dtype=np.float32).reshape(-1)[:n]
    return bool(np.allclose(x, y, rtol=1e-4, atol=tol))


def _name_variants(name):
    yield name
    if name.startswith("action_head."):
        yield name[len("action_head."):]
    else:
        yield f"action_head.{name}"
    yield name.split(".")[-1]


def _released_value(idx, name):
    for cand in _name_variants(name):
        if cand in idx:
            shape, loader = idx[cand]
            try:
                return loader()
            except Exception:
                return None
    return None


# ---------------- acquisition + training spies ------------------------


def _install_spies(state):
    """Record weight-file reads, HF downloads, optimizer instances and
    device usage while finetune runs."""
    patched = state["patched"]

    def patch(obj, attr, fn):
        try:
            old = getattr(obj, attr)
        except Exception:
            return
        patched.append((obj, attr, old))
        setattr(obj, attr, fn)

    def record_path(p):
        state["reads"].append(str(p))

    try:
        import safetensors.torch as stt
        for fname in ("load_file", "load"):
            fn = getattr(stt, fname, None)
            if callable(fn):
                def wrap_load(fn):
                    def w(p, *a, **k):
                        record_path(p)
                        return fn(p, *a, **k)
                    return w
                patch(stt, fname, wrap_load(fn))
    except Exception:
        pass

    try:
        import huggingface_hub as hf
        for fname in ("snapshot_download", "hf_hub_download"):
            fn = getattr(hf, fname, None)
            if callable(fn):
                def wrap_hf(fn):
                    def w(*a, **k):
                        state["hf"].append(
                            {"fn": getattr(fn, "__name__", "?"),
                             "repo": kwargs_repo(a, k)})
                        return fn(*a, **k)
                    return w
                patch(hf, fname, wrap_hf(fn))
    except Exception:
        pass

    try:
        import torch
        orig_tl = torch.load

        orig_rg = torch.nn.Module.requires_grad_

        def rg(self, requires_grad=True):
            r = orig_rg(self, requires_grad)
            try:
                if not requires_grad:
                    for n, p in self.named_parameters():
                        state["frozen"].add(n)
            except Exception:
                pass
            return r
        patch(torch.nn.Module, "requires_grad_", rg)

        def tl(f, *a, **k):
            try:
                record_path(f)
            except Exception:
                pass
            return orig_tl(f, *a, **k)
        patch(torch, "load", tl)

        orig_init = torch.optim.Optimizer.__init__

        def opt_init(self, params, *a, **k):
            orig_init(self, params, *a, **k)
            snap = state["param_snap"]
            try:
                for g in self.param_groups:
                    for p in g.get("params", []):
                        if isinstance(p, torch.Tensor) \
                                and p.is_floating_point() \
                                and p.numel() <= 20_000_000:
                            snap[id(p)] = (
                                p,
                                p.detach().float().clone().cpu())
                            state["devices"].add(str(p.device))
            except Exception:
                pass
            state["opts"].append(self)
            orig_step = self.step

            def step(*a, **k):
                state["opt_steps"] += 1
                return orig_step(*a, **k)
            try:
                object.__setattr__(self, "step", step)
            except Exception:
                pass
        patch(torch.optim.Optimizer, "__init__", opt_init)
    except Exception:
        pass


def kwargs_repo(args, kwargs):
    for k in ("repo_id", "model_id"):
        if isinstance(kwargs.get(k), str):
            return kwargs[k]
    for a in args:
        if isinstance(a, str) and "/" in a \
                and not a.startswith(("/", ".", "http")):
            return a
    return None


def _restore(state):
    for obj, attr, old in reversed(state["patched"]):
        try:
            setattr(obj, attr, old)
        except Exception:
            pass


def _rss():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM"):
                return float(line.split()[1]) * 1024.0
    except Exception:
        pass
    return 0.0


# ---------------- driving finetune ------------------------------------


def _find_entry(mod):
    cands = []
    for name in ENTRY_NAMES:
        f = getattr(mod, name, None)
        if callable(f):
            cands.append(f)
    for name, f in sorted(vars(mod).items()):
        if callable(f) and inspect.isfunction(f) \
                and f.__module__ == mod.__name__ \
                and re.search(r"train|finetun|tune|post|fit", name) \
                and f not in cands:
            cands.append(f)
    for name in ("main", "run"):
        f = getattr(mod, name, None)
        if callable(f) and f not in cands:
            cands.append(f)
    return cands


def _fill_kwargs(sig, dataset, out_dir):
    kwargs, vkw = {}, False
    for p in sig.parameters.values():
        nml = p.name.lower()
        if p.kind == p.VAR_KEYWORD:
            vkw = True
            continue
        if p.kind == p.VAR_POSITIONAL:
            continue
        if nml in DATA_NAMES or any(k in nml for k in DATA_NAMES):
            kwargs[p.name] = str(dataset)
        elif nml in OUT_NAMES or any(k in nml for k in OUT_NAMES):
            kwargs[p.name] = str(out_dir)
        elif nml in STEP_NAMES or any(k == nml for k in STEP_NAMES):
            kwargs[p.name] = 2
        elif nml in BATCH_NAMES:
            kwargs[p.name] = 1
        elif nml in BASE_NAMES:
            kwargs[p.name] = "nvidia/GR00T-N1.5-3B"
    if vkw:
        for k in DATA_NAMES[:6]:
            kwargs.setdefault(k, str(dataset))
        for k in OUT_NAMES[:6]:
            kwargs.setdefault(k, str(out_dir))
        for k in ("steps", "max_steps", "epochs"):
            kwargs.setdefault(k, 2)
        kwargs.setdefault("batch_size", 1)
    return kwargs


def _drive_finetune(mod, dataset, out_dir):
    entries = _find_entry(mod)
    assert entries, (
        "harmless.policies.finetune has no train/finetune/main entry")
    errs = []
    for fn in entries:
        attempts = []
        try:
            sig = inspect.signature(fn)
        except Exception:
            sig = None
        if sig is not None:
            attempts.append(((), _fill_kwargs(sig, dataset, out_dir)))
        attempts += [
            ((str(dataset), str(out_dir)), {}),
            ((str(dataset),), {"out_dir": str(out_dir)}),
            ((str(dataset),), {}),
            ((), {}),
            ((["--dataset", str(dataset), "--out", str(out_dir),
               "--steps", "2"],), {}),
            ((["--data", str(dataset), "--checkpoint", str(out_dir),
               "--epochs", "1"],), {}),
        ]
        for args, kw in attempts:
            try:
                return fn(*args, **kw)
            except SystemExit as exc:
                if exc.code in (0, None):
                    return None
                errs.append(f"{fn.__name__} exit {exc.code}")
                continue
            except (TypeError, ValueError, FileNotFoundError) as exc:
                errs.append(f"{fn.__name__}{args}: {exc}")
                continue
        for argv in (
                ["--dataset", str(dataset), "--out", str(out_dir),
                 "--steps", "2"],
                ["--dataset", str(dataset), "--checkpoint",
                 str(out_dir), "--epochs", "1"],
                ["--data", str(dataset), "--out-dir", str(out_dir)]):
            old_argv = sys.argv
            try:
                sys.argv = [getattr(fn, "__name__", "finetune")] + argv
                return fn()
            except SystemExit as exc:
                if exc.code in (0, None):
                    return None
                errs.append(f"{fn.__name__} exit {exc.code}")
            except (TypeError, ValueError, FileNotFoundError) as exc:
                errs.append(f"{fn.__name__}() argv: {exc}")
            finally:
                sys.argv = old_argv
    pytest.fail(
        "no finetune.py entry point could be driven on the recorded "
        "episodes:\n" + "\n".join(errs[:8]))


def _git_tracked(path):
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "ls-files", "-z",
             "--error-unmatch", str(path)],
            stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return False


def _git_ignored(path):
    r = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q", str(path)])
    return r.returncode == 0


def _under(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except Exception:
        return False


def _find_written_ckpt(mod, result, out_dir):
    cands = []
    vals = [result]
    if isinstance(result, (list, tuple)):
        vals += list(result)
    if isinstance(result, Mapping):
        vals += list(result.values())
    for v in vals:
        if isinstance(v, (str, os.PathLike)) and Path(v).exists():
            cands.append(Path(v))
    if Path(out_dir).exists():
        leaves = [p for p in Path(out_dir).rglob("*")
                  if p.is_file() and p.suffix.lower() in WEIGHT_EXT]
        if leaves:
            cands.append(Path(out_dir))
    for name, v in vars(mod).items():
        if isinstance(v, (str, os.PathLike)) and Path(v).exists() \
                and re.search(r"ckpt|checkpoint|out|save|export|dir",
                              name, re.I):
            cands.append(Path(v))
    for c in cands:
        c = c.parent if c.is_file() else c
        try:
            if any(p.is_file() and p.suffix.lower() in WEIGHT_EXT
                   for p in c.rglob("*")):
                return c
        except Exception:
            continue
    return None


def _harvest_modules(*roots):
    import torch
    found, seen = [], set()

    def walk(obj, depth):
        if depth > 4 or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            found.append(obj)
            return
        if isinstance(obj, Mapping):
            for v in list(obj.values())[:200]:
                walk(v, depth + 1)
        elif isinstance(obj, (list, tuple)) and len(obj) <= 64:
            for v in obj:
                walk(v, depth + 1)
        elif not inspect.ismodule(obj) and not inspect.isclass(obj) \
                and not inspect.isfunction(obj) \
                and not inspect.ismethod(obj):
            try:
                for v in list(vars(obj).values()):
                    walk(v, depth + 1)
            except TypeError:
                pass

    for r in roots:
        walk(r, 0)
    return found


def _record_one_episode(demos, out_dir):
    """Ask demos.py for a single episode — the same bounded-entry
    probing POL-5.1 exercises, minimized. Returns out_dir or None;
    a run that ignores the bound is cut off at the deadline and the
    partial dataset is still usable."""
    COUNT = ("episodes", "n_episodes", "num_episodes", "episode_count",
             "count", "n", "num", "total")
    OUT = ("out_dir", "output_dir", "output", "out", "path", "root",
           "dest", "dataset_dir", "save_dir", "dir", "dataset")
    for name in ("EPISODES", "N_EPISODES", "NUM_EPISODES",
                 "EPISODE_COUNT", "TOTAL_EPISODES", "DEFAULT_EPISODES"):
        if isinstance(getattr(demos, name, None), int):
            setattr(demos, name, 1)
    for env in ("NOHARM_DEMO_EPISODES", "DEMOS_EPISODES", "N_EPISODES",
                "NUM_EPISODES"):
        os.environ[env] = "1"

    class _OT(Exception):
        pass

    def handler(signum, frame):
        raise _OT("record exceeded the deadline")
    try:
        old_sig = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, 300)
        armed = True
    except (ValueError, OSError, AttributeError):
        armed = False
    try:
        for name in ("record_episodes", "record_demos", "record_dataset",
                     "record", "collect_episodes", "collect", "generate",
                     "run", "main"):
            fn = getattr(demos, name, None)
            if not callable(fn):
                continue
            try:
                sig = inspect.signature(fn)
            except Exception:
                sig = None
            tries = []
            if sig is not None:
                kw = {}
                vkw = False
                for p in sig.parameters.values():
                    nml = p.name.lower()
                    if p.kind == p.VAR_KEYWORD:
                        vkw = True
                        continue
                    if p.kind == p.VAR_POSITIONAL:
                        continue
                    if any(k == nml or k in nml or nml in k
                           for k in COUNT):
                        kw[p.name] = 1
                    elif any(k == nml or k in nml or nml in k
                             for k in OUT):
                        kw[p.name] = str(out_dir)
                if vkw:
                    kw.setdefault("episodes", 1)
                    kw.setdefault("out_dir", str(out_dir))
                tries.append(((), kw))
            tries += [((1, str(out_dir)), {}),
                      ((1,), {"out_dir": str(out_dir)}), ((1,), {})]
            for args, k in tries:
                try:
                    fn(*args, **k)
                    return out_dir
                except _OT:
                    return out_dir
                except SystemExit as exc:
                    if exc.code in (0, None):
                        return out_dir
                except (TypeError, ValueError, FileNotFoundError):
                    continue
    finally:
        if armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_sig)
    return None


def test_pol_6_1_finetune_posttrains_gr00t_frozen_vl_one_8gb_gpu(
        tmp_path):
    finetune = _import("harmless.policies.finetune")

    # ---- a real episode dataset from demos.py --------------------------
    demos = _import("harmless.policies.demos")
    dataset_dir = tmp_path / "episodes"
    got = _record_one_episode(demos, dataset_dir)
    assert got is not None or dataset_dir.exists(), (
        "demos.py could not record one episode for finetune to train "
        "on")

    # ---- spies: weights reads, downloads, optimizers, devices ---------
    state = {"patched": [], "reads": [], "hf": [], "opts": [],
             "opt_steps": 0, "param_snap": {}, "devices": set(),
             "frozen": set()}
    out_dir = tmp_path / "ckpt"
    _install_spies(state)
    rss0 = _rss()
    import torch
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    result = None
    try:
        result = _drive_finetune(finetune, dataset_dir, out_dir)
    finally:
        _restore(state)
    rss1 = _rss()

    # ---- post-trains the released GR00T N checkpoint -------------------
    released = _hf_gr00t_snapshot()
    read_gr00t = any(
        GR00T_N.search(p) and _under(p, r)
        for p in state["reads"] for r in
        [Path.home() / ".cache" / "huggingface" / "hub",
         Path(os.environ.get("HF_HUB_CACHE", "/nonexistent"))])
    dl_gr00t = any(GR00T_N.search(str(c.get("repo") or ""))
                   or GR00T_REPO.search(str(c.get("repo") or ""))
                   for c in state["hf"])
    rel_idx = _tensor_index(released) if released else {}

    assert read_gr00t or dl_gr00t or (
        released is not None and any(
            _under(p, released) for p in state["reads"])), (
        "finetune never loaded the released GR00T N checkpoint — no "
        "weight-file read inside the HF cache snapshot, no HF download "
        "naming a GR00T-N repo "
        f"({len(state['reads'])} reads, hf calls: "
        f"{state['hf'][:5]})")

    # ---- training actually moved parameters ---------------------------
    changed = 0
    for pid, (p, snap) in state["param_snap"].items():
        cur = p.detach().float().cpu().numpy()
        if cur.shape == snap.shape and not _same(cur, snap):
            changed += 1
    trained_in_process = changed > 0

    ckpt = _find_written_ckpt(finetune, result, out_dir)
    assert ckpt is not None, (
        "finetune wrote no checkpoint (no weight files under "
        f"{out_dir} and no path returned)")
    out_idx = _tensor_index(ckpt)
    assert out_idx, (
        f"the written checkpoint at {ckpt} has no readable weight "
        "tensors")

    differs_from_release = []
    vl_named_out = []
    for name, (shape, loader) in out_idx.items():
        if not shape:
            continue
        rel = _released_value(rel_idx, name) if rel_idx else None
        if rel is None:
            continue
        try:
            cur = loader()
        except Exception:
            continue
        if VL_NAME.search(name):
            vl_named_out.append((name, cur, rel))
        elif not _same(cur, rel):
            differs_from_release.append(name)

    assert trained_in_process or differs_from_release, (
        "no parameter moved during finetune — no optimizer step "
        "changed a weight and the written checkpoint is bit-identical "
        "to the released GR00T N tensors it shares names with "
        f"(compared {len(out_idx)} tensors)")

    # ---- the vision-language backbone stayed frozen --------------------
    modules = _harvest_modules(result, vars(finetune))
    named = {}
    for mod in modules:
        try:
            for pn, p in mod.named_parameters():
                named.setdefault(pn, p)
        except Exception:
            pass
    opt_params = set()
    for opt in state["opts"]:
        try:
            for g in opt.param_groups:
                for p in g.get("params", []):
                    opt_params.add(id(p))
        except Exception:
            pass
    vl_params = {n: p for n, p in named.items() if VL_NAME.search(n)}
    if vl_params:
        bad = []
        for n, p in vl_params.items():
            if id(p) in opt_params and getattr(p, "requires_grad", True):
                rel = _released_value(rel_idx, n) if rel_idx else None
                if rel is None or not _same(
                        p.detach().float().cpu().numpy(), rel):
                    bad.append(n)
        assert not bad, (
            "these vision-language backbone tensors received gradient "
            "updates — the line requires the VL backbone frozen: "
            f"{bad[:6]}")
        for n, cur, rel in vl_named_out:
            assert _same(cur, rel, tol=1e-4), (
                f"written checkpoint's VL tensor {n} differs from the "
                "released GR00T N value — the backbone was not frozen")
    else:
        # no trainable VL stack at all: the backbone is only expressible
        # as fixed feature projectors — they must carry no gradients
        frozen_vl = [n for n in state["frozen"] if VL_NAME.search(n)]
        proj = [n for n, v in vars(finetune).items()
                if re.search(r"proj|backbone|vision|language", n, re.I)]
        fixed = [n for n, p in named.items()
                 if not getattr(p, "requires_grad", True)]
        assert proj or fixed or vl_named_out or frozen_vl, (
            "no vision-language backbone was loaded or frozen — "
            "finetune trained weights with no VL component present")

    # ---- one 8 GB GPU ---------------------------------------------------
    cuda_devices = {d for d in state["devices"] if d.startswith("cuda")}
    assert len(cuda_devices) <= 1, (
        f"training spread across devices {sorted(cuda_devices)} — the "
        "line says one 8 GB GPU")
    if cuda and cuda_devices:
        peak = torch.cuda.max_memory_allocated()
        assert peak <= GPU_BUDGET_B, (
            f"training peaked at {peak / 1e9:.2f} GB of device memory — "
            "over one 8 GB GPU's budget")
    elif not cuda_devices:
        assert rss1 - rss0 <= RSS_BUDGET_B or rss1 == 0 or rss0 == 0, (
            f"CPU training grew RSS by {(rss1 - rss0) / 1e9:.2f} GB — "
            "a run that cannot fit one 8 GB GPU")

    # ---- the checkpoint is written outside git --------------------------
    # the declared default destination must also resolve outside git —
    # the builder may not touch .gitignore, so a default under the
    # worktree has to land in an already-ignored subtree (demo/, runs/)
    for fn in _find_entry(finetune):
        try:
            sig = inspect.signature(fn)
        except Exception:
            continue
        for p in sig.parameters.values():
            nml = p.name.lower()
            if p.default is inspect.Parameter.empty \
                    or p.default is None:
                continue
            if not (nml in OUT_NAMES
                    or any(k in nml for k in OUT_NAMES)):
                continue
            if isinstance(p.default, (str, os.PathLike)):
                dflt = Path(p.default)
                if not dflt.is_absolute():
                    dflt = REPO / dflt
                if _under(dflt, REPO):
                    assert _git_ignored(dflt), (
                        f"finetune's default output {p.default!r} "
                        "resolves inside the worktree on a tracked "
                        "path — the line requires the checkpoint be "
                        "written outside git")
    for name, v in vars(finetune).items():
        if name.startswith("__"):
            continue
        if isinstance(v, (str, os.PathLike)) and re.search(
                r"ckpt|checkpoint|out_dir|output_dir|save_dir|export",
                name, re.I):
            dflt = Path(v)
            if not dflt.is_absolute():
                dflt = REPO / dflt
            if _under(dflt, REPO):
                assert _git_ignored(dflt), (
                    f"finetune's declared output {name}={v!r} resolves "
                    "inside the worktree on a tracked path — the line "
                    "requires the checkpoint be written outside git")

    ckpt = Path(ckpt).resolve()
    assert not _under(ckpt, REPO / ".git"), (
        f"the checkpoint {ckpt} resolves inside .git")
    if _under(ckpt, REPO):
        assert _git_ignored(ckpt), (
            f"the checkpoint {ckpt} sits inside the worktree on a "
            "tracked path — the line requires it written outside git")
    tracked = [str(p) for p in ckpt.rglob("*")
               if p.is_file() and p.suffix.lower() in WEIGHT_EXT
               and _git_tracked(p)]
    assert not tracked, (
        f"checkpoint weight files are git-tracked: {tracked[:4]}")
