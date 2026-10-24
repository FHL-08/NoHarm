"""POL-5.1@1 demos.py records at least 200 G1 episodes in LeRobot format
(head-camera frames, joint state, 50 Hz joint_pos actions) from
randomized starts, each ending in a measured palm-bowl contact.

The contract under test is harmless/policies/demos.py (a file this
requirement touches) — the demonstration recorder for the bowl touch:

  records ... in LeRobot format: a bounded recording run (the test asks
    for a small number of episodes, not the full corpus) must leave a
    dataset on disk in LeRobot layout — a JSON info manifest declaring
    fps=50, a per-episode data artifact (parquet/jsonl/npz) carrying
    joint state and joint_pos action vectors, and a head-camera video or
    frame artifact per episode. When a parquet reader is installed the
    rows themselves are checked: ~20 ms frame spacing and an action
    column at least as wide as the G1 joint_pos space.

  at least 200 G1 episodes: the recorder's own episode target — the
    count default on its record entry point, a module-level episode
    constant, or a CLI flag default — must be >= 200. Recording all 200
    inside a test is not feasible, so the per-episode contract is
    verified on a bounded run while the declared target is checked
    directly.

  head-camera frames + joint state + 50 Hz joint_pos actions: the
    dataset declares (and, when readable, stores) per-frame joint state
    and joint-position action vectors at the 50 Hz control rate, plus a
    decodable head-camera video per episode.

  randomized starts: measured through the simulator — the test tapes
    every physics step the recorder runs and requires the per-episode
    start states (pelvis free joint, arm joints, bowl pose) to differ.

  each ending in a measured palm-bowl contact: measured through the
    simulator, not the recorder's self-report. The tape sees a real
    palm-bowl contact (a MuJoCo contact pair between a bowl geom and a
    wrist/hand robot link, a palm site pressed onto the bowl surface, or
    a hand touch sensor firing next to the bowl) inside the last second
    of at least as many recorded segments as there are saved episodes.

harmless/policies/demos.py does not exist yet — failing to import it is
the expected pre-implementation failure.
"""

import importlib
import inspect
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]
SCENE = REPO / "kitchen.xml"
EPISODES_REQUIRED = 200
TEST_EPISODES = 2
FPS = 50.0
END_CONTACT_WINDOW_S = 1.0
RUN_DEADLINE_S = 600

COUNT_NAMES = (
    "episodes", "n_episodes", "num_episodes", "episode_count", "count",
    "n_demos", "num_demos", "demo_count", "n", "num", "total",
    "total_episodes", "n_samples", "num_samples", "n_record",
)
OUT_NAMES = (
    "out_dir", "output_dir", "output", "out", "path", "out_path",
    "root", "root_dir", "dest", "destination", "dataset_dir",
    "dataset_path", "save_dir", "dir", "write_to", "data_dir", "dataset",
)
TARGET_NAMES = ("task", "target", "obj", "object", "prop", "name",
                "item", "object_name")
ENTRY_NAMES = (
    "record_episodes", "record_demos", "record_dataset", "record",
    "collect_episodes", "collect_demos", "collect", "generate_demos",
    "generate_dataset", "generate", "build_dataset", "make_dataset",
    "run", "main",
)
HAND_BODY = re.compile(r"wrist|hand|palm|elbow", re.I)


def _import(name, required=True):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        if not required:
            return None
        pytest.fail(
            f"{name} cannot be imported ({type(exc).__name__}: {exc}) "
            "— the demonstration recorder does not exist yet")


# ---------------- measured palm-bowl contact --------------------------


def _in_robot(ex, bid):
    while bid > 0:
        if bid == ex.pelvis_body:
            return True
        bid = ex.model.body_parentid[bid]
    return False


def _palm_bowl_contact(ex):
    """Measured palm-bowl contact on live sim state: a real contact pair
    between a bowl geom and a wrist/hand link, a palm site pressed on the
    bowl surface, or a hand touch sensor firing next to the bowl."""
    import mujoco
    m, d = ex.model, ex.data
    bowl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bowl")
    if bowl >= 0:
        for c in range(int(d.ncon)):
            con = d.contact[c]
            b1, b2 = int(m.geom_bodyid[con.geom1]), \
                int(m.geom_bodyid[con.geom2])
            if bowl not in (b1, b2):
                continue
            other = b2 if b1 == bowl else b1
            name = mujoco.mj_id2name(
                m, mujoco.mjtObj.mjOBJ_BODY, other) or ""
            if HAND_BODY.search(name) and _in_robot(ex, other):
                return True
    try:
        gaps = ex.palm_gaps("bowl")
        if any(float(g) <= 0.02 for g in gaps.values()):
            return True
    except Exception:
        gaps = {}
    try:
        sens = ex.touch_sensors
        for side, site in (("right", "touch_rhand"),
                           ("left", "touch_lhand")):
            if site in sens and gaps.get(side, 9.0) <= 0.15 and \
                    float(d.sensordata[sens[site]]) > 0.5:
                return True
    except Exception:
        pass
    return False


def _start_sig(ex):
    """Start-state signature of an episode segment: pelvis free joint,
    arm joints, bowl pose — anything a randomizer could vary."""
    m, d = ex.model, ex.data
    sig = list(np.asarray(d.qpos[:7], float))
    try:
        sig += list(np.asarray(ex.pos("bowl"), float))
    except Exception:
        sig += [0.0, 0.0, 0.0]
    for j in list(getattr(ex, "arm_jids", []) or []):
        sig.append(float(d.qpos[m.jnt_qposadr[j]]))
    return np.asarray(sig, float)


# ---------------- the tape: measured state while recording ------------


def _tape_note(state, ex, t):
    segs = state["tape"].setdefault(id(ex), [])
    if not segs or t < segs[-1]["last_t"] - 1e-9 \
            or segs[-1]["last_t"] == float("inf"):
        segs.append({"start": _start_sig(ex), "last_t": t,
                     "samples": []})
    seg = segs[-1]
    seg["last_t"] = t
    seg["samples"].append((t, _palm_bowl_contact(ex)))


def _make_mj_step_wrap(state):
    def wrap(orig):
        def w(m, d, *a, **k):
            r = orig(m, d, *a, **k)
            ex = state["owners"].get(id(d))
            if ex is not None:
                _tape_note(state, ex, float(d.time))
            return r
        return w
    return wrap


def _install_tape(state):
    """Tape measured sim state while the recorder runs: every physics
    step of every Executor is stamped (sim time + palm-bowl contact) and
    segmented into episodes on new instances and sim-time resets."""
    import mujoco
    import harmless.executor as exmod

    wrap = _make_mj_step_wrap(state)
    patched = state["patched"]

    def patch(obj, attr, fn):
        try:
            old = getattr(obj, attr)
        except Exception:
            return
        patched.append((obj, attr, old))
        setattr(obj, attr, fn)

    state["orig_mj_step"] = mujoco.mj_step
    patch(mujoco, "mj_step", wrap(mujoco.mj_step))

    orig_init = exmod.Executor.__init__

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        state["owners"][id(self.data)] = self
    patch(exmod.Executor, "__init__", __init__)

    orig_reset = getattr(exmod.Executor, "reset_trial", None)
    if orig_reset is not None:
        def reset_trial(self, *a, **k):
            r = orig_reset(self, *a, **k)
            segs = state["tape"].setdefault(id(self), [])
            if segs:
                segs[-1]["last_t"] = float("inf")     # close segment
            return r
        patch(exmod.Executor, "reset_trial", reset_trial)


def _sweep_mj_step(state):
    """Rebind from-imported copies of the original mj_step inside any
    already-loaded module so no caller bypasses the tape."""
    orig = state.get("orig_mj_step")
    if orig is None:
        return
    wrap = _make_mj_step_wrap(state)
    for mod in list(sys.modules.values()):
        try:
            d = vars(mod)
        except Exception:
            continue
        for attr, v in list(d.items()):
            if v is orig:
                try:
                    state["patched"].append((mod, attr, v))
                    setattr(mod, attr, wrap(v))
                except Exception:
                    pass


def _adopt_existing_executors(state):
    """Register any Executor built before the tape was installed."""
    import gc
    import harmless.executor as exmod
    for obj in gc.get_objects():
        try:
            if isinstance(obj, exmod.Executor):
                state["owners"][id(obj.data)] = obj
        except Exception:
            pass


def _restore(state):
    for obj, attr, old in reversed(state["patched"]):
        try:
            setattr(obj, attr, old)
        except Exception:
            pass


# ---------------- driving the recorder --------------------------------


def _find_entry(mod):
    cands = []
    for name in ENTRY_NAMES:
        f = getattr(mod, name, None)
        if callable(f):
            cands.append(f)
    for name, f in sorted(vars(mod).items()):
        if callable(f) and inspect.isfunction(f) \
                and f.__module__ == mod.__name__ \
                and re.search(r"record|collect|demos?|dataset", name) \
                and f not in cands:
            cands.append(f)
    for name in ("main", "run"):
        f = getattr(mod, name, None)
        if callable(f) and f not in cands:
            cands.append(f)
    return cands


def _declared_episode_target(mod, entries):
    """The recorder's own episode target: a default on the record entry,
    a module constant, or a CLI flag default — must say >= 200."""
    found = []
    for f in entries:
        try:
            sig = inspect.signature(f)
        except Exception:
            continue
        for p in sig.parameters.values():
            nml = p.name.lower()
            if p.default is inspect.Parameter.empty:
                continue
            if isinstance(p.default, int) and not isinstance(
                    p.default, bool) and any(
                    k == nml or k in nml or nml in k for k in COUNT_NAMES):
                found.append((f"{f.__name__}({p.name}="
                              f"{p.default})", int(p.default)))
    for name, v in vars(mod).items():
        if isinstance(v, int) and not isinstance(v, bool) and \
                re.search(r"episode|demo|count|total|num", name, re.I) \
                and not re.search(r"fps|hz|rate|dim", name, re.I):
            found.append((name, int(v)))
    try:
        src = inspect.getsource(mod)
    except Exception:
        src = ""
    for m in re.finditer(
            r"--?[a-zA-Z_-]*episodes?[\"'\s\),][^\n]*?"
            r"default\s*=\s*(\d+)", src):
        found.append(("cli default", int(m.group(1))))
    for m in re.finditer(r"range\(\s*(\d{3,4})\s*\)", src):
        found.append(("range() literal", int(m.group(1))))
    return found


def _fill_kwargs(sig, n, out_dir):
    kwargs, vkw = {}, False
    for p in sig.parameters.values():
        nml = p.name.lower()
        if p.kind == p.VAR_KEYWORD:
            vkw = True
            continue
        if p.kind == p.VAR_POSITIONAL:
            continue
        if any(k == nml or k in nml or nml in k for k in COUNT_NAMES):
            kwargs[p.name] = n
        elif any(k == nml or k in nml or nml in k for k in OUT_NAMES):
            kwargs[p.name] = str(out_dir)
        elif nml in TARGET_NAMES:
            kwargs[p.name] = "bowl"
        elif re.search(r"scene|xml|mjcf", nml):
            kwargs[p.name] = str(SCENE)
        elif re.search(r"^(seed|rng)$", nml):
            kwargs[p.name] = None
    if vkw:
        for k in ("episodes", "n_episodes", "num_episodes", "count",
                  "n"):
            kwargs.setdefault(k, n)
        for k in ("out_dir", "output_dir", "out"):
            kwargs.setdefault(k, str(out_dir))
    return kwargs


def _bounded_call(fn, n, out_dir):
    """Try plausible spellings of a bounded n-episode record call.
    Returns the entry's result, or raises TypeError if none fit."""
    errs = []
    attempts = []
    try:
        sig = inspect.signature(fn)
    except Exception:
        sig = None
    if sig is not None:
        attempts.append(((), _fill_kwargs(sig, n, out_dir)))
        pos = []
        ok = True
        for p in sig.parameters.values():
            nml = p.name.lower()
            if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                continue
            if any(k == nml or k in nml or nml in k for k in COUNT_NAMES):
                pos.append(n)
            elif any(k == nml or k in nml or nml in k
                     for k in OUT_NAMES):
                pos.append(str(out_dir))
            elif nml in TARGET_NAMES:
                pos.append("bowl")
            elif p.default is inspect.Parameter.empty:
                ok = False
                break
        if ok:
            attempts.append((tuple(pos), {}))
    attempts += [
        ((n, str(out_dir)), {}),
        ((str(out_dir), n), {}),
        ((n,), {"out_dir": str(out_dir)}),
        ((n,), {}),
        ((), {}),
        ((["--episodes", str(n), "--out", str(out_dir)],), {}),
        ((["--episodes", str(n), "--out-dir", str(out_dir)],), {}),
        ((["-n", str(n), "-o", str(out_dir)],), {}),
    ]
    saw_system_exit = None
    for args, kw in attempts:
        try:
            return fn(*args, **kw)
        except SystemExit as exc:
            if exc.code in (0, None):
                return None
            saw_system_exit = exc
            continue
        except (TypeError, ValueError, FileNotFoundError) as exc:
            errs.append(f"{fn.__name__}{args}: {exc}")
            continue
    # a zero-arg main() that parses sys.argv
    for argv in (["--episodes", str(n), "--out", str(out_dir)],
                 ["--episodes", str(n), "--out-dir", str(out_dir)],
                 ["-n", str(n), "-o", str(out_dir)],
                 [str(n), str(out_dir)]):
        old_argv = sys.argv
        try:
            sys.argv = [getattr(fn, "__name__", "demos")] + argv
            return fn()
        except SystemExit as exc:
            if exc.code in (0, None):
                return None
            saw_system_exit = exc
        except (TypeError, ValueError, FileNotFoundError) as exc:
            errs.append(f"{fn.__name__}() argv={argv}: {exc}")
        finally:
            sys.argv = old_argv
    raise TypeError(
        f"{fn.__name__} could not be driven for a bounded {n}-episode "
        "run: " + "; ".join(errs[:6])
        + (f"; SystemExit({saw_system_exit.code})" if saw_system_exit
           else ""))


class _OutOfTime(Exception):
    pass


def _deadline(seconds):
    """SIGALRM deadline wrapping an in-process call (main thread only —
    degrades to a no-op off it)."""
    class _Ctx:
        def __init__(self):
            self._armed = False
            self._old = None

        def __enter__(self):
            def handler(signum, frame):
                raise _OutOfTime(
                    f"the recording run exceeded {seconds} s")
            try:
                self._old = signal.signal(signal.SIGALRM, handler)
                signal.setitimer(signal.ITIMER_REAL, seconds)
                self._armed = True
            except (ValueError, OSError):
                pass
            return self

        def __exit__(self, *a):
            if self._armed:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, self._old)
            return False
    return _Ctx()


def _set_count_knobs(mod, n):
    for name in ("EPISODES", "N_EPISODES", "NUM_EPISODES",
                 "EPISODE_COUNT", "TOTAL_EPISODES", "DEFAULT_EPISODES",
                 "N_DEMOS", "NUM_DEMOS", "DEMOS", "COUNT"):
        if isinstance(getattr(mod, name, None), int):
            setattr(mod, name, n)
    for env in ("NOHARM_DEMO_EPISODES", "DEMOS_EPISODES", "N_EPISODES",
                "NUM_EPISODES"):
        os.environ[env] = str(n)


def _dataset_root(out_dir, result, since=0.0):
    """Where the run wrote the dataset: the out dir we passed, a path
    the call returned, or a lerobot-manifest dir under it — falling back
    to a fresh info.json written under the usual dataset roots during
    the run."""
    cands = []
    vals = [result]
    if isinstance(result, (list, tuple)):
        vals += list(result)
    if isinstance(result, dict):
        vals += list(result.values())
    for v in vals:
        if isinstance(v, (str, os.PathLike)) and Path(v).exists():
            cands.append(Path(v))
    cands.append(Path(out_dir))
    for c in cands:
        if c.is_file():
            c = c.parent
        if not c.is_dir():
            continue
        if (c / "meta" / "info.json").exists() \
                or (c / "info.json").exists() \
                or list(c.rglob("info.json")):
            return c
    for root in (REPO / "demo", REPO / "demos", REPO / "data",
                 REPO / "datasets", REPO / "runs", REPO,
                 Path.home() / ".cache", Path("/tmp")):
        if not root.is_dir():
            continue
        try:
            for p in root.rglob("info.json"):
                if p.stat().st_mtime >= since:
                    return p.parent.parent if p.parent.name == "meta" \
                        else p.parent
        except Exception:
            continue
    return Path(out_dir)


# ---------------- LeRobot dataset verification ------------------------


def _info_manifest(root):
    for p in [root / "meta" / "info.json", root / "info.json"]:
        if p.is_file():
            return json.loads(p.read_text()), p
    hits = sorted(root.rglob("info.json"))
    if hits:
        return json.loads(hits[0].read_text()), hits[0]
    return None, None


def _episode_data_files(root):
    return [p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower()
            in (".parquet", ".npz", ".jsonl", ".pt", ".pkl")
            and "meta" not in p.parts]


def _video_files(root):
    return [p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower()
            in (".mp4", ".mkv", ".webm", ".avi")]


def _head_named(path):
    return bool(re.search(r"head|ego", str(path), re.I))


def _read_rows(path):
    """Per-frame rows from an episode data file, when a reader for its
    format is installed. Returns list of dicts / dict of arrays / None."""
    ext = path.suffix.lower()
    if ext == ".parquet":
        try:
            import pyarrow.parquet as pq
            return pq.read_table(str(path)).to_pylist()
        except Exception:
            try:
                import pandas as pd
                return pd.read_parquet(str(path)).to_dict("records")
            except Exception:
                return None
    if ext == ".jsonl":
        try:
            return [json.loads(l) for l in
                    path.read_text().splitlines() if l.strip()]
        except Exception:
            return None
    if ext == ".npz":
        try:
            z = np.load(str(path))
            return {k: z[k] for k in z.files}
        except Exception:
            return None
    return None


def _row_action(row):
    if isinstance(row, dict):
        for k in ("action", "actions", "joint_pos"):
            if k in row:
                return np.asarray(row[k], dtype=float)
        for k, v in row.items():
            if "action" in str(k).lower():
                return np.asarray(v, dtype=float)
    return None


def _check_lerobot(root, n_expect, n_joints):
    root = Path(root)
    assert root.is_dir(), (
        f"the recording run left no dataset directory at {root}")
    info, ipath = _info_manifest(root)
    assert info is not None, (
        f"no LeRobot info manifest (info.json) under {root} — "
        "the dataset is not in LeRobot format")
    fps = info.get("fps") or info.get("frame_rate") or info.get("fps_hz")
    if fps is None and isinstance(info.get("video"), dict):
        fps = info["video"].get("fps")
    assert fps is not None and abs(float(fps) - FPS) < 0.5, (
        f"{ipath} declares fps={fps} — joint_pos actions must be "
        "recorded at the 50 Hz control rate")
    feats = info.get("features") or info.get("modalities") or {}
    keys = [str(k) for k in feats]
    img_keys = [k for k in keys if re.search(r"image|video|cam|rgb",
                                            k, re.I)]
    head_ok = any(_head_named(k) for k in img_keys) or any(
        _head_named(p) for p in _video_files(root))
    assert head_ok, (
        "the dataset has no head-camera feature/video (feature keys: "
        f"{keys[:10]}) — head-camera frames are required")
    assert any(re.search(r"state|joint", k, re.I) for k in keys), (
        "the dataset declares no joint-state feature "
        f"(keys: {keys[:10]})")
    act_keys = [k for k in keys if re.search(r"action", k, re.I)]
    assert act_keys, (
        "the dataset declares no action feature "
        f"(keys: {keys[:10]})")
    for k in act_keys:
        f = feats.get(k)
        if isinstance(f, dict) and f.get("shape"):
            assert int(np.asarray(f["shape"]).reshape(-1)[-1]) \
                >= n_joints, (
                f"action feature {k} has shape {f['shape']} — "
                f"narrower than the {n_joints}-joint G1 joint_pos "
                "space")
    data_files = _episode_data_files(root)
    assert len(data_files) >= n_expect, (
        f"only {len(data_files)} episode data file(s) under {root} "
        f"after a {n_expect}-episode run")
    videos = _video_files(root)
    if videos:
        head_videos = [v for v in videos if _head_named(v)] or videos
        try:
            import imageio.v3 as iio
            for v in head_videos[:n_expect]:
                n_fr = sum(1 for _ in iio.imiter(str(v)))
                assert n_fr >= 2, (
                    f"head-camera video {v.name} holds {n_fr} "
                    "frame(s)")
        except ImportError:
            for v in head_videos[:n_expect]:
                assert v.stat().st_size > 10_000, (
                    f"head-camera artifact {v.name} is only "
                    f"{v.stat().st_size} B — not real frames")
    for f in data_files[:n_expect]:
        if f.suffix.lower() == ".parquet":
            with open(f, "rb") as fh:
                head = fh.read(4)
                fh.seek(-4, 2)
                tail = fh.read(4)
            assert head == b"PAR1" and tail == b"PAR1", (
                f"{f.name} is not a parquet file (no PAR1 magic)")
        rows = _read_rows(f)
        if rows is None:
            continue
        if isinstance(rows, dict):
            ts = rows.get("timestamp") or rows.get("timestamps")
            if ts is not None and len(ts) > 2:
                dt = np.diff(np.asarray(ts, dtype=float))
                assert np.abs(dt - 1.0 / FPS).max() < 0.005, (
                    f"{f.name} frame spacing is {dt.mean():.4f} s "
                    "— not 50 Hz")
            acts = rows.get("action")
            if acts is not None:
                arr = np.asarray(acts, dtype=float)
                assert arr.ndim == 2 and arr.shape[1] >= n_joints, (
                    f"{f.name} action rows are {arr.shape} — "
                    f"narrower than the {n_joints}-joint joint_pos "
                    "space")
            continue
        if not isinstance(rows, list) or len(rows) < 2:
            continue
        a0 = _row_action(rows[0])
        if a0 is not None:
            assert a0.reshape(-1).size >= n_joints, (
                f"{f.name} action rows carry {a0.size} values — "
                f"narrower than the {n_joints}-joint joint_pos space")
        ts = [r.get("timestamp") for r in rows
              if isinstance(r, dict) and r.get("timestamp") is not None]
        if len(ts) > 2:
            dt = np.diff(np.asarray(ts, dtype=float))
            assert np.abs(dt - 1.0 / FPS).max() < 0.005, (
                f"{f.name} frame spacing is {dt.mean():.4f} s — the "
                "actions are not on the 50 Hz control rate")


# ---------------- subprocess fallback ---------------------------------


def _driver_path(tmp_path):
    """A subprocess driver that replays this test's tape while running
    the recorder — used only when no in-process entry can be bounded."""
    p = tmp_path / "_demos_driver.py"
    p.write_text(textwrap.dedent(f"""\
        import importlib.util, sys
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "pol51_test", r"{Path(__file__).resolve()}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod._drive_subprocess(int(sys.argv[1]), Path(sys.argv[2]),
                              Path(sys.argv[3]))
    """))
    return p


def _dump_tape(state, tape_out):
    dump = {}
    for exid, segs in state["tape"].items():
        dump[str(exid)] = [{
            "start": np.asarray(s["start"]).tolist(),
            "last_t": s["last_t"] if s["last_t"] != float("inf") else 1e30,
            "samples": s["samples"],
        } for s in segs]
    Path(tape_out).write_text(json.dumps(dump))


def _drive_subprocess(n, out_dir, tape_out):
    """Runs inside the driver subprocess: installs the tape, drives the
    record entry, keeps flushing the tape to tape_out as JSON."""
    demos = _import("harmless.policies.demos")
    state = {"owners": {}, "tape": {}, "patched": []}
    _install_tape(state)
    _set_count_knobs(demos, n)

    orig_note = _tape_note

    def note(st, ex, t):
        before = sum(len(s) for s in st["tape"].values())
        orig_note(st, ex, t)
        after = sum(len(s) for s in st["tape"].values())
        if after != before:
            _dump_tape(st, tape_out)
    globals()["_tape_note"] = note

    _sweep_mj_step(state)
    _adopt_existing_executors(state)
    try:
        entries = _find_entry(demos)
        for fn in entries:
            try:
                _bounded_call(fn, n, Path(out_dir))
                break
            except TypeError:
                continue
    finally:
        _dump_tape(state, tape_out)


def _run_subprocess(n, out_dir, tmp_path):
    """Bounded run in a subprocess: the driver installs the same tape;
    we poll the dataset dir and stop once n episode files exist."""
    tape_out = tmp_path / "tape.json"
    driver = _driver_path(tmp_path)
    py = REPO / ".venv" / "bin" / "python"
    py = py if py.exists() else sys.executable
    env = dict(os.environ, NOHARM_HEADLESS="1", MUJOCO_GL="egl")
    proc = subprocess.Popen(
        [str(py), str(driver), str(n), str(out_dir), str(tape_out)],
        cwd=str(REPO), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + RUN_DEADLINE_S
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            files = _episode_data_files(out_dir) \
                if out_dir.exists() else []
            if len(files) >= n:
                newest = max(f.stat().st_mtime for f in files)
                if time.time() - newest > 2.0:
                    proc.terminate()
                    break
            time.sleep(1.0)
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
    finally:
        if proc.poll() is None:
            proc.kill()
    if tape_out.exists():
        raw = json.loads(tape_out.read_text())
        tape = {}
        for exid, segs in raw.items():
            tape[exid] = [{
                "start": np.asarray(s["start"]),
                "last_t": s["last_t"],
                "samples": [(t, c) for t, c in s["samples"]],
            } for s in segs]
        return tape
    return {}


def _contact_terminated_segments(tape):
    """Segments whose last END_CONTACT_WINDOW_S of sim time contain a
    measured palm-bowl contact."""
    out = []
    for segs in tape.values():
        for s in segs:
            samples = s["samples"]
            if not samples:
                continue
            t_end = samples[-1][0]
            if any(c for (t, c) in samples
                   if t_end - t <= END_CONTACT_WINDOW_S):
                out.append(s)
    return out


def test_pol_5_1_demos_record_lerobot_episodes_ending_in_bowl_contact(
        tmp_path):
    demos = _import("harmless.policies.demos")

    # ---- the recorder targets at least 200 episodes -------------------
    entries = _find_entry(demos)
    found = _declared_episode_target(demos, entries)
    assert any(v >= EPISODES_REQUIRED for _, v in found), (
        "demos.py declares no episode target of 200+ — the line "
        "requires at least 200 recorded G1 episodes (found: "
        f"{found or 'no count default/constant/flag'})")

    # ---- a bounded recording run under the measured-state tape --------
    state = {"owners": {}, "tape": {}, "patched": []}
    out_dir = tmp_path / "demos"
    t_start = time.time()
    _install_tape(state)
    _set_count_knobs(demos, TEST_EPISODES)
    ran = False
    try:
        _sweep_mj_step(state)
        _adopt_existing_executors(state)
        for fn in entries:
            try:
                with _deadline(RUN_DEADLINE_S):
                    _bounded_call(fn, TEST_EPISODES, out_dir)
                ran = True
                break
            except TypeError:
                continue
            except _OutOfTime:
                ran = bool(_episode_data_files(out_dir))
                break
        _adopt_existing_executors(state)
    finally:
        _restore(state)

    tape = state["tape"]
    if not ran and not _episode_data_files(out_dir):
        # nothing could be bounded in-process: run the subprocess driver
        tape = _run_subprocess(TEST_EPISODES, out_dir, tmp_path)

    root = _dataset_root(out_dir, None, since=t_start - 1)

    # ---- the dataset is LeRobot format --------------------------------
    try:
        from harmless.embodiments.g1 import G1Embodiment
        n_joints = len(G1Embodiment().action_space.joints)
    except Exception:
        n_joints = 29
    _check_lerobot(root, TEST_EPISODES, n_joints)

    # ---- every saved episode ended in a measured palm-bowl contact ----
    n_saved = len(_episode_data_files(root))
    good = _contact_terminated_segments(tape)
    assert len(good) >= min(n_saved, TEST_EPISODES), (
        f"the tape saw {len(good)} episode segment(s) ending in a "
        f"measured palm-bowl contact but {n_saved} episode file(s) "
        "were saved — episodes were recorded without the run ending "
        "in a real palm-on-bowl contact")

    # ---- starts are randomized ----------------------------------------
    sigs = [s["start"] for segs in tape.values()
            for s in segs if s["samples"]]
    assert len(sigs) >= 2, (
        "the tape saw fewer than two episode segments — cannot verify "
        "randomized starts")
    pairs = [float(np.abs(a - b).max())
             for i, a in enumerate(sigs) for b in sigs[i + 1:]
             if a.shape == b.shape]
    spread = max(pairs) if pairs else 0.0
    assert spread > 1e-3, (
        "every episode started from an identical measured state "
        f"(max deviation {spread:.2e}) — the starts are not "
        "randomized")
