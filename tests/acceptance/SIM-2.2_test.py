"""SIM-2.2@1 A scripted stand episode runs 5 seconds of sim with the robot
upright.

The contract under test is the MuJoCo reference backend living under
harmless/sim/** (the sim code this requirement may touch). The episode is
scripted — the repo's ScriptedPolicy, a deterministic hold-the-measured-
pose controller filling the same policy slot the model policies fill —
and the robot it drives is the G1 of kitchen.xml, spawned at the stand
keyframe. `harmless.sim` does not exist yet — failing to import it is the
expected pre-implementation failure.

The test first looks for a dedicated scripted-stand entry point under
harmless.sim (stand/scripted episode-named callables) and, when it can be
driven on an observable MjModel/MjData pair — ours or the one it returns —
uses it. Otherwise the episode is composed through the repo's public
contracts exactly the way a caller would: the sim Executor supplies the
live MjModel/MjData, harmless.runner.Runner binds ScriptedPolicy to the
G1Embodiment adapter, and the executor's own stepping entry point advances
physics while the runner applies the scripted hold at the embodiment's
50 Hz control rate.

The line is judged on measured state after the run: about five seconds of
sim time must have elapsed, and the G1's pelvis must still be at standing
height with its up axis pointing up — not a fall, a sit, or a scripted
freeze of sim time.
"""

import importlib
import inspect
import os
import pkgutil
import re
from pathlib import Path

import mujoco
import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]
KITCHEN_XML = REPO / "kitchen.xml"

_PATH_PARAMS = ("xml_path", "scene", "scene_path", "scene_xml", "xml",
                "path", "model_path", "mjcf", "xml_file", "mjcf_path")
_HEADLESS_KW = {"record": False, "recording": False, "record_video": False,
                "headless": True, "viewer": False, "show_viewer": False,
                "gui": False, "render": False, "rendering": False,
                "attach_viewer": False, "window": False, "visible": False}
_OUT_PARAMS = ("out_dir", "output_dir", "run_dir", "out", "log_dir",
               "outdir", "output")
_WS_PARAMS = ("skills_path", "learned_skills", "skills", "ws_dir",
              "workspace", "workspace_dir")
_STEP_SECONDS_FNS = ("run_seconds", "step_seconds", "advance_seconds",
                     "simulate_seconds", "run_for", "forward_seconds")
_STEP_FNS = ("step", "advance", "simulate", "forward", "tick")
_EPISODE_NAME = re.compile(
    r"stand|scripted", re.I)
_EPISODE_NAME_EXCLUDE = re.compile(
    r"test|helper|util|import|load|path", re.I)


def _sim_modules():
    """Every importable module under harmless/sim — the backend package."""
    sim = importlib.import_module("harmless.sim")
    modules = [sim]
    path = getattr(sim, "__path__", None)
    if path is not None:
        for info in pkgutil.walk_packages(path, prefix="harmless.sim."):
            try:
                modules.append(importlib.import_module(info.name))
            except Exception:
                continue
    return modules


def _find_executor(modules):
    """The Executor the sibling line names, wherever under harmless.sim
    it lives."""
    named = []
    factories = []
    for mod in modules:
        ex = getattr(mod, "Executor", None)
        if inspect.isclass(ex) or callable(ex):
            return ex
        for attr, value in vars(mod).items():
            low = attr.lower()
            if "executor" in low and inspect.isclass(value):
                named.append(value)
            elif ("executor" in low
                  or low in ("make_executor", "create_executor",
                             "load_executor", "get_executor")) \
                    and callable(value):
                factories.append(value)
    if named:
        return named[0]
    if factories:
        return factories[0]
    return None


def _make_executor(factory, tmp_path):
    """Construct the Executor on kitchen.xml with headless-friendly kwargs
    wherever its signature accepts them."""
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        params = {}
    kw = {}
    for name in params:
        if name in _PATH_PARAMS:
            kw[name] = str(KITCHEN_XML)
        elif name in _HEADLESS_KW:
            kw[name] = _HEADLESS_KW[name]
        elif name in _OUT_PARAMS:
            kw[name] = str(tmp_path)
        elif name in _WS_PARAMS:
            kw[name] = str(tmp_path / name)
    errors = []
    for args in ((), (str(KITCHEN_XML),)):
        try:
            return factory(*args, **kw)
        except Exception as e:  # noqa: BLE001 - reported if all spellings fail
            errors.append(f"{factory}{args}+{sorted(kw)}: {e}")
    pytest.fail("could not construct the sim Executor on kitchen.xml:\n"
                + "\n".join(errors))


def _body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _run_seconds(ex, seconds):
    for name in _STEP_SECONDS_FNS:
        fn = getattr(ex, name, None)
        if callable(fn):
            fn(seconds)
            return
    for name in _STEP_FNS:
        fn = getattr(ex, name, None)
        if callable(fn):
            fn(int(round(seconds / float(ex.model.opt.timestep))))
            return
    pytest.fail("the sim Executor exposes no way to step sim time "
                "(no run_seconds/step-style method)")


def _mj_pair(obj, seen=None):
    """An (MjModel, MjData) pair reachable from obj — the object itself,
    its attributes, or values if it is a mapping."""
    if seen is None:
        seen = set()
    if id(obj) in seen or obj is None:
        return None
    seen.add(id(obj))
    if isinstance(obj, mujoco.MjData):
        return obj
    if isinstance(obj, dict):
        for value in obj.values():
            found = _mj_pair(value, seen)
            if found is not None:
                return found
        return None
    data = getattr(obj, "data", None)
    if isinstance(data, mujoco.MjData):
        return data
    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = _mj_pair(value, seen)
            if found is not None:
                return found
    return None


def _find_stand_entry(modules):
    """A dedicated scripted/stand episode entry under harmless.sim."""
    for mod in modules:
        for attr, value in vars(mod).items():
            if not callable(value) or inspect.isclass(value):
                continue
            if _EPISODE_NAME.search(attr) \
                    and not _EPISODE_NAME_EXCLUDE.search(attr):
                return value
    return None


def _drive_entry(entry, ex, tmp_path):
    """Drive a scripted-stand entry point with whatever its signature
    accepts; returns the MjData the episode ran on, or None when the call
    leaves nothing observable."""
    try:
        params = inspect.signature(entry).parameters
    except (TypeError, ValueError):
        params = {}
    from harmless.policy import ScriptedPolicy
    kw = {}
    for name in params:
        low = name.lower()
        if low in ("executor", "ex", "sim", "backend", "executor_"):
            kw[name] = ex
        elif low in ("policy", "controller"):
            kw[name] = "scripted"
        elif "policy" in low:
            kw[name] = ScriptedPolicy()
        elif low in ("seconds", "duration", "sim_seconds", "time",
                     "episode_seconds", "secs"):
            kw[name] = 5.0
        elif name in _PATH_PARAMS:
            kw[name] = str(KITCHEN_XML)
        elif name in _HEADLESS_KW:
            kw[name] = _HEADLESS_KW[name]
        elif name in _OUT_PARAMS:
            kw[name] = str(tmp_path)
    try:
        result = entry(**kw)
    except Exception:  # noqa: BLE001 - not the line's entry; compose instead
        return None
    return _mj_pair(result) or (
        ex.data if any(v is ex for v in kw.values()) else None)


def _scripted_episode(ex):
    """Run the scripted stand episode through the repo's contracts:
    Runner binds the scripted hold to the G1 embodiment on the executor's
    live MjModel/MjData; the executor steps physics between control
    ticks at the embodiment's 50 Hz. The script is the stand pose: the
    joint positions measured at spawn (the stand keyframe the scene
    applies), replayed as fixed targets every tick — a live echo would
    ratify the robot's sag into the command and is not a stand."""
    from harmless.embodiments.g1 import G1Embodiment
    from harmless.policy import ScriptedPolicy
    from harmless.runner import Runner

    model, data = ex.model, ex.data
    emb = G1Embodiment(model, data)
    runner = Runner(policy=ScriptedPolicy(), embodiment=emb,
                    model=model, data=data)
    stand = {}
    for name in emb.action_space.joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            stand[name] = float(data.qpos[model.jnt_qposadr[jid]])
    dt = float(model.opt.timestep)
    steps_per_tick = max(1, int(round(0.02 / dt)))   # 50 Hz control
    start = float(data.time)
    while data.time - start < 5.0 - 1e-9:
        runner.step(observation={"joints": stand})
        step = getattr(ex, "step", None)
        if callable(step):
            step(steps_per_tick)
        else:
            _run_seconds(ex, steps_per_tick * dt)
    emb.close()


def _assert_upright(data, model):
    pelvis = _body_id(model, "pelvis")
    assert pelvis >= 0, "the episode ran without the G1 (no pelvis body)"
    up = data.xmat[pelvis].reshape(3, 3)[:, 2]
    assert data.xpos[pelvis][2] > 0.55, (
        f"after 5s of scripted stand the pelvis is at "
        f"z={data.xpos[pelvis][2]:.3f} — the robot is down")
    assert up[2] > 0.85, (
        f"after 5s of scripted stand the pelvis up-axis is "
        f"{np.round(up, 3).tolist()} — the robot is not upright")


def test_sim_2_2_scripted_stand_episode_five_seconds_upright(tmp_path):
    modules = _sim_modules()
    factory = _find_executor(modules)
    assert factory is not None, (
        "harmless.sim exposes no Executor (no Executor class, "
        "executor-named class, or executor factory)")
    ex = _make_executor(factory, tmp_path)
    try:
        entry = _find_stand_entry(modules)
        if entry is not None:
            data = _drive_entry(entry, ex, tmp_path)
            if data is not None:
                assert data.time >= 4.9, (
                    f"the scripted stand episode ran {data.time:.2f}s "
                    "of sim, expected 5")
                _assert_upright(data, ex.model)
                return
        _scripted_episode(ex)
        assert ex.data.time >= 4.9, (
            f"the scripted stand episode ran {ex.data.time:.2f}s of "
            "sim, expected 5")
        _assert_upright(ex.data, ex.model)
    finally:
        close = getattr(ex, "close", None)
        if callable(close):
            close()
