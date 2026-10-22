"""NPC-2.2@1 The person stops rather than walking into the robot, measured
by person-robot distance holding above 0.4 m while the robot is
stationary.

The contract under test is the walking-person module this requirement
adds, ``harmless/person.py`` (named in the requirement's touches). It must
expose a runnable walking person — a module-level instance, a
zero-argument factory, or a constructible class — that lets a caller
advance simulated time (step/update/advance/tick-like method, or a bare
MjModel/MjData pair the test can step) and observe the person's
horizontal position (pos/position/xy accessor or a person-named body in
its MjData).

The stationary robot is presented to the walker as an xy point on the
floor, through whichever channel the module provides: a method such as
set_obstacle / add_obstacle / avoid / robot_at, a writable attribute such
as obstacle / robot / robot_pos, a robot-like parameter on its step
method, a matching constructor keyword, or a module-level obstacle
collection. At least one channel must exist: a walker that cannot be told
about a stationary robot can never stop for one.

The test first lets the person walk ~2 s to measure its heading, then
places the robot 1.6 m dead ahead on that heading and keeps it there,
stationary, for 30 s of simulated time. The line is true only if the
person

  - actually closes on the robot (minimum distance under 1.4 m — it was
    on a collision course, not parked somewhere else),
  - keeps person-robot distance above 0.4 m for the whole run (the
    hold-above threshold the line names),
  - never passes through or beyond the robot along its heading, and
  - is stopped at the end: under 8 cm of movement over the final 2 s.

Nothing about this exists yet: there is no harmless/person.py, and the
person currently in the kitchen scene is a rigid body that glides its
circuit at 0.35 m/s and "never stops or yields". The expected
pre-implementation failure is the missing module.
"""

import importlib
import inspect
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[2]

TARGET_SPEED = 0.25
DT = 0.02
HEADING_S = 2.0
RUN_S = 30.0
ROBOT_AHEAD_M = 1.6
MIN_APPROACH_M = 1.4
MIN_CLEARANCE_M = 0.4
MAX_PAST_M = 0.15
STOP_WINDOW_S = 2.0
STOP_MOVE_M = 0.08

STEP_NAMES = ("step", "update", "advance", "tick", "simulate", "run",
              "forward", "advance_time")
POSE_NAMES = ("pose_at", "apply_clip", "set_clip", "clip_at", "set_time",
              "animate", "play", "set_pose")
POS_NAMES = ("pos", "position", "xy", "location", "pos_xy", "get_pos",
             "get_position", "where", "root_pos", "base_pos")
TIME_NAMES = ("t", "time", "sim_time", "t_s", "now", "clock")
MD_PAIRS = (("model", "data"), ("m", "d"), ("mj_model", "mj_data"),
            ("mjmodel", "mjdata"))
CONTAINER_NAMES = ("sim", "mj", "env", "world", "physics", "mujoco")
BODY_KEYS = ("person", "npc", "walker", "human", "pedestrian", "bip01",
             "biped")
FACTORY_WORDS = ("person", "walker", "npc", "human", "pedestrian", "biped")

ROBOT_PARAMS = ("robot", "robot_pos", "robot_xy", "obstacle",
                "obstacle_xy", "obstacles", "blocker", "avoid", "hazard",
                "bystander", "other", "pedestrian_obstacle")
DT_PARAMS = ("dt", "time", "delta", "step_s", "timestep", "elapsed",
             "duration", "seconds")
INJECT_METHODS = ("set_obstacle", "add_obstacle", "register_obstacle",
                  "obstacle_at", "avoid", "blocked_by", "stop_for",
                  "set_robot", "robot_at", "halt_for", "present_obstacle",
                  "set_blocker", "add_robot")
INJECT_ATTRS = ("obstacle", "obstacles", "robot", "robot_pos", "robot_xy",
                "blocker", "avoid", "hazard", "stop_at", "avoid_xy")


def _mujoco():
    import mujoco
    return mujoco


def _try_call(fn, *args):
    """Call fn with as many leading args as it will accept."""
    for k in range(len(args), -1, -1):
        try:
            return fn(*args[:k])
        except TypeError:
            continue
        except Exception:
            return None
    return None


def _first_callable(obj, names):
    for n in names:
        f = getattr(obj, n, None)
        if callable(f):
            return f
    return None


def _md_pair(obj):
    try:
        mujoco = _mujoco()
    except Exception:
        return (None, None)
    boxes = [obj] + [getattr(obj, a, None) for a in CONTAINER_NAMES]
    for box in boxes:
        if box is None:
            continue
        for a, b in MD_PAIRS:
            m, d = getattr(box, a, None), getattr(box, b, None)
            if isinstance(m, mujoco.MjModel) and isinstance(d, mujoco.MjData):
                return (m, d)
    return (None, None)


def _person_pos(w):
    for n in POS_NAMES:
        if hasattr(w, n):
            v = getattr(w, n)
            if callable(v):
                v = _try_call(v)
            try:
                a = np.asarray(v, float).ravel()
            except Exception:
                continue
            if a.size >= 2 and np.all(np.isfinite(a[:2])):
                return np.array(a[:2], float)
    m, d = _md_pair(w)
    if m is not None:
        mujoco = _mujoco()
        for i in range(m.nbody):
            nm = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
                  or "").lower()
            if any(k in nm for k in BODY_KEYS):
                mid = int(m.body_mocapid[i])
                if mid >= 0:
                    return np.array(d.mocap_pos[mid][:2], float)
                return np.array(d.xpos[i][:2], float)
        for i in range(m.nbody):
            mid = int(m.body_mocapid[i])
            if mid >= 0:
                return np.array(d.mocap_pos[mid][:2], float)
        if m.nbody > 1:
            return np.array(d.xpos[1][:2], float)
    return None


def _time_of(w):
    for n in TIME_NAMES:
        if hasattr(w, n):
            v = getattr(w, n)
            if callable(v):
                v = _try_call(v)
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    m, d = _md_pair(w)
    if d is not None:
        return float(d.time)
    return None


def _advance(w, dt, robot=None):
    step = _first_callable(w, STEP_NAMES)
    if step is not None:
        try:
            params = [p for p in inspect.signature(step).parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY,
                                    p.POSITIONAL_OR_KEYWORD,
                                    p.KEYWORD_ONLY)]
        except (TypeError, ValueError):
            params = []
        if robot is not None and params:
            args, kwargs = [], {}
            for p in params:
                low = p.name.lower()
                if any(k in low for k in ROBOT_PARAMS):
                    val = np.asarray(robot, float)
                elif any(k in low for k in DT_PARAMS):
                    val = dt
                elif p.default is not p.empty:
                    continue
                else:
                    val = dt
                if p.kind == p.KEYWORD_ONLY:
                    kwargs[p.name] = val
                else:
                    args.append(val)
            try:
                step(*args, **kwargs)
                return
            except TypeError:
                pass
        _try_call(step, dt)
        return
    pose = _first_callable(w, POSE_NAMES)
    m, d = _md_pair(w)
    if pose is not None:
        t = _time_of(w)
        _try_call(pose, (0.0 if t is None else t) + dt)
        if m is not None:
            _mujoco().mj_forward(m, d)
        return
    if m is not None:
        n = max(1, int(round(dt / m.opt.timestep)))
        _mujoco().mj_step(m, d, n)


def _ctor_kwargs(cls, robot=None):
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return {}
    kw = {}
    for name, p in params.items():
        low = name.lower()
        required = p.default is p.empty and p.kind in (
            p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        if any(k in low for k in ("route", "waypoint", "path", "circuit")):
            try:
                from harmless import executor
                kw[name] = np.asarray(executor.PERSON_ROUTE, float)
            except Exception:
                if required:
                    return None
        elif "speed" in low or "velocity" in low:
            kw[name] = TARGET_SPEED
        elif any(k in low for k in ROBOT_PARAMS):
            if robot is None:
                if required:
                    return None
            else:
                kw[name] = np.asarray(robot, float)
        elif required:
            return None
    return kw


def _looks_like_walker(o):
    if o is None or inspect.ismodule(o) or inspect.isclass(o) or \
            inspect.isroutine(o):
        return False
    if isinstance(o, (str, bytes, int, float, bool, list, tuple, dict,
                      np.ndarray)):
        return False
    if _first_callable(o, STEP_NAMES) is not None:
        return True
    return _md_pair(o)[0] is not None


def _make_walker(mod, robot=None):
    for name in dir(mod):
        if name.startswith("_"):
            continue
        try:
            o = getattr(mod, name)
        except Exception:
            continue
        if _looks_like_walker(o):
            return o
    for _name, cls in inspect.getmembers(mod, inspect.isclass):
        kw = _ctor_kwargs(cls, robot)
        if kw is None:
            continue
        try:
            o = cls(**kw)
        except Exception:
            continue
        if _looks_like_walker(o):
            return o
    for name, fn in inspect.getmembers(mod):
        if name.startswith("_") or inspect.isclass(fn) or not callable(fn):
            continue
        if not any(wd in name.lower() for wd in FACTORY_WORDS):
            continue
        fkw = {}
        if robot is not None:
            try:
                for p in inspect.signature(fn).parameters.values():
                    if any(k in p.name.lower() for k in ROBOT_PARAMS):
                        fkw[p.name] = np.asarray(robot, float)
            except (TypeError, ValueError):
                pass
        for args in ((), (None,)):
            try:
                o = fn(*args, **fkw)
            except Exception:
                continue
            if _looks_like_walker(o):
                return o
    return None


def _step_has_robot_param(w):
    step = _first_callable(w, STEP_NAMES)
    if step is None:
        return False
    try:
        for p in inspect.signature(step).parameters.values():
            if any(k in p.name.lower() for k in ROBOT_PARAMS):
                return True
    except (TypeError, ValueError):
        pass
    return False


def _present_robot(mod, w, robot):
    """Tell the walker about the stationary robot at xy. Returns the list
    of channels that accepted it (non-empty when any exists)."""
    channels = []
    xy = np.asarray(robot, float)
    for n in INJECT_METHODS:
        f = getattr(w, n, None)
        if callable(f):
            try:
                f(xy)
                channels.append(f"method {n}")
            except TypeError:
                try:
                    f(float(xy[0]), float(xy[1]))
                    channels.append(f"method {n}")
                except Exception:
                    pass
            except Exception:
                pass
    for src, tag in ((w, "attr"), (mod, "module attr")):
        for n in INJECT_ATTRS:
            if not hasattr(src, n):
                continue
            cur = getattr(src, n)
            if callable(cur):
                continue
            try:
                if isinstance(cur, list):
                    cur.clear()
                    cur.append(xy.copy())
                elif isinstance(cur, np.ndarray) and cur.ndim == 2:
                    setattr(src, n, xy.reshape(1, 2))
                else:
                    setattr(src, n, xy.copy())
                channels.append(f"{tag} {n}")
            except Exception:
                pass
    if _step_has_robot_param(w):
        channels.append("step parameter")
    return channels


def _record(w, seconds, robot=None, dt=DT):
    n = max(1, int(round(seconds / dt)))
    pos = np.full((n, 2), np.nan)
    tim = np.full(n, np.nan)
    for i in range(n):
        _advance(w, dt, robot)
        p = _person_pos(w)
        if p is not None:
            pos[i] = p
        t = _time_of(w)
        if t is not None:
            tim[i] = t
    if np.isnan(tim).all() or (np.nanmax(tim) - np.nanmin(tim)) < 0.5 * seconds:
        tim = np.arange(n) * dt
    return tim, pos


def test_npc_2_2_person_stops_for_stationary_robot():
    try:
        mod = importlib.import_module("harmless.person")
    except Exception as e:
        pytest.fail(
            "harmless/person.py does not exist or does not import — the "
            f"walking-person module the line requires is missing ({e})")

    w = _make_walker(mod)
    if w is None:
        w = _make_walker(mod, robot=np.array([999.0, 999.0]))
    assert w is not None, (
        "harmless.person exposes no runnable walking person: expected a "
        "module-level instance, a zero-arg factory, or a constructible "
        "class offering step/update/advance/tick (or a MjModel+MjData "
        "pair) plus an observable horizontal position")

    p0 = _person_pos(w)
    assert p0 is not None, "person position is not observable"
    _tim, pos = _record(w, HEADING_S)
    p1 = pos[-1]
    d = p1 - p0
    assert np.linalg.norm(d) >= 0.25, (
        f"the person did not start walking ({np.linalg.norm(d):.2f} m in "
        f"{HEADING_S} s) — a parked person can never 'stop' for the robot")
    u = d / np.linalg.norm(d)
    robot = p1 + u * ROBOT_AHEAD_M

    channels = _present_robot(mod, w, robot)
    if not channels and not _step_has_robot_param(w):
        w2 = _make_walker(mod, robot=robot)
        assert w2 is not None, (
            "the walking person offers no channel to be told about a "
            "stationary robot ahead: no set_obstacle/avoid/robot_at-like "
            "method, no obstacle/robot/robot_pos attribute, no "
            "robot-like step parameter, no matching constructor keyword, "
            "and no module-level obstacle collection")
        w = w2
        _tim, pos = _record(w, HEADING_S)
        d = pos[-1] - pos[0]
        assert np.linalg.norm(d) >= 0.1, (
            "the person did not walk after construction with the robot "
            "in place")
        u = d / np.linalg.norm(d)

    tim, pos = _record(w, RUN_S, robot=robot)
    assert np.isfinite(pos).all(), (
        "the person's horizontal position was not observable during the "
        "run")

    dist = np.linalg.norm(pos - robot, axis=1)
    min_d = float(dist.min())
    assert min_d <= MIN_APPROACH_M, (
        f"the person never closed on the stationary robot (minimum "
        f"distance {min_d:.2f} m) — there was nothing to stop for")
    assert min_d >= MIN_CLEARANCE_M - 1e-6, (
        f"person-robot distance fell to {min_d:.3f} m, under the "
        f"{MIN_CLEARANCE_M} m the line holds it above")

    proj = (pos - robot) @ u
    assert float(proj.max()) <= MAX_PAST_M, (
        f"the person walked {float(proj.max()):.2f} m past the robot "
        "along its heading — it went through rather than stopping")

    late = pos[tim >= tim[-1] - STOP_WINDOW_S]
    moved = float(np.linalg.norm(late[-1] - late[0]))
    assert moved <= STOP_MOVE_M, (
        f"the person kept moving in the last {STOP_WINDOW_S} s "
        f"({moved:.2f} m) instead of stopping short of the robot")
