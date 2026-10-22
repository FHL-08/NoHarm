"""NPC-2.1@1 The person follows its aisle route at 0.25 m/s with a looping
walk clip moving legs and arms.

The contract under test is the walking-person module this requirement adds,
``harmless/person.py`` (named in the requirement's touches). It must expose
a runnable walking person: a module-level instance, a zero-argument
factory, or a class the test can construct. The object must let a caller
advance simulated time (a step/update/advance/tick-like method, or a bare
MjModel/MjData pair the test can step, or a pose_at(t)-style clip setter
on such a pair) and it must expose the person's horizontal position
(pos/position/xy attribute or method, or a person/npc/human/Bip01-named
body inside its MjData).

"Follows its aisle route" is measured against the route the module or the
walker declares (a ROUTE / PERSON_ROUTE / WAYPOINTS / PATH attribute, a
list of xy waypoints); when none is declared, the aisle route the scene
already defines — harmless.executor.PERSON_ROUTE, the rectangle around
the kitchen island — is the reference. The person's ground track must
stay inside a 0.7 m corridor of that polyline while it accumulates metres
of travel.

"At 0.25 m/s" is the median horizontal speed while advancing: a statue, a
parked walker, and the old 0.35 m/s glide all fail.

"A looping walk clip moving legs and arms" means at least two leg joints
(thigh/calf/knee/ankle/shin/foot/toe/leg names; the Bip01 'Footsteps'
helper is excluded) and at least two arm joints (arm/clavicle/shoulder/
elbow/wrist/hand/finger names) each swing while the person advances, and
the dominant leg joint repeats itself — a normalised autocorrelation
peak at a nonzero gait lag — rather than holding one static pose or
playing a single swing once.

Nothing about this exists yet: there is no harmless/person.py, and the
person currently in the kitchen scene is a rigid body that glides at
0.35 m/s in a fixed pose. The expected pre-implementation failure is the
missing module.
"""

import importlib
import inspect
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[2]

TARGET_SPEED = 0.25
SPEED_TOL = 0.06
MEAN_SPEED_MIN = 0.15
CORRIDOR_M = 0.7
ROUTE_FRACTION = 0.80
MIN_TRAVEL_M = 3.0
RUN_S = 24.0
DT = 0.02
HINGE_MOVE_RAD = 0.07        # ~4 deg
QUAT_MOVE = 0.04             # per-component swing on a ball/quaternion joint
PERIOD_MIN_S = 0.3
PERIOD_MAX_S = 6.0
AUTOCORR_MIN = 0.45
MIN_ZERO_CROSSINGS = 4

LEG_KEYS = ("thigh", "calf", "knee", "ankle", "shin", "foot", "toe", "leg")
ARM_KEYS = ("arm", "clavicle", "shoulder", "elbow", "wrist", "hand",
            "finger", "thumb")
JOINT_EXCLUDE = ("footstep",)

STEP_NAMES = ("step", "update", "advance", "tick", "simulate", "run",
              "forward", "advance_time")
POSE_NAMES = ("pose_at", "apply_clip", "set_clip", "clip_at", "set_time",
              "animate", "play", "set_pose")
POS_NAMES = ("pos", "position", "xy", "location", "pos_xy", "get_pos",
             "get_position", "where", "root_pos", "base_pos")
TIME_NAMES = ("t", "time", "sim_time", "t_s", "now", "clock")
ROUTE_NAMES = ("ROUTE", "PERSON_ROUTE", "WAYPOINTS", "PATH", "AISLE_ROUTE",
               "route", "waypoints", "path", "circuit", "points",
               "route_xy", "walk_route")
MD_PAIRS = (("model", "data"), ("m", "d"), ("mj_model", "mj_data"),
            ("mjmodel", "mjdata"))
CONTAINER_NAMES = ("sim", "mj", "env", "world", "physics", "mujoco")
BODY_KEYS = ("person", "npc", "walker", "human", "pedestrian", "bip01",
             "biped")
FACTORY_WORDS = ("person", "walker", "npc", "human", "pedestrian", "biped")


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


def _advance(w, dt):
    step = _first_callable(w, STEP_NAMES)
    if step is not None:
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


def _joint_snapshot(w):
    """{leg|arm: {joint_name: qpos slice}} for one instant."""
    out = {"leg": {}, "arm": {}}
    m, d = _md_pair(w)
    if m is not None:
        mujoco = _mujoco()
        nq = {int(mujoco.mjtJoint.mjJNT_BALL): 4,
              int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
              int(mujoco.mjtJoint.mjJNT_HINGE): 1}
        for j in range(m.njnt):
            jt = int(m.jnt_type[j])
            if jt == int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            nm = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                  or "").lower()
            if any(x in nm for x in JOINT_EXCLUDE):
                continue
            grp = ("leg" if any(k in nm for k in LEG_KEYS)
                   else "arm" if any(k in nm for k in ARM_KEYS)
                   else None)
            if grp is None:
                continue
            adr = int(m.jnt_qposadr[j])
            out[grp][nm] = np.array(d.qpos[adr:adr + nq[jt]], float)
        return out
    for n in ("joint_angles", "angles", "joint_state", "joints",
              "pose_map", "qpos_map"):
        if hasattr(w, n):
            v = getattr(w, n)
            if callable(v):
                v = _try_call(v)
            if isinstance(v, Mapping):
                for k, x in v.items():
                    low = str(k).lower()
                    if any(e in low for e in JOINT_EXCLUDE):
                        continue
                    grp = ("leg" if any(q in low for q in LEG_KEYS)
                           else "arm" if any(q in low for q in ARM_KEYS)
                           else None)
                    if grp is None:
                        continue
                    try:
                        out[grp][str(k)] = np.atleast_1d(
                            np.asarray(x, float))
                    except Exception:
                        pass
                if out["leg"] or out["arm"]:
                    return out
    return out


def _fallback_route():
    try:
        from harmless import executor
        return np.asarray(executor.PERSON_ROUTE, float)
    except Exception:
        return None


def _declared_route(mod, w):
    """(route, closed): the walker's declared route, else the executor's
    PERSON_ROUTE — a known closed circuit around the island."""
    for src in (w, mod):
        for n in ROUTE_NAMES:
            if hasattr(src, n):
                v = getattr(src, n)
                if callable(v):
                    v = _try_call(v)
                try:
                    a = np.asarray(v, float)
                except Exception:
                    continue
                if a.ndim == 2 and a.shape[0] >= 2 and a.shape[1] >= 2:
                    r = a[:, :2]
                    return r, bool(np.linalg.norm(r[0] - r[-1]) < 0.3)
    r = _fallback_route()
    return (r, True) if r is not None else (None, False)


def _ctor_kwargs(cls, robot=None):
    """Constructor kwargs the test can fill from parameter names; None when
    a required parameter cannot be filled."""
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
            kw[name] = _fallback_route()
        elif "speed" in low or "velocity" in low:
            kw[name] = TARGET_SPEED
        elif any(k in low for k in ("obstacle", "robot", "block", "avoid")):
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
    if _md_pair(o)[0] is not None:
        return True
    return False


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
        for args in ((), (None,)):
            try:
                o = fn(*args)
            except Exception:
                continue
            if _looks_like_walker(o):
                return o
    return None


def _record(w, seconds, dt=DT):
    n = max(1, int(round(seconds / dt)))
    pos = np.full((n, 2), np.nan)
    tim = np.full(n, np.nan)
    joints = {"leg": {}, "arm": {}}
    have_joints = False
    for i in range(n):
        _advance(w, dt)
        p = _person_pos(w)
        if p is not None:
            pos[i] = p
        t = _time_of(w)
        if t is not None:
            tim[i] = t
        snap = _joint_snapshot(w)
        for grp in ("leg", "arm"):
            for k, v in snap[grp].items():
                joints[grp].setdefault(k, []).append(v)
                have_joints = True
    if np.isnan(tim).all() or (np.nanmax(tim) - np.nanmin(tim)) < 0.5 * seconds:
        tim = np.arange(n) * dt
    return tim, pos, joints, have_joints


def _polyline_dist(pts, poly, closed):
    segs = [(poly[i], poly[i + 1]) for i in range(len(poly) - 1)]
    if closed:
        segs.append((poly[-1], poly[0]))
    out = np.empty(len(pts))
    for i, p in enumerate(pts):
        best = np.inf
        for a, b in segs:
            ab = b - a
            den = float(np.dot(ab, ab))
            t = 0.0 if den == 0 else float(
                np.clip(np.dot(p - a, ab) / den, 0.0, 1.0))
            best = min(best, float(np.linalg.norm(p - (a + t * ab))))
        out[i] = best
    return out


def _moving(series_by_name, scalar_rad, per_comp):
    """Joints whose angle swings while walking; name -> strongest series."""
    moved = {}
    for name, vals in series_by_name.items():
        a = np.asarray(vals, float)
        if a.ndim == 1:
            a = a[:, None]
        rng = a.max(axis=0) - a.min(axis=0)
        if a.shape[1] == 1:
            if rng[0] >= scalar_rad:
                moved[name] = a[:, 0]
        else:
            k = int(np.argmax(rng))
            if rng[k] >= per_comp:
                moved[name] = a[:, k]
    return moved


def _autocorr_peak(s, dt):
    s = np.asarray(s, float).ravel()
    if s.size < 8 or s.std() < 1e-9:
        return 0.0, None
    x = np.arange(s.size)
    s = s - np.polyval(np.polyfit(x, s, 1), x)
    s = s - s.mean()
    den = float(np.dot(s, s)) + 1e-12
    ac = np.correlate(s, s, "full")[s.size - 1:] / den
    lo = max(1, int(round(PERIOD_MIN_S / dt)))
    hi = min(s.size - 1, int(round(PERIOD_MAX_S / dt)))
    if hi <= lo:
        return 0.0, None
    lag = lo + int(np.argmax(ac[lo:hi]))
    return float(ac[lag]), lag * dt


def test_npc_2_1_person_follows_aisle_route_with_looping_walk_clip():
    try:
        mod = importlib.import_module("harmless.person")
    except Exception as e:
        pytest.fail(
            "harmless/person.py does not exist or does not import — the "
            f"walking-person module the line requires is missing ({e})")

    w = _make_walker(mod)
    assert w is not None, (
        "harmless.person exposes no runnable walking person: expected a "
        "module-level instance, a zero-arg factory, or a constructible "
        "class offering step/update/advance/tick (or a MjModel+MjData "
        "pair) plus an observable horizontal position")

    route, closed = _declared_route(mod, w)
    assert route is not None and len(route) >= 2, (
        "no aisle route found: harmless.person declares no "
        "ROUTE/WAYPOINTS/PATH waypoints and the executor's PERSON_ROUTE "
        "could not be loaded as a reference")

    tim, pos, joints, have_joints = _record(w, RUN_S)
    assert np.isfinite(pos).all(), (
        "the person's horizontal position was not observable while it "
        "walked (no pos/position/xy accessor and no person-named body in "
        "its MjData)")

    dist = _polyline_dist(pos, route, closed)
    on = dist <= CORRIDOR_M
    assert on.mean() >= ROUTE_FRACTION, (
        f"person left its aisle route: only {on.mean():.0%} of sampled "
        f"positions lie within {CORRIDOR_M} m of the route polyline "
        f"(median off-route distance {np.median(dist):.2f} m)")

    step_len = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    travel = float(step_len.sum())
    assert travel >= MIN_TRAVEL_M, (
        f"person covered only {travel:.2f} m in {RUN_S:.0f} s — it is not "
        "following the route")

    dts = np.diff(tim)
    valid = dts > 1e-9
    assert valid.any(), "simulated time did not advance while stepping"
    speeds = step_len[valid] / dts[valid]
    med = float(np.median(speeds))
    mean = float(np.mean(speeds))
    assert mean >= MEAN_SPEED_MIN, (
        f"person is effectively parked: mean speed {mean:.3f} m/s")
    assert abs(med - TARGET_SPEED) <= SPEED_TOL, (
        f"person walks at {med:.3f} m/s (median), not "
        f"{TARGET_SPEED} m/s ± {SPEED_TOL}")

    assert have_joints, (
        "the walking person exposes no leg/arm joint state: no MuJoCo "
        "joints named for legs or arms, and no joint_angles/angles/pose "
        "mapping — a looping walk clip moving legs and arms cannot be "
        "seen at all")

    dt_eff = float(np.median(dts[valid])) if valid.any() else DT
    for grp, label in (("leg", "leg"), ("arm", "arm")):
        scalar = HINGE_MOVE_RAD
        moved = _moving(joints[grp], scalar, QUAT_MOVE)
        assert len(moved) >= 2, (
            f"fewer than two {label} joints move while the person walks "
            f"(moving: {sorted(moved) or 'none'}; seen: "
            f"{sorted(joints[grp]) or 'none'}) — the walk clip is not "
            f"moving its {label}s")

    legs = _moving(joints["leg"], HINGE_MOVE_RAD, QUAT_MOVE)
    name = max(legs, key=lambda k: float(np.ptp(legs[k])))
    s = legs[name] - legs[name].mean()
    zc = int(np.sum(s[:-1] * s[1:] < 0))
    peak, period = _autocorr_peak(legs[name], dt_eff)
    assert zc >= MIN_ZERO_CROSSINGS and peak >= AUTOCORR_MIN, (
        f"leg joint '{name}' does not loop: {zc} oscillations and "
        f"autocorrelation peak {peak:.2f}"
        + (f" at {period:.2f}s lag" if period else "")
        + " — expected a repeating walk-clip cycle, not a static pose or "
        "a single swing")
