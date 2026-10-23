"""CTRL-2.1@1 grasp(obj) drives the G1's three fingers onto the object
and holds it by contact forces only, with no equality weld activated.

The contract under test is the executor's public grasp(name) entry
point — the same call the program() bridge exposes — closing the G1's
dexterous three-fingered hand (the {side}_hand_{thumb,index,middle}_N
joint chains of the Unitree dex hand, each driven by its own actuator)
onto an authored prop and keeping it held by measured contact forces
alone. The weld_<prop>/weldL_<prop> equalities in kitchen.xml are
vestigial scene baggage: they must stay inactive for the whole grasp.

Everything is checked against measured state, never self-report:

- the compiled scene model must contain three actuated finger chains on
  at least one hand — the bundled g1.xml rubber-hand robot has none, so
  a scene without dexterous fingers fails here first (the expected
  pre-implementation failure)
- grasp("apple") is exercised from the repo's established pregrasp a
  caller builds with public motion: robot stood before the prop, both
  palms brought to its opposite sides a few cm off the surface
- the object's world position is recorded every physics step: a snap
  between the palms is a kinematic teleport and shows as a single-step
  jump far beyond anything contact physics can produce
- every equality constraint involving the object body is sampled per
  step: none may be active at any point during or after the grasp
- after grasp() returns, each of the three finger groups on some hand
  must have flexed measurably, and at least two distinct finger links
  must be in measured contact with the object's collision geoms
- the hold is proven physically: both palms lift 8 cm and the object
  must rise with the hand while nothing but contact can carry it —
  still touching fingers, still unwelded

On a tree with the weld-snap grasp the test fails at the finger-chain
assertion; a grasp that teleports or welds the object fails the
per-step jump/equality probes; a grip that never closes or never
touches fails the flexion and contact probes; a prop merely resting on
the counter fails the lift.
"""

import math
import os
import re
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]
SCENE = REPO / "kitchen.xml"

FINGER_JNT = re.compile(r"^(left|right)_hand_(thumb|index|middle)_\d+",
                        re.I)
HAND_BODY = re.compile(r"^(left|right)_hand_(palm|thumb|index|middle)",
                       re.I)
FINGERS = ("thumb", "index", "middle")

# at dt=0.002 s a physical move is < 1 cm; a between-palms snap is cm-scale
JUMP_M = 0.025


def _mujoco():
    import mujoco
    return mujoco


def _executor(tmp_path):
    try:
        from harmless.executor import Executor
    except Exception as exc:
        pytest.fail(
            f"harmless.executor cannot be imported "
            f"({type(exc).__name__}: {exc})")
    try:
        return Executor(str(SCENE), record=False,
                        out_dir=str(tmp_path / "episode"),
                        skills_path=None,
                        ws_dir=str(tmp_path / "ws"))
    except Exception as exc:
        pytest.fail(
            f"the kitchen scene failed to load in the Executor "
            f"({type(exc).__name__}: {exc})")


def _teleport(ex, name, xyz, yaw=None):
    """Arrange the trial's initial condition: place a free body at rest.
    Test setup only — every later motion is the physics engine's."""
    mujoco = _mujoco()
    m, d = ex.model, ex.data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    jnt = m.body_jntadr[bid]
    qa, va = m.jnt_qposadr[jnt], m.jnt_dofadr[jnt]
    d.qpos[qa:qa + 3] = xyz
    if yaw is not None:
        d.qpos[qa + 3:qa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    d.qvel[va:va + 6] = 0
    mujoco.mj_forward(m, d)


def _park_person(ex):
    """Hold the person on her tether at the route start so she cannot
    walk into the measurement area mid-trial (determinism, not policy)."""
    if hasattr(ex, "_fall"):
        ex._fall = (float(ex.data.time), np.array([1.0, 0.0, 0.0]))


def _palm(ex, side):
    return np.asarray(ex.pos(f"{side}_palm"), float)


def _place_palms(ex, name, standoff=0.035, timeout=6.0):
    """A caller-built pregrasp: both palms to the object's left and right
    sides a few cm off the skin, driven through the public hands_path
    primitive (up and over the worktop edge, then down onto the sides)."""
    info = ex.geom_info(name)
    lo, hi = np.array(info["aabb_min"]), np.array(info["aabb_max"])
    c = (lo + hi) / 2
    R = ex.data.xmat[ex.pelvis_body].reshape(3, 3)
    left = R[:, 1].copy()
    left[2] = 0.0
    left /= max(np.linalg.norm(left), 1e-9)
    half = float(np.abs(left[:2]) @ ((hi - lo)[:2] / 2))
    z = float(hi[2]) - 0.03
    off = half + standoff
    lift = z + 0.10
    tgts = {"left": np.r_[c[:2] + left[:2] * off, z],
            "right": np.r_[c[:2] - left[:2] * off, z]}
    paths = {}
    for s, p in tgts.items():
        cur = _palm(ex, s)
        paths[s] = [np.r_[cur[:2], max(lift, cur[2])],
                    np.r_[p[:2], lift], p]
    ex.hands_path(paths["right"], paths["left"])
    t0 = float(ex.data.time)
    errs = {"right": 9.0, "left": 9.0}
    while float(ex.data.time) - t0 < timeout:
        ex.run_seconds(0.1)
        st = ex.arm_status()
        errs = {s: st.get(s, {}).get("palm_error_m", 9.0) for s in tgts}
        if all(st.get(s, {}).get("via_points_left") == 0 and e < 0.03
               for s, e in errs.items()):
            break
    return errs


def _watch(ex, name):
    """Wrap ex.step: record, per physics step, the object's world position
    and the activation state of every equality constraint that involves
    its body. Returns the log dict; detach with ex.step = log['step']."""
    mujoco = _mujoco()
    m, d = ex.model, ex.data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    prop_eqs = [i for i in range(m.neq)
                if int(m.eq_objtype[i]) == int(mujoco.mjtObj.mjOBJ_BODY)
                and bid in (int(m.eq_obj1id[i]), int(m.eq_obj2id[i]))]
    log = {"step": ex.step, "jump": 0.0, "prop_eq": set(),
           "last": d.xpos[bid].copy(), "eq_ids": prop_eqs}
    orig = ex.step

    def step(n=1):
        orig(n)
        p = d.xpos[bid]
        log["jump"] = max(log["jump"],
                          float(np.linalg.norm(p - log["last"])))
        log["last"] = p.copy()
        for i in prop_eqs:
            if d.eq_active[i]:
                log["prop_eq"].add(i)

    ex.step = step
    return log


def _eq_name(ex, i):
    mujoco = _mujoco()
    return mujoco.mj_id2name(ex.model, mujoco.mjtObj.mjOBJ_EQUALITY, i) \
        or f"eq{i}"


def _finger_joints(ex):
    """{(side, finger): [joint ids]} for the dexterous hand chains."""
    mujoco = _mujoco()
    out = {}
    for j in range(ex.model.njnt):
        nm = mujoco.mj_id2name(ex.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        mm = FINGER_JNT.match(nm)
        if mm:
            out.setdefault((mm.group(1).lower(), mm.group(2).lower()),
                           []).append(j)
    return out


def _finger_q(ex, fingers):
    return {(s, f): np.array([ex.data.qpos[ex.model.jnt_qposadr[j]]
                              for j in js])
            for (s, f), js in fingers.items()}


def _hand_contacts(ex, name):
    """Measured contact between hand/finger links and the object's geoms:
    {(side, link): total normal force N}."""
    mujoco = _mujoco()
    m, d = ex.model, ex.data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    hand_of = {}
    for b in range(m.nbody):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        mm = HAND_BODY.match(nm)
        if mm:
            hand_of[b] = (mm.group(1).lower(), mm.group(2).lower())
    obj_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == bid}
    f6 = np.zeros(6)
    hits = {}
    for c in range(d.ncon):
        con = d.contact[c]
        for g_h, g_o in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
            bh = int(m.geom_bodyid[g_h])
            if bh not in hand_of or int(g_o) not in obj_geoms:
                continue
            mujoco.mj_contactForce(m, d, c, f6)
            k = hand_of[bh]
            hits[k] = hits.get(k, 0.0) + abs(float(f6[0]))
    return hits


def test_ctrl_2_1_grasp_drives_three_fingers_contact_hold_no_weld(
        tmp_path):
    ex = _executor(tmp_path)
    m, d = ex.model, ex.data

    # --- the robot in the scene must actually carry the dexterous hands:
    # three finger chains (thumb/index/middle), each actuator-driven, on
    # at least one hand
    fingers = _finger_joints(ex)
    sides = sorted({s for s, _ in fingers})
    three = [s for s in sides
             if all((s, f) in fingers for f in FINGERS)]
    assert three, (
        "the compiled scene has no three-fingered G1 hand — no "
        "{side}_hand_{thumb,index,middle}_* joints exist, so grasp() has "
        f"no fingers to drive onto the object (saw hand joints: "
        f"{sorted(fingers) or 'none'})")
    driven = [s for s in three
              if all(any(ex.ctrl_for_jid.get(j) is not None for j in
                         fingers[s, f]) for f in FINGERS)]
    assert driven, (
        f"the finger chains on {three} are present but have no actuator "
        "in the scene — grasp() cannot drive them closed")

    # --- grasp is a program() bridge verb
    res = ex.program("grasp")
    assert "error" not in res, (
        f"grasp is not bound in the program() bridge: {res.get('error')}")

    # --- arrange the trial: apple on the island's south edge, robot
    # stood square in front of it facing it (setup state, then physics
    # owns everything)
    _park_person(ex)
    _teleport(ex, "apple", [0.15, -0.95, 1.10])
    _teleport(ex, "pelvis", [0.15, -1.30, 0.793], yaw=math.pi / 2)
    ex.run_seconds(0.5)

    errs = _place_palms(ex, "apple")
    assert max(errs.values()) < 0.10, (
        f"pregrasp setup failed to place the palms beside the apple "
        f"(errors {errs}) — the trial cannot start")

    # --- the call under test, with per-step instrumentation
    q0 = _finger_q(ex, fingers)
    log = _watch(ex, "apple")
    inner = ex.step
    qmin = {k: v.copy() for k, v in q0.items()}
    qmax = {k: v.copy() for k, v in q0.items()}

    def step(n=1):
        inner(n)
        for k, js in fingers.items():
            q = np.array([d.qpos[m.jnt_qposadr[j]] for j in js])
            qmin[k] = np.minimum(qmin[k], q)
            qmax[k] = np.maximum(qmax[k], q)

    ex.step = step
    res = ex.grasp("apple")
    assert isinstance(res, dict) and "error" not in res, (
        f"grasp('apple') failed from a valid pregrasp: {res}")
    ex.run_seconds(0.4)   # let the hold settle under physics

    # --- no snap: the apple never teleported mid-grasp
    assert log["jump"] < JUMP_M, (
        f"the apple jumped {log['jump']:.3f} m inside one physics step "
        "during grasp() — a snap between the palms is a kinematic "
        "teleport, not finger motion")

    # --- no weld: no equality constraint on the apple may ever activate
    assert not log["prop_eq"], (
        f"grasp() activated equality weld(s) "
        f"{[_eq_name(ex, i) for i in sorted(log['prop_eq'])]} on the "
        "apple — the line requires contact forces only")
    for i in log["eq_ids"]:
        assert not d.eq_active[i], (
            f"equality {_eq_name(ex, i)!r} is still active after "
            "grasp() — the hold is welded, not contact")

    # --- the three fingers were really driven onto the apple: each
    # group on some hand moved measurably during the grasp (mid-motion
    # excursion, so an open-then-return wiggle does not count)
    moved = {k: float(np.maximum(qmax[k] - q0[k], q0[k] - qmin[k]).max())
             for k in fingers}
    per_finger = {f"{k[0]}_{k[1]}": round(v, 3) for k, v in moved.items()}
    assert any(all(moved.get((s, f), 0.0) >= 0.08 for f in FINGERS)
               for s in driven), (
        f"grasp() did not drive all three fingers of either hand onto "
        f"the apple (max finger excursion per finger: {per_finger})")

    # --- the hold is measured contact: >= two distinct finger links
    # touching the apple with real normal force
    hits = _hand_contacts(ex, "apple")
    f_hits = {k: v for k, v in hits.items()
              if k[1] in FINGERS and v > 0.05}
    f_report = {f"{k[0]}_{k[1]}": round(v, 2) for k, v in f_hits.items()}
    assert len({k[1] for k in f_hits}) >= 2 and \
        sum(f_hits.values()) >= 0.5, (
            f"finger contact on the apple is {f_report or 'none'} — "
            "three-fingered contact, not a weld, must hold it")

    # --- prove the hold physically: lift both palms 8 cm; the apple
    # must come up off the worktop carried by contact alone
    z0 = float(ex.pos("apple")[2])
    for s in ("right", "left"):
        p = _palm(ex, s)
        ex.arm_reach(p[0], p[1], p[2] + 0.08, side=s)
    t0 = float(d.time)
    while float(d.time) - t0 < 4.0:
        ex.run_seconds(0.1)
        st = ex.arm_status()
        if all(st.get(s, {}).get("palm_error_m", 9) < 0.04
               for s in ("right", "left")):
            break
    ex.run_seconds(0.4)
    z1 = float(ex.pos("apple")[2])
    assert z1 >= z0 + 0.025, (
        f"the palms rose 8 cm but the apple moved {z1 - z0:+.3f} m — "
        "it was resting on the counter, not held by the fingers")
    near = min(float(np.linalg.norm(ex.pos("apple") - _palm(ex, s)))
               for s in ("right", "left"))
    assert near < 0.30, (
        f"the lifted apple sits {near:.3f} m from the nearest palm — "
        "it is not in the hand")
    air_hits = _hand_contacts(ex, "apple")
    assert any(k[1] in FINGERS and v > 0.05
               for k, v in air_hits.items()), (
        "the airborne apple touches no finger link — whatever holds it "
        "is not finger contact")
    assert not log["prop_eq"], (
        f"an equality weld activated late: "
        f"{[_eq_name(ex, i) for i in sorted(log['prop_eq'])]}")
    assert log["jump"] < JUMP_M, (
        f"the apple teleported {log['jump']:.3f} m in one step while "
        "being lifted")
    ex.step = log["step"]
