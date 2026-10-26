"""CTRL-2.2@1 A grasped bowl survives a 0.3 m carry at 0.2 m/s and
releases as a real drop under gravity.

The contract under test is the end-to-end hold a caller builds from the
public bridge: place the palms beside the bowl (hands_path), grasp()
it, carry it by walking at a commanded 0.2 m/s, then release() it —
every phase scored from measured simulation state.

The pregrasp is an aisle-side approach, revised after review found the
up-and-over route dragged forearms across the island's edge: the bowl
is staged on the island's south edge overhanging it slightly (a real
pickup spot), the robot stands square in the aisle facing it and leans
at the waist, and each palm rises at rim height while still in the
aisle, then sets down onto the bowl's rim from just above — fingers
over the lip, forearms never crossing the counter edge.

- the person is parked so she cannot walk into the trial
- from grasp() onward every physics step is recorded: the bowl's world
  position may never jump between steps (a between-palms snap is a
  teleport, not motion) and no weld equality involving the bowl may
  ever switch on — the carry must be a real contact hold, not a
  constraint
- the carry is the learned walker's job: walk(0, -0.2, 0) sidesteps the
  robot east along the open aisle until the bowl has travelled well
  past the 0.3 m the line requires AND past the island's east edge, so
  at release the bowl is over open floor; afterwards it must still be
  held — near a palm and far above the floor — and the robot upright
- release() must free it into a genuine drop: measured downward speed
  exceeding 0.8 m/s (free fall, not a lowered placement or a grip that
  never lets go) and the bowl coming to rest on the floor below

On a tree with the weld-snap grasp the test fails at the weld/jump
probes the moment grasp() returns; a grasp whose fingers never reach
the bowl fails the contact checks; a grip that cannot hold through the
carry fails the displacement/held checks; a release that sets the bowl
back on the counter or never lets go fails the free-fall and
rest-height checks.
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

# at dt=0.002 s a physical move is < 1 cm; a between-palms snap is cm-scale
JUMP_M = 0.025
ISLAND_EAST_EDGE = 0.51          # standardislandheight top slab, +x side
HAND_BODY = re.compile(r"^(left|right)_hand_(palm|thumb|index|middle)",
                       re.I)


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


def _place_palms(ex, name, half_span=0.055, timeout=9.0):
    """A caller-built pregrasp: both palms onto the rim from just above,
    fingers over the lip. The palms rise at rim height while still in
    the aisle (south of the worktop edge), then come down a few cm onto
    the rim — no forearm ever reaches across the counter edge."""
    info = ex.geom_info(name)
    lo, hi = np.array(info["aabb_min"]), np.array(info["aabb_max"])
    c = (lo + hi) / 2
    z = float(hi[2]) + 0.01                    # just above the rim
    tgts = {"left": np.r_[c[0] - half_span, c[1], z],
            "right": np.r_[c[0] + half_span, c[1], z]}
    for s, p in tgts.items():
        cur = _palm(ex, s)
        ex.arm_path([np.r_[cur[:2], z],                # up at the body
                     np.r_[p[0] * 1.6, c[1] - 0.20, z],  # forward, aisle air
                     p], side=s)                        # down onto the rim
        t0 = float(ex.data.time)
        while float(ex.data.time) - t0 < timeout:
            ex.run_seconds(0.1)
            st = ex.arm_status().get(s, {})
            if st.get("via_points_left") == 0 and \
                    st.get("palm_error_m", 9.0) < 0.04:
                break
    st = ex.arm_status()
    return {s: st.get(s, {}).get("palm_error_m", 9.0) for s in tgts}


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


def _hand_contacts(ex, name):
    """Measured contact between hand/finger links and the object's geoms:
    {(side, link): total normal force N} — read off MjData.contact, never
    a predicate from the code under test."""
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


def _tilt(ex):
    up = ex.data.xmat[ex.pelvis_body].reshape(3, 3)[2, 2]
    return float(np.degrees(np.arccos(np.clip(up, -1.0, 1.0))))


def test_ctrl_2_2_grasped_bowl_survives_carry_and_drops_free(tmp_path):
    mujoco = _mujoco()
    ex = _executor(tmp_path)
    m, d = ex.model, ex.data

    # --- stage the pickup: bowl on the island's south edge, robot stood
    # in the aisle square in front of it facing it, leaning in at the
    # waist (embodiment posture, not a reach)
    _park_person(ex)
    _teleport(ex, "bowl", [0.0, -0.93, 1.10])
    _teleport(ex, "pelvis", [0.0, -1.25, 0.793], yaw=math.pi / 2)
    ex.run_seconds(0.5)
    ex.set("waist_pitch_joint", 0.35)
    ex.run_seconds(1.0)

    errs = _place_palms(ex, "bowl")
    gaps = ex.palm_gaps("bowl")
    assert max(errs.values()) < 0.10 and max(gaps.values()) < 0.10, (
        f"pregrasp setup failed to place the palms at the bowl's rim "
        f"(palm errors {errs}, surface gaps {gaps}) — the trial cannot "
        "start")

    # --- grasp it, per-step watched: no snap, no weld
    log = _watch(ex, "bowl")
    res = ex.grasp("bowl")
    assert isinstance(res, dict) and "error" not in res, (
        f"grasp('bowl') failed from a valid rim pregrasp: {res}")
    ex.run_seconds(0.4)
    assert not log["prop_eq"], (
        f"grasp('bowl') activated equality weld(s) "
        f"{[_eq_name(ex, i) for i in sorted(log['prop_eq'])]} — the "
        "line requires the bowl held by contact forces only")
    assert log["jump"] < JUMP_M, (
        f"the bowl jumped {log['jump']:.3f} m inside one physics step "
        "during grasp() — a snap between the palms is a kinematic "
        "teleport, not finger motion")
    contacts = {k: v for k, v in _hand_contacts(ex, "bowl").items()
                if v > 0.05}
    assert contacts, (
        "grasp('bowl') reported success but no hand link touches the "
        "bowl — the hold must be measured contact")

    # --- the carry: sidestep east along the aisle at a commanded
    # 0.2 m/s until the bowl has travelled >= 0.3 m AND is clear of the
    # island's east edge, over open floor for the drop (bounded by gait
    # quality, capped at 10 s)
    bowl_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bowl")
    bowl_va = m.jnt_dofadr[m.body_jntadr[bowl_bid]]
    bowl0 = ex.pos("bowl").copy()
    pel0 = d.xpos[ex.pelvis_body].copy()
    z_held0 = float(bowl0[2])
    ex.walk(0.0, -0.2, 0.0)
    t0 = float(d.time)
    while float(d.time) - t0 < 10.0:
        ex.run_seconds(0.1)
        p = ex.pos("bowl")
        if p[0] > ISLAND_EAST_EDGE + 0.12 and \
                np.linalg.norm(p[:2] - bowl0[:2]) >= 0.30:
            break
    ex.walk(0.0, 0.0, 0.0)
    ex.run_seconds(0.6)

    bowl_disp = float(np.linalg.norm(ex.pos("bowl")[:2] - bowl0[:2]))
    pel_disp = float(np.linalg.norm(
        d.xpos[ex.pelvis_body][:2] - pel0[:2]))
    assert pel_disp >= 0.15, (
        f"the robot only moved {pel_disp:.3f} m while commanded to "
        "sidestep at 0.2 m/s — there was no carry to survive")
    assert bowl_disp >= 0.30, (
        f"the bowl moved only {bowl_disp:.3f} m horizontally during the "
        f"carry (robot moved {pel_disp:.3f} m) — it was left behind, "
        "not carried")

    z_held = float(ex.pos("bowl")[2])
    near = min(float(np.linalg.norm(ex.pos("bowl") - _palm(ex, s)))
               for s in ("right", "left"))
    assert z_held >= z_held0 - 0.25 and near <= 0.35, (
        f"the bowl did not survive the carry: z {z_held0:.3f} -> "
        f"{z_held:.3f} m, nearest palm {near:.3f} m away — it slipped "
        "or was dropped")
    assert _tilt(ex) < 30, (
        f"the robot fell over during the carry (tilt {_tilt(ex):.0f} "
        "deg) — the trial is void")
    assert not log["prop_eq"] and log["jump"] < JUMP_M, (
        f"the carry was not a real hold: welds "
        f"{[_eq_name(ex, i) for i in sorted(log['prop_eq'])]}, max "
        f"one-step jump {log['jump']:.3f} m")

    # --- release() must free the bowl into a real drop under gravity;
    # the carry left it over the open aisle east of the island
    assert float(ex.pos("bowl")[0]) > ISLAND_EAST_EDGE, (
        f"the carry ended with the bowl at x={ex.pos('bowl')[0]:.3f} — "
        "still over the counter, so a release could only set it down, "
        "not drop it")
    res = ex.release()
    assert not (isinstance(res, dict) and "error" in res), (
        f"release() failed while holding the bowl: {res}")
    min_vz, final_z = 0.0, float(ex.pos("bowl")[2])
    for _ in range(int(round(2.5 / 0.05))):
        ex.run_seconds(0.05)
        min_vz = min(min_vz, float(d.qvel[bowl_va + 2]))
        final_z = float(ex.pos("bowl")[2])
    assert min_vz <= -0.8, (
        f"after release() the bowl's fastest fall was {min_vz:.2f} m/s "
        "— no free fall happened; it was lowered, carried down, or "
        "never let go")
    assert final_z <= 0.35, (
        f"after release() the bowl came to rest at z={final_z:.3f} m "
        "above open floor — it is still held up (or re-welded), not "
        "dropped under gravity")
    assert not log["prop_eq"] and log["jump"] < JUMP_M, (
        f"the drop was not free: welds "
        f"{[_eq_name(ex, i) for i in sorted(log['prop_eq'])]}, max "
        f"one-step jump {log['jump']:.3f} m")
    ex.step = log["step"]
