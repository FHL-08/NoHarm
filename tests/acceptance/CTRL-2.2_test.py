"""CTRL-2.2@1 A grasped bowl survives a 0.3 m carry at 0.2 m/s and
releases as a real drop under gravity.

The contract under test is the end-to-end hold a caller builds from the
public bridge: place the palms beside the bowl (hands_path), grasp()
it, carry it 0.3 m by walking at a commanded 0.2 m/s, then release()
it — every phase scored from measured simulation state.

- the bowl is staged on the island's south edge with the robot stood
  square in front (the authored pickup spot geometry); the person is
  parked so she cannot walk into the trial
- from grasp() onward every physics step is recorded: the bowl's world
  position may never jump between steps (a between-palms snap is a
  teleport, not motion) and no weld equality involving the bowl may ever
  switch on — the carry must be a real contact hold, not a constraint
- the carry is the learned walker's job: walk(0, 0.2, 0) sidesteps the
  robot ~0.3 m along the open aisle while the bowl rides in the hands;
  afterwards the bowl must still be held — near a palm and far above
  the floor — and the robot still upright
- release() must free it into a genuine drop: measured downward speed
  exceeding 0.8 m/s (free fall, not a lowered placement or a grip that
  never lets go) and the bowl coming to rest on the floor below

On a tree with the weld-snap grasp the test fails at the weld/jump
probes the moment grasp() returns; a grasp that cannot hold through the
carry fails the displacement/held checks; a release that keeps the bowl
cradled or welded fails the free-fall and rest-height checks.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]
SCENE = REPO / "kitchen.xml"

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


def _place_palms(ex, name, standoff=0.04, timeout=6.0):
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


def _tilt(ex):
    up = ex.data.xmat[ex.pelvis_body].reshape(3, 3)[2, 2]
    return float(np.degrees(np.arccos(np.clip(up, -1.0, 1.0))))


def test_ctrl_2_2_grasped_bowl_survives_carry_and_drops_free(tmp_path):
    mujoco = _mujoco()
    ex = _executor(tmp_path)
    m, d = ex.model, ex.data

    # --- stage the pickup: bowl on the island's south edge, robot stood
    # in the aisle square in front of it facing it
    _park_person(ex)
    _teleport(ex, "bowl", [0.0, -0.93, 1.10])
    _teleport(ex, "pelvis", [0.0, -1.28, 0.793], yaw=math.pi / 2)
    ex.run_seconds(0.5)

    errs = _place_palms(ex, "bowl")
    assert max(errs.values()) < 0.10, (
        f"pregrasp setup failed to place the palms beside the bowl "
        f"(errors {errs}) — the trial cannot start")

    # --- grasp it, per-step watched: no snap, no weld
    log = _watch(ex, "bowl")
    res = ex.grasp("bowl")
    assert isinstance(res, dict) and "error" not in res, (
        f"grasp('bowl') failed from a valid pregrasp: {res}")
    ex.run_seconds(0.4)
    assert not log["prop_eq"], (
        f"grasp('bowl') activated equality weld(s) "
        f"{[_eq_name(ex, i) for i in sorted(log['prop_eq'])]} — the "
        "line requires the bowl held by contact forces only")
    assert log["jump"] < JUMP_M, (
        f"the bowl jumped {log['jump']:.3f} m inside one physics step "
        "during grasp() — a snap between the palms is a kinematic "
        "teleport, not finger motion")

    # --- the 0.3 m carry: sidestep at a commanded 0.2 m/s until the bowl
    # has travelled 0.3 m (bounded by gait quality, capped at 5 s)
    bowl_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bowl")
    bowl_va = m.jnt_dofadr[m.body_jntadr[bowl_bid]]
    bowl0 = ex.pos("bowl").copy()
    pel0 = d.xpos[ex.pelvis_body].copy()
    z_held0 = float(bowl0[2])
    ex.walk(0.0, 0.2, 0.0)
    t0 = float(d.time)
    while float(d.time) - t0 < 5.0:
        ex.run_seconds(0.1)
        if np.linalg.norm(ex.pos("bowl")[:2] - bowl0[:2]) >= 0.30:
            break
    ex.walk(0.0, 0.0, 0.0)
    ex.run_seconds(0.6)

    bowl_disp = float(np.linalg.norm(ex.pos("bowl")[:2] - bowl0[:2]))
    pel_disp = float(np.linalg.norm(
        d.xpos[ex.pelvis_body][:2] - pel0[:2]))
    assert pel_disp >= 0.15, (
        f"the robot only moved {pel_disp:.3f} m while commanded to "
        "sidestep at 0.2 m/s — there was no carry to survive")
    assert bowl_disp >= 0.20, (
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

    # --- release() must free the bowl into a real drop under gravity
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
