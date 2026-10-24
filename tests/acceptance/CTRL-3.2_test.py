"""CTRL-3.2@1 A scripted reach lands the palm within 5 cm of a
counter-top target in under 10 simulated seconds.

The contract under test is the executor's scripted reach capability the
way a caller drives it: after walking to the south counter under the
public walk() velocity command (real physics, measured staging), a reach
command issued to a point on the counter top must bring the measured
palm within 0.05 m of that target within 10 simulated seconds of being
issued.

Everything is checked against measured simulation state: the counter-top
target is verified to lie on standardcounterheightwidth's collision AABB
top; the palm position is read back from site_xpos after the reach; the
budget is data.time sim-seconds.

The reach is exercised through the bridge's reach entries:

- reach_xyz(x, y, z) — the bounded reach that reports an outcome: on the
  base tree it refuses the counter-top target with "unreachable
  (residual ~0.1 m)" after a one-shot IK plan, never moving the palm —
  so the measured palm lands ~0.67 m off, not within 5 cm. That is the
  failure this line closes: a scripted reach must land the palm, not a
  plan verdict.

- arm_reach(x, y, z, side) — the palm primitive must also land the
  measured palm within 5 cm inside the budget.

A reach layer that drives the arm, re-reads the physical palm and
re-solves from measured state lands this same target to ~1 cm.
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
COUNTER = "standardcounterheightwidth"

# a point on the south counter top (slab y -2.8..-2.17, top z ~0.9),
# ~13 cm in from its north edge — where a reach would touch down
TARGET = np.array([1.0, -2.3, 0.92])
REACH_TOL_M = 0.05
REACH_BUDGET_S = 10.0


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


def _palm(ex, side="right"):
    mujoco = _mujoco()
    sid = mujoco.mj_name2id(ex.model, mujoco.mjtObj.mjOBJ_SITE,
                            f"{side}_palm")
    return np.asarray(ex.data.site_xpos[sid], dtype=float)


def _yaw(ex):
    w, x, y, z = np.asarray(ex.data.qpos[3:7], dtype=float)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _aabb(ex, name):
    """World collision AABB of a named body subtree, measured."""
    mujoco = _mujoco()
    m, d = ex.model, ex.data
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    assert root >= 0, f"no body named {name!r} in the scene"
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        b = m.geom_bodyid[g]
        while b > 0 and b != root:
            b = m.body_parentid[b]
        if b != root:
            continue
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        R = d.geom_xmat[g].reshape(3, 3)
        wc = d.geom_xpos[g] + R @ c
        wh = np.abs(R) @ h
        lo, hi = np.minimum(lo, wc - wh), np.maximum(hi, wc + wh)
    return lo, hi


def _stage_at_counter(ex):
    """Walk the robot to face the south counter — real physics through
    the public walk() velocity command, closed-loop on the measured
    pelvis/yaw."""
    dt = float(ex.model.opt.timestep)
    ex.walk(0.0, 0.0, 0.6)
    t0 = float(ex.data.time)
    while float(ex.data.time) - t0 < 4.0:
        ex.step(int(round(0.05 / dt)))
        if abs(_yaw(ex) - (-math.pi / 2)) < 0.12:
            break
    ex.walk(0.25, 0.0, 0.0)
    t0 = float(ex.data.time)
    while float(ex.data.time) - t0 < 8.0:
        ex.step(int(round(0.05 / dt)))
        if float(ex.data.xpos[ex.pelvis_body][1]) <= -1.75:
            break
    ex.walk(0.0, 0.0, 0.0)
    ex.run_seconds(0.4)


def _best_within(ex, seconds):
    """Min measured palm->target distance over the next `seconds` of sim
    time, and the sim time it was first under REACH_TOL_M (or None)."""
    dt = float(ex.model.opt.timestep)
    t0 = float(ex.data.time)
    best, hit = np.inf, None
    while float(ex.data.time) - t0 < seconds:
        ex.step(max(1, int(round(0.1 / dt))))
        err = float(np.linalg.norm(_palm(ex) - TARGET))
        best = min(best, err)
        if err < REACH_TOL_M and hit is None:
            hit = float(ex.data.time) - t0
    return best, hit


def test_ctrl_3_2_scripted_reach_palm_within_5cm_under_10s(tmp_path):
    ex = _executor(tmp_path)
    mujoco = _mujoco()
    m, d = ex.model, ex.data

    # the target really is a counter-top point: on the counter's
    # collision AABB, inside its xy footprint, at its top surface
    lo, hi = _aabb(ex, COUNTER)
    assert lo[0] - 0.01 <= TARGET[0] <= hi[0] + 0.01 and \
        lo[1] - 0.01 <= TARGET[1] <= hi[1] + 0.01 and \
        0.0 <= TARGET[2] - hi[2] <= 0.06, (
            f"{TARGET.tolist()} is not on the {COUNTER} top "
            f"(AABB {np.round(lo, 2)}..{np.round(hi, 2)})")

    # staged at the counter under real locomotion
    _stage_at_counter(ex)
    pel_y = float(d.xpos[ex.pelvis_body][1])
    assert pel_y <= -1.70, (
        f"staging failed: pelvis y={pel_y:.2f}, the robot never "
        f"reached the counter front")
    staged_qpos = d.qpos.copy()
    staged_qvel = d.qvel.copy()
    staged_ctrl = d.ctrl.copy()

    # ---- scripted reach 1: the bounded reach that reports ------------
    t_issue = float(d.time)
    res = ex.program("reach_xyz(%.3f, %.3f, %.3f)" % tuple(TARGET))
    spent = float(d.time) - t_issue
    best, hit = _best_within(ex, max(0.0, REACH_BUDGET_S - spent))
    assert hit is not None or (
        spent < REACH_BUDGET_S and best < REACH_TOL_M), (
        f"the scripted reach to the counter-top target left the "
        f"measured palm {best:.3f} m away "
        f"(result {res.get('error', res)}) — the reach never landed "
        f"within {REACH_TOL_M:.2f} m in under {REACH_BUDGET_S:.0f} "
        f"simulated seconds")

    # ---- scripted reach 2: the palm primitive ------------------------
    # back to the staged posture with no arm motion in flight, so this
    # leg's reach is the only thing moving the palm
    ex.arm_hold()
    d.qpos[:] = staged_qpos
    d.qvel[:] = staged_qvel
    d.ctrl[:] = staged_ctrl
    mujoco.mj_forward(m, d)
    res = ex.program(
        "arm_reach(%.3f, %.3f, %.3f, side='right')" % tuple(TARGET))
    assert "error" not in res, (
        f"arm_reach is not callable through the program() bridge: "
        f"{res.get('error')}")
    best, hit = _best_within(ex, REACH_BUDGET_S)
    assert hit is not None, (
        f"arm_reach left the measured palm {best:.3f} m from the "
        f"counter-top target for the whole {REACH_BUDGET_S:.0f} s — "
        f"the scripted reach never landed within {REACH_TOL_M:.2f} m")
