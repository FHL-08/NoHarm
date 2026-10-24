"""CTRL-3.1@1 After each arm motion the executor re-reads the physical
palm position and re-solves IK from the measured state before reporting
failure.

The contract under test is the executor's arm-motion surface the way a
caller drives it — the public reach entries exposed on the executor and
through the program() bridge (arm_reach / arm_path for the palm sites,
reach / reach_xyz for the bounded EE reach that reports an outcome).

Two halves are checked on the real kitchen scene, all against measured
simulation state:

- The motion runs on measured state, not a stale plan. With the arm
  physically displaced to a different pose than any plan would assume
  (the test shoves the arm joints — a real configuration change, like a
  bump), a scripted arm_reach must still land the measured palm at the
  target, and arm_status() must report the live measured palm error —
  proving IK is re-solved from where the arm actually is.

- Failure may only be reported after that measured correction loop ran.
  From a staged stance at the south counter, reach_xyz is sent to a
  counter-top target the one-shot IK plan calls "unreachable" while the
  measured closed loop still drives the palm most of the way to it.
  The executor may report failure — but only after it has re-read the
  physical palm and re-solved IK from the measured state, i.e. the palm
  must actually have been driven toward the target under correction,
  and any residual the report quotes must be the measured distance,
  not a plan-side estimate. A refusal that fires before the arm moves,
  quoting a scratch-solver residual, fails this line: it reports
  failure from the plan, never having re-read the physical palm.

On the base tree the second leg fails: reach_xyz returns
"unreachable (residual ~0.2-0.3 m)" with the palm unmoved, so neither
the re-read nor the re-solve from measured state happened.
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
COUNTER = "standardcounterheightwidth"

# a counter-top target deep on the south counter slab: one-shot IK
# calls it unreachable (~0.2-0.3 m residual) while a measured
# correction loop still drives the palm most of the way toward it
COUNTER_TARGET = np.array([0.9, -2.5, 0.92])
# a comfortable in-workspace target for the measured-state probe
NEAR_TARGET = np.array([0.8, -0.95, 0.95])


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


def _palm_sid(ex, side="right"):
    mujoco = _mujoco()
    return mujoco.mj_name2id(ex.model, mujoco.mjtObj.mjOBJ_SITE,
                             f"{side}_palm")


def _palm(ex, side="right"):
    return np.asarray(
        ex.data.site_xpos[_palm_sid(ex, side)], dtype=float)


def _yaw(ex):
    w, x, y, z = np.asarray(ex.data.qpos[3:7], dtype=float)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _pelvis_y(ex):
    return float(ex.data.xpos[ex.pelvis_body][1])


def _stage_at_counter(ex):
    """Walk the robot to face the south counter — real physics through
    the public walk() velocity command, closed-loop on the measured
    pelvis/yaw. Ends facing the counter, pelvis ~0.4-0.5 m off the
    counter top's north edge."""
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
        if _pelvis_y(ex) <= -1.75:
            break
    ex.walk(0.0, 0.0, 0.0)
    ex.run_seconds(0.4)


def _run_until(ex, cond, timeout_s, poll_s=0.1):
    """Step the sim until cond() is true or timeout; returns the first
    sim-time cond held, or None."""
    dt = float(ex.model.opt.timestep)
    t0 = float(ex.data.time)
    while float(ex.data.time) - t0 < timeout_s:
        ex.step(max(1, int(round(poll_s / dt))))
        if cond():
            return float(ex.data.time) - t0
    return None


def _numbers(text):
    return [float(v) for v in
            re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", str(text))]


def test_ctrl_3_1_measured_palm_resolved_before_failure_report(tmp_path):
    ex = _executor(tmp_path)
    mujoco = _mujoco()
    m, d = ex.model, ex.data

    # ---- leg A: the reach loop re-solves IK from the measured state ---
    # displace the physical arm to a pose no plan assumed (a real bump):
    # an open-loop plan would land the palm ~10 cm off; a loop that
    # re-reads the measured palm and re-solves recovers it.
    shove = 0.25
    for jid in ex.arm_jids[:4]:
        d.qpos[m.jnt_qposadr[jid]] += shove
        d.qvel[m.jnt_dofadr[jid]] = 0.0
    mujoco.mj_forward(m, d)
    displaced = _palm(ex).copy()
    res = ex.program(
        "arm_reach(%.3f, %.3f, %.3f, side='right')" % tuple(NEAR_TARGET))
    assert "error" not in res, (
        f"arm_reach is not callable through the program() bridge: "
        f"{res.get('error')}")
    hit = _run_until(
        ex, lambda: np.linalg.norm(_palm(ex) - NEAR_TARGET) < 0.04, 6.0)
    measured = float(np.linalg.norm(_palm(ex) - NEAR_TARGET))
    assert hit is not None and measured < 0.06, (
        f"after a physical displacement of {displaced[0] - NEAR_TARGET[0]:+.2f} m "
        f"the palm settled {measured:.3f} m from the target — the reach "
        f"did not re-solve IK from the measured palm position")
    st = ex.arm_status().get("right", {})
    if "palm_error_m" in st:
        assert abs(float(st["palm_error_m"]) - measured) < 0.01, (
            f"arm_status reports palm_error_m={st['palm_error_m']} but "
            f"the measured palm error is {measured:.3f} m — the report "
            f"is not the re-read physical palm position")
    ex.arm_hold("right")

    # ---- leg B: failure is reported only after the measured loop ------
    _stage_at_counter(ex)
    assert _pelvis_y(ex) <= -1.70, (
        f"staging failed: pelvis y={_pelvis_y(ex):.2f}, could not reach "
        f"the counter front to pose the reach")
    d0 = float(np.linalg.norm(_palm(ex) - COUNTER_TARGET))
    res = ex.program(
        "reach_xyz(%.3f, %.3f, %.3f)" % tuple(COUNTER_TARGET))
    # let any post-report correction run briefly, then measure
    ex.run_seconds(0.5)
    d1 = float(np.linalg.norm(_palm(ex) - COUNTER_TARGET))

    # before any failure report, the palm must have been driven toward
    # the target under measured-state correction — a refusal that leaves
    # the palm unmoved reports the plan's verdict, not a measured outcome
    assert d1 < 0.05 or d1 <= d0 - 0.15, (
        f"reach_xyz reported {res.get('error', res)} while the measured "
        f"palm is still {d1:.3f} m from the target (moved "
        f"{d0 - d1:+.3f} m) — the executor reported failure without "
        f"re-reading the physical palm and re-solving IK from the "
        f"measured state")

    # and a reported residual must quantify the measured palm distance,
    # not a plan-side estimate
    err_text = res.get("error")
    if err_text:
        nums = [v for v in _numbers(err_text) if v < d0]
        if nums:
            assert any(abs(v - d1) < 0.03 for v in nums), (
                f"the failure report {err_text!r} quotes no residual "
                f"near the measured {d1:.3f} m — the executor reported "
                f"a plan-side estimate, not the re-read palm position")
