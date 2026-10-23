"""CTRL-1.1@1 walk(vx, vy, yaw_rate) commands the Unitree pretrained RL
walking policy producing PD torques at 50 Hz.

The contract under test is the executor's public walk(vx, vy, yaw_rate)
entry point — the same call the program() bridge exposes — commanding
the pretrained Unitree G1 walking policy: the TorchScript artifact at
assets/policies/g1_walk_unitree_rl_gym.pt (unitreerobotics/
unitree_rl_gym deploy policy, 47-d observation -> 12 leg-joint actions).
The policy's leg position targets must be tracked by PD servos so the
gait is produced by real joint torques, and policy inference must run
on a 50 Hz cadence against the sim timestep.

Everything is checked against measured state, never self-report:

- the artifact loads as TorchScript and keeps the deployed 47 -> 12 I/O
- the object the executor engages for walk() runs a TorchScript module
  fingerprint-identical to that artifact (a hand-rolled gait script or a
  different network fails here)
- while walking, the leg actuators' position targets in MjData.ctrl only
  change on a uniform cadence whose period is ~0.02 s (50 Hz)
- each leg actuator shows PD behaviour on measured data: displacing a
  joint produces a restoring torque (stiffness) and joint velocity
  produces an opposing torque (damping) in MjData.qfrc_actuator, and
  those torques are large while the gait runs
- the robot really locomotes: commanding vx translates the pelvis free
  body along its own forward axis while it stays upright, and yaw_rate
  rotates it — motion through contact physics, not a gliding base

On a tree without velocity-commanded learned locomotion the test fails
at the walk()/walker assertions; a kinematic base mover fails the
torque, PD and cadence probes; a differently-cadenced controller fails
the 50 Hz check.
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
POLICY_PATH = (REPO / "assets" / "policies" /
               "g1_walk_unitree_rl_gym.pt")

OBS_DIM = 47          # unitree_rl_gym g1 deploy obs: omega(3) + gravity(3)
                      # + cmd(3) + q(12) + dq(12) + action(12) + phase(2)
ACT_DIM = 12          # 12 leg joints (hip/ knee/ ankle x2)
LEG_NAME = re.compile(r"hip|knee|ankle", re.I)


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


def _walker_of(ex):
    """The low-level gait object walk() engages — wherever the executor
    keeps it (implementation may live under harmless/control/)."""
    w = getattr(ex, "walker", None)
    if w is not None:
        return w
    try:
        vals = list(vars(ex).values())
    except TypeError:
        vals = [getattr(ex, n) for n in dir(ex)]
    for v in vals:
        if hasattr(v, "tick") and hasattr(v, "on") and \
                (hasattr(v, "set_cmd") or hasattr(v, "cmd")):
            return v
    return None


def _leg_actuators(model):
    """(actuator index, joint id, dof adr, qpos adr) for the 12 leg
    servos — named hip/knee/ankle joints."""
    mujoco = _mujoco()
    out = []
    for a in range(int(model.nu)):
        jid = int(model.actuator_trnid[a, 0])
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if LEG_NAME.search(name):
            out.append((a, jid, int(model.jnt_dofadr[jid]),
                        int(model.jnt_qposadr[jid])))
    return out


def _jit_modules(obj, torch):
    mods = []
    for n in dir(obj):
        try:
            v = getattr(obj, n)
        except Exception:
            continue
        if isinstance(v, torch.jit.ScriptModule):
            mods.append((n, v))
    return mods


def _fingerprint(mod):
    """Shape+mass fingerprint over the module's parameter tensors (the
    trained weights). Buffers like a recurrent policy's hidden state are
    excluded — they legitimately change every inference."""
    out = []
    for name, t in sorted(mod.named_parameters()):
        t = t.detach().float().reshape(-1)
        out.append((name, tuple(t.shape),
                    round(float(t.abs().sum()), 3)))
    return out


def _pelvis(ex):
    bid = getattr(ex, "pelvis_body", None)
    if bid is None or bid < 0:
        mujoco = _mujoco()
        bid = mujoco.mj_name2id(
            ex.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return int(bid)


def _forward_xy(ex):
    """Pelvis +x axis in world xy — the direction vx must move it."""
    w, x, y, z = np.asarray(ex.data.qpos[3:7], dtype=float)
    f = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w)])
    n = np.linalg.norm(f)
    return f / n if n > 1e-9 else f


def _yaw(ex):
    w, x, y, z = np.asarray(ex.data.qpos[3:7], dtype=float)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _pd_probe(ex, legs):
    """Perturb each leg joint on measured data: a PD servo fights the
    displacement (stiffness) and fights the velocity (damping). Returns
    (stiff_count, damped_count, failures)."""
    mujoco = _mujoco()
    m, d = ex.model, ex.data
    qpos0, qvel0, ctrl0 = d.qpos.copy(), d.qvel.copy(), d.ctrl.copy()
    stiff = damped = 0
    fails = []
    for a, jid, vadr, qadr in legs:
        d.qpos[:] = qpos0
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        f0 = float(d.qfrc_actuator[vadr])
        d.qpos[qadr] = qpos0[qadr] + 0.05
        mujoco.mj_forward(m, d)
        f_up = float(d.qfrc_actuator[vadr])
        d.qpos[qadr] = qpos0[qadr] - 0.05
        mujoco.mj_forward(m, d)
        f_dn = float(d.qfrc_actuator[vadr])
        d.qpos[qadr] = qpos0[qadr]
        d.qvel[vadr] = 0.5
        mujoco.mj_forward(m, d)
        fv_up = float(d.qfrc_actuator[vadr])
        d.qvel[vadr] = -0.5
        mujoco.mj_forward(m, d)
        fv_dn = float(d.qfrc_actuator[vadr])
        d.qvel[vadr] = 0.0
        name = _mujoco().mj_id2name(m, _mujoco().mjtObj.mjOBJ_JOINT, jid)
        if f_up < f0 - 0.4 and f_dn > f0 + 0.4:
            stiff += 1
        else:
            fails.append(f"{name}: +0.05 rad -> {f_up - f0:+.2f} N, "
                         f"-0.05 rad -> {f_dn - f0:+.2f} N (not restoring)")
        if fv_up < fv_dn - 0.4:
            damped += 1
        else:
            fails.append(f"{name}: +0.5 rad/s -> {fv_up:+.2f} N vs "
                         f"-0.5 rad/s -> {fv_dn:+.2f} N (not damping)")
    d.qpos[:] = qpos0
    d.qvel[:] = qvel0
    d.ctrl[:] = ctrl0
    mujoco.mj_forward(m, d)
    return stiff, damped, fails


def test_ctrl_1_1_walk_runs_unitree_rl_policy_pd_torques_at_50hz(tmp_path):
    import torch

    # --- the pretrained Unitree artifact exists and keeps its deploy I/O
    assert POLICY_PATH.is_file(), (
        f"the pretrained Unitree G1 walking policy is missing at "
        f"{POLICY_PATH.relative_to(REPO)}")
    try:
        ref = torch.jit.load(str(POLICY_PATH), map_location="cpu")
    except Exception as exc:
        pytest.fail(
            f"{POLICY_PATH.name} is not a TorchScript module the runtime "
            f"can load ({type(exc).__name__}: {exc})")
    try:
        out = ref(torch.zeros(1, OBS_DIM))
    except Exception:
        out = ref(torch.zeros(OBS_DIM))
    assert int(out.reshape(-1).numel()) == ACT_DIM, (
        f"the walking policy maps the {OBS_DIM}-d observation to "
        f"{out.reshape(-1).numel()} outputs, not the {ACT_DIM} leg-joint "
        f"actions of the Unitree deploy policy")

    ex = _executor(tmp_path)
    dt = float(ex.model.opt.timestep)
    legs = _leg_actuators(ex.model)
    leg_act = np.array([a for a, _, _, _ in legs])
    leg_dof = np.array([v for _, _, v, _ in legs])
    assert len(legs) == ACT_DIM, (
        f"expected {ACT_DIM} leg (hip/knee/ankle) servos in the scene, "
        f"found {len(legs)}")

    # --- public entry: walk(vx, vy, yaw_rate), also via the program bridge
    walk = getattr(ex, "walk", None)
    assert callable(walk), "the executor exposes no walk(vx, vy, yaw_rate)"
    res = ex.program("walk(0.0, 0.0, 0.0)")
    assert "error" not in res, (
        f"walk() is not callable through the program() bridge: "
        f"{res.get('error')}")

    st = walk(0.3, 0.0, 0.0)
    w = _walker_of(ex)
    assert w is not None, (
        "walk(vx, vy, yaw_rate) engaged no low-level walker object on "
        "the executor — the command is not handed to a gait controller")

    # --- the walker runs the pretrained TorchScript artifact
    mods = _jit_modules(w, torch) or _jit_modules(ex, torch)
    assert mods, (
        "the walker does not run a TorchScript (torch.jit) module — "
        "the gait is not driven by the pretrained RL policy")
    pol = None
    for name, m in mods:
        try:
            o = m(torch.zeros(1, OBS_DIM))
        except Exception:
            try:
                o = m(torch.zeros(OBS_DIM))
            except Exception:
                continue
        if int(o.reshape(-1).numel()) == ACT_DIM:
            pol = (name, m)
            break
    assert pol is not None, (
        f"no TorchScript module on the walker consumes the {OBS_DIM}-d "
        f"Unitree observation and emits {ACT_DIM} leg actions "
        f"(saw jit attrs {[n for n, _ in mods]})")
    assert _fingerprint(pol[1]) == _fingerprint(ref), (
        f"the walker's jit module '{pol[0]}' is not the pretrained "
        f"artifact assets/policies/g1_walk_unitree_rl_gym.pt")

    # --- 50 Hz: leg position targets only change on the policy cadence
    prev = ex.data.ctrl[leg_act].copy()
    changed = []
    n_steps = int(round(1.5 / dt))
    for i in range(1, n_steps + 1):
        ex.step(1)
        cur = ex.data.ctrl[leg_act]
        if not np.allclose(cur, prev, atol=1e-7):
            changed.append(i)
            prev = cur.copy()
    assert changed, (
        "walking produced no leg position-target updates in "
        f"{n_steps} steps — the policy is not driving the leg servos")
    gaps = [g for g in np.diff(changed).tolist() if g > 0]
    assert len(gaps) >= 2, (
        f"leg position targets changed only {len(changed)} time(s) in "
        f"{n_steps} steps — no measurable inference cadence")
    base = math.gcd(*gaps)
    assert all(g % base == 0 for g in gaps) and base < min(gaps) * 2, (
        f"leg target updates are not on a uniform cadence: steps "
        f"{changed[:12]}")
    period = base * dt
    assert 0.015 <= period <= 0.025, (
        f"the walking policy infers every {period * 1000:.1f} ms "
        f"({1 / period:.1f} Hz) — the line requires a 50 Hz cadence "
        f"(~20 ms)")

    # --- PD torques: real, measured, restoring + damping on all 12 legs
    peak_tau = float(np.abs(ex.data.qfrc_actuator[leg_dof]).max())
    assert peak_tau > 1.0, (
        f"leg actuator torques are ~zero while walking "
        f"(peak {peak_tau:.3f} N*m) — the gait is not torque-driven")
    stiff, damped, fails = _pd_probe(ex, legs)
    assert stiff == len(legs) and damped == len(legs), (
        f"leg servos are not PD-tracked ({stiff}/{len(legs)} restoring, "
        f"{damped}/{len(legs)} damping): " + "; ".join(fails[:4]))

    # --- vx locomotes the free pelvis forward, upright, under torque
    pel = _pelvis(ex)
    fwd = _forward_xy(ex)
    p0 = ex.data.xpos[pel].copy()
    walk(0.4, 0.0, 0.0)
    travelled = 0.0
    min_z = float(p0[2])
    for _ in range(int(round(4.0 / dt))):
        ex.step(1)
        p = ex.data.xpos[pel]
        min_z = min(min_z, float(p[2]))
        travelled = float(np.dot((p - p0)[:2], fwd))
        if travelled >= 0.3:
            break
    assert travelled >= 0.3, (
        f"walk(0.4, 0, 0) moved the pelvis {travelled:.3f} m along its "
        f"forward axis in 4 s — no real locomotion (moved "
        f"{np.round((ex.data.xpos[pel] - p0), 3).tolist()})")
    assert min_z > 0.55, (
        f"the pelvis dropped to z={min_z:.3f} m under the walking "
        f"policy — it fell or slid instead of walking upright")

    # --- yaw_rate turns the body: the command is a rate, not a position
    walk(0.0, 0.0, 0.6)
    y0 = _yaw(ex)
    ex.step(int(round(2.0 / dt)))
    dyaw = abs(_wrap(_yaw(ex) - y0))
    assert dyaw >= 0.3, (
        f"walk(0, 0, 0.6) changed the body yaw by {dyaw:.3f} rad in "
        f"2 s — the yaw_rate channel does not steer the gait")

    # --- the command is one 3-vector (vx, vy, yaw_rate) on the walker
    walk(0.31, 0.12, -0.4)
    want = np.array([0.31, 0.12, -0.4])
    hits = []
    for n in dir(w):
        try:
            v = np.asarray(getattr(w, n), dtype=float)
        except Exception:
            continue
        if v.shape == (3,) and np.allclose(v, want, atol=5e-3):
            hits.append(n)
    assert hits, (
        "walk(vx, vy, yaw_rate) set no 3-vector velocity command on the "
        "walker — the (vx, vy, yaw_rate) channels are not the input")
