"""Learned low-level locomotion for the G1 — the hybrid split GPT-as-Policy
uses (a learned policy does motor control, GPT supervises).

Policy: Unitree's pretrained G1 walking policy from unitreerobotics/
unitree_rl_gym (deploy/pre_train/g1/motion.pt, BSD-3-Clause, commit
276801e), run exactly as its deploy_mujoco.py sim2sim does: 50 Hz, 47-d
observation (body angular velocity, projected gravity, velocity command,
12 leg joint positions/velocities, last action, gait phase), 12 leg
position targets tracked by PD with the policy's own gains. It produces
real joint torques through MuJoCo; nothing is teleported.

GPT-6-astra only sets the command (vx, vy, yaw_rate); it still owns the
arms, waist, grasping, navigation and avoidance.
"""
from __future__ import annotations

import os

import numpy as np

POLICY = os.path.join(os.path.dirname(__file__), "..", "assets", "policies",
                      "g1_walk_unitree_rl_gym.pt")
LEGS = [f"{s}_{j}_joint" for s in ("left", "right")
        for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
KPS = np.array([100, 100, 100, 150, 40, 40] * 2, dtype=np.float32)
KDS = np.array([2, 2, 2, 4, 2, 2] * 2, dtype=np.float32)
DEFAULT = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2, dtype=np.float32)
ACTION_SCALE, ANG_VEL_SCALE, DOF_VEL_SCALE = 0.25, 0.25, 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
CMD_MAX = np.array([0.8, 0.4, 1.0], dtype=np.float32)   # m/s, m/s, rad/s
DECIMATION_S = 0.02                                     # 50 Hz
PERIOD = 0.8                                            # gait phase period


def _gravity(q):
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x),
                     1 - 2 * (w * w + z * z)], dtype=np.float32)


class Walker:
    def __init__(self, ex):
        import mujoco
        import torch
        self.torch = torch
        self.policy = torch.jit.load(POLICY)
        self.ex, m = ex, ex.model
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEGS]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.vadr = np.array([m.jnt_dofadr[j] for j in jid])
        self.act = np.array([ex.ctrl_for_jid[j] for j in jid])
        self.base_q = m.jnt_qposadr[0]          # pelvis free joint is joint 0
        self.base_v = m.jnt_dofadr[0]
        self.saved = (m.actuator_gainprm[self.act].copy(),
                      m.actuator_biasprm[self.act].copy())
        self.every = max(1, int(round(DECIMATION_S / m.opt.timestep)))
        self.on = False
        self.cmd = np.zeros(3, dtype=np.float32)
        self.action = np.zeros(12, dtype=np.float32)
        self.t0 = 0.0

    def enable(self):
        m = self.ex.model
        # PD gains of the policy on the 12 leg servos (MuJoCo position
        # actuator: force = kp*(ctrl - q) - kd*qdot)
        m.actuator_gainprm[self.act, 0] = KPS
        m.actuator_biasprm[self.act, 1] = -KPS
        m.actuator_biasprm[self.act, 2] = -KDS
        self.action[:] = 0
        self.t0 = float(self.ex.data.time)
        self.on = True
        self._infer()

    def disable(self):
        m = self.ex.model
        m.actuator_gainprm[self.act], m.actuator_biasprm[self.act] = (
            self.saved[0].copy(), self.saved[1].copy())
        self.on = False
        self.cmd[:] = 0

    def set_cmd(self, vx, vy, yaw_rate):
        self.cmd[:] = np.clip([vx, vy, yaw_rate], -CMD_MAX, CMD_MAX)

    def _infer(self):
        d = self.ex.data
        q = d.qpos[self.qadr].astype(np.float32)
        dq = d.qvel[self.vadr].astype(np.float32)
        quat = d.qpos[self.base_q + 3:self.base_q + 7]
        omega = d.qvel[self.base_v + 3:self.base_v + 6].astype(np.float32)
        phase = ((float(d.time) - self.t0) % PERIOD) / PERIOD
        obs = np.concatenate([
            omega * ANG_VEL_SCALE, _gravity(quat), self.cmd * CMD_SCALE,
            q - DEFAULT, dq * DOF_VEL_SCALE, self.action,
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]]).astype(np.float32)
        with self.torch.no_grad():
            self.action = self.policy(self.torch.from_numpy(obs)[None]).numpy()[0]
        d.ctrl[self.act] = self.action * ACTION_SCALE + DEFAULT

    def tick(self, step_count):
        if self.on and step_count % self.every == 0:
            self._infer()

    def status(self):
        return {"on": self.on, "cmd": [round(float(c), 3) for c in self.cmd],
                "limits": CMD_MAX.tolist()}
