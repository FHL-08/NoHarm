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


# --------------------------------------------------------------------------
# Pretrained right-arm reacher (luckyrobots/g1-manipulation-challenge,
# right_reacher.onnx, commit 0e9d1f9): 36-d obs -> 7 right-arm joint
# targets at 50 Hz, run as its run.py does (pelvis-frame palm target,
# rate-limited targets). It moves the palm to a point; GPT picks the point.
REACHER = os.path.join(os.path.dirname(POLICY), "right_reacher.onnx")
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
       "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
       "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
ARM_DEFAULT = np.array([0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0], dtype=np.float32)
ARM_SCALE = np.array([0.438577] * 5 + [0.074501] * 2, dtype=np.float32)
ARM_MAX_DELTA = 0.012                      # rad per 50 Hz policy step
REACH_BOX = (np.array([-0.3, -0.6, -0.4]), np.array([0.6, 0.3, 0.6]))  # pelvis frame


def _qinv_apply(q, v):
    w, xyz = q[0], q[1:4]
    t = np.cross(xyz, v) * 2
    return v - w * t + np.cross(xyz, t)


# Left arm = the same policy through the robot's left/right mirror symmetry
# (reflection across the pelvis x-z plane): roll/yaw joints and lateral
# coordinates flip sign. Standard morphological-symmetry reuse.
MIRROR_J = np.array([1, -1, -1, 1, -1, 1, -1], dtype=np.float32)
MIRROR_V = np.array([1, -1, 1], dtype=np.float32)       # positions / gravity
MIRROR_R = np.array([-1, 1, -1], dtype=np.float32)      # roll, pitch, yaw


class Reacher:
    def __init__(self, ex, side="right"):
        import mujoco
        import onnxruntime as ort
        self.mj = mujoco
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(REACHER, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.ex, m = ex, ex.model
        self.side = side
        left = side == "left"
        self.sj = MIRROR_J if left else np.ones(7, np.float32)
        self.sv = MIRROR_V if left else np.ones(3, np.float32)
        self.sr = MIRROR_R if left else np.ones(3, np.float32)
        names = [n.replace("right_", "left_") for n in ARM] if left else ARM
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.vadr = np.array([m.jnt_dofadr[j] for j in jid])
        self.act = np.array([ex.ctrl_for_jid[j] for j in jid])
        self.palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_palm")
        self.every = max(1, int(round(DECIMATION_S / m.opt.timestep)))
        self.on = False
        self.world_target = None
        self.rpy = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(7, dtype=np.float32)
        self.last_target = None

    def _base(self):
        d = self.ex.data
        return d.qpos[0:3].copy(), d.qpos[3:7].copy()

    def target_in_pelvis(self):
        """Target in the (mirrored, for the left arm) pelvis frame, clipped
        to the policy's workspace."""
        p, q = self._base()
        t = _qinv_apply(q, np.asarray(self.world_target, float) - p) * self.sv
        return np.clip(t, *REACH_BOX)

    def palm_in_pelvis(self):
        p, q = self._base()
        return _qinv_apply(q, self.ex.data.site_xpos[self.palm] - p)

    def _palm_rpy(self):
        _, pq = self._base()
        palm_q = np.zeros(4)
        self.mj.mju_mat2Quat(palm_q, self.ex.data.site_xmat[self.palm])
        rel = np.zeros(4)
        self.mj.mju_mulQuat(rel, np.array([pq[0], -pq[1], -pq[2], -pq[3]]), palm_q)
        w, x, y, z = rel
        return np.array([np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                         np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
                         np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))],
                        dtype=np.float32)

    def set_target(self, xyz_world):
        self.world_target = np.asarray(xyz_world, float)
        if not self.on:
            self.on = True
            self.last_target = self.ex.data.qpos[self.qadr].astype(np.float32)

    def stop(self):
        self.on = False      # arm keeps its last targets

    def tick(self, step_count):
        if not self.on or step_count % self.every:
            return
        d = self.ex.data
        _, q = self._base()
        grav = _qinv_apply(q, np.array([0, 0, -1.0]))
        obs = np.concatenate([
            self.target_in_pelvis(), self.rpy, self.palm_in_pelvis() * self.sv,
            self._palm_rpy() * self.sr,
            self.sj * d.qpos[self.qadr] - ARM_DEFAULT, self.sj * d.qvel[self.vadr],
            self.last_action, grav * self.sv]).astype(np.float32)[None]
        a = self.sess.run(None, {self.inp: obs})[0][0]
        tgt = self.sj * (ARM_DEFAULT + a * ARM_SCALE)   # back to this arm's joints
        tgt = self.last_target + np.clip(tgt - self.last_target, -ARM_MAX_DELTA, ARM_MAX_DELTA)
        self.last_target, self.last_action = tgt.astype(np.float32), a.astype(np.float32)
        d.ctrl[self.act] = tgt

    def status(self):
        if self.world_target is None:
            return {"on": self.on}
        err = float(np.linalg.norm(self.ex.data.site_xpos[self.palm] - self.world_target))
        return {"on": self.on, "target_world": np.round(self.world_target, 3).tolist(),
                "target_pelvis_clipped": np.round(self.target_in_pelvis(), 3).tolist(),
                "palm_world": np.round(self.ex.data.site_xpos[self.palm], 3).tolist(),
                "palm_error_m": round(err, 3), "reach_box_pelvis": [b.tolist() for b in REACH_BOX]}
