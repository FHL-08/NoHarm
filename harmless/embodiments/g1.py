"""Unitree G1 humanoid adapter for the embodiment contract.

Bound to a loaded MjModel/MjData pair, the adapter exposes the G1's
position-actuated joints as a `joint_pos` action space — targets bounded by
each actuator's control range, accepted at the 50 Hz rate the on-board
controller runs — and produces observations bundling measured per-joint
position/velocity with offscreen renders of every named scene camera.
"""
from __future__ import annotations

import mujoco
import numpy as np

from harmless.embodiment import ActionSpace, Embodiment

G1_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


class G1Embodiment(Embodiment):
    name = "g1"
    description = "Unitree G1 humanoid: 29 joint_pos targets at 50 Hz"
    control_hz = 50.0
    render_height = 480
    render_width = 640

    def __init__(self, model=None, data=None):
        super().__init__(model, data)
        joints, bounds = self._declared_space()
        self.action_space = ActionSpace(
            "joint_pos", joints, bounds, rate_hz=self.control_hz)
        self.cameras = self._camera_names()
        self._renderer = None

    def _declared_space(self):
        if self.model is None:
            return list(G1_JOINTS), {j: (-np.pi, np.pi) for j in G1_JOINTS}
        joints, bounds = [], {}
        for a in range(self.model.nu):
            jid = self.model.actuator_trnid[a, 0]
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            lo, hi = (float(v) for v in self.model.actuator_ctrlrange[a])
            if not lo < hi:
                lo, hi = (float(v) for v in self.model.jnt_range[jid])
            if not lo < hi:
                lo, hi = -np.pi, np.pi
            joints.append(name)
            bounds[name] = (lo, hi)
        return joints, bounds

    def _camera_names(self):
        if self.model is None:
            return ["head_cam", "scene_cam"]
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, c)
            for c in range(self.model.ncam)
        ]

    def _render(self, camera):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.render_height,
                width=self.render_width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def observation(self):
        if self.model is None or self.data is None:
            raise RuntimeError(
                "G1Embodiment is not bound to a MjModel/MjData pair")
        qpos, qvel = {}, {}
        for name in self.action_space.joints:
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos[name] = float(self.data.qpos[self.model.jnt_qposadr[jid]])
            qvel[name] = float(self.data.qvel[self.model.jnt_dofadr[jid]])
        return {
            "joints": qpos,
            "joint_pos": np.array(
                [qpos[n] for n in self.action_space.joints]),
            "joint_vel": np.array(
                [qvel[n] for n in self.action_space.joints]),
            "cameras": {cam: self._render(cam) for cam in self.cameras},
            "time": float(self.data.time),
        }

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
