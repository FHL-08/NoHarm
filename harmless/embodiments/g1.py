"""Unitree G1 humanoid adapter for the embodiment contract.

Bound to a loaded MjModel/MjData pair, the adapter exposes the G1's
position-actuated joints as a `joint_pos` action space — targets bounded by
each actuator's control range, accepted at the 50 Hz rate the on-board
controller runs — and produces observations bundling measured per-joint
position/velocity, the pelvis free-base pose, the model's gyro and
accelerometer readings, its touch-sensor normal forces, and offscreen
renders of every named scene camera.
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

    def _pelvis_id(self):
        pelvis = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis >= 0:
            return pelvis
        for j in range(self.model.njnt):
            if int(self.model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
                return int(self.model.jnt_bodyid[j])
        return -1

    def _sensor_channels(self):
        imu, touch = {}, {}
        imu_kinds = (mujoco.mjtSensor.mjSENS_GYRO,
                     mujoco.mjtSensor.mjSENS_ACCELEROMETER)
        for i in range(self.model.nsensor):
            adr = int(self.model.sensor_adr[i])
            dim = int(self.model.sensor_dim[i])
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, i) or f"sensor{i}"
            kind = int(self.model.sensor_type[i])
            if kind == mujoco.mjtSensor.mjSENS_TOUCH:
                touch[name] = float(self.data.sensordata[adr])
            elif kind in imu_kinds:
                imu[name] = np.array(
                    self.data.sensordata[adr:adr + dim], dtype=float)
        return imu, touch

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
        imu, touch = self._sensor_channels()
        obs = {
            "joints": qpos,
            "joint_pos": np.array(
                [qpos[n] for n in self.action_space.joints]),
            "joint_vel": np.array(
                [qvel[n] for n in self.action_space.joints]),
            "imu": imu,
            "touch": touch,
            "cameras": {cam: self._render(cam) for cam in self.cameras},
            "time": float(self.data.time),
        }
        pelvis = self._pelvis_id()
        if pelvis >= 0:
            obs["pelvis"] = {
                "pos": np.array(self.data.xpos[pelvis], dtype=float),
                "quat": np.array(self.data.xquat[pelvis], dtype=float),
            }
        return obs

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
