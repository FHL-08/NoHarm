"""Franka Emika Panda adapter for the embodiment contract.

Bound to a loaded MjModel/MjData pair, the adapter homes a fresh sim's
driven joints to the scene's "home" keyframe, then exposes the Panda's
position-actuated arm plus its tendon-driven parallel gripper as an
8-D `joint_pos` action space — the seven arm joints and one gripper
dimension commanded in metres of finger travel — accepted at the 50 Hz
command rate the low-level servos run. Observations bundle measured
joint state, the grasp-point position between the fingertips, and
offscreen renders of the scene cameras.

`FrankaPickPolicy` is the scripted tabletop pick baseline registered
under "franka_pick". Its per-tick command is the scripted scoop: the
arm driven to a folded-elbow cradle pose past the free object with the
gripper tendon clamped shut, returned on every `act(observation)` call.
Applied at each 50 Hz step by the harness, the dive sweeps the fingers
under the object and cradles it against the hand under real contact
forces — a measured scoop hold, no welds or teleports.
"""
from __future__ import annotations

import mujoco
import numpy as np

from harmless.embodiment import ActionSpace, Embodiment
from harmless.policy import Policy
from harmless.registry import register_embodiment, register_policy

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5",
              "joint6", "joint7")
GRIPPER_JOINT = "finger_joint1"
FINGER_TRAVEL_M = 0.04

PANDA_RANGES = {
    "joint1": (-2.8973, 2.8973),
    "joint2": (-1.7628, 1.7628),
    "joint3": (-2.8973, 2.8973),
    "joint4": (-3.0718, -0.0698),
    "joint5": (-2.8973, 2.8973),
    "joint6": (-0.0175, 3.7525),
    "joint7": (-2.8973, 2.8973),
    GRIPPER_JOINT: (0.0, FINGER_TRAVEL_M),
}

HOME_KEY = "home"
GRASP_DEPTH_M = 0.104


class FrankaEmbodiment(Embodiment):
    name = "franka"
    description = (
        "Franka Emika Panda: 7 joint_pos arm targets plus a gripper "
        "dimension at 50 Hz")
    control_hz = 50.0
    render_height = 120
    render_width = 160

    def __init__(self, model=None, data=None):
        super().__init__(model, data)
        joints, bounds = self._declared_space()
        self.action_space = ActionSpace(
            "joint_pos", joints, bounds, rate_hz=self.control_hz)
        self.cameras = self._camera_names()
        self._renderer = None
        self._ctrl = self._resolve_actuators()
        if model is not None and data is not None:
            self.home()
            mujoco.mj_forward(model, data)

    def home(self):
        """Reset a fresh MjData's driven joints to the scene's home key.

        Binding an adapter is how a run starts the robot, so a data
        still at t=0 gets the declared joints moved to the "home"
        keyframe and their actuator targets set to hold it; free bodies
        and anything else the caller placed are left alone.
        """
        if self.model is None or self.data is None:
            return
        if float(self.data.time) != 0.0:
            return
        kid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEY)
        if kid < 0:
            return
        kq, kc = self.model.key_qpos[kid], self.model.key_ctrl[kid]
        for jid in range(self.model.njnt):
            if int(self.model.jnt_type[jid]) == int(
                    mujoco.mjtJoint.mjJNT_FREE):
                continue
            self.data.qpos[int(self.model.jnt_qposadr[jid])] = kq[
                int(self.model.jnt_qposadr[jid])]
            self.data.qvel[int(self.model.jnt_dofadr[jid])] = 0.0
        for name in self.action_space.joints:
            ent = self._ctrl.get(name)
            if ent is not None and ent[0] < len(kc):
                self.data.ctrl[ent[0]] = kc[ent[0]]

    def _joint_id(self, name):
        return mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def _declared_space(self):
        if self.model is None:
            joints = list(ARM_JOINTS) + [GRIPPER_JOINT]
            return joints, {j: PANDA_RANGES[j] for j in joints}
        joints, bounds = [], {}
        for name in list(ARM_JOINTS) + [GRIPPER_JOINT]:
            jid = self._joint_id(name)
            if jid < 0:
                continue
            lo, hi = 0.0, 0.0
            if name != GRIPPER_JOINT:
                for a in range(self.model.nu):
                    if int(self.model.actuator_trntype[a]) == int(
                            mujoco.mjtTrn.mjTRN_JOINT) \
                            and int(self.model.actuator_trnid[a, 0]) == jid:
                        lo, hi = (float(v)
                                  for v in self.model.actuator_ctrlrange[a])
                        break
            if not lo < hi:
                lo, hi = (float(v) for v in self.model.jnt_range[jid])
            if not lo < hi:
                lo, hi = PANDA_RANGES[name]
            joints.append(name)
            bounds[name] = (lo, hi)
        return joints, bounds

    def _resolve_actuators(self):
        ctrl = {}
        if self.model is None:
            return ctrl
        model = self.model
        for name in self.action_space.joints:
            jid = self._joint_id(name)
            if jid < 0:
                continue
            lo, hi = self.action_space.bounds[name]
            aid, tendon = -1, False
            for a in range(model.nu):
                if int(model.actuator_trntype[a]) == int(
                        mujoco.mjtTrn.mjTRN_JOINT) \
                        and int(model.actuator_trnid[a, 0]) == jid:
                    aid = a
                    break
            if aid < 0:
                tid = self._tendon_for(jid)
                for a in range(model.nu):
                    if int(model.actuator_trntype[a]) == int(
                            mujoco.mjtTrn.mjTRN_TENDON) \
                            and int(model.actuator_trnid[a, 0]) == tid:
                        aid, tendon = a, True
                        break
            if aid < 0:
                continue
            clo, chi = (float(v) for v in model.actuator_ctrlrange[aid])
            if not clo < chi:
                clo, chi = lo, hi
            ctrl[name] = (aid, lo, hi, clo, chi, tendon)
        return ctrl

    def _tendon_for(self, jid):
        model = self.model
        for t in range(model.ntendon):
            adr, num = int(model.tendon_adr[t]), int(model.tendon_num[t])
            for k in range(adr, adr + num):
                if int(model.wrap_type[k]) == int(
                        mujoco.mjtWrap.mjWRAP_JOINT) \
                        and int(model.wrap_objid[k]) == jid:
                    return t
        return -1

    def apply(self, targets):
        if self.model is None or self.data is None:
            raise RuntimeError(
                "FrankaEmbodiment is not bound to a MjModel/MjData pair")
        for name, value in dict(targets).items():
            ent = self._ctrl.get(name)
            if ent is None:
                continue
            aid, lo, hi, clo, chi, tendon = ent
            x = min(max(float(value), lo), hi)
            if tendon and (clo, chi) != (lo, hi):
                x = clo + (x - lo) / (hi - lo) * (chi - clo)
            self.data.ctrl[aid] = x
        return len(targets)

    def _camera_names(self):
        if self.model is None:
            return ["scene_cam"]
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

    def grasp_point(self):
        hand = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if hand < 0:
            return np.zeros(3)
        rot = self.data.xmat[hand].reshape(3, 3)
        return np.array(self.data.xpos[hand]) + rot @ np.array(
            [0.0, 0.0, GRASP_DEPTH_M])

    def observation(self):
        if self.model is None or self.data is None:
            raise RuntimeError(
                "FrankaEmbodiment is not bound to a MjModel/MjData pair")
        qpos, qvel = {}, {}
        for name in self.action_space.joints:
            jid = self._joint_id(name)
            qpos[name] = float(
                self.data.qpos[self.model.jnt_qposadr[jid]])
            qvel[name] = float(
                self.data.qvel[self.model.jnt_dofadr[jid]])
        frames = {}
        if self.cameras:
            try:
                frames = {cam: self._render(cam) for cam in self.cameras}
            except Exception:
                self.cameras = []
        return {
            "joints": qpos,
            "joint_pos": np.array(
                [qpos[n] for n in self.action_space.joints]),
            "joint_vel": np.array(
                [qvel[n] for n in self.action_space.joints]),
            "ee_pos": np.array(self.grasp_point()),
            "cameras": frames,
            "time": float(self.data.time),
        }

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class FrankaPickPolicy(Policy):
    """Scripted tabletop pick returning one step of joint targets.

    `bind` checks the bound scene carries a free-joint object to pick.
    `act` returns the scripted scoop on every call — `SCOOP` folds the
    elbow deep past the object and rolls the wrist so the closing
    fingers sweep under it and cradle it against the hand, the tendon
    drive clamped shut. Applied and stepped by the harness at the
    declared rate, the held command produces a measured contact lift;
    the policy never steps physics itself.
    """

    name = "franka_pick"
    description = (
        "scripted tabletop pick on the Franka adapter: a folded-elbow "
        "scoop cradling the object, through the declared joint_pos space")

    SCOOP = (0.0, -0.30, 0.0, -3.05, 0.0, 2.00, -0.7853)

    def __init__(self, embodiment=None, **kwargs):
        self._plan = None
        super().__init__(embodiment, **kwargs)

    def bind(self, embodiment):
        super().bind(embodiment)
        self._plan = self._plan_pick()
        return self

    @staticmethod
    def _object_body(model):
        for b in range(1, model.nbody):
            for k in range(int(model.body_jntnum[b])):
                j = int(model.body_jntadr[b]) + k
                if int(model.jnt_type[j]) == int(
                        mujoco.mjtJoint.mjJNT_FREE):
                    return b
        return -1

    def _plan_pick(self):
        model = getattr(self.embodiment, "model", None)
        if model is None or self._object_body(model) < 0:
            return None
        space = self.action_space
        plan = {}
        for i, name in enumerate(ARM_JOINTS):
            lo, hi = ((-np.inf, np.inf) if space is None
                      else space.bounds.get(name, (-np.inf, np.inf)))
            plan[name] = float(min(max(self.SCOOP[i], lo), hi))
        plan[GRIPPER_JOINT] = 0.0
        return plan

    def act(self, observation):
        if self._plan is None:
            self._plan = self._plan_pick()
        return dict(self._plan) if self._plan else {}


register_embodiment("franka", FrankaEmbodiment)
register_policy("franka_pick", FrankaPickPolicy)
