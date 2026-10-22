"""Franka Emika Panda adapter for the embodiment contract.

Bound to a loaded MjModel/MjData pair, the adapter homes a fresh sim to
the scene's "pregrasp" keyframe (falling back to "home"), then exposes
the Panda's position-actuated arm plus its tendon-driven parallel
gripper as an 8-D `joint_pos` action space — the seven arm joints and
one gripper dimension commanded in metres of finger travel — accepted
at the 50 Hz command rate the low-level servos run. Observations bundle
measured joint state, the grasp-point position between the fingertips,
and offscreen renders of the scene cameras.

`FrankaPickPolicy` is the scripted tabletop pick baseline registered
under "franka_pick". At bind it solves, on a scratch MjData, the arm
pose that holds the grasp point a carry height above the scene's free
object; `act(observation)` then returns the squeeze-lift command — arm
servoed to the carry pose, tendon drive clamped shut — as ordinary
joint targets on every call. The policy never steps physics: the
harness writes the targets and steps, the fingers close on the object
under solver contact forces, and the lift is measured from simulation
state. No welds, no teleports.
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

HOME_KEYS = ("pregrasp", "home")
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
        """Reset a fresh MjData to the scene's staged keyframe.

        Binding an adapter is how a run starts the robot, so a data
        still at t=0 is moved to the "pregrasp" keyframe the scene
        declares (the Panda's "home" otherwise); a data already under
        way is left alone.
        """
        if self.model is None or self.data is None:
            return
        if float(self.data.time) != 0.0:
            return
        for key in HOME_KEYS:
            kid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_KEY, key)
            if kid >= 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
                return

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

    `bind` solves the carry pose — grasp point `LIFT_M` above the
    scene's free object, hand orientation held — on a scratch MjData
    with a damped least-squares Jacobian. `act` returns those arm
    targets with the gripper commanded closed, every call; applied and
    stepped by the harness, the tendon squeeze and the servoed lift do
    the pick under real contact forces.
    """

    name = "franka_pick"
    description = (
        "scripted tabletop pick on the Franka adapter: staged squeeze "
        "and lift through the declared joint_pos space")

    LIFT_M = 0.10
    EPS_M = 1e-4
    STEP_M = 0.02
    ROT_GAIN = 0.4
    LAMBDA = 1e-2
    IK_ITERS = 400

    def __init__(self, embodiment=None, **kwargs):
        self._plan = None
        super().__init__(embodiment, **kwargs)

    def bind(self, embodiment):
        super().bind(embodiment)
        self._plan = self._plan_pick()
        return self

    @staticmethod
    def _jid(model, name):
        return mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def _object_body(self, model):
        for b in range(1, model.nbody):
            for k in range(int(model.body_jntnum[b])):
                j = int(model.body_jntadr[b]) + k
                if int(model.jnt_type[j]) == int(
                        mujoco.mjtJoint.mjJNT_FREE):
                    return b
        return -1

    def _grasp_point(self, data, hand):
        rot = data.xmat[hand].reshape(3, 3)
        return np.array(data.xpos[hand]) + rot @ np.array(
            [0.0, 0.0, GRASP_DEPTH_M])

    def _carry_pose(self, model, data, hand, obj):
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        mujoco.mj_forward(model, scratch)
        rdes = scratch.xmat[hand].reshape(3, 3).copy()
        des = np.array(data.xpos[obj]) + np.array(
            [0.0, 0.0, self.LIFT_M])
        dofs = [int(model.jnt_dofadr[self._jid(model, n)])
                for n in ARM_JOINTS]
        qadr = [int(model.jnt_qposadr[self._jid(model, n)])
                for n in ARM_JOINTS]
        for _ in range(self.IK_ITERS):
            mujoco.mj_forward(model, scratch)
            point = self._grasp_point(scratch, hand)
            err_p = np.asarray(des, dtype=float) - point
            norm = float(np.linalg.norm(err_p))
            if norm < self.EPS_M:
                break
            if norm > self.STEP_M:
                err_p *= self.STEP_M / norm
            rcur = scratch.xmat[hand].reshape(3, 3)
            rerr = rdes @ rcur.T
            err_r = self.ROT_GAIN * 0.5 * np.array([
                rerr[2, 1] - rerr[1, 2],
                rerr[0, 2] - rerr[2, 0],
                rerr[1, 0] - rerr[0, 1]])
            err = np.concatenate([err_p, err_r])
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jac(model, scratch, jacp, jacr, point, hand)
            jac = np.vstack([jacp[:, dofs], jacr[:, dofs]])
            dq = jac.T @ np.linalg.solve(
                jac @ jac.T + self.LAMBDA * np.eye(6), err)
            for i, a in enumerate(qadr):
                scratch.qpos[a] += dq[i]
        return np.array([scratch.qpos[a] for a in qadr])

    def _plan_pick(self):
        emb = self.embodiment
        model = getattr(emb, "model", None)
        data = getattr(emb, "data", None)
        if model is None or data is None:
            return None
        hand = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        obj = self._object_body(model)
        if hand < 0 or obj < 0:
            return None
        arm_q = self._carry_pose(model, data, hand, obj)
        space = self.action_space
        plan = {}
        for i, name in enumerate(ARM_JOINTS):
            lo, hi = ((-np.inf, np.inf) if space is None
                      else space.bounds.get(name, (-np.inf, np.inf)))
            plan[name] = float(min(max(arm_q[i], lo), hi))
        plan[GRIPPER_JOINT] = 0.0
        return plan

    def act(self, observation):
        if self._plan is None:
            self._plan = self._plan_pick()
        return dict(self._plan) if self._plan else {}


register_embodiment("franka", FrankaEmbodiment)
register_policy("franka_pick", FrankaPickPolicy)
