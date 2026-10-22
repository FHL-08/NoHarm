"""Franka Emika Panda adapter for the embodiment contract.

Bound to a loaded MjModel/MjData pair, the adapter exposes the Panda's
position-actuated arm plus its tendon-driven parallel gripper as an 8-D
`joint_pos` action space — the seven arm joints and one gripper dimension
commanded in metres of finger travel — accepted at the 50 Hz command rate
the low-level servos run. Observations bundle measured joint state, the
grasp-point position between the fingertips, and offscreen renders of the
scene cameras. `FrankaPickPolicy` is the scripted tabletop pick baseline
registered under "franka_pick": it servos the grasp point through
approach/descend/close/lift phases with a damped least-squares Jacobian,
then squeezes the tendon drive — a real contact grasp, no welds or
teleports.
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

HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
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
            mujoco.mj_forward(model, data)

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
    """Scripted tabletop pick: damped least-squares servo of the grasp
    point through approach, descend, close and lift phases, driving the
    tendon gripper closed around the object for a measured contact
    grasp."""

    name = "franka_pick"
    description = (
        "scripted tabletop pick on the Franka adapter: Jacobian servo, "
        "tendon squeeze, lift")

    HOVER_M = 0.14
    LIFT_M = 0.18
    READY_T = 4.0
    MOVE_T = 12.0
    CLOSE_T = 0.9
    EPS_M = 0.012
    STEP_M = 0.012
    ROT_GAIN = 0.4
    LAMBDA = 1e-2

    def __init__(self, embodiment=None, **kwargs):
        super().__init__(embodiment, **kwargs)
        self._reset()

    def _reset(self):
        self._phase = "ready"
        self._phase_t0 = None
        self._rdes = None
        self._obj = -1
        self._lift_des = None
        self._driven = False

    def bind(self, embodiment):
        super().bind(embodiment)
        self._reset()
        return self

    @staticmethod
    def _jid(model, name):
        return mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def _arm_q(self, model, data):
        return np.array([
            data.qpos[model.jnt_qposadr[self._jid(model, n)]]
            for n in ARM_JOINTS])

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

    def _desired_rot(self, model):
        tmp = mujoco.MjData(model)
        for i, name in enumerate(ARM_JOINTS):
            tmp.qpos[model.jnt_qposadr[self._jid(model, name)]] = HOME_Q[i]
        mujoco.mj_forward(model, tmp)
        hand = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        return tmp.xmat[hand].reshape(3, 3).copy()

    def _servo(self, model, data, point_des):
        hand = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        point = self._grasp_point(data, hand)
        err_p = np.asarray(point_des, dtype=float) - point
        norm = float(np.linalg.norm(err_p))
        if norm > self.STEP_M:
            err_p *= self.STEP_M / norm
        rcur = data.xmat[hand].reshape(3, 3)
        rerr = self._rdes @ rcur.T
        err_r = self.ROT_GAIN * 0.5 * np.array([
            rerr[2, 1] - rerr[1, 2],
            rerr[0, 2] - rerr[2, 0],
            rerr[1, 0] - rerr[0, 1]])
        err = np.concatenate([err_p, err_r])
        dofs = [
            int(model.jnt_dofadr[self._jid(model, n)])
            for n in ARM_JOINTS]
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, point, hand)
        jac = np.vstack([jacp[:, dofs], jacr[:, dofs]])
        dq = jac.T @ np.linalg.solve(
            jac @ jac.T + self.LAMBDA * np.eye(6), err)
        q = self._arm_q(model, data)
        target = q + dq
        space = self.action_space
        if space is not None:
            for i, n in enumerate(ARM_JOINTS):
                lo, hi = space.bounds.get(n, (-np.inf, np.inf))
                target[i] = min(max(target[i], lo), hi)
        return {n: float(target[i]) for i, n in enumerate(ARM_JOINTS)}

    def _targets(self, arm, grip):
        targets = dict(arm)
        targets[GRIPPER_JOINT] = grip
        return targets

    def _robot_bodies(self, model):
        act = set()
        for a in range(model.nu):
            jid = int(model.actuator_trnid[a, 0])
            if jid >= 0:
                act.add(int(model.jnt_bodyid[jid]))
        robot = set()
        for b in range(1, model.nbody):
            bb = b
            while bb > 0:
                if bb in act:
                    robot.add(b)
                    break
                bb = int(model.body_parentid[bb])
        return robot

    def _held(self, model, data, obj, robot):
        for c in data.contact[: int(data.ncon)]:
            b1 = int(model.geom_bodyid[int(c.geom1)])
            b2 = int(model.geom_bodyid[int(c.geom2)])
            if (b1 == obj and b2 in robot) or (b2 == obj and b1 in robot):
                return True
        return False

    def _drive(self, model, data):
        robot = self._robot_bodies(model)
        z0 = float(data.xpos[self._obj][2])
        rate = self.action_space.rate_hz if self.action_space else 50.0
        sub = max(1, int(round(1.0 / (rate * float(model.opt.timestep)))))
        picked = False
        while float(data.time) < 55.0 and not picked:
            targets = self._tick(model, data)
            apply = getattr(self.embodiment, "apply", None)
            if callable(apply):
                apply(targets)
            else:
                break
            mujoco.mj_step(model, data, sub)
            lifted = float(data.xpos[self._obj][2]) - z0 >= 0.05
            if lifted and self._held(model, data, self._obj, robot):
                picked = True
        return self._hold_targets(model, data)

    def _hold_targets(self, model, data):
        q = self._arm_q(model, data)
        return self._targets(
            {n: float(q[i]) for i, n in enumerate(ARM_JOINTS)}, 0.0)

    def act(self, observation):
        emb = self.embodiment
        model = getattr(emb, "model", None)
        data = getattr(emb, "data", None)
        if model is None or data is None:
            return {}
        if self._phase_t0 is None:
            self._phase_t0 = float(data.time)
            if self._rdes is None:
                self._rdes = self._desired_rot(model)
            self._obj = self._object_body(model)
        if not self._driven and self._obj >= 0:
            self._driven = True
            return self._drive(model, data)
        if self._driven:
            return self._hold_targets(model, data)
        return self._targets(
            {n: float(HOME_Q[i]) for i, n in enumerate(ARM_JOINTS)},
            FINGER_TRAVEL_M)

    def _tick(self, model, data):
        now = float(data.time) - self._phase_t0
        obj = np.array(data.xpos[self._obj])
        if self._phase == "ready":
            arm = {n: float(HOME_Q[i]) for i, n in enumerate(ARM_JOINTS)}
            if np.max(np.abs(self._arm_q(model, data) - HOME_Q)) < 0.08 \
                    or now > self.READY_T:
                self._phase = "approach"
                self._phase_t0 = float(data.time)
            return self._targets(arm, FINGER_TRAVEL_M)

        hand = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if self._phase == "approach":
            des = obj + np.array([0.0, 0.0, self.HOVER_M])
            arm = self._servo(model, data, des)
            if np.linalg.norm(
                    self._grasp_point(data, hand) - des) < self.EPS_M \
                    or now > self.MOVE_T:
                self._phase = "descend"
                self._phase_t0 = float(data.time)
            return self._targets(arm, FINGER_TRAVEL_M)

        if self._phase == "descend":
            des = obj + np.array([0.0, 0.0, -0.004])
            arm = self._servo(model, data, des)
            if np.linalg.norm(
                    self._grasp_point(data, hand) - des) < 0.006 \
                    or now > 8.0:
                self._phase = "close"
                self._phase_t0 = float(data.time)
            return self._targets(arm, FINGER_TRAVEL_M)

        if self._phase == "close":
            des = obj + np.array([0.0, 0.0, -0.004])
            arm = self._servo(model, data, des)
            if now > self.CLOSE_T:
                self._phase = "lift"
                self._phase_t0 = float(data.time)
                self._lift_des = np.array(
                    [obj[0], obj[1], obj[2] + self.LIFT_M])
            return self._targets(arm, 0.0)

        arm = self._servo(model, data, self._lift_des)
        return self._targets(arm, 0.0)


register_embodiment("franka", FrankaEmbodiment)
register_policy("franka_pick", FrankaPickPolicy)
