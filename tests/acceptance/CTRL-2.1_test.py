"""CTRL-2.1@2 With the arms driven by policy joint actions (no scripted
IK reach), grasp(obj) closes the G1's three fingers on the object and
holds it by contact forces only, with no equality weld activated.

The contract under test is the executor's public grasp(name) entry
point — the same call the program() bridge exposes — exercised the way
the @2 line requires: every arm motion in the trial comes from a POLICY
emitting joint_pos actions (joint-name -> position target), applied
through set_joints() at the 50 Hz action cadence. No scripted IK reach
entry point (arm_reach / arm_path / hands_path / reach / reach_xyz /
pick_up / put_down, or the private IK plumbing behind them) may run at
any point — the test tripwires all of them and audits the executor's
reacher registry.

The policy used is the test's own _ArmJointPolicy: a waypoint-following
controller bound to the live model that emits the next arm-joint
targets from the measured state on every act() call — a joint_pos
policy filling the same act(observation) contract slot the registry
policies (scripted, gr00t VLA) fill. The executor's only role in arm
motion is applying those position-actuator targets to the sim.

Everything else is checked against measured state, never self-report:

- the compiled scene must contain three actuated finger chains on at
  least one hand — {side}_hand_{thumb,index,middle} joints each backed
  by an actuator — the bundled rubber-hand g1.xml has none, so a scene
  without dexterous fingers fails here first (the expected
  pre-implementation failure)
- grasp("apple") runs from a flank-side pregrasp the policy drives a
  few cm off the apple's surface: the palms rise to flank height while
  still in the aisle (south of the worktop edge, no forearm across the
  counter) and settle beside the apple at its mid-height, where the
  closing fingers can wrap the body
- the apple's world position is recorded every physics step from grasp()
  to the end: a snap between the palms is a kinematic teleport and
  shows as a single-step jump far beyond anything contact physics can
  produce
- every equality constraint involving the apple's body is sampled per
  step: none may be active at any point during or after the grasp
- after grasp() returns, each of the three finger groups on some hand
  must have flexed measurably, and at least two distinct finger links
  must touch the apple's collision geoms with real normal force
- the hold is proven physically: the policy lifts both palms 8 cm and
  the apple must rise with the hand while nothing but contact can
  carry it — still touching fingers, still unwelded

On a tree with the weld-snap grasp the test fails at the finger-chain
assertion; a grasp that teleports or welds the object fails the
per-step jump/equality probes; arm motion through scripted IK fails the
tripwire; a grip that never closes or never touches fails the flexion
and contact probes; a prop merely resting on the counter fails the lift.
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

ARM_DOF = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
           "wrist_roll", "wrist_pitch", "wrist_yaw")
FINGER_JNT = re.compile(r"^(left|right)_hand_(thumb|index|middle)", re.I)
HAND_BODY = re.compile(r"^(left|right)_hand_(palm|thumb|index|middle)",
                       re.I)
FINGERS = ("thumb", "index", "middle")

# every scripted IK reach entry point on the executor: the @2 clause
# forbids any of them moving the arms during this trial
SCRIPTED_REACH = ("arm_reach", "arm_path", "hands_path", "reach",
                  "reach_xyz", "pick_up", "put_down", "_converge_to",
                  "_ik", "_move_arm_to", "_move_grip_to", "_wait_palms")

# at dt=0.002 s a physical move is < 1 cm; a between-palms snap is cm-scale
JUMP_M = 0.025
ACTION_HZ = 50.0        # joint_pos actions per second of sim time


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


def _teleport(ex, name, xyz, yaw=None):
    """Arrange the trial's initial condition: place a free body at rest.
    Test setup only — every later motion is the physics engine's."""
    mujoco = _mujoco()
    m, d = ex.model, ex.data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    jnt = m.body_jntadr[bid]
    qa, va = m.jnt_qposadr[jnt], m.jnt_dofadr[jnt]
    d.qpos[qa:qa + 3] = xyz
    if yaw is not None:
        d.qpos[qa + 3:qa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    d.qvel[va:va + 6] = 0
    mujoco.mj_forward(m, d)


def _park_person(ex):
    """Hold the person on her tether at the route start so she cannot
    walk into the measurement area mid-trial (determinism, not policy)."""
    if hasattr(ex, "_fall"):
        ex._fall = (float(ex.data.time), np.array([1.0, 0.0, 0.0]))


def _palm(ex, side):
    return np.asarray(ex.pos(f"{side}_palm"), float)


def _tripwire(ex):
    """Wrap every scripted IK reach entry point (public and private) so
    any call is recorded. Returns the list that fills with the names of
    any scripted-reach code path that ran during the trial."""
    fired = []
    for name in SCRIPTED_REACH:
        fn = getattr(ex, name, None)
        if not callable(fn):
            continue

        def wrap(fn=fn, name=name):
            def inner(*a, **k):
                fired.append(name)
                return fn(*a, **k)
            return inner
        setattr(ex, name, wrap())
    return fired


class _ArmJointPolicy:
    """The trial's joint_pos policy: the only thing that drives the arms.

    act() consumes the measured state and emits the next arm-joint
    position targets (a joint_pos action, the same action family the G1
    embodiment contract and every registry policy use). Its internal
    computation — waypoint-following with a damped-least-squares step on
    the measured palm — is the policy's own world model; the executor
    contributes nothing but applying the emitted targets through
    set_joints(). No executor IK or reach code is consulted.
    """

    ARM = ARM_DOF
    SPEED = 0.30          # m/s the palm travels along a straight segment
    DT = 1.0 / ACTION_HZ

    def __init__(self, ex):
        mujoco = _mujoco()
        m = ex.model
        self.ex = ex
        self._scratch = mujoco.MjData(m)
        act_of = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
        self.chains = {}
        for side in ("left", "right"):
            jid = np.array([
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                  f"{side}_{j}_joint")
                for j in self.ARM])
            if (jid < 0).any():
                continue
            act = np.array([act_of.get(int(j), -1) for j in jid])
            if (act < 0).any():
                continue
            self.chains[side] = {
                "jid": jid, "acts": act,
                "names": [f"{side}_{j}_joint" for j in self.ARM],
                "qadr": m.jnt_qposadr[jid],
                "dofs": m.jnt_dofadr[jid],
                "lo": m.jnt_range[jid, 0],
                "hi": m.jnt_range[jid, 1],
                "site": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE,
                                          f"{side}_palm"),
                "q_nom": ex.stand_qpos[m.jnt_qposadr[jid]].copy(),
                "path": [], "target": None, "carrot": None,
                "q_goal": None, "tol": 0.04,
            }

    # ---- the policy's internal model-based computation (scratch state)
    def _solve_from(self, ch, xyz, seed, iters):
        mujoco = _mujoco()
        m, s = self.ex.model, self._scratch
        s.qpos[:] = self.ex.data.qpos
        s.qpos[ch["qadr"]] = seed
        jacp = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, s)
            mujoco.mj_comPos(m, s)
            err = np.asarray(xyz) - s.site_xpos[ch["site"]]
            if np.linalg.norm(err) < 0.003:
                break
            mujoco.mj_jacSite(m, s, jacp, None, ch["site"])
            J = jacp[:, ch["dofs"]]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.005 * np.eye(3), err)
            dq += 0.005 * (ch["q_nom"] - s.qpos[ch["qadr"]])
            s.qpos[ch["qadr"]] = np.clip(
                s.qpos[ch["qadr"]] + np.clip(dq, -0.2, 0.2),
                ch["lo"], ch["hi"])
        mujoco.mj_kinematics(m, s)
        return s.qpos[ch["qadr"]].copy(), float(
            np.linalg.norm(xyz - s.site_xpos[ch["site"]]))

    def _solve(self, side, xyz, iters=150):
        ch = self.chains[side]
        roll = -0.15 if side == "right" else 0.15
        seeds = [ch["q_nom"],
                 np.array([-1.2, roll, 0, 0.4, 0, 0, 0]),
                 np.array([-0.8, roll, 0, 0.9, 0, 0, 0]),
                 np.array([-1.5, 2 * roll, 0, 0.2, 0, 0, 0])]
        best = None
        for seed in seeds:
            q, e = self._solve_from(
                ch, xyz, np.clip(seed, ch["lo"], ch["hi"]), iters)
            if best is None or e < best[1]:
                best = (q, e)
            if e < 0.005:
                break
        return best

    def _begin(self, side):
        ch = self.chains[side]
        ch["carrot"] = self.ex.data.site_xpos[ch["site"]].copy()
        ch["q_goal"], _ = self._solve(side, ch["target"])

    def set_path(self, side, points, tol=0.04):
        """Policy intent: follow these world waypoints with this palm."""
        ch = self.chains[side]
        pts = [np.asarray(p, float) for p in points]
        ch["target"], ch["path"], ch["tol"] = pts[0], pts[1:], tol
        self._begin(side)

    def _track(self, ch, xyz, iters=12):
        mujoco = _mujoco()
        m, s = self.ex.model, self._scratch
        s.qpos[:] = self.ex.data.qpos
        jacp = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, s)
            mujoco.mj_comPos(m, s)
            err = xyz - s.site_xpos[ch["site"]]
            if np.linalg.norm(err) < 0.003:
                break
            mujoco.mj_jacSite(m, s, jacp, None, ch["site"])
            J = jacp[:, ch["dofs"]]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.005 * np.eye(3), err)
            dq += 0.02 * (ch["q_goal"] - s.qpos[ch["qadr"]])
            s.qpos[ch["qadr"]] = np.clip(
                s.qpos[ch["qadr"]] + np.clip(dq, -0.1, 0.1),
                ch["lo"], ch["hi"])
        return s.qpos[ch["qadr"]].copy()

    def act(self, observation=None):
        """Emit the next joint_pos action for both arms.

        The observation is the live measured state (the policy is bound
        to the embodiment); the action is a {joint: position target}
        dict — pure joint actions, the only kind the @2 line allows."""
        mujoco = _mujoco()
        d = self.ex.data
        out = {}
        jacp = np.zeros((3, self.ex.model.nv))
        for side, ch in self.chains.items():
            if ch["target"] is None:
                continue
            palm = d.site_xpos[ch["site"]]
            err = ch["target"] - palm
            if ch["path"] and np.linalg.norm(err) < ch["tol"]:
                ch["target"], ch["path"] = ch["path"][0], ch["path"][1:]
                self._begin(side)
            seg = ch["target"] - ch["carrot"]
            n = float(np.linalg.norm(seg))
            if n > 1e-6 and np.linalg.norm(ch["carrot"] - palm) < 0.06:
                ch["carrot"] = ch["carrot"] + seg / n * min(
                    n, self.SPEED * self.DT)
            q_cmd = self._track(ch, ch["carrot"])
            cur = d.ctrl[ch["acts"]]
            tgt = np.clip(cur + np.clip(q_cmd - cur, -0.04, 0.04),
                          ch["lo"], ch["hi"])
            if n <= 1e-6 and np.linalg.norm(err) <= 0.03:
                # at the final point: fine closed-loop correction on the
                # measured palm — joints pinned at a limit cannot help
                q = d.qpos[ch["qadr"]]
                mujoco.mj_jacSite(self.ex.model, d, jacp, None, ch["site"])
                J = jacp[:, ch["dofs"]]
                free = ~(((q <= ch["lo"] + 0.02) & (J.T @ err < 0)) |
                         ((q >= ch["hi"] - 0.02) & (J.T @ err > 0)))
                Jf = J * free
                Jpinv = Jf.T @ np.linalg.inv(
                    Jf @ Jf.T + 0.01 * np.eye(3))
                dq = Jpinv @ err
                dq += (np.eye(7) - Jpinv @ Jf) @ (0.3 * (ch["q_goal"] - q))
                dq = np.clip(dq, -0.03, 0.03)         # <= 1.5 rad/s
                cmd = np.clip(q + dq, ch["lo"], ch["hi"])
                tgt = np.clip(cmd + np.clip(cur - q, -0.05, 0.05) * 0.5,
                              ch["lo"], ch["hi"])
            for name, v in zip(ch["names"], tgt):
                out[name] = float(v)
        return out

    def error(self, side):
        ch = self.chains[side]
        if ch["target"] is None:
            return 0.0
        return float(np.linalg.norm(
            ch["target"] - self.ex.data.site_xpos[ch["site"]]))

    def busy(self, side):
        ch = self.chains[side]
        return bool(ch["path"]) or self.error(side) > ch["tol"]


def _drive(ex, pol, seconds, sides=("left", "right")):
    """Run the policy: each tick it emits a joint_pos action, the
    executor applies it with set_joints(), physics steps 20 ms — the
    arms move under policy joint actions at the 50 Hz action cadence."""
    n_tick = max(1, int(round(pol.DT / ex.model.opt.timestep)))
    t0 = float(ex.data.time)
    while float(ex.data.time) - t0 < seconds:
        ex.set_joints(pol.act())
        ex.step(n_tick)
        if not any(pol.busy(s) for s in sides if s in pol.chains):
            break
    return {s: pol.error(s) for s in sides if s in pol.chains}


def _pregrasp_targets(ex, name, half_span=0.10):
    """Palm waypoints beside the object's left and right flanks at
    mid-height: rise to flank height while still in the aisle (south of
    the worktop edge — no forearm reaches across the counter), then come
    in a few cm off the sides so the closing fingers can wrap the body
    (a grip above the rim pinches air; a grip at the equator wraps it).
    Geometry the caller plans; the policy flies it."""
    info = ex.geom_info(name)
    lo, hi = np.array(info["aabb_min"]), np.array(info["aabb_max"])
    c = (lo + hi) / 2
    R = ex.data.xmat[ex.pelvis_body].reshape(3, 3)
    left = R[:, 1].copy()
    left[2] = 0.0
    left /= max(np.linalg.norm(left), 1e-9)
    fwd = np.cross(left, np.array([0.0, 0.0, 1.0]))
    z = float(c[2]) + 0.02
    paths = {}
    for s, sgn in (("left", 1.0), ("right", -1.0)):
        flank = np.r_[c[:2] + left[:2] * (sgn * half_span), z]
        paths[s] = [np.r_[_palm(ex, s)[:2], z],
                    flank - fwd * 0.20,
                    flank]
    return paths


def _surface_gaps(ex, name):
    """Measured distance of each palm to the object's collision AABB
    surface (0 when touching/inside)."""
    info = ex.geom_info(name)
    lo, hi = np.array(info["aabb_min"]), np.array(info["aabb_max"])
    out = {}
    for s in ("left", "right"):
        p = _palm(ex, s)
        out[s] = float(np.linalg.norm(p - np.clip(p, lo, hi)))
    return out


def _watch(ex, name):
    """Wrap ex.step: record, per physics step, the object's world position
    and the activation state of every equality constraint that involves
    its body. A "snap" is flagged only when a single-step jump lands the
    object essentially ON the palms' midpoint — the teleport's signature;
    a contact-driven squeeze ejection can also displace it fast but lands
    elsewhere, and is honest (failed) physics, not a snap. Returns the
    log dict; detach with ex.step = log['step']."""
    mujoco = _mujoco()
    m, d = ex.model, ex.data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    prop_eqs = [i for i in range(m.neq)
                if int(m.eq_objtype[i]) == int(mujoco.mjtObj.mjOBJ_BODY)
                and bid in (int(m.eq_obj1id[i]), int(m.eq_obj2id[i]))]
    log = {"step": ex.step, "jump": 0.0, "prop_eq": set(), "snap": False,
           "last": d.xpos[bid].copy(), "eq_ids": prop_eqs}
    orig = ex.step

    def step(n=1):
        orig(n)
        p = d.xpos[bid]
        mid = (_palm(ex, "left") + _palm(ex, "right")) / 2
        step_m = float(np.linalg.norm(p - log["last"]))
        if step_m > JUMP_M and np.linalg.norm(p - mid) < 0.05:
            log["snap"] = True
        log["jump"] = max(log["jump"], step_m)
        log["last"] = p.copy()
        for i in prop_eqs:
            if d.eq_active[i]:
                log["prop_eq"].add(i)

    ex.step = step
    return log


def _eq_name(ex, i):
    mujoco = _mujoco()
    return mujoco.mj_id2name(ex.model, mujoco.mjtObj.mjOBJ_EQUALITY, i) \
        or f"eq{i}"


def _finger_joints(ex):
    """{(side, finger): [joint ids]} for the dexterous hand chains."""
    mujoco = _mujoco()
    out = {}
    for j in range(ex.model.njnt):
        nm = mujoco.mj_id2name(ex.model, mujoco.mjtObj.mjOBJ_JOINT, j) \
            or ""
        mm = FINGER_JNT.match(nm)
        if mm:
            out.setdefault((mm.group(1).lower(), mm.group(2).lower()),
                           []).append(j)
    return out


def _actuated_joints(ex):
    return {int(ex.model.actuator_trnid[a, 0])
            for a in range(ex.model.nu)}


def _finger_q(ex, fingers):
    return {(s, f): np.array([ex.data.qpos[ex.model.jnt_qposadr[j]]
                              for j in js])
            for (s, f), js in fingers.items()}


def _hand_contacts(ex, name):
    """Measured contact between hand/finger links and the object's geoms:
    {(side, link): total normal force N}."""
    mujoco = _mujoco()
    m, d = ex.model, ex.data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    hand_of = {}
    for b in range(m.nbody):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        mm = HAND_BODY.match(nm)
        if mm:
            hand_of[b] = (mm.group(1).lower(), mm.group(2).lower())
    obj_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == bid}
    f6 = np.zeros(6)
    hits = {}
    for c in range(d.ncon):
        con = d.contact[c]
        for g_h, g_o in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
            bh = int(m.geom_bodyid[g_h])
            if bh not in hand_of or int(g_o) not in obj_geoms:
                continue
            mujoco.mj_contactForce(m, d, c, f6)
            k = hand_of[bh]
            hits[k] = hits.get(k, 0.0) + abs(float(f6[0]))
    return hits


def test_ctrl_2_1_policy_arms_grasp_three_fingers_contact_no_weld(
        tmp_path):
    ex = _executor(tmp_path)
    m, d = ex.model, ex.data

    # --- the robot in the scene must actually carry the dexterous hands:
    # three finger chains (thumb/index/middle), each actuator-driven, on
    # at least one hand — grasp() has no fingers to close without them
    fingers = _finger_joints(ex)
    sides = sorted({s for s, _ in fingers})
    three = [s for s in sides
             if all((s, f) in fingers for f in FINGERS)]
    assert three, (
        "the compiled scene has no three-fingered G1 hand — no "
        "{side}_hand_{thumb,index,middle} joints exist, so grasp() has "
        f"no fingers to close on the object (saw hand joints: "
        f"{sorted(fingers) or 'none'})")
    actuated = _actuated_joints(ex)
    driven = [s for s in three
              if all(any(j in actuated for j in fingers[s, f])
                     for f in FINGERS)]
    assert driven, (
        f"the finger chains on {three} are present but have no actuator "
        "in the scene — grasp() cannot drive them closed")

    # --- grasp is a program() bridge verb
    res = ex.program("grasp")
    assert "error" not in res, (
        f"grasp is not bound in the program() bridge: {res.get('error')}")

    # --- tripwire every scripted IK reach path: the @2 clause allows
    # only policy joint actions to move the arms in this trial
    fired = _tripwire(ex)

    # --- arrange the trial: apple on the island's south edge, robot
    # stood square in front of it facing it and leaning in at the waist
    # (embodiment posture, not an arm reach — the @2 clause binds the
    # arms). Setup state, then physics owns everything.
    _park_person(ex)
    _teleport(ex, "apple", [0.15, -0.93, 1.10])
    _teleport(ex, "pelvis", [0.15, -1.25, 0.793], yaw=math.pi / 2)
    ex.run_seconds(0.5)
    ex.set("waist_pitch_joint", 0.35)
    ex.run_seconds(1.0)

    # --- the pregrasp, under policy joint actions alone: the policy
    # emits joint_pos targets each 20 ms tick and the executor merely
    # applies them — no scripted IK reach may engage
    pol = _ArmJointPolicy(ex)
    assert set(pol.chains) == {"left", "right"}, (
        "the policy found no actuated arm chains to drive")
    paths = _pregrasp_targets(ex, "apple")
    for s, pts in paths.items():
        pol.set_path(s, pts)
    errs = _drive(ex, pol, 9.0)
    gaps = _surface_gaps(ex, "apple")
    assert min(gaps.values()) < 0.10, (
        f"the policy-driven pregrasp left the palms {gaps} from the "
        f"apple's surface (palm errors {errs}) — no hand is inside the "
        "grasp gate, so the trial cannot start")

    # --- the call under test, with per-step instrumentation
    q0 = _finger_q(ex, fingers)
    log = _watch(ex, "apple")
    inner = ex.step
    qmin = {k: v.copy() for k, v in q0.items()}
    qmax = {k: v.copy() for k, v in q0.items()}

    def step(n=1):
        inner(n)
        for k, js in fingers.items():
            q = np.array([d.qpos[m.jnt_qposadr[j]] for j in js])
            qmin[k] = np.minimum(qmin[k], q)
            qmax[k] = np.maximum(qmax[k], q)

    ex.step = step
    res = ex.grasp("apple")
    assert isinstance(res, dict) and "error" not in res, (
        f"grasp('apple') failed from a valid pregrasp: {res}")
    ex.run_seconds(0.4)   # let the hold settle under physics

    # --- no snap: the apple never teleported between the palms
    assert not log["snap"], (
        f"the apple jumped {log['jump']:.3f} m inside one physics step "
        "toward the palms' midpoint during grasp() — a snap between the "
        "palms is a kinematic teleport, not finger motion")

    # --- no weld: no equality constraint on the apple may ever activate
    assert not log["prop_eq"], (
        f"grasp() activated equality weld(s) "
        f"{[_eq_name(ex, i) for i in sorted(log['prop_eq'])]} on the "
        "apple — the line requires contact forces only")
    for i in log["eq_ids"]:
        assert not d.eq_active[i], (
            f"equality {_eq_name(ex, i)!r} is still active after "
            "grasp() — the hold is welded, not contact")

    # --- the three fingers were really driven onto the apple: each
    # group on some hand moved measurably during the grasp (mid-motion
    # excursion, so an open-then-return wiggle does not count)
    moved = {k: float(np.maximum(qmax[k] - q0[k], q0[k] - qmin[k]).max())
             for k in fingers}
    per_finger = {f"{k[0]}_{k[1]}": round(v, 3) for k, v in moved.items()}
    assert any(all(moved.get((s, f), 0.0) >= 0.08 for f in FINGERS)
               for s in driven), (
        f"grasp() did not drive all three fingers of either hand onto "
        f"the apple (max finger excursion per finger: {per_finger})")

    # --- the hold is measured contact: >= two distinct finger links
    # touching the apple with real normal force
    hits = _hand_contacts(ex, "apple")
    f_hits = {k: v for k, v in hits.items()
              if k[1] in FINGERS and v > 0.05}
    f_report = {f"{k[0]}_{k[1]}": round(v, 2) for k, v in f_hits.items()}
    assert len({k[1] for k in f_hits}) >= 2 and \
        sum(f_hits.values()) >= 0.5, (
            f"finger contact on the apple is {f_report or 'none'} — "
            "three-fingered contact, not a weld, must hold it")

    # --- prove the hold physically: the policy lifts both palms 8 cm
    # with the same joint_pos actions; the apple must come up off the
    # worktop carried by contact alone
    z0 = float(ex.pos("apple")[2])
    for s in ("right", "left"):
        p = _palm(ex, s)
        pol.set_path(s, [p + np.array([0.0, 0.0, 0.08])], tol=0.04)
    _drive(ex, pol, 4.0)
    ex.run_seconds(0.4)
    z1 = float(ex.pos("apple")[2])
    assert z1 >= z0 + 0.025, (
        f"the palms rose 8 cm but the apple moved {z1 - z0:+.3f} m — "
        "it was resting on the counter, not held by the fingers")
    near = min(float(np.linalg.norm(ex.pos("apple") - _palm(ex, s)))
               for s in ("right", "left"))
    assert near < 0.30, (
        f"the lifted apple sits {near:.3f} m from the nearest palm — "
        "it is not in the hand")
    air_hits = _hand_contacts(ex, "apple")
    assert any(k[1] in FINGERS and v > 0.05
               for k, v in air_hits.items()), (
        "the airborne apple touches no finger link — whatever holds it "
        "is not finger contact")
    assert not log["prop_eq"], (
        f"an equality weld activated late: "
        f"{[_eq_name(ex, i) for i in sorted(log['prop_eq'])]}")
    assert not log["snap"], (
        f"the apple teleported {log['jump']:.3f} m in one step toward "
        "the palms' midpoint while being lifted")

    # --- the whole trial ran on policy joint actions: no scripted IK
    # reach entry point ever fired and the executor's reacher registry
    # stayed empty
    assert not fired, (
        f"scripted IK reach ran during the trial: {sorted(set(fired))} "
        "— the @2 line requires the arms driven by policy joint "
        "actions only")
    assert not getattr(ex, "reacher", None), (
        "the executor engaged its scripted arm reacher during the "
        "trial — the arms must be driven by policy joint actions")
    ex.step = log["step"]
