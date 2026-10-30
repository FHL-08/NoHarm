"""CTRL-2.2@2 At authored bowl pose, model lifts with joint actions and
finger contact, carries bowl 0.3 m under 0.2 m/s walk command, then releases
into gravity drop; no fixed waypoints, weld, snap, or teleport.

Contract: run GPT-6-astra through harmless.agent.run_episode on untouched
authored scene state. Inspect emitted joint actions, walker commands, MuJoCo
contacts, prop motion, equality state, and the physical drop. The test never
supplies a policy, arm trajectory, object pose, or mock model.
"""

import ast
import json
import os
import re
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]
SCENE = REPO / "kitchen.xml"
HAND_BODY = re.compile(
    r"^(left|right)_hand_(thumb|index|middle)(?:_|$)", re.I)
FINGERS = ("thumb", "index", "middle")
FORBIDDEN_MOTION = {
    "arm_reach", "arm_path", "hands_path", "reach", "reach_xyz",
    "pick_up", "put_down", "move_joints",
}


def _mujoco():
    import mujoco
    return mujoco


def _body_tree(model, root):
    result = set()
    for body in range(model.nbody):
        parent = body
        while parent >= 0:
            if parent == root:
                result.add(body)
                break
            parent = int(model.body_parentid[parent])
    return result


def _actuated_fingers(model, mujoco):
    actuated = {int(model.actuator_trnid[a, 0]) for a in range(model.nu)}
    groups = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        match = re.match(
            r"^(left|right)_hand_(thumb|index|middle)(?:_|$)", name, re.I)
        if match and jid in actuated:
            groups.setdefault(match.group(1).lower(), set()).add(
                match.group(2).lower())
    return groups


def _preflight(tmp_path):
    from harmless.executor import Executor
    mujoco = _mujoco()
    ex = Executor(str(SCENE), record=False,
                  out_dir=str(tmp_path / "preflight"), skills_path=None,
                  ws_dir=str(tmp_path / "ws"))
    try:
        bid = mujoco.mj_name2id(ex.model, mujoco.mjtObj.mjOBJ_BODY, "bowl")
        assert bid >= 0, "authored kitchen scene has no bowl body"
        authored_pos = ex.data.xpos[bid].copy()
        fingers = _actuated_fingers(ex.model, mujoco)
        assert any(set(FINGERS) <= groups for groups in fingers.values()), (
            "G1 scene has no actuated thumb, index, and middle finger "
            "chains; model cannot lift bowl by finger contact"
        )
        return authored_pos
    finally:
        ex.close()


def _hand_contacts(ex, object_bodies):
    mujoco = _mujoco()
    model, data = ex.model, ex.data
    hand_links = {}
    for body in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        match = HAND_BODY.match(name)
        if match:
            hand_links[body] = match.group(2).lower()
    object_geoms = {geom for geom in range(model.ngeom)
                    if int(model.geom_bodyid[geom]) in object_bodies}
    forces = {finger: 0.0 for finger in FINGERS}
    wrench = np.zeros(6)
    for ci in range(data.ncon):
        con = data.contact[ci]
        for hand_geom, object_geom in ((con.geom1, con.geom2),
                                       (con.geom2, con.geom1)):
            finger = hand_links.get(int(model.geom_bodyid[hand_geom]))
            if finger is None or int(object_geom) not in object_geoms:
                continue
            mujoco.mj_contactForce(model, data, ci, wrench)
            forces[finger] += abs(float(wrench[0]))
    return forces


def _watch_executor(tracker):
    from harmless.executor import Executor
    original_init = Executor.__init__
    original_step = Executor.step
    original_program = Executor.program
    original_set_joints = Executor.set_joints
    original_walk = Executor.walk
    original_release = Executor.release
    original_close = Executor.close

    def bind(self):
        mujoco = _mujoco()
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bowl")
        tracker["executor"] = self
        tracker["body"] = bid
        tracker["bodies"] = _body_tree(self.model, bid)
        tracker["last"] = self.data.xpos[bid].copy()
        tracker["start"] = tracker["last"].copy()
        tracker["release_pos"] = None
        tracker["release_z"] = None
        tracker["release_drop_speed"] = 0.0
        tracker["final_pos"] = None
        tracker["eqs"] = []
        for ei in range(self.model.neq):
            ids = (int(self.model.eq_obj1id[ei]), int(self.model.eq_obj2id[ei]))
            if int(self.model.eq_objtype[ei]) == int(mujoco.mjtObj.mjOBJ_BODY) \
                    and any(i in tracker["bodies"] for i in ids):
                tracker["eqs"].append(ei)
        tracker["active_eqs"] = set()
        tracker["positions"] = [tracker["last"].copy()]
        tracker["max_jump"] = 0.0
        tracker["finger_forces"] = {finger: 0.0 for finger in FINGERS}
        tracker["walk_commands"] = []
        tracker["joint_targets"] = []

    def observe(self, boundary=False):
        if tracker.get("executor") is not self:
            return
        pos = self.data.xpos[tracker["body"]].copy()
        tracker["max_jump"] = max(
            tracker["max_jump"], float(np.linalg.norm(pos - tracker["last"])))
        tracker["last"] = pos
        tracker["positions"].append(pos)
        for ei in tracker["eqs"]:
            if self.data.eq_active[ei]:
                tracker["active_eqs"].add(ei)
        if tracker["release_pos"] is not None:
            free = self.model.jnt_dofadr[
                self.model.body_jntadr[tracker["body"]]]
            speed = -float(self.data.qvel[free + 2])
            tracker["release_drop_speed"] = max(
                tracker["release_drop_speed"], speed)
        if not boundary:
            forces = _hand_contacts(self, tracker["bodies"])
            for finger, force in forces.items():
                tracker["finger_forces"][finger] = max(
                    tracker["finger_forces"][finger], force)

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        bind(self)

    def step(self, n=1):
        if tracker.get("executor") is not self:
            return original_step(self, n)
        for _ in range(int(n)):
            original_step(self, 1)
            observe(self)

    def program(self, code):
        observe(self, boundary=True)
        result = original_program(self, code)
        observe(self, boundary=True)
        return result

    def set_joints(self, targets):
        if tracker.get("executor") is self:
            tracker["joint_targets"].append(dict(targets))
        return original_set_joints(self, targets)

    def walk(self, vx=0.0, vy=0.0, yaw_rate=0.0):
        if tracker.get("executor") is self:
            tracker["walk_commands"].append(
                (float(vx), float(vy), float(yaw_rate)))
        return original_walk(self, vx, vy, yaw_rate)

    def release(self):
        if tracker.get("executor") is self:
            contact = _hand_contacts(self, tracker["bodies"])
            if self.held == "bowl" or max(contact.values()) > 0.05:
                tracker["release_pos"] = self.data.xpos[tracker["body"]].copy()
                tracker["release_z"] = float(tracker["release_pos"][2])
        result = original_release(self)
        if tracker.get("executor") is self and tracker["release_pos"] is not None:
            self.run_seconds(1.5)
            tracker["final_pos"] = self.data.xpos[tracker["body"]].copy()
        return result

    def close(self):
        if tracker.get("executor") is self:
            observe(self, boundary=True)
            tracker["final_pos"] = self.data.xpos[tracker["body"]].copy()
        return original_close(self)

    Executor.__init__ = init
    Executor.step = step
    Executor.program = program
    Executor.set_joints = set_joints
    Executor.walk = walk
    Executor.release = release
    Executor.close = close
    return [(Executor, "__init__", original_init),
            (Executor, "step", original_step),
            (Executor, "program", original_program),
            (Executor, "set_joints", original_set_joints),
            (Executor, "walk", original_walk),
            (Executor, "release", original_release),
            (Executor, "close", original_close)]


def _restore(patched):
    for cls, name, value in patched:
        setattr(cls, name, value)


def _model_actions(transcript):
    programs = []
    for turn in transcript:
        call = turn.get("call", {})
        if call.get("skill") != "program":
            continue
        code = call.get("args", {}).get("code", "")
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise AssertionError(f"GPT-6-astra emitted invalid code: {exc}")
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        names = {node.func.id if isinstance(node.func, ast.Name)
                 else node.func.attr if isinstance(node.func, ast.Attribute)
                 else "" for node in calls}
        assert not names.intersection(FORBIDDEN_MOTION), (
            "model used scripted IK or fixed-waypoint motion: "
            f"{sorted(names.intersection(FORBIDDEN_MOTION))}"
        )
        observed = set()
        assignments = []
        loops = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                assignments.append((node.targets, node.value))
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr,
                                   ast.AugAssign)):
                assignments.append(([node.target], node.value))
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                loops.append((node.target, node.iter))
            elif isinstance(node, ast.comprehension):
                loops.append((node.target, node.iter))
        observation_names = {"joint_state", "state", "sense", "qpos", "qvel"}
        for _ in range(len(assignments) + len(loops) + 1):
            prior = set(observed)
            for targets, value in assignments:
                value_names = {n.id for n in ast.walk(value)
                               if isinstance(n, ast.Name)}
                value_calls = {n.func.id if isinstance(n.func, ast.Name)
                               else n.func.attr if isinstance(n.func, ast.Attribute)
                               else "" for n in ast.walk(value)
                               if isinstance(n, ast.Call)}
                if value_names.intersection(observed) or \
                        value_calls.intersection(observation_names):
                    for target in targets:
                        observed.update(_target_names(target))
            for target, iterator in loops:
                iter_names = {n.id for n in ast.walk(iterator)
                              if isinstance(n, ast.Name)}
                iter_calls = {n.func.id if isinstance(n.func, ast.Name)
                              else n.func.attr if isinstance(n.func, ast.Attribute)
                              else "" for n in ast.walk(iterator)
                              if isinstance(n, ast.Call)}
                if iter_names.intersection(observed) or \
                        iter_calls.intersection(observation_names):
                    observed.update(_target_names(target))
            if prior == observed:
                break
        feedback = False
        for node in calls:
            if _call_name(node.func) != "set_joints" or not node.args:
                continue
            value = node.args[0]
            value_names = {n.id for n in ast.walk(value)
                           if isinstance(n, ast.Name)}
            value_calls = {n.func.id if isinstance(n.func, ast.Name)
                           else n.func.attr if isinstance(n.func, ast.Attribute)
                           else "" for n in ast.walk(value)
                           if isinstance(n, ast.Call)}
            feedback |= bool(value_names.intersection(observed) or
                             value_calls.intersection(observation_names))
        if feedback:
            programs.append(code)
    return programs


def _target_names(node):
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Subscript):
        return _target_names(node.value)
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(item) for item in node.elts))
    return set()


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def test_ctrl_2_2_model_carries_and_drops_authored_bowl_by_contact(tmp_path):
    from harmless.executor import Executor
    mujoco = _mujoco()
    preflight = Executor(str(SCENE), record=False,
                         out_dir=str(tmp_path / "preflight"),
                         skills_path=None, ws_dir=str(tmp_path / "ws"))
    try:
        bid = mujoco.mj_name2id(preflight.model,
                                mujoco.mjtObj.mjOBJ_BODY, "bowl")
        assert bid >= 0, "authored kitchen scene has no bowl body"
        authored_pos = preflight.data.xpos[bid].copy()
        fingers = _actuated_fingers(preflight.model, mujoco)
        assert any(set(FINGERS) <= groups for groups in fingers.values()), (
            "G1 scene has no actuated thumb, index, and middle finger "
            "chains; model cannot lift bowl by finger contact"
        )
    finally:
        preflight.close()

    tracker = {}
    patched = _watch_executor(tracker)
    try:
        from harmless.agent import run_episode
        out_dir = tmp_path / "bowl_episode"
        run_episode(
            task=(
                "At the untouched authored kitchen pose, pick up the bowl. "
                "Observe joint_state(), state(), pos(), geom_info(), and "
                "contacts(). Use program() with observation-driven "
                "set_joints() targets to close the fingers on the bowl. "
                "Carry it at a 0.2 m/s walk command until its measured "
                "horizontal travel reaches at least 0.3 m, then release it "
                "over open floor and let gravity drop it. Do not use "
                "arm_reach, arm_path, hands_path, reach, reach_xyz, pick_up, "
                "put_down, move_joints, fixed waypoint lists, welds, snaps, "
                "or object pose edits."
            ),
            max_steps=14,
            budget_s=1200,
            out_dir=str(out_dir),
            live=False,
            model="gpt-6-astra",
            memory_path=str(tmp_path / "no_memory.md"),
            skills_path=None,
            scene=str(SCENE),
        )
        transcript = json.loads((out_dir / "transcript.json").read_text())
    finally:
        _restore(patched)

    assert tracker["executor"] is not None, "agent loop created no executor"
    assert np.allclose(tracker["start"], authored_pos, atol=0.005), (
        "agent run changed bowl from authored scene pose before control"
    )
    assert _model_actions(transcript), (
        "GPT-6-astra emitted no observation-driven set_joints() program"
    )
    assert tracker["joint_targets"], "model emitted no joint-position actions"
    assert max(tracker["finger_forces"].values()) > 0.05, (
        "MuJoCo measured no finger-to-bowl contact: "
        f"{tracker['finger_forces']}"
    )
    assert tracker["walk_commands"], "model never commanded the learned walker"
    speeds = [float(np.linalg.norm(command[:2]))
              for command in tracker["walk_commands"]]
    assert any(abs(speed - 0.2) <= 0.02 for speed in speeds), (
        f"model never commanded a 0.2 m/s walk: {speeds}"
    )
    assert tracker["release_pos"] is not None, (
        "model never released the grasped bowl"
    )
    horizontal_travel = float(np.linalg.norm(
        tracker["release_pos"][:2] - tracker["start"][:2]))
    assert horizontal_travel >= 0.3, (
        f"bowl traveled {horizontal_travel:.3f} m before release; needs 0.3 m"
    )
    assert tracker["release_drop_speed"] > 0.8, (
        "bowl did not gain downward speed under gravity after release: "
        f"{tracker['release_drop_speed']:.3f} m/s"
    )
    assert tracker["final_pos"] is not None and tracker["final_pos"][2] < 0.2, (
        f"released bowl did not fall to floor: {tracker['final_pos']}"
    )
    assert not tracker["active_eqs"], (
        f"bowl grasp activated equality constraints {tracker['active_eqs']}"
    )
    assert tracker["max_jump"] < 0.025, (
        f"bowl moved {tracker['max_jump']:.3f} m in one physics step or "
        "program boundary"
    )
