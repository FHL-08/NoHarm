"""CTRL-2.1@3 At authored apple pose, foundation model emits joint-position
actions from observations; grasp('apple') lifts by three-finger contact,
without scripted IK, authored waypoints, teleport, or weld.

Contract: run GPT-6-astra through harmless.agent.run_episode on the untouched
kitchen keyframe. Inspect emitted program code and measured MuJoCo state.
The test never supplies a policy, arm trajectory, object pose, or mock model.
"""

import ast
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]
SCENE = REPO / "kitchen.xml"
FINGERS = ("thumb", "index", "middle")
HAND_BODY = re.compile(
    r"^(left|right)_hand_(thumb|index|middle)(?:_|$)", re.I)
FORBIDDEN_MOTION = {
    "arm_reach", "arm_path", "hands_path", "reach", "reach_xyz",
    "pick_up", "put_down", "move_joints",
}


def _mujoco():
    import mujoco
    return mujoco


def _executor(tmp_path):
    from harmless.executor import Executor
    return Executor(str(SCENE), record=False,
                    out_dir=str(tmp_path / "preflight"),
                    skills_path=None, ws_dir=str(tmp_path / "ws"))


def _body_tree(model, root):
    return {body for body in range(model.nbody)
            if _is_descendant(model, body, root)}


def _is_descendant(model, body, root):
    while body >= 0:
        if body == root:
            return True
        body = int(model.body_parentid[body])
    return False


def _actuated_fingers(model, mujoco):
    actuated = {int(model.actuator_trnid[a, 0]) for a in range(model.nu)}
    groups = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        match = re.match(
            r"^(left|right)_hand_(thumb|index|middle)(?:_|$)", name, re.I)
        if match and jid in actuated:
            side, finger = match.group(1).lower(), match.group(2).lower()
            groups.setdefault(side, set()).add(finger)
    return groups


def _preflight(tmp_path):
    mujoco = _mujoco()
    ex = _executor(tmp_path)
    try:
        bid = mujoco.mj_name2id(ex.model, mujoco.mjtObj.mjOBJ_BODY, "apple")
        assert bid >= 0, "authored kitchen scene has no apple body"
        authored_pos = ex.data.xpos[bid].copy()
        fingers = _actuated_fingers(ex.model, mujoco)
        assert any(set(FINGERS) <= groups for groups in fingers.values()), (
            "G1 scene has no actuated thumb, index, and middle finger "
            "chains; foundation model cannot make three-finger contact"
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
    original_grasp = Executor.grasp
    original_program = Executor.program
    original_set_joints = Executor.set_joints
    original_close = Executor.close

    def bind(self):
        mujoco = _mujoco()
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "apple")
        tracker["executor"] = self
        tracker["body"] = bid
        tracker["bodies"] = _body_tree(self.model, bid)
        tracker["last"] = self.data.xpos[bid].copy()
        tracker["start"] = tracker["last"].copy()

        eqs = []
        for ei in range(self.model.neq):
            ids = (int(self.model.eq_obj1id[ei]), int(self.model.eq_obj2id[ei]))
            if int(self.model.eq_objtype[ei]) == int(mujoco.mjtObj.mjOBJ_BODY) \
                    and any(i in tracker["bodies"] for i in ids):
                eqs.append(ei)
        tracker["eqs"] = eqs
        tracker["finger_forces"] = {finger: 0.0 for finger in FINGERS}
        tracker["lifted_three_finger"] = False
        tracker["active_eqs"] = set()
        tracker["positions"] = [tracker["last"].copy()]
        tracker["max_jump"] = 0.0
        tracker["grasp_calls"] = []
        tracker["joint_targets"] = []

    def observe(self, boundary=False):
        if tracker.get("executor") is not self:
            return
        body = tracker["body"]
        pos = self.data.xpos[body].copy()
        tracker["max_jump"] = max(
            tracker["max_jump"], float(np.linalg.norm(pos - tracker["last"])))
        tracker["last"] = pos
        tracker["positions"].append(pos)
        for ei in tracker["eqs"]:
            if self.data.eq_active[ei]:
                tracker["active_eqs"].add(ei)
        if not boundary:
            forces = _hand_contacts(self, tracker["bodies"])
            for finger, force in forces.items():
                tracker["finger_forces"][finger] = max(
                    tracker["finger_forces"][finger], force)
            if pos[2] >= tracker["start"][2] + 0.04 and all(
                    force > 0.05 for force in forces.values()):
                tracker["lifted_three_finger"] = True

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        bind(self)

    def step(self, n=1):
        if tracker.get("executor") is not self:
            return original_step(self, n)
        for _ in range(int(n)):
            original_step(self, 1)
            observe(self)

    def grasp(self, name, *args, **kwargs):
        if tracker.get("executor") is self and name == "apple":
            tracker["grasp_calls"].append(name)
        result = original_grasp(self, name, *args, **kwargs)
        observe(self, boundary=True)
        return result

    def program(self, code):
        observe(self, boundary=True)
        result = original_program(self, code)
        observe(self, boundary=True)
        return result

    def set_joints(self, targets):
        if tracker.get("executor") is self:
            tracker["joint_targets"].append(dict(targets))
        return original_set_joints(self, targets)

    def close(self):
        if tracker.get("executor") is self:
            observe(self, boundary=True)
        return original_close(self)

    Executor.__init__ = init
    Executor.step = step
    Executor.grasp = grasp
    Executor.program = program
    Executor.set_joints = set_joints
    Executor.close = close
    return [(Executor, "__init__", original_init),
            (Executor, "step", original_step), (Executor, "grasp", original_grasp),
            (Executor, "program", original_program),
            (Executor, "set_joints", original_set_joints),
            (Executor, "close", original_close)]


def _restore(patched):
    for cls, name, value in patched:
        setattr(cls, name, value)


def _model_actions(transcript):
    programs = []
    for turn in transcript:
        call = turn.get("call", {})
        if call.get("skill") == "program":
            code = call.get("args", {}).get("code", "")
            try:
                tree = ast.parse(code)
            except SyntaxError as exc:
                pytest.fail(f"GPT-6-astra emitted invalid program code: {exc}")
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            names = {node.func.id if isinstance(node.func, ast.Name)
                     else node.func.attr if isinstance(node.func, ast.Attribute)
                     else "" for node in calls}
            assert not names.intersection(FORBIDDEN_MOTION), (
                "foundation model used scripted IK or fixed-waypoint motion: "
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


def test_ctrl_2_1_model_uses_observations_and_lifts_authored_apple_by_contact(
        tmp_path):
    authored_pos = _preflight(tmp_path)
    tracker = {}
    patched = _watch_executor(tracker)
    try:
        from harmless.agent import run_episode
        out_dir = tmp_path / "apple_episode"
        run_episode(
            task=(
                "At the untouched authored kitchen pose, pick up the apple. "
                "Observe joint_state(), state(), pos(), and geom_info(). Use "
                "program() with observation-driven set_joints() targets to "
                "position the hands and close the thumb, index, and middle "
                "finger joints. Call grasp('apple') only to close on physical "
                "contact. Lift the apple at least 4 cm. Do not use arm_reach, "
                "arm_path, hands_path, reach, reach_xyz, pick_up, put_down, "
                "fixed waypoint lists, welds, or object pose edits."
            ),
            max_steps=8,
            budget_s=900,
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
        "agent run changed apple from authored scene pose before control"
    )
    assert _model_actions(transcript), (
        "GPT-6-astra emitted no observation-driven set_joints() program"
    )
    assert tracker["grasp_calls"], "GPT-6-astra never called grasp('apple')"
    finger_names = {key for targets in tracker["joint_targets"] for key in targets}
    assert any(all(any(re.match(
        rf"^{side}_hand_{finger}(?:_|$)", name, re.I)
        for name in finger_names) for finger in FINGERS)
        for side in ("left", "right")), (
        "model emitted no joint-position actions for all three finger groups"
    )
    assert tracker["lifted_three_finger"], (
        "apple never rose 4 cm while thumb, index, and middle contacted it; "
        f"measured peak finger forces {tracker['finger_forces']}"
    )
    assert not tracker["active_eqs"], (
        f"grasp activated apple equality constraints {tracker['active_eqs']}"
    )
    assert tracker["max_jump"] < 0.025, (
        f"apple moved {tracker['max_jump']:.3f} m in one physics step or "
        "program boundary"
    )
