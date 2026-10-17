"""Load the NoHarm MJCF scene into Newton with the right collision setup and
start pose.

What add_mjcf alone gets wrong for this scene:
  - Self-collision: kitchen.xml is ONE import, so
    enable_self_collisions=False filters robot-vs-room and prop-vs-counter
    pairs too (everything free-falls). We keep it on and reproduce
    MuJoCo's filtering instead: no static-static contacts, no contacts
    inside a welded body group, no G1 self-collision.
  - Contact gap: the 0.1 m default yields ~50k contacts at rest; 5 mm
    gives ~1.7k (fits SOLVER_KW's nconmax).
  - Keyframes: the <key name="kitchen"> start pose is ignored. We copy its
    qpos into joint_q / joint_target_q (MuJoCo wxyz -> Newton xyzw for free
    joints).
"""
from __future__ import annotations

import mujoco
import numpy as np
import newton

ROBOT_ROOT = "pelvis"
RIGID_GAP = 0.005   # m
# SolverMuJoCo settings verified with this scene: G1 stands, every prop
# rests (newton 1.7, ~1.7k contacts at rest)
SOLVER_KW = dict(use_mujoco_contacts=False, njmax=20000, nconmax=8000)


def build(xml_path="kitchen.xml", keyframe="kitchen"):
    """Returns (builder, model) with contacts and the start pose set up."""
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    # contact detection gap: Newton's 0.1 m default turns every pair of props
    # standing side by side on a counter into thousands of mesh contacts
    b.rigid_gap = RIGID_GAP
    b.add_mjcf(xml_path, enable_self_collisions=True)

    # Negative collision groups collide with everything except their own
    # group (newton broad_phase_common.test_group_pair):
    #   STATIC: room fixtures + mocap person (bodies welded to world). MuJoCo
    #           never makes static-static contacts; without this Newton finds
    #           ~60k fixture-vs-fixture contacts and drops the real ones.
    #   ROBOT:  G1 links, i.e. self-collision off as in Menagerie usage.
    #   Weld groups: MuJoCo also skips contacts between bodies rigidly joined
    #   (no joint between them), e.g. the parts of one prop; Newton keeps
    #   them as FIXED joints, so those shape pairs are filtered explicitly.
    parent_fixed = {b.joint_child[j]: b.joint_parent[j]
                    for j in range(len(b.joint_type))
                    if b.joint_type[j] == newton.JointType.FIXED}

    def weld_root(body):
        while body in parent_fixed:
            body = parent_fixed[body]
        return body          # -1 = welded to the world (static)

    robot = {i for i, lab in enumerate(b.body_label)
             if f"/{ROBOT_ROOT}/" in lab or lab.endswith(f"/{ROBOT_ROOT}")}
    used = {-g for g in b.shape_collision_group if g < 0}
    g_static = -(max(used, default=0) + 1)
    g_robot = g_static - 1
    welded = {}
    for s in range(b.shape_count):
        if not b.shape_flags[s] & newton.ShapeFlags.COLLIDE_SHAPES \
                or b.shape_collision_group[s] == 0:
            continue
        body = b.shape_body[s]
        root = weld_root(body)
        if root == -1:
            b.shape_collision_group[s] = g_static
        elif body in robot:
            b.shape_collision_group[s] = g_robot
        else:
            welded.setdefault(root, []).append(s)
    for shapes in welded.values():
        if len({b.shape_body[s] for s in shapes}) < 2:
            continue
        for k, a in enumerate(shapes):
            for c in shapes[k + 1:]:
                if b.shape_body[a] != b.shape_body[c]:
                    b.add_shape_collision_filter_pair(a, c)

    m = b.finalize()
    if keyframe:
        apply_keyframe(m, b, xml_path, keyframe)
    return b, m


def apply_keyframe(m, b, xml_path, keyframe):
    """Copy an MJCF keyframe's qpos into Newton joint coordinates by joint
    name (so the ordering of the two models never matters)."""
    mj = mujoco.MjModel.from_xml_path(xml_path)
    kid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if kid < 0:
        return
    kq = mj.key_qpos[kid]
    q = m.joint_q.numpy()
    qstart = m.joint_q_start.numpy()
    jtype = m.joint_type.numpy()
    for j, lab in enumerate(b.joint_label):
        name = lab.rsplit("/", 1)[-1]
        mjj = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, name)
        if mjj < 0:
            continue
        a, s = mj.jnt_qposadr[mjj], qstart[j]
        if jtype[j] == newton.JointType.FREE:
            w, x, y, z = kq[a + 3:a + 7]
            q[s:s + 7] = [*kq[a:a + 3], x, y, z, w]
        elif jtype[j] in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
            q[s] = kq[a]
    m.joint_q.assign(q)
    if hasattr(m, "joint_target_q"):
        m.joint_target_q.assign(q)
