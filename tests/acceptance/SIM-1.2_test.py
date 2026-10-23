"""SIM-1.2@1 The kitchen keyframe qpos is applied to the Newton model so
the G1 spawns standing in the aisle.

The expected pose is the MJCF ``kitchen`` key's qpos read with mujoco
(MuJoCo fills values the <key> omits with qpos0, so the array covers
every joint). Each newton joint is matched to the MJCF joint of the same
name; free-joint quaternions convert MuJoCo wxyz to newton xyzw. Forward
kinematics must then place the pelvis where the keyframe puts it,
upright — the aisle spot east of the island."""

import warnings
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
KITCHEN_XML = REPO / "kitchen.xml"
KEYFRAME = "kitchen"


def test_sim_1_2_kitchen_keyframe_qpos_applied_to_newton_model():
    from harmless import newton_scene
    import newton

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = newton_scene.build(str(KITCHEN_XML))
    model = built[-1] if isinstance(built, tuple) else built

    mj = mujoco.MjModel.from_xml_path(str(KITCHEN_XML))
    key = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME)
    assert key >= 0, "kitchen.xml has no 'kitchen' keyframe"
    kq = mj.key_qpos[key]

    q = model.joint_q.numpy()
    starts = model.joint_q_start.numpy()
    types = model.joint_type.numpy()
    FREE = int(newton.JointType.FREE)
    HINGE = (int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC))

    compared = 0
    for j, label in enumerate(model.joint_label):
        name = label.rsplit("/", 1)[-1]
        mjj = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, name)
        if mjj < 0:
            continue
        adr, start = mj.jnt_qposadr[mjj], starts[j]
        if types[j] == FREE:
            expected = [kq[adr], kq[adr + 1], kq[adr + 2],
                        kq[adr + 4], kq[adr + 5], kq[adr + 6], kq[adr + 3]]
            np.testing.assert_allclose(
                q[start:start + 7], expected, atol=1e-6,
                err_msg=f"joint_q of free joint '{name}' is not the keyframe qpos")
            compared += 1
        elif types[j] in HINGE:
            assert abs(q[start] - kq[adr]) <= 1e-6, (
                f"joint_q of '{name}' is {q[start]}, "
                f"the keyframe has {kq[adr]}")
            compared += 1
    assert compared >= 30, (
        f"only {compared} newton joints match an MJCF joint name — "
        "the keyframe cannot be verified applied to the G1")

    # Forward kinematics: the pelvis must land where the keyframe puts
    # the robot — aisle spot at standing height, yaw-only orientation.
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pelvis = next(
        i for i, label in enumerate(model.body_label)
        if label.rsplit("/", 1)[-1] == "pelvis")
    pose = state.body_q.numpy()[pelvis]
    np.testing.assert_allclose(
        pose[:3], kq[:3], atol=1e-4,
        err_msg="the G1 pelvis does not spawn at the keyframe position")
    assert pose[2] > 0.5, (
        f"the G1 spawns with its pelvis at z={pose[2]:.3f} — not standing")
    np.testing.assert_allclose(
        np.abs(pose[3:7]), np.abs([kq[4], kq[5], kq[6], kq[3]]), atol=1e-4,
        err_msg="the G1 pelvis does not spawn with the keyframe orientation")
