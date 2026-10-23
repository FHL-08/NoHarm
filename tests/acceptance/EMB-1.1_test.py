"""EMB-1.1@1 The G1 adapter reports joint_pos actions over the 29
actuated joints plus pelvis pose in observations.

The contract under test is the G1 embodiment adapter
(harmless/embodiments/g1.py — the file this requirement touches) bound
to the real G1 MJCF under assets/unitree_g1/. Its declared action space
must be of joint_pos kind spanning exactly the 29 actuated joints of the
loaded model, and its observation() must report the pelvis pose — the
free-base body's measured position and orientation, checked against
MjData.xpos/xquat after the pelvis free joint is moved off its spawn
state so identity quaternions or zero vectors cannot pass by accident.

The current adapter already declares 29 joint_pos targets but its
observation() reports only joint state, cameras and time — the missing
pelvis pose is the expected pre-implementation failure.
"""

import importlib
import inspect
import os
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[2]
ATOL_POS = 5e-4
ATOL_ORI = 1e-3

G1_NAME = re.compile(r"g1|unitree|humanoid", re.I)


def _mujoco():
    import mujoco
    return mujoco


def _g1_module():
    try:
        return importlib.import_module("harmless.embodiments.g1")
    except Exception as exc:
        pytest.fail(
            f"harmless.embodiments.g1 cannot be imported "
            f"({type(exc).__name__}: {exc}) — the G1 adapter module is "
            f"broken")


def _resolve_adapter():
    _g1_module()
    try:
        from harmless import registry
        names = []
        try:
            names = list(registry.list_embodiments())
        except Exception:
            for attr in ("EMBODIMENTS", "embodiments", "EMBODIMENT_REGISTRY"):
                reg = getattr(registry, attr, None)
                if isinstance(reg, dict):
                    names = list(reg)
                    break
        hits = [n for n in names if G1_NAME.search(str(n))]
        for n in hits + ["g1"]:
            try:
                return registry.get_embodiment(n)
            except Exception:
                for attr in ("EMBODIMENTS", "embodiments",
                             "EMBODIMENT_REGISTRY"):
                    reg = getattr(registry, attr, None)
                    if isinstance(reg, dict) and reg.get(n) is not None:
                        return reg[n]
    except Exception:
        pass
    mod = _g1_module()
    try:
        from harmless.embodiment import Embodiment
        base = Embodiment
    except Exception:
        base = None
    for name, obj in sorted(vars(mod).items()):
        if inspect.isclass(obj) and obj.__module__ == mod.__name__:
            if base is None or issubclass(obj, base):
                return obj
    pytest.fail(
        "no G1 adapter found — harmless.registry has no g1-named "
        "embodiment and harmless.embodiments.g1 declares no Embodiment "
        "subclass")


def _scene_candidates():
    out = [REPO / "assets" / "unitree_g1" / "scene.xml", REPO / "kitchen.xml"]
    g1dir = REPO / "assets" / "unitree_g1"
    if g1dir.is_dir():
        out += sorted(g1dir.rglob("*.xml"))
    return out


def _actuated_joint_names(model):
    mujoco = _mujoco()
    names = []
    for a in range(int(model.nu)):
        jid = int(model.actuator_trnid[a, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        names.append(name or f"joint{jid}")
    return names


def _load_g1():
    mujoco = _mujoco()
    for path in _scene_candidates():
        if not path.is_file():
            continue
        try:
            model = mujoco.MjModel.from_xml_path(str(path))
        except Exception:
            continue
        pelvis = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis < 0 or len(_actuated_joint_names(model)) != 29:
            continue
        return path, model
    pytest.fail(
        "no MJCF under assets/unitree_g1/ or kitchen.xml loads a model "
        "with a pelvis body and 29 actuators — the G1 scene is missing")


def _bind(entry, model, data):
    if not inspect.isclass(entry) and not callable(entry):
        return entry
    for call in (lambda: entry(model, data),
                 lambda: entry(model=model, data=data),
                 lambda: entry(model)):
        try:
            emb = call()
        except Exception:
            continue
        if getattr(emb, "data", None) is None:
            try:
                emb.data = data
            except Exception:
                pass
        return emb
    pytest.fail(
        f"the registered G1 adapter {entry!r} could not be bound to a "
        f"MjModel/MjData pair")


def _observe(emb):
    for name in ("observation", "observe", "get_observation", "obs"):
        f = getattr(emb, name, None)
        if callable(f):
            return f()
    pytest.fail(
        f"the G1 adapter {emb!r} exposes no observation() call")


def _flatten(obs, prefix="", out=None, depth=0):
    if out is None:
        out = []
    if depth > 6:
        return out
    items = None
    if isinstance(obs, Mapping):
        items = list(obs.items())
    elif isinstance(obs, (tuple, list)) and prefix == "":
        items = [(str(i), v) for i, v in enumerate(obs)]
    elif hasattr(obs, "__dict__") and not np.isscalar(obs):
        items = [(k, v) for k, v in vars(obs).items()
                 if not k.startswith("_")]
    if items is not None:
        for k, v in items:
            _flatten(v, f"{prefix}.{k}" if prefix else str(k), out,
                     depth + 1)
        return out
    try:
        arr = np.asarray(obs, dtype=float)
    except Exception:
        return out
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1 and 1 <= arr.size <= 64 and np.all(np.isfinite(arr)):
        out.append((prefix.lower(), arr.astype(float)))
    return out


def _space_of(emb):
    space = getattr(emb, "action_space", None)
    if space is None:
        for attr in ("space", "actions", "action_spec"):
            space = getattr(emb, attr, None)
            if space is not None:
                break
    if space is None and getattr(emb, "joints", None) is not None:
        space = emb
    return space


def _field(space, keys):
    for k in keys:
        v = getattr(space, k, None)
        if v is not None:
            return v
        if isinstance(space, Mapping) and space.get(k) is not None:
            return space[k]
    return None


def _quat_match(vec, quat_wxyz):
    v = np.asarray(vec, dtype=float)
    if v.shape != (4,):
        return False
    n = np.linalg.norm(v)
    if n < 1e-6:
        return False
    v = v / n
    q = np.asarray(quat_wxyz, dtype=float)
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    for cand in (q, -q, q_xyzw, -q_xyzw):
        if np.allclose(v, cand, atol=ATOL_ORI):
            return True
    return False


def _euler_match(vec, xmat):
    v = np.asarray(vec, dtype=float)
    if v.shape != (3,) or np.max(np.abs(v)) > 2 * np.pi + 0.1:
        return False
    cx, sx = np.cos(v[0]), np.sin(v[0])
    cy, sy = np.cos(v[1]), np.sin(v[1])
    cz, sz = np.cos(v[2]), np.sin(v[2])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    expected = np.asarray(xmat, dtype=float).reshape(3, 3)
    for m in (rz @ ry @ rx, rx @ ry @ rz):
        if np.allclose(m, expected, atol=ATOL_ORI):
            return True
    return False


def _pose_found(flat, pos, quat, xmat):
    pos_ok = ori_ok = False
    for key, arr in flat:
        if arr.shape == (3,) and np.allclose(arr, pos, atol=ATOL_POS):
            pos_ok = True
            continue
        if arr.shape == (4,) and _quat_match(arr, quat):
            ori_ok = True
            continue
        if arr.shape == (3,) and _euler_match(arr, xmat):
            ori_ok = True
            continue
        if arr.shape == (9,) and np.allclose(
                arr.reshape(3, 3), np.asarray(xmat).reshape(3, 3),
                atol=ATOL_ORI):
            ori_ok = True
            continue
        if arr.shape == (7,):
            head_pos = np.allclose(arr[:3], pos, atol=ATOL_POS)
            tail_pos = np.allclose(arr[4:], pos, atol=ATOL_POS)
            if head_pos and _quat_match(arr[3:], quat):
                pos_ok = ori_ok = True
            elif tail_pos and _quat_match(arr[:4], quat):
                pos_ok = ori_ok = True
    return pos_ok, ori_ok


def test_emb_1_1_g1_adapter_reports_29d_joint_pos_and_pelvis_pose():
    entry = _resolve_adapter()
    path, model = _load_g1()
    data = _mujoco().MjData(model)

    mujoco = _mujoco()
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    jid = int(model.body_jntadr[pelvis])
    qadr = int(model.jnt_qposadr[jid])
    data.qpos[qadr:qadr + 3] = [0.12, -0.07, 0.78]
    ang = np.array([0.21, -0.13, 0.34])
    half = ang / 2.0
    data.qpos[qadr + 3:qadr + 7] = np.concatenate(
        [[np.cos(np.linalg.norm(half))],
         np.sin(np.linalg.norm(half)) * ang / np.linalg.norm(ang)])
    mujoco.mj_forward(model, data)

    emb = _bind(entry, model, data)
    space = _space_of(emb)
    assert space is not None, (
        f"the G1 adapter on {path.name} declares no action_space")

    kind = _field(space, ("kind", "type", "action_kind", "family"))
    assert isinstance(kind, str) and "joint" in kind.lower() \
        and "pos" in kind.lower(), (
            f"the G1 action space kind is {kind!r}, not joint_pos")

    joints = _field(space, ("joints", "joint_names", "names", "dofs",
                            "axes"))
    joints = list(joints) if joints is not None else []
    if not joints:
        bounds0 = _field(space, ("bounds", "ranges", "limits"))
        if isinstance(bounds0, Mapping):
            joints = list(bounds0)
    actuated = _actuated_joint_names(model)
    assert len(joints) == 29, (
        f"the G1 joint_pos space declares {len(joints)} dims "
        f"{joints[:12]}, not the 29 actuated joints")
    assert set(joints) == set(actuated), (
        "the declared joint_pos joints do not match the model's 29 "
        f"actuated joints — missing {sorted(set(actuated) - set(joints))}, "
        f"extra {sorted(set(joints) - set(actuated))}")

    obs = _observe(emb)
    flat = _flatten(obs)
    pos = np.asarray(data.xpos[pelvis], dtype=float)
    quat = np.asarray(data.xquat[pelvis], dtype=float)
    xmat = np.asarray(data.xmat[pelvis], dtype=float)
    pos_ok, ori_ok = _pose_found(flat, pos, quat, xmat)
    keys = [k for k, _ in flat]
    assert pos_ok and ori_ok, (
        "the G1 observation() reports no measured pelvis pose: expected "
        f"a position leaf ≈{np.round(pos, 4).tolist()} and an "
        f"orientation leaf ≈quat{np.round(quat, 4).tolist()} (wxyz or "
        f"xyzw, either sign), a matching euler/rotation-matrix leaf, or a "
        f"7-vec pose — got leaves keyed {keys[:40]}")
