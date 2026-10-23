"""EMB-1.2@1 Observations include pelvis IMU (gyro and accel), joint
state, and touch-sensor normal forces on pelvis, torso, elbows, hands,
shins, and feet.

The contract under test is the observation() bundle of the G1 adapter
(harmless/embodiments/g1.py) bound to the G1 MJCF under
assets/unitree_g1/, whose <sensor> block declares pelvis and torso
gyro/accelerometer sensors plus ten touch sites: pelvis, torso, both
elbows, both hands, both feet and both shins.

Every claimed channel is pinned to measured physics: the pelvis free
joint is given angular velocity and the waist is bent before
observation() is called, so pelvis and torso IMU readings provably
diverge — an adapter reporting the torso IMU, constants, or zeros fails.
Expected values are read straight from MjData.sensordata/qpos/qvel at
the observed state. Joint state must cover every actuated joint's
measured position and velocity; each of the model's touch sensors must
appear as a keyed scalar matching its measured normal force (or as an
ordered vector covering them all).

The current observation() reports joints, joint_pos/joint_vel arrays,
cameras and time — no IMU and no touch channels — so the missing
channels are the expected pre-implementation failure.
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
ATOL_JOINT = 1e-4
ATOL_GYRO = 5e-2
ATOL_ACCEL = 3e-1
ATOL_TOUCH = 5e-3

G1_NAME = re.compile(r"g1|unitree|humanoid", re.I)
GENERIC_TOKS = {"touch", "sensor", "force", "normal", "site", "imu",
                "read", "reading", "value", "contact", "skin"}


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


def _toks(s):
    raw = re.findall(r"[a-z0-9]+", s.lower())
    toks = set()
    for t in raw:
        m = re.fullmatch(
            r"([lr])(hand|foot|feet|arm|leg|elbow|shin|knee|wrist|ankle"
            r"|shank)", t)
        if m:
            toks.add("left" if m.group(1) == "l" else "right")
            t = m.group(2)
        elif t in ("l", "left", "lft", "lhs"):
            toks.add("left")
            continue
        elif t in ("r", "right", "rhs"):
            toks.add("right")
            continue
        if t == "feet":
            t = "foot"
        toks.add(t)
        if len(t) > 1 and t.endswith("s"):
            toks.add(t[:-1])
    return toks


def _sensors(model, kind):
    mujoco = _mujoco()
    out = []
    for i in range(int(model.nsensor)):
        if int(model.sensor_type[i]) != int(kind):
            continue
        sid = int(model.sensor_objid[i])
        site = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_SITE, sid) or f"site{sid}"
        adr, dim = int(model.sensor_adr[i]), int(model.sensor_dim[i])
        out.append((i, site, adr, dim))
    return out


def _is_pelvis_site(model, site):
    mujoco = _mujoco()
    if "pelvis" in site.lower():
        return True
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    if sid < 0:
        return False
    body = int(model.site_bodyid[sid])
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
    return "pelvis" in name.lower()


def _joint_state_ok(flat, model, data, joints):
    mujoco = _mujoco()
    pos_exp, vel_exp = {}, {}
    for name in joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            return False, f"declared joint {name!r} absent from model"
        pos_exp[name] = float(data.qpos[int(model.jnt_qposadr[jid])])
        vel_exp[name] = float(data.qvel[int(model.jnt_dofadr[jid])])
    pos_vec = np.array([pos_exp[n] for n in joints])
    vel_vec = np.array([vel_exp[n] for n in joints])

    vec_pos = vec_vel = False
    named_pos, named_vel = set(), set()
    for key, arr in flat:
        if arr.shape == (len(joints),):
            vec_pos = vec_pos or np.allclose(
                arr, pos_vec, atol=ATOL_JOINT)
            vec_vel = vec_vel or np.allclose(
                arr, vel_vec, atol=ATOL_JOINT)
        if arr.shape != (1,):
            continue
        leaf = key.rsplit(".", 1)[-1]
        for name in joints:
            if leaf != name.lower() and name.lower() not in key:
                continue
            if abs(arr[0] - pos_exp[name]) < ATOL_JOINT:
                named_pos.add(name)
            if abs(arr[0] - vel_exp[name]) < ATOL_JOINT:
                named_vel.add(name)
    pos_ok = vec_pos or len(named_pos) == len(joints)
    vel_ok = vec_vel or len(named_vel) == len(joints)
    missing = [n for n in joints if n not in named_pos]
    return pos_ok and vel_ok, (
        f"pos_ok={pos_ok} (vec={vec_pos} missing_named={missing[:6]}) "
        f"vel_ok={vel_ok} (vec={vec_vel})")


def _imu_ok(flat, data, pelvis, others, atol):
    exp = data.sensordata[pelvis[2]:pelvis[2] + pelvis[3]].astype(float)
    other_vecs = [data.sensordata[o[2]:o[2] + o[3]].astype(float)
                  for o in others]
    for key, arr in flat:
        if arr.shape != exp.shape:
            continue
        if not np.allclose(arr, exp, atol=atol):
            continue
        if "pelvis" in _toks(key) or "base" in _toks(key):
            return True
        if not any(np.allclose(arr, o, atol=atol) for o in other_vecs):
            return True
    return False


def _touch_ok(flat, data, touches):
    exp_all = np.array(
        [float(data.sensordata[t[2]]) for t in touches])
    if any(np.allclose(arr, exp_all, atol=ATOL_TOUCH)
           for _, arr in flat if arr.shape == (len(touches),)):
        return []
    missing = []
    for _, site, adr, _ in touches:
        want = _toks(site) - GENERIC_TOKS
        val = float(data.sensordata[adr])
        hit = any(
            want <= _toks(key)
            and not ({"joint", "joints", "qpos", "qvel", "pos", "vel"}
                     & _toks(key))
            and arr.shape == (1,)
            and abs(arr[0] - val) < ATOL_TOUCH
            for key, arr in flat)
        if not hit:
            missing.append(site)
    return missing


def test_emb_1_2_observation_reports_pelvis_imu_joint_state_and_touch():
    entry = _resolve_adapter()
    path, model = _load_g1()
    data = _mujoco().MjData(model)

    mujoco = _mujoco()
    waist = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch_joint")
    if waist >= 0:
        data.qpos[int(model.jnt_qposadr[waist])] = 0.25
    data.qvel[3:6] = [0.4, -0.2, 0.3]
    mujoco.mj_forward(model, data)

    emb = _bind(entry, model, data)
    obs = _observe(emb)
    flat = _flatten(obs)

    gyros = _sensors(model, mujoco.mjtSensor.mjSENS_GYRO)
    accels = _sensors(model, mujoco.mjtSensor.mjSENS_ACCELEROMETER)
    touches = _sensors(model, mujoco.mjtSensor.mjSENS_TOUCH)
    pelvis_gyro = [s for s in gyros if _is_pelvis_site(model, s[1])]
    pelvis_acc = [s for s in accels if _is_pelvis_site(model, s[1])]
    assert pelvis_gyro and pelvis_acc, (
        f"{path.name} declares no gyro+accelerometer pair attached to "
        f"the pelvis — gyros {[s[1] for s in gyros]}, accels "
        f"{[s[1] for s in accels]}")
    assert len(touches) >= 10, (
        f"{path.name} declares only {len(touches)} touch sensors, the "
        "G1 needs ten: pelvis, torso, elbows, hands, shins, feet")

    space = getattr(emb, "action_space", None)
    joints = list(getattr(space, "joints", None) or [])
    if not joints:
        joints = _actuated_joint_names(model)

    js_ok, js_why = _joint_state_ok(flat, model, data, joints)
    gyro_ok = _imu_ok(flat, data, pelvis_gyro[0],
                      [s for s in gyros if s != pelvis_gyro[0]],
                      ATOL_GYRO)
    accel_ok = _imu_ok(flat, data, pelvis_acc[0],
                       [s for s in accels if s != pelvis_acc[0]],
                       ATOL_ACCEL)
    missing_touch = _touch_ok(flat, data, touches)

    keys = [k for k, _ in flat]
    assert js_ok and gyro_ok and accel_ok and not missing_touch, (
        "the G1 observation() is missing measured channels: "
        f"joint_state ok={js_ok} ({js_why}), pelvis gyro ok={gyro_ok}, "
        f"pelvis accel ok={accel_ok}, touch sites unmatched="
        f"{missing_touch or 'none'} — observation leaves keyed "
        f"{keys[:40]}")
