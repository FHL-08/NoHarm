"""NPC-3.1@2 In a kitchen episode, an Executor-measured person contact of
50 N or more triggers the faint clip, and her body then collides with the
iTHOR floor and furniture.

The contract under test is wired into the episode executor itself — an
earlier round let a private Person world fake this and was rejected —
so every measurement below is taken from a live ``Executor`` built on
``kitchen.xml``: the real MuJoCo scene with the iTHOR room, the robot
and the tethered walking person. The contact that triggers the faint
must be measured by that executor, the collapse must play out in that
scene, and the collisions must be read off its ``data.contact``.

The faint clip. "The faint clip" is a thing: the test requires a
faint/collapse-named clip artifact on the touched surface — a clip dict
or keyframe list on harmless.person, a faint entry of a clip mapping,
a faint clip object on the executor, or a clip file under env/person/.
A procedural shove is not a clip.

The trigger. The contact is Executor-measured: a real robot-pelvis
impact staged in the sim must show up as a person contact at or above
50 N on the executor's own measurement (stats.person_peak_force, also
readable through harmless.receipts). Baseline sanity: while she only
walks and nothing touches her, no faint starts — she stays upright and
keeps moving. (No staged sub-50 touch exists: welded to the mocap
driver she is an immovable anchor, so every real contact in this scene
measures hundreds of newtons; the gate's lower side is covered by the
no-contact baseline.)

The collapse. After the measured >=50 N contact she must go down in the
kitchen sim — her head site falling progressively through intermediate
heights (a played collapse, not a teleport) to below 0.5 m. It must also
be a faint, not the pre-existing ballistic knockdown: the current code
injects a fixed 0.6 m/s sideways push plus a 2 rad/s tip the instant
contact registers, visible in her free-body qvel on the trigger step; a
clip-driven collapse starts at clip pace, so the test bounds her
horizontal and angular speed in the window around the trigger.

The collisions. Her body collides with the iTHOR floor and furniture,
measured as contacts between her geoms and the scene: staged beside the
south counter row (her torso already grazing the counter face), the
test requires person<->furniture contacts during the collapse and
settle, and sustained person<->floor contacts once she lies at rest.
She must never tunnel: none of her geoms may end below the floor.

Expected failure before implementation: nothing faint-named exists on
the touched modules or under env/person/ — the person today just gets
shoved loose of her tether by any contact over ~1 N, with no clip.
"""

import importlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]

FAINT_FORCE_N = 50.0
DOWN_HEAD_Z = 0.5
UPRIGHT_HEAD_Z = 1.4
STAGE_XY = (-0.85, -2.02)     # grazing the south counter run
ROBOT_STAGE_XY = (-0.85, -1.62)  # open floor just north of her
ROBOT_HIT_VEL = (0.0, -4.0, 0.0)
ROBOT_CLEAR_XY = (3.0, 0.0)
MAX_FALL_S = 8.0
REST_S = 1.0

FAINT_WORDS = ("faint", "collapse", "collaps", "swoon", "crumple",
               "crumbl", "unconscious", "fall")
CLIP_ATTRS = ("FAINT_CLIP", "faint_clip", "COLLAPSE_CLIP",
              "collapse_clip", "FALL_CLIP", "fall_clip", "SWOON_CLIP",
              "swoon_clip", "CRUMPLE_CLIP", "crumple_clip",
              "FAINT", "COLLAPSE", "SWOON", "CLIPS", "clips",
              "ANIMATIONS", "animations")
CLIP_DIRS = (REPO / "env" / "person", REPO / "assets" / "person")
CLIP_SUFFIXES = (".json", ".npy", ".npz", ".yaml", ".yml", ".glb",
                 ".gltf", ".txt", ".csv")
ROOM_WORDS = ("floor", "ground", "plane", "wall", "ceiling", "window",
              "door_", "_door", "world", "mesh", "electricswitch",
              "room", "shell")


def _mujoco():
    import mujoco
    return mujoco


def _executor(tmp_path, tag="run"):
    from harmless.executor import Executor
    return Executor(str(REPO / "kitchen.xml"), record=False,
                    out_dir=str(tmp_path / tag), skills_path=None)


def _head_sid(ex):
    mujoco = _mujoco()
    return mujoco.mj_name2id(ex.model, mujoco.mjtObj.mjOBJ_SITE,
                             "person_head_site")


def _head_z(ex):
    return float(ex.data.site_xpos[_head_sid(ex)][2])


def _person_min_z(ex):
    return min(float(ex.data.geom_xpos[g][2]) for g in ex.person_geoms)


def _move_person(ex, xy):
    """Park the walking person at xy: the mocap driver and her tethered
    free body moved together so the weld stays consistent."""
    mujoco = _mujoco()
    per_qa = ex.model.jnt_qposadr[ex.model.body_jntadr[ex.person_body]]
    ex.data.mocap_pos[ex.person_mocap][:2] = xy
    ex.data.qpos[per_qa:per_qa + 3] = (float(xy[0]), float(xy[1]), 0.0)
    ex.data.qpos[per_qa + 3:per_qa + 7] = (1, 0, 0, 0)
    ex.data.qvel[:] = 0.0
    mujoco.mj_forward(ex.model, ex.data)


def _pelvis_to(ex, xy, z=0.80, vel=None):
    mujoco = _mujoco()
    pj = ex.model.body_jntadr[ex.pelvis_body]
    qa = ex.model.jnt_qposadr[pj]
    da = ex.model.jnt_dofadr[pj]
    ex.data.qpos[qa:qa + 3] = (float(xy[0]), float(xy[1]), z)
    ex.data.qpos[qa + 3:qa + 7] = (1, 0, 0, 0)
    ex.data.qvel[:] = 0.0
    if vel is not None:
        ex.data.qvel[da:da + 3] = vel
    mujoco.mj_forward(ex.model, ex.data)


def _other_body(ex, con):
    """The non-person body on the other side of a person contact."""
    g = con.geom2 if con.geom1 in ex.person_geoms else con.geom1
    return int(ex.model.geom_bodyid[g])


def _person_scene_contacts(ex):
    """person<->scene contacts right now (robot excluded): {body_name:
    max normal force N}. This is where floor and furniture hits show."""
    mujoco = _mujoco()
    out = {}
    f6 = np.zeros(6)
    for c in range(ex.data.ncon):
        con = ex.data.contact[c]
        if con.geom1 not in ex.person_geoms and \
                con.geom2 not in ex.person_geoms:
            continue
        b = _other_body(ex, con)
        if b == ex.person_body or ex._robot_body(b):
            continue
        mujoco.mj_contactForce(ex.model, ex.data, c, f6)
        name = mujoco.mj_id2name(ex.model, mujoco.mjtObj.mjOBJ_BODY, b) \
            or f"body_{b}"
        out[name] = max(out.get(name, 0.0), float(np.linalg.norm(f6[:3])))
    return out


def _is_room(name):
    n = name.lower()
    return any(w in n for w in ROOM_WORDS)


def _measured_peak(ex):
    """The executor's own measured person-contact peak (N): the episode
    stat step() maintains, cross-checked by the receipts module."""
    peak = float(getattr(getattr(ex, "stats", None),
                         "person_peak_force", 0.0) or 0.0)
    try:
        from harmless import receipts
        rec = receipts.person_harm(ex)
        peak = max(peak, float(rec.get("person_peak_force_N", 0.0)))
    except Exception:
        pass
    return peak


def _faintish(name):
    n = name.lower()
    return any(w in n for w in FAINT_WORDS)


def _carries_poses(v, depth=0):
    """True when v looks like clip data: a joint/keyframe mapping or a
    sequence of poses — not a flag, a string, or an empty shell."""
    if depth > 3 or v is None or isinstance(v, bool) \
            or isinstance(v, (str, bytes)):
        return False
    if isinstance(v, np.ndarray):
        return v.size >= 2 and np.issubdtype(v.dtype, np.number)
    if isinstance(v, Mapping):
        if not v:
            return False
        if _faintish(" ".join(str(k) for k in v)):
            return True
        return any(_carries_poses(x, depth + 1) for x in v.values())
    if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
        seq = list(v)
        if len(seq) >= 2 and all(
                isinstance(x, (int, float, np.integer, np.floating))
                for x in seq):
            return True
        return len(seq) >= 2 and any(
            _carries_poses(x, depth + 1) for x in seq)
    if callable(v):
        return False
    if hasattr(v, "__dict__"):
        return any(_faintish(k) or _carries_poses(x, depth + 1)
                   for k, x in vars(v).items() if not k.startswith("_"))
    return False


def _clip_artifacts():
    """Faint-clip artifacts on the touched surface: module attributes
    named for faint/collapse clips (or faint entries inside a CLIPS
    mapping) on harmless.person / harmless.executor, person objects
    they vend, and clip files under env/person/ and assets/person/."""
    found = []
    for modname in ("harmless.person", "harmless.executor"):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for name in dir(mod):
            if name.startswith("__"):
                continue
            try:
                obj = getattr(mod, name)
            except Exception:
                continue
            label = f"{modname}.{name}"
            if _faintish(name) and _carries_poses(obj):
                found.append(label)
                continue
            if name in CLIP_ATTRS and isinstance(obj, Mapping):
                for k, v in obj.items():
                    if _faintish(str(k)) and _carries_poses(v):
                        found.append(f"{label}[{k!r}]")
            if isinstance(obj, Mapping) and any(
                    _faintish(str(k)) for k in obj):
                for k, v in obj.items():
                    if _faintish(str(k)) and _carries_poses(v):
                        found.append(f"{label}[{k!r}]")
        for name in dir(mod):
            obj = getattr(mod, name, None)
            if obj is None or callable(obj) or \
                    isinstance(obj, type) or name.startswith("_"):
                continue
            for a in CLIP_ATTRS:
                try:
                    v = getattr(obj, a, None)
                except Exception:
                    continue
                if v is None:
                    continue
                if _faintish(a) and _carries_poses(v):
                    found.append(f"{modname}.{name}.{a}")
                elif isinstance(v, Mapping):
                    for k, e in v.items():
                        if _faintish(str(k)) and _carries_poses(e):
                            found.append(
                                f"{modname}.{name}.{a}[{k!r}]")
    for d in CLIP_DIRS:
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in CLIP_SUFFIXES:
                continue
            if _faintish(f.name):
                found.append(str(f.relative_to(REPO)))
                continue
            if f.suffix.lower() == ".json" and f.stat().st_size < 2_000_000:
                try:
                    data = json.loads(f.read_text())
                except Exception:
                    continue
                if isinstance(data, Mapping) and any(
                        _faintish(str(k)) for k in data):
                    found.append(
                        f"{f.relative_to(REPO)} keys={sorted(data)[:6]}")
    return found


def test_npc_3_1_measured_50n_contact_triggers_faint_clip_and_collision(
        tmp_path):
    # --- the faint clip must exist on the touched surface -------------
    artifacts = _clip_artifacts()
    assert artifacts, (
        "no faint clip artifact found: nothing faint/collapse-named "
        "carrying pose data on harmless.person or harmless.executor, and "
        "no faint clip file under env/person/ or assets/person/ — "
        "there is no clip for the 50 N contact to trigger")

    ex = _executor(tmp_path)
    head_sid = _head_sid(ex)
    assert head_sid >= 0, "kitchen scene has no person_head_site"

    # --- baseline: walking, untouched, upright — no faint -------------
    ex.step(250)   # ~0.5 s of her stroll
    assert _measured_peak(ex) == 0.0, \
        "staging assumption broken: she registered a contact untouched"
    assert _head_z(ex) > UPRIGHT_HEAD_Z, \
        "she is not upright while simply walking"
    assert not bool(getattr(ex.stats, "person_down", False)), \
        "person_down reads true before any contact — not measured"

    # --- stage her beside the south counter row, then hit her ---------
    _move_person(ex, STAGE_XY)
    _pelvis_to(ex, ROBOT_STAGE_XY, vel=ROBOT_HIT_VEL)

    per_da = ex.model.jnt_dofadr[ex.model.body_jntadr[ex.person_body]]

    def _body_speeds():
        v = ex.data.qvel[per_da:per_da + 3]
        w = ex.data.qvel[per_da + 3:per_da + 6]
        return float(np.linalg.norm(v[:2])), float(np.linalg.norm(w))

    vmax_h = wmax = 0.0
    triggered = False
    for _ in range(60):
        ex.step(1)
        vmax_h, wmax = map(max, (vmax_h, wmax), _body_speeds())
        if _measured_peak(ex) >= FAINT_FORCE_N:
            triggered = True
            break
    assert triggered, (
        "staging failed: the robot impact never measured >=50 N on the "
        "executor (peak "
        f"{_measured_peak(ex):.1f} N) — the line requires an "
        "Executor-measured person contact of 50 N or more")

    # --- a faint, not a ballistic shove ---------------------------------
    # The pre-existing knockdown injects a fixed push (0.6 m/s sideways +
    # a 2 rad/s tip) the instant contact registers — present in qvel on
    # the trigger step itself. A triggered clip starts the collapse at
    # clip pace. Clear the robot so the next 0.35 s of her motion is
    # hers alone, then keep sampling.
    _pelvis_to(ex, ROBOT_CLEAR_XY)
    ts = ex.model.opt.timestep
    for _ in range(int(0.35 / ts)):
        ex.step(1)
        vmax_h, wmax = map(max, (vmax_h, wmax), _body_speeds())
    assert vmax_h < 0.55 and wmax < 1.8, (
        f"the collapse started with a ballistic shove, not a clip: her "
        f"free body took |v| {vmax_h:.2f} m/s horizontal and |w| "
        f"{wmax:.2f} rad/s at the trigger — the old 0.6 m/s + 2 rad/s "
        "impulse, not a faint clip playing")

    # --- the collapse: played out, ending down -------------------------
    max_steps = int(MAX_FALL_S / ts)
    mid_steps = 0
    furniture_hits = {}
    floor_hits = {}
    min_geom_z = np.inf
    head_now = _head_z(ex)
    went_down = False
    for i in range(max_steps):
        ex.step(1)
        head_now = _head_z(ex)
        min_geom_z = min(min_geom_z, _person_min_z(ex))
        if DOWN_HEAD_Z < head_now < UPRIGHT_HEAD_Z + 0.05:
            mid_steps += 1
        for name, f in _person_scene_contacts(ex).items():
            if _is_room(name):
                floor_hits[name] = max(floor_hits.get(name, 0.0), f)
            else:
                furniture_hits[name] = max(
                    furniture_hits.get(name, 0.0), f)
        if head_now < DOWN_HEAD_Z:
            went_down = True
            break

    assert mid_steps >= 15, (
        f"her head went from standing to down in {mid_steps} sampled "
        "steps — that is a teleport or a latch, not a played collapse")

    # --- she ends down and stays colliding with the scene -------------
    rest_force = {}
    if went_down:
        for _ in range(int(REST_S / ts)):
            ex.step(1)
            min_geom_z = min(min_geom_z, _person_min_z(ex))
            for name, f in _person_scene_contacts(ex).items():
                rest_force[name] = max(rest_force.get(name, 0.0), f)
        head_now = _head_z(ex)

    assert head_now < DOWN_HEAD_Z, (
        f"after a measured {_measured_peak(ex):.0f} N contact and "
        f"{MAX_FALL_S:.0f} s her head site still reads "
        f"{head_now:.2f} m — the faint clip never put her down")

    assert min_geom_z > -0.05, (
        f"her geoms reached z={min_geom_z:.3f} m — she tunnelled through "
        "the iTHOR floor instead of colliding with it")

    assert floor_hits or any(f > 0 for f in rest_force.values()), (
        "no person<->floor contact was ever measured while she went down "
        f"and lay still (seen: {sorted(rest_force) or 'none'}) — her "
        "body does not collide with the iTHOR floor")

    assert furniture_hits, (
        "no person<->furniture contact was measured during the collapse: "
        "staged grazing the south counter run she went down without ever "
        f"touching a cabinet/counter (scene contacts seen: "
        f"{sorted(rest_force) or sorted(floor_hits) or 'none'}) — her "
        "body does not collide with iTHOR furniture")
