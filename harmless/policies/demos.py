"""G1 bowl-touch demonstrations recorded as a LeRobot dataset.

record_episodes() resets the kitchen to a randomized start (pelvis stance
around the bowl, perturbed arm joints, jittered bowl), lets a scripted
expert drive the right palm onto the bowl rim through the executor's
closed-loop arm reach, and samples every 0.02 s control tick: the
head-camera frame, the G1 joint state and the joint_pos targets the
actuators received. Only episodes that end in a measured palm-bowl
contact are written; the run is resumable, so the 200-episode corpus can
be filled across several invocations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

import mujoco
import numpy as np

from harmless.executor import Executor

REPO = Path(__file__).resolve().parents[2]
SCENE = REPO / "kitchen.xml"
DEFAULT_OUT = REPO / "demo" / "g1_bowl_touch"
EPISODES = 200
FPS = 50
CTRL_DT = 1.0 / FPS
HORIZON_S = 8.0
RETARGET_S = 3.0
HOLD_S = 0.3
IMAGE_SIZE = 224
CHUNK = 1000
TASK = "approach and touch the bowl"
HAND_BODY = ("wrist", "hand", "palm", "elbow")
HEAD_KEY = "observation.images.head_cam"
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = ("videos/chunk-{episode_chunk:03d}/{video_key}/"
              "episode_{episode_index:06d}.mp4")
STANCE_ANGLE_DEG = (-50.0, -40.0)
STANCE_RADIUS_M = (0.44, 0.50)
YAW_JITTER = 0.08
ARM_JITTER = 0.12
BOWL_JITTER_M = 0.02
MAX_TRIES = 5


def _body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _in_robot(ex, bid):
    while bid > 0:
        if bid == ex.pelvis_body:
            return True
        bid = int(ex.model.body_parentid[bid])
    return False


def palm_bowl_contact(ex):
    m, d = ex.model, ex.data
    bowl = _body_id(m, "bowl")
    for c in range(int(d.ncon)):
        b1 = int(m.geom_bodyid[d.contact[c].geom1])
        b2 = int(m.geom_bodyid[d.contact[c].geom2])
        if bowl not in (b1, b2):
            continue
        other = b2 if b1 == bowl else b1
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other) or ""
        if any(k in name for k in HAND_BODY) and _in_robot(ex, other):
            return True
    return min(ex.palm_gaps("bowl").values()) <= 0.02


def _arm_jids(ex):
    m = ex.model
    jids = list(getattr(ex, "arm_jids", []) or [])
    if jids:
        return jids
    return [j for j in range(m.njnt)
            if re.search(r"shoulder|elbow|wrist",
                         mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                         or "")]


def _bowl_qadr(ex):
    m = ex.model
    jid = int(m.body_jntadr[_body_id(m, "bowl")])
    if jid < 0 or m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        return None
    return int(m.jnt_qposadr[jid])


def randomize_start(ex, rng):
    m, d = ex.model, ex.data
    adr = _bowl_qadr(ex)
    if adr is not None:
        d.qpos[adr:adr + 2] += rng.uniform(-BOWL_JITTER_M, BOWL_JITTER_M, 2)
        mujoco.mj_forward(m, d)
    b = ex.pos("bowl")
    a = math.radians(rng.uniform(*STANCE_ANGLE_DEG))
    r = rng.uniform(*STANCE_RADIUS_M)
    px, py = b[0] + r * math.cos(a), b[1] + r * math.sin(a)
    yaw = math.atan2(b[1] - py, b[0] - px) + rng.uniform(-YAW_JITTER,
                                                          YAW_JITTER)
    d.qpos[0:2] = [px, py]
    d.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    for j in _arm_jids(ex):
        q = int(m.jnt_qposadr[j])
        lo, hi = m.jnt_range[j]
        d.qpos[q] = np.clip(d.qpos[q] + rng.uniform(-ARM_JITTER, ARM_JITTER),
                            lo, hi)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)


def _rim_targets(ex):
    o = ex._object_info("bowl")
    lo, hi = np.array(o["aabb_min"]), np.array(o["aabb_max"])
    c = (lo + hi) / 2
    toward = ex.data.qpos[0:2] - c[:2]
    toward /= max(np.linalg.norm(toward), 1e-6)
    rim = [c[0] + toward[0] * 0.07, c[1] + toward[1] * 0.07, hi[2] + 0.01]
    inner = [c[0] + toward[0] * 0.07, c[1] + toward[1] * 0.07, hi[2] - 0.03]
    lift = [(ex.data.qpos[0] + rim[0]) / 2, (ex.data.qpos[1] + rim[1]) / 2,
            hi[2] + 0.06]
    return lift, rim, inner


def _joint_ids(model):
    return [int(model.actuator_trnid[a, 0]) for a in range(model.nu)]


def _sample(ex, renderer, jids):
    m, d = ex.model, ex.data
    renderer.update_scene(d, camera="head_cam")
    return (renderer.render().copy(),
            np.array([d.qpos[m.jnt_qposadr[j]] for j in jids], np.float32),
            np.array([d.qvel[m.jnt_dofadr[j]] for j in jids], np.float32),
            np.array(d.ctrl[:m.nu], np.float32))


def demonstrate(ex, renderer, rng):
    m = ex.model
    jids = _joint_ids(m)
    ex.reset_trial()
    randomize_start(ex, rng)
    lift, rim, inner = _rim_targets(ex)
    ex.arm_path([lift, rim], side="right", tol=0.05)
    per_tick = max(1, int(round(CTRL_DT / m.opt.timestep)))
    frames, states, vels, actions = [], [], [], []
    t0 = float(ex.data.time)
    touched_at, retargeted = None, False
    while float(ex.data.time) - t0 < HORIZON_S:
        image, q, qd, _ = _sample(ex, renderer, jids)
        ex.step(per_tick)
        frames.append(image)
        states.append(q)
        vels.append(qd)
        actions.append(np.array(ex.data.ctrl[:m.nu], np.float32))
        elapsed = float(ex.data.time) - t0
        if ex.data.qpos[2] < 0.5:
            return None
        if touched_at is None and palm_bowl_contact(ex):
            touched_at = elapsed
        if touched_at is not None and elapsed - touched_at >= HOLD_S:
            if not palm_bowl_contact(ex):
                touched_at = None
                continue
            return {"frames": frames, "state": np.stack(states),
                    "velocity": np.stack(vels), "action": np.stack(actions)}
        if not retargeted and elapsed >= RETARGET_S:
            ex.arm_reach(*inner, side="right")
            retargeted = True
    return None


def _features(joints):
    vec = {"dtype": "float32", "shape": [len(joints)], "names": joints}
    scalar = {"dtype": "int64", "shape": [1], "names": None}
    return {
        HEAD_KEY: {
            "dtype": "video", "shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.fps": float(FPS), "video.codec": "h264",
                     "video.pix_fmt": "yuv420p", "has_audio": False}},
        "observation.state": dict(vec),
        "observation.velocity": dict(vec),
        "action": dict(vec),
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": dict(scalar),
        "episode_index": dict(scalar),
        "index": dict(scalar),
        "task_index": dict(scalar),
    }


def _read_episodes(root):
    path = root / "meta" / "episodes.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _write_info(root, joints, episodes):
    total = sum(e["length"] for e in episodes)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "unitree_g1",
        "total_episodes": len(episodes),
        "total_frames": total,
        "total_tasks": 1,
        "total_videos": len(episodes),
        "total_chunks": len(episodes) // CHUNK + 1,
        "chunks_size": CHUNK,
        "fps": FPS,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": _features(joints),
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))


def _save_episode(root, index, start, demo):
    import imageio.v2 as imageio
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = len(demo["action"])
    chunk = index // CHUNK
    data = root / DATA_PATH.format(episode_chunk=chunk, episode_index=index)
    video = root / VIDEO_PATH.format(episode_chunk=chunk,
                                     episode_index=index, video_key=HEAD_KEY)
    data.parent.mkdir(parents=True, exist_ok=True)
    video.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(video, demo["frames"], fps=FPS, macro_block_size=16)
    table = pa.table({
        "observation.state": list(demo["state"]),
        "observation.velocity": list(demo["velocity"]),
        "action": list(demo["action"]),
        "timestamp": np.arange(n, dtype=np.float32) / FPS,
        "frame_index": np.arange(n, dtype=np.int64),
        "episode_index": np.full(n, index, dtype=np.int64),
        "index": np.arange(start, start + n, dtype=np.int64),
        "task_index": np.zeros(n, dtype=np.int64),
    })
    pq.write_table(table, data)
    with open(root / "meta" / "episodes.jsonl", "a") as fh:
        fh.write(json.dumps({"episode_index": index, "tasks": [TASK],
                             "length": n}) + "\n")


def record_episodes(episodes=EPISODES, out_dir=DEFAULT_OUT, seed=0):
    root = Path(out_dir)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": TASK}) + "\n")
    done = _read_episodes(root)
    ex = Executor(str(SCENE), record=False, out_dir=str(root / ".scratch"))
    renderer = mujoco.Renderer(ex.model, IMAGE_SIZE, IMAGE_SIZE)
    joints = [mujoco.mj_id2name(ex.model, mujoco.mjtObj.mjOBJ_JOINT, j)
              for j in _joint_ids(ex.model)]
    _write_info(root, joints, done)
    attempt = 0
    try:
        while len(done) < int(episodes):
            rng = np.random.default_rng([int(seed or 0), len(done), attempt])
            attempt += 1
            if attempt > MAX_TRIES * int(episodes) + 10:
                raise RuntimeError(
                    f"the expert reached the bowl in only {len(done)} of "
                    f"{attempt - 1} attempts")
            demo = demonstrate(ex, renderer, rng)
            if demo is None:
                continue
            start = sum(e["length"] for e in done)
            _save_episode(root, len(done), start, demo)
            done = _read_episodes(root)
            _write_info(root, joints, done)
    finally:
        renderer.close()
        ex.close()
    return str(root)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=EPISODES)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    print(record_episodes(args.episodes, args.out, args.seed))


if __name__ == "__main__":
    main()
