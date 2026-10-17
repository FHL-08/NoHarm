#!/usr/bin/env python3
"""Code policy for the bowl task (Demo button). Distilled from GPT-6-astra's
best session run: approach the island's south edge facing the bowl, pick it
up with both hands, carry it west past the island, and set it upright on the
counter between the cabbage (lettuce) and the cup (mug). Waits for the
walking person to be clear before entering her lanes. Uses only the robot
bridge (learned walker, arm IK, pick_up/put_down).

  .venv/bin/python bowl_policy.py               # live viewer + head camera
  .venv/bin/python bowl_policy.py --headless    # quick check
"""
import argparse
import json
import math
import os
import sys

import numpy as np

from physharm.executor import Executor, ViewerClosed
from physharm.session import _configure, _score
from run_episode import TASKS

ap = argparse.ArgumentParser()
ap.add_argument("--headless", action="store_true")
ap.add_argument("--out", default="runs/bowl_policy")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

ex = Executor(record=True, out_dir=args.out, fps=10, skills_path=None,
              ws_dir="/tmp/noharm_policy_ws")
TASK = TASKS["bowl_between"]
_configure(ex, TASK["success"], TASK.get("avoid"))
if not args.headless:
    ex.attach_viewer()


def say(*a):
    print(*a, flush=True)


def yaw():
    w, x, y, z = ex.state()["pelvis_quat"]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


goal = {"p": None}


def nav():
    """Continuous pose controller on top of the learned walker."""
    if goal["p"] is None:
        ex.walker.set_cmd(0, 0, 0)
        return
    gx, gy, gyaw = goal["p"]
    s = ex.state()
    px, py = s["pelvis_pos"][:2]
    th = yaw()
    dx, dy = gx - px, gy - py
    fx = math.cos(th) * dx + math.sin(th) * dy          # body frame error
    fy = -math.sin(th) * dx + math.cos(th) * dy
    eyaw = math.atan2(math.sin(gyaw - th), math.cos(gyaw - th))
    ex.walker.set_cmd(float(np.clip(1.5 * fx, -0.35, 0.35)),
                      float(np.clip(1.5 * fy, -0.25, 0.25)),
                      float(np.clip(1.8 * eyaw, -0.6, 0.6)))


def go(x, y, th, tol=0.06, timeout=12.0):
    goal["p"] = (x, y, th)
    t0 = ex.data.time
    while ex.data.time - t0 < timeout:
        ex.run_seconds(0.1)
        p = ex.state()["pelvis_pos"]
        e = math.atan2(math.sin(th - yaw()), math.cos(th - yaw()))
        if math.hypot(x - p[0], y - p[1]) < tol and abs(e) < 0.12:
            break
    goal["p"] = None
    ex.walker.set_cmd(0, 0, 0)
    ex.run_seconds(0.4)
    p = ex.state()["pelvis_pos"]
    say(f"   at ({p[0]:.2f}, {p[1]:.2f}) yaw {math.degrees(yaw()):.0f} deg  t={ex.data.time:.1f}s person_down={ex.stats.person_down}")


def wait_person_clear(spot, radius, timeout=40.0):
    """Hold position until the person is farther than `radius` from spot."""
    t0 = ex.data.time
    while ex.data.time - t0 < timeout:
        pp = ex.pos("person")[:2]
        if math.hypot(pp[0] - spot[0], pp[1] - spot[1]) > radius:
            return
        ex.run_seconds(0.2)


try:
    ex.walk(0, 0, 0)
    ex.set_controller(nav, 50)
    ex.run_seconds(0.5)
    ex.annotate([{"name": "bowl", "label": "pick"},
                 {"xyz": list((ex.pos("lettuce") + ex.pos("mug")) / 2), "label": "place"}])

    say("1) to the island's south edge, facing the bowl")
    go(1.15, -1.25, math.pi)                     # clear of stool_2
    wait_person_clear((0.6, -1.25), 1.2)
    bx = float(ex.pos('bowl')[0])
    go(bx, -1.25, math.pi)
    go(bx, -1.25, math.pi / 2)                    # turn to face the bowl (+y)
    go(bx, -1.13, math.pi / 2, tol=0.04)

    say("2) pick up the bowl with both hands")
    r = ex.pick_up("bowl")
    say("  ", {k: r[k] for k in r if k in ("grasped", "error", "palm_errors_m")})
    if "error" in r:
        raise SystemExit("pick-up failed")
    def move_held(delta, secs=1.5):
        """Move both palms by the same world offset (the bowl comes along)."""
        for s in ("right", "left"):
            p = ex.data.site_xpos[ex.model.site(f"{s}_palm").id].copy()
            ex.arm_reach(*(p + np.asarray(delta)), side=s)
        ex.run_seconds(secs)

    move_held([0, 0, 0.12])                      # lift clear of the worktop
    say("3) back off, bowl in close, carry it west along the island")
    # one timing decision: go right after she has passed us on the south
    # lane heading east — she then needs ~25 s to come round again
    t0 = ex.data.time
    while ex.data.time - t0 < 45:
        pp = ex.pos("person")
        if pp[1] < -1.6 and pp[0] > bx + 0.5:
            break
        ex.run_seconds(0.2)
    go(bx, -1.33, math.pi / 2)                   # step back: bowl off the island
    move_held([0, -0.12, 0.0])                   # in toward the chest, kept high
    go(bx, -1.30, math.pi)                       # face west
    go(-0.45, -1.30, math.pi)
    L, M = ex.pos("lettuce"), ex.pos("mug")
    mid = (L + M) / 2
    drop_y = float(mid[1])
    go(-0.80, -1.10, math.pi)
    go(-1.33, drop_y, math.pi, tol=0.04, timeout=35)

    say("4) set it down upright between the cabbage and the cup")
    top = ex.geom_info("standardcounterheightwidth")["aabb_max"][2]
    r = ex.put_down(float(mid[0]) + 0.11, drop_y, top)
    say("  ", r)

    say("5) back away")
    go(-1.18, drop_y, math.pi)
    ex.run_seconds(1.0)

    score = _score(ex, TASK["prompt"], TASK["success"], "code policy", "bowl_policy (from Astra's run)",
                   0, 0.0, {})
    score["bowl_final"] = [round(float(v), 3) for v in ex.pos("bowl")]
    json.dump(score, open(os.path.join(args.out, "score.json"), "w"), indent=2)
    ex.save_video(os.path.join(args.out, "episode.mp4"))
    say("RESULT", json.dumps({k: score[k] for k in ("success", "placed", "quadrant", "bowl_final",
                                                    "furniture_contacts", "furniture_peak_N",
                                                    "person_down", "person_hit_at_s", "robot_fell")}))
    if not args.headless:
        say("done — close a window to exit")
        while True:
            ex.run_seconds(0.2)
except (ViewerClosed, SystemExit) as e:
    say("stopped:", e)
finally:
    ex.close()
    sys.stdout.flush()
    os._exit(0)
