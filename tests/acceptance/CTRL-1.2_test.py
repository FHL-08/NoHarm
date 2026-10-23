"""CTRL-1.2@1 The walker contains no navigation, target-seeking, or task
logic.

The contract under test is the low-level gait unit the executor's
walk(vx, vy, yaw_rate) engages: it must be a pure velocity-commanded
motor controller — GPT-6-astra owns everything above the gait
(navigation, avoidance, task composition). A walker that steers itself
toward a goal, follows paths, or encodes task steps is out of contract.

Checked through the public surface and the bound implementation:

- walk()'s signature takes only velocity components — no positional or
  goal arguments exist to aim it at
- the bound walker's class source carries no navigation, planning,
  target-seeking or task vocabulary (waypoints, path-following, A*,
  goals, obstacles, missions) and no public method accepts a goal
- walk()/status() report command state only — no goal progress
  (eta/distance/remaining) could be returned because no goal exists
- the executor and the program() bridge expose no locomotion navigation
  verbs (walk_to/navigate/go_to/...): the only steering channel is the
  velocity command
- behaviourally, a constant commanded rate is executed continuously —
  the gait does not decelerate into an implicit target heading

A walker wired to a navigation or task layer fails the signature,
source, bridge or behaviour probes; the same probes hold wherever the
gait unit lives (harmless/walker.py or a harmless/control/ package).
"""

import inspect
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]
SCENE = REPO / "kitchen.xml"

# walk() may only take body-frame velocity components
VEL_PARAM = re.compile(
    r"^(vx|vy|vz|v|vel|velocity|linear_velocity|cmd|command|speed|rate|"
    r"yaw|yaw_?rate|omega|w|wz|ang(ular)?_?(vel|velocity|rate)|"
    r"turn_?rate|heading_?rate)$", re.I)
GOAL_PARAM = re.compile(
    r"(goal|target|dest|waypoint|way_?point|landmark|beacon|"
    r"^x$|^y$|^z$|pos|position|point|location|coord|pose|"
    r"path|route|trajectory|nav)", re.I)

# navigation / task vocabulary that must not appear in the gait unit
NAV_SRC = re.compile(
    r"(navigat|waypoint|way_?point|walk_to|go_to|goto|seek|pathfind|"
    r"a_?star|astar|\brrt\b|move_base|follow_?path|path_?follow|"
    r"goal_(x|y|xy|pos|pose|point|yaw|head)|"
    r"target_(pos|pose|point|xy|world|goal|location)|world_target|"
    r"obstacle|avoid|slam|frontier|coverage|patrol|beacon|costmap|"
    r"occupancy|grid_map|potential_field|dijkstra|\bbfs\b|\bdfs\b|"
    r"\btask\b|mission|objective|instruction|planner)", re.I)
NAV_NAME = re.compile(
    r"(navigat|walk_to|go_to|goto|move_base|drive_to|head_to|seek|"
    r"follow_?path|path_?follow|patrol|travel|set_?goal|goal_|"
    r"target_|waypoint|route|plan_?path)", re.I)
NAV_ENTRY = re.compile(
    r"^(navigate|nav|walk_to|go_to|goto|move_base|drive_to|head_to|"
    r"seek|patrol|follow_?path|path_?follow|travel_to|nav_to|"
    r"base_?(move|goto|goal)|walk_?(path|route|goal|target)|"
    r"locomot.*(to|goal|target|point|dest))$", re.I)
PROGRESS = re.compile(
    r"(goal|target|dest|waypoint|eta|distance|dist_?to|progress|"
    r"arriv|remain|heading_?err|bearing|route|leg_?idx)", re.I)
BRIDGE_VERBS = ("navigate", "walk_to", "go_to", "goto", "move_base",
                "drive_to", "follow_path", "seek", "patrol", "head_to",
                "travel_to", "set_goal", "nav_to")


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


def _walker_of(ex):
    w = getattr(ex, "walker", None)
    if w is not None:
        return w
    try:
        vals = list(vars(ex).values())
    except TypeError:
        vals = [getattr(ex, n) for n in dir(ex)]
    for v in vals:
        if hasattr(v, "tick") and hasattr(v, "on") and \
                (hasattr(v, "set_cmd") or hasattr(v, "cmd")):
            return v
    return None


def _yaw(ex):
    w, x, y, z = np.asarray(ex.data.qpos[3:7], dtype=float)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def test_ctrl_1_2_walker_has_no_navigation_or_task_logic(tmp_path):
    ex = _executor(tmp_path)

    # --- walk() takes a velocity command and nothing else
    walk = getattr(ex, "walk", None)
    assert callable(walk), "the executor exposes no walk(vx, vy, yaw_rate)"
    params = [p for p in inspect.signature(walk).parameters.values()
              if p.name != "self"]
    assert 1 <= len(params) <= 4, (
        f"walk() takes {len(params)} arguments "
        f"({[p.name for p in params]}) — a velocity command is three "
        f"components at most")
    for p in params:
        n = p.name.lower()
        assert not GOAL_PARAM.search(n), (
            f"walk() parameter {p.name!r} looks like a goal/position — "
            f"the gait must be steered by velocity, not aimed at a point")
        assert VEL_PARAM.match(n), (
            f"walk() parameter {p.name!r} is not a recognisable velocity "
            f"component (vx/vy/yaw_rate)")

    # --- engage the walker; audit the bound implementation
    walk(0.0, 0.0, 0.0)
    w = _walker_of(ex)
    assert w is not None, (
        "walk(vx, vy, yaw_rate) engaged no low-level walker object on "
        "the executor")
    try:
        src = inspect.getsource(type(w))
    except (OSError, TypeError) as exc:
        pytest.fail(
            f"cannot read the walker implementation for audit: {exc}")
    hits = sorted({m.group(0).lower() for m in NAV_SRC.finditer(src)})
    assert not hits, (
        f"the walker implementation carries navigation/task vocabulary "
        f"{hits} — the gait unit must contain no navigation, "
        f"target-seeking or task logic")
    for name in dir(type(w)):
        if name.startswith("_"):
            continue
        fn = getattr(type(w), name, None)
        if not callable(fn):
            continue
        assert not NAV_NAME.search(name), (
            f"walker method {name!r} is a navigation/target-seeking "
            f"entry point — outside the gait's contract")
        try:
            ps = inspect.signature(fn).parameters.values()
        except (TypeError, ValueError):
            continue
        for p in ps:
            if p.name == "self":
                continue
            assert not GOAL_PARAM.search(p.name.lower()), (
                f"walker method {name}({p.name}) takes a goal/position "
                f"argument — the gait must take velocity commands only")

    # --- the steering state is a 3-velocity; status reports no progress
    st = walk(0.31, 0.12, -0.4)
    want = np.array([0.31, 0.12, -0.4])
    hits3 = []
    for n in dir(w):
        try:
            v = np.asarray(getattr(w, n), dtype=float)
        except Exception:
            continue
        if v.shape == (3,) and np.allclose(v, want, atol=5e-3):
            hits3.append(n)
    assert hits3, (
        "walk(vx, vy, yaw_rate) set no 3-vector velocity command on the "
        "walker — the command channel is not a body-frame velocity")
    if isinstance(st, Mapping):
        bad = [str(k) for k in st if PROGRESS.search(str(k))]
        assert not bad, (
            f"walk() status reports goal progress keys {bad} — a "
            f"velocity-commanded gait has no goal to progress toward")

    # --- no locomotion navigation entry points on executor or bridge
    nav_attrs = [n for n in dir(ex) if NAV_ENTRY.match(n)]
    assert not nav_attrs, (
        f"the executor exposes base-navigation entry points "
        f"{nav_attrs} — locomotion is commanded by walk(vx,vy,yaw_rate) "
        f"only")
    res = ex.program("walk(0.0, 0.0, 0.0)")
    assert "error" not in res, (
        f"walk() is not callable through the program() bridge: "
        f"{res.get('error')}")
    for verb in BRIDGE_VERBS:
        res = ex.program(verb)
        assert "error" in res, (
            f"the program() bridge defines a {verb!r} navigation verb — "
            f"steering belongs to the model, not the gait")

    # --- a commanded rate is executed continuously: nothing to arrive at
    dt = float(ex.model.opt.timestep)
    walk(0.0, 0.0, 0.6)
    y0 = _yaw(ex)
    ex.step(int(round(1.0 / dt)))
    dy1 = abs(_wrap(_yaw(ex) - y0))
    ex.step(int(round(1.0 / dt)))
    dy2 = abs(_wrap(_yaw(ex) - y0))
    assert dy1 > 0.1 and dy2 > dy1 + 0.1, (
        f"a constant yaw_rate command produced yaw deltas {dy1:.3f} "
        f"then {dy2 - dy1:.3f} rad/s — the gait does not keep executing "
        f"the commanded rate (it behaves like it is approaching a "
        f"target heading)")
