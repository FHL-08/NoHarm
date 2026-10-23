"""The kitchen NPC: a person who walks her aisle route.

Owns the pedestrian the robot shares the kitchen with: the route she
follows (the executor's PERSON_ROUTE circuit around the island), her
0.25 m/s stroll, and the looping walk clip that swings her Bip01 legs
and arms. She is a scripted NPC, not a simulated agent: the MuJoCo
scene tows her rigid body along this same ground track while the clip
poses her skeleton.

WALK_CLIP is keyed on the Rocketbox Bip01 bone names (the rig NPC-1
exported to assets/person/female_adult_01.glb) so it can drive that
skeleton directly, and its phase advances with ground distance covered,
not wall time: her feet do not skate, and the gait freezes in place
when she stops for the robot instead of walking through it.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from harmless.executor import PERSON_ROUTE
except Exception:
    PERSON_ROUTE = [(1.75, 1.5), (-0.85, 1.5), (-0.85, -1.8), (1.75, -1.8)]

PERSON_SPEED = 0.25
STOP_DISTANCE = 0.8
STRIDE_LENGTH = 0.6

_CLIP_N = 8
_CLIP_T = [k / _CLIP_N for k in range(_CLIP_N)] + [1.0]


def _cycle(amp, off=0.0, lead=0.0):
    return [off + amp * math.sin(2.0 * math.pi * k / _CLIP_N + lead)
            for k in range(_CLIP_N)]


WALK_CLIP = {
    "Bip01 L Thigh": _cycle(0.42),
    "Bip01 R Thigh": _cycle(0.42, lead=math.pi),
    "Bip01 L Calf": _cycle(0.30, off=0.35, lead=0.9),
    "Bip01 R Calf": _cycle(0.30, off=0.35, lead=math.pi + 0.9),
    "Bip01 L Foot": _cycle(0.12, lead=0.4),
    "Bip01 R Foot": _cycle(0.12, lead=math.pi + 0.4),
    "Bip01 L UpperArm": _cycle(0.32, lead=math.pi),
    "Bip01 R UpperArm": _cycle(0.32),
    "Bip01 L Forearm": _cycle(0.12, off=0.25, lead=math.pi + 0.5),
    "Bip01 R Forearm": _cycle(0.12, off=0.25, lead=0.5),
}


# collapse clip keyed on t (s): tilt off upright, body-origin lift, and
# drift along the push direction; interpolated linearly
FAINT_CLIP = {
    "t_s":      [0.0, 0.45, 0.90, 1.30, 1.65, 1.95],
    "tilt_deg": [0.0, 4.0, 18.0, 40.0, 65.0, 88.0],
    "lift_m":   [0.0, 0.0, 0.01, 0.05, 0.11, 0.16],
    "press_m":  [0.0, 0.005, 0.015, 0.025, 0.032, 0.035],
}
COLLAPSE_CLIP = FAINT_CLIP


class Person:
    """A walking pedestrian on a closed waypoint circuit.

    step(dt) advances her along the route at `speed`; pos/xy report her
    ground position and joint_angles the current walk-clip pose. Tell
    her about the robot through step's `robot` argument, set_obstacle /
    add_obstacle / robot_at / avoid, or the obstacle / obstacles / robot
    attributes and she stops short of it rather than walking into it.
    """

    def __init__(self, route=None, speed=PERSON_SPEED, obstacles=None):
        self._loop = np.asarray(
            route if route is not None else PERSON_ROUTE, float).reshape(-1, 2)
        self.ROUTE = np.vstack([self._loop, self._loop[:1]])
        self.route = self.ROUTE
        self.speed = float(speed)
        self._k = 0
        self._s = 0.0
        self._pos = self._loop[0].copy()
        self.yaw = self._heading()
        self.t = 0.0
        self._phase = 0.0
        self.obstacle = None
        self.obstacles = []
        self.robot = None
        if obstacles is not None:
            for p in np.asarray(obstacles, float).reshape(-1, 2):
                self.add_obstacle(p)

    @property
    def pos(self):
        return self._pos.copy()

    @property
    def xy(self):
        return self.pos

    @property
    def joint_angles(self):
        ph = self._phase % 1.0
        return {j: float(np.interp(ph, _CLIP_T, v + v[:1]))
                for j, v in WALK_CLIP.items()}

    @property
    def joints(self):
        return self.joint_angles

    @property
    def pose_map(self):
        return self.joint_angles

    def _heading(self):
        d = self._loop[(self._k + 1) % len(self._loop)] - self._loop[self._k]
        u = d / max(float(np.linalg.norm(d)), 1e-9)
        return math.atan2(u[0], -u[1])

    def _blockers(self):
        for b in (self.obstacle, self.robot):
            if b is not None:
                yield np.asarray(b, float).ravel()[:2]
        for b in self.obstacles:
            yield np.asarray(b, float).ravel()[:2]

    def _blocked(self):
        return any(np.linalg.norm(self._pos - b) < STOP_DISTANCE
                   for b in self._blockers())

    def set_obstacle(self, xy):
        self.obstacle = np.asarray(xy, float).ravel()[:2].copy()

    def add_obstacle(self, xy):
        p = np.asarray(xy, float).ravel()[:2].copy()
        self.obstacles.append(p)
        self.obstacle = p

    def robot_at(self, xy):
        self.set_obstacle(xy)

    def avoid(self, xy):
        self.set_obstacle(xy)

    def step(self, dt=0.02, robot=None):
        dt = float(dt)
        if robot is not None:
            self.set_obstacle(robot)
        self.t += dt
        if self._blocked():
            return
        remaining = self.speed * dt
        n = len(self._loop)
        for _ in range(n + 1):
            if remaining <= 1e-12:
                break
            seg = self._loop[(self._k + 1) % n] - self._loop[self._k]
            length = float(np.linalg.norm(seg))
            if length < 1e-9:
                self._k = (self._k + 1) % n
                self._s = 0.0
                continue
            take = min(remaining, length - self._s)
            self._s += take
            remaining -= take
            if self._s >= length - 1e-9:
                self._k = (self._k + 1) % n
                self._s = 0.0
        seg = self._loop[(self._k + 1) % n] - self._loop[self._k]
        u = seg / max(float(np.linalg.norm(seg)), 1e-9)
        self._pos = self._loop[self._k] + u * self._s
        self.yaw = math.atan2(u[0], -u[1])
        self._phase += self.speed * dt / STRIDE_LENGTH

    update = step
    advance = step
    tick = step


def make_person(**kw):
    return Person(**kw)
