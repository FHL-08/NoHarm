#!/usr/bin/env python3
"""Scripted check of the bridge's two-handed pickup, shown LIVE in the
MuJoCo viewer + head-camera window (this is NOT GPT-6-astra — it verifies
the primitives Astra uses). The robot starts at the island's south edge
facing the bowl, leans in, puts both palms on the bowl's sides, grasps,
lifts it, and steps back. The window stays open until you close it."""
import os
import numpy as np
import mujoco
from harmless.executor import Executor, ViewerClosed

ex = Executor(record=False)
ex.body_avoid = {"standardislandheight"}
ex.overlay_objects = ["bowl"]
ex.data.qpos[0:3] = [0.28, -1.12, 0.79]                       # island's south edge
ex.data.qpos[3:7] = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]   # facing +y (the bowl)
mujoco.mj_forward(ex.model, ex.data)
ex.attach_viewer()


def say(*a):
    print(*a, flush=True)


try:
    ex.run_seconds(1.0)
    o = ex._object_info("bowl")
    lo, hi = np.array(o["aabb_min"]), np.array(o["aabb_max"])
    c = (lo + hi) / 2
    R = [hi[0] + 0.01, c[1], hi[2] - 0.03]      # right palm: bowl's +x side
    L = [lo[0] - 0.01, c[1], hi[2] - 0.03]      # left palm:  bowl's -x side
    ex.annotate([{"xyz": R, "label": "R palm target"}, {"xyz": L, "label": "L palm target"}])
    say("1) lean in"); ex.run_seconds(0.2)
    say("2) both palms to the bowl's sides"); ex.arm_reach(*R, side="right"); ex.arm_reach(*L, side="left")
    ex.run_seconds(3.0)
    say("   palm gaps", ex.palm_gaps("bowl"))
    say("3) grasp:", ex.grasp("bowl"))
    say("4) lift 12 cm")
    ex.arm_reach(R[0], R[1], R[2] + 0.12, side="right"); ex.arm_reach(L[0], L[1], L[2] + 0.12, side="left")
    ex.run_seconds(2.5)
    say("   bowl at", np.round(ex.pos("bowl"), 3), "held", ex.held)
    say("5) straighten up and step back with it")
    ex.set("waist_pitch_joint", 0.0); ex.run_seconds(1.0)
    ex.walk(-0.25, 0, 0); ex.run_seconds(2.0); ex.walk(0, 0, 0); ex.run_seconds(1.0)
    say("   bowl at", np.round(ex.pos("bowl"), 3), "held", ex.held,
        "robot upright", ex.data.xpos[ex.pelvis_body][2] > 0.6)
    say("done — close either window to exit")
    while True:
        ex.run_seconds(0.2)
except ViewerClosed:
    pass
finally:
    ex.close()
    os._exit(0)
