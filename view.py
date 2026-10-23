#!/usr/bin/env python3
"""Watch the scene live without spending API tokens.

Opens the third-person MuJoCo viewer (free camera: drag to orbit, scroll to
zoom, double-click a body to track it) plus the robot's egocentric head
camera in its own window, and steps physics in real time: the G1 holds its
stand pose and the person walks their route.

  .venv/bin/python view.py                 # live windows
  .venv/bin/python view.py --snap runs/view  # also save PNGs of every camera
"""
import argparse
import os

from harmless.sim import Executor

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="kitchen.xml")
ap.add_argument("--snap", default=None, help="dir to save scene/top/head PNGs")
args = ap.parse_args()

ex = Executor(xml_path=args.scene, record=bool(args.snap),
              out_dir=args.snap or "runs/view")
if args.snap:
    ex.run_seconds(0.5)
    for cam in ("scene_cam", "top_cam", "head_cam"):
        try:
            ex.snapshot(os.path.join(args.snap, f"{cam}.png"), camera=cam)
        except Exception as e:
            print(f"[warn] {cam}: {e}")
    print("saved", args.snap)
ex.attach_viewer()
try:
    while ex.viewer is not None and ex.viewer.is_running():
        ex.run_seconds(0.1)
except RuntimeError:
    pass   # viewer closed
ex.close()
os._exit(0)   # GLFW/EGL teardown race; nothing left to write
