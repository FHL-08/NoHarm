#!/usr/bin/env python3
"""Continuous GPT-6-astra session in ONE live scene (viewer + head-camera
window stay open the whole time). Trials reset in place; the conversation,
notes and self-written skills persist. Close either window to stop.

  .venv/bin/python live.py --task bowl_between
  .venv/bin/python live.py --task bowl_between --trials 100 --turns 60
"""
import argparse
import os
import sys

from physharm.session import run_session
from run_episode import TASKS

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="bowl_between", choices=list(TASKS))
ap.add_argument("--trials", type=int, default=50)
ap.add_argument("--turns", type=int, default=40, help="model turns per trial")
ap.add_argument("--out", default="runs/session")
ap.add_argument("--model", default=None)
args = ap.parse_args()

t = TASKS[args.task]
import time
run_session(t["prompt"], success_spec=t["success"], avoid=t.get("avoid"),
            trials=args.trials, turns_per_trial=args.turns,
            out_dir=os.path.join(args.out, f"{args.task}_{time.strftime('%m%d_%H%M%S')}"),
            model=args.model)
sys.stdout.flush()
os._exit(0)   # GLFW/EGL teardown race; everything is written
