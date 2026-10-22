#!/usr/bin/env python3
"""Replay a GPT-6-astra run in the live viewer, no API calls.

Re-executes the model's OWN programs from a saved transcript, in order,
with the same physics time between them that elapsed while it was
thinking (the sim kept running then too), using the skills and workspace
it had saved. Physics is deterministic, so the run plays out the same.

  .venv/bin/python demo_replay.py                       # golden bowl run
  .venv/bin/python demo_replay.py --run runs/golden/bowl_between_trial1
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

from harmless.executor import Executor, ViewerClosed
from harmless.session import _configure, _score
from run_episode import TASKS

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/golden/bowl_between_trial1")
ap.add_argument("--task", default="bowl_between")
ap.add_argument("--out", default="runs/demo_replay")
ap.add_argument("--headless", action="store_true")
args = ap.parse_args()

src = args.run
transcript = json.load(open(os.path.join(src, "transcript.json")))
os.makedirs(args.out, exist_ok=True)
ws = tempfile.mkdtemp(prefix="noharm_ws_")
if os.path.isdir(os.path.join(src, "workspace")):
    shutil.copytree(os.path.join(src, "workspace"), ws, dirs_exist_ok=True)
skills = os.path.join(src, "learned_skills.json")

ex = Executor(record=True, out_dir=args.out, fps=10,
              skills_path=skills if os.path.exists(skills) else None, ws_dir=ws)
ex.save_skill = lambda *a, **k: {"saved_skill": "(replay: not persisted)"}
t = TASKS[args.task]
_configure(ex, t["success"], t.get("avoid"))
if not args.headless:
    ex.attach_viewer()
print(f"replaying {len(transcript)} turns of GPT-6-astra from {src}", flush=True)
try:
    for x in transcript:
        idle = float((x.get("while_you_thought") or {}).get("sim_s") or 0.0)
        if idle > 0:
            ex.step(int(round(idle / ex.model.opt.timestep)))   # it was thinking
        call = x["call"]
        if call.get("skill") == "done":
            print("model said done:", call.get("args", {}).get("summary", "")[:200], flush=True)
            break
        r = ex.run_skill(call)
        print(f"turn {x['turn']}: {json.dumps(r['result'])[:120]}", flush=True)
    score = _score(ex, t["prompt"], t["success"], "replay", "gpt-6-astra (replayed)",
                   len(transcript), 0.0, {})
    score["bowl_final"] = [round(float(v), 3) for v in ex.pos("bowl")]
    json.dump(score, open(os.path.join(args.out, "score.json"), "w"), indent=2)
    ex.save_video(os.path.join(args.out, "episode.mp4"))
    print("RESULT", json.dumps({k: score[k] for k in ("placed", "success", "quadrant", "bowl_final",
                                                      "furniture_contacts", "person_down")}), flush=True)
    if not args.headless:
        print("replay finished — close a window to exit", flush=True)
        while True:
            ex.run_seconds(0.2)
except ViewerClosed:
    pass
finally:
    ex.close()
    sys.stdout.flush()
    os._exit(0)
