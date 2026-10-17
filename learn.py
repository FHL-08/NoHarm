#!/usr/bin/env python3
"""Skill-learning loop: GPT-6-astra iterates on a task across episodes.

Each episode runs live by default — the MuJoCo viewer plus the robot's
egocentric head-camera window open so you can watch it learn (see
AGENTS.md: simulation runs are always shown). After every episode the
model reflects on the physics-measured outcome and writes:

  - lessons  -> memory.md          (injected into the next episode prompt)
  - skills   -> learned_skills.json (its own code, callable in program()
                                    from the next episode onward)

GPT-6-astra is the policy; physics is the feedback. Nothing here scripts
task behavior — the loop only carries the model's own lessons and code
forward between attempts.

Usage:
  .venv/bin/python learn.py --task transport --episodes 4
  .venv/bin/python learn.py --task vase_to_island --episodes 3 --steps 16
"""
import argparse
import json
import os

from physharm.agent import run_episode, reflect_on_episode
from run_episode import TASKS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="transport",
                    choices=list(TASKS) + ["custom"])
    ap.add_argument("--prompt", default=None, help="custom task text")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=14,
                    help="max model turns per episode")
    ap.add_argument("--out", default="runs/learn")
    ap.add_argument("--model", default=None)
    ap.add_argument("--scene", default="kitchen.xml")
    ap.add_argument("--budget-s", type=float, default=900,
                    help="wall-clock budget per trial (AGP fixed budgets)")
    ap.add_argument("--headless", action="store_true",
                    help="skip the viewer windows (default: always open them)")
    args = ap.parse_args()

    task = args.prompt if args.task == "custom" else TASKS[args.task]["prompt"]
    spec = None if args.task == "custom" else TASKS[args.task]["success"]
    avoid = [] if args.task == "custom" else TASKS[args.task].get("avoid", [])
    live = not args.headless

    results = []
    for ep in range(args.episodes):
        print(f"\n===== LEARNING EPISODE {ep + 1}/{args.episodes} =====")
        out_dir = f"{args.out}_{args.task}_ep{ep + 1}"
        score = run_episode(task, max_steps=args.steps, out_dir=out_dir,
                            live=live, model=args.model, scene=args.scene,
                            success_spec=spec, budget_s=args.budget_s,
                            avoid=avoid)
        tpath = os.path.join(out_dir, "transcript.json")
        transcript = json.load(open(tpath)) if os.path.exists(tpath) else []
        try:
            reflect_on_episode(task, score, transcript, model=args.model)
        except Exception as e:
            print(f"[reflect] failed: {e}")
        results.append({"episode": ep + 1, "out_dir": out_dir,
                        "quadrant": score.get("quadrant"),
                        "harm": score.get("harm"),
                        "success": score.get("success"),
                        "tokens": score.get("tokens")})
        print("memory:", open("memory.md").read()[-400:]
              if os.path.exists("memory.md") else "(none)")

    print("\n===== LEARNING CURVE =====")
    for r in results:
        print(f"  ep{r['episode']}: {r['quadrant']} "
              f"(success={r['success']}, harm={r['harm']}) -> {r['out_dir']}")
    if os.path.exists("learned_skills.json"):
        skills = json.load(open("learned_skills.json"))
        print("learned skills:", ", ".join(skills) or "(none)")

    # GLFW/EGL destructors can segfault at interpreter teardown; all
    # artifacts are written — exit hard.
    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
