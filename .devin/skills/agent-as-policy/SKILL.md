# Agent as Policy (AGP) pipeline

Distilled from arXiv:2609.12541 — *Agent as Policy for Robotic
Manipulation* (Jia, Lin, Zhang, Zhang, Liu, Jiang). They ran GPT-6 Astra
at high thinking effort as the direct robot policy; this repo follows the
same architecture. **Every agent working in this repo must keep the code
conformant to this pipeline.**

## The pipeline (all stages mandatory)

1. **Preparation agent → task definition** happens offline (us): pick the
   task, initial scene, budgets, success criteria. Never runtime.
2. **Fresh session per trial.** Each episode starts with an empty model
   context: system prompt (interface docs) + task + initial observation.
   Prior *attempt logs* are excluded; persistent *experience* (below) is
   the only carryover.
3. **Bridge interface, not a policy.** The executor exposes observation
   and action primitives only. The model decides what to observe, how to
   interpret it, and which motion to request.
   - Robot→agent: scene frame + egocentric head camera + proprioception
     (`sense()`: pelvis pose/vel, IMU gyro+accel, feet contacts, grip/tip,
     person position) + measured receipts after every action.
   - Robot→agent geometry/touch: `pos`, `scene_geoms`, `geom_info`,
     `ray`, `contacts`, and `touch_N` (pelvis/torso/elbows/hands/shins/
     feet) — the analogue of AGP's calibrated geometric queries.
   - Agent→robot: `program(code)` against the bridge (`set`,
     `set_joints`, `move_joints`, `reach_xyz`, `grasp`/`release`, `step`,
     `wait`). Like AGP's `move_joints`/`move_ee`/`gripper`: IK + joint
     trajectories turn targets into motion; physics executes them. There
     are deliberately NO task skills (no sweep/strike/walk/place).
4. **Runtime programming.** The model writes Python mid-execution to
   interpret observations, compute targets, and submit motions — not just
   pick canned skills. `program()` is the primary call.
5. **Feedback and adaptation.** After each action the model reads the
   receipt (measured forces, positions, contact counts, falls) plus fresh
   frames, and revises estimates/approach. Recovery is expected inside the
   trial budget.
6. **Fixed trial budgets.** `max_steps` action requests + `budget_s`
   wall-clock. A fall or unrecoverable state ends the trial; the score
   is recorded, not retried silently.
7. **Persistent experience across trials.** Fresh context every episode,
   but three stores persist and are injected next episode:
   - `memory.md` — distilled lessons (what worked/failed + measured why)
   - `learned_skills.json` — reusable `def name(...)` code the model
     wrote, callable inside `program()` from then on. The model writes
     these itself mid-episode with `save_skill(name, code, description)`
     (or at reflection); the assistant never authors entries.
   - `workspace/` — JSON key/value store (`save`/`load`/`workspace`)
     for measurements, fitted estimates, procedures, notes
   Reflection (`reflect_on_episode`) runs after each episode and writes
   the first two; the workspace is the model's own to maintain.
8. **Physical scoring, never self-grading.** Success criteria are
   measured state (`eval_success` specs: apple released near the bowl,
   mug resting on the counter) and harm is measured
   contact force / knockdown / near-miss / falls. The model's summary is
   narrative only.
9. **Efficiency discipline.** Instruct the model to batch consecutive
   operations into one `program()` call and `wait()` for completion —
   AGP showed this is where most wall-time is saved. Keep this
   instruction in the system prompt.
10. **Show the run.** Any simulation run opens the MuJoCo viewer + the
    egocentric head-camera window (`--live` is the default; `--headless`
    only for CI).

## Invariants to enforce when editing

- The assistant never adds task-solving logic to the executor. Helpers
  are *motions primitives* (IK reach, weld grasp); sequencing, balance,
  gait, and recovery are the model's job.
- Every number the model sees about the world must be physically
  measured (`sense()`, receipts). No fabricated ground truth beyond
  object positions, which are legitimately readable.
- Changing the API surface means updating the SYSTEM prompt in
  `harmless/agent.py` AND the REFLECT_SYSTEM allowlist to match.
- Model config is GPT-6 Astra (`gpt-6-astra`) by default; do not
  silently swap it.
