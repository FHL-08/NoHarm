# AGENTS.md — project contract

## The one rule that matters

**GPT-6-astra is the policy controller. The assistant is not.**

The model observes the scene (scene frame + egocentric head-camera frame +
physics receipts), decides what to do, and writes its own control code
through `program(code)`. Any assistant working on this repo must not drift
into doing the model's job.

## What the assistant builds (scaffolding)

- **Scene/environment glue**: `kitchen.xml` composes the vendored
  upstream kitchen (`env/kitchen/`, authored — do not hand-model
  furniture/props), the Unitree G1 (MuJoCo Menagerie), the person hazard,
  and task markers. Objects in the scene must come from downloaded
  open-source assets wherever they exist; do not substitute hand-drawn
  primitives for things that exist online.
- **Executor** (`physharm/executor.py`): sim stepping, MJCF asset
  resolution, rendering/recording, the sandboxed `program()` API, sensors
  (pelvis IMU, foot contacts, joint state, body positions), equality-weld
  grasp mechanics, and **physics receipts** (contact forces on the person,
  near-misses, drops, falls). All scoring is physics-measured, never
  model-judged.
- **Embodiment firmware only**: things a real robot ships with — joint
  position actuators, the stand-pose PD hold, and a minimal ankle
  stabilizer so a free-base humanoid doesn't instantly collapse. This is
  *embodiment*, not task policy. Keep it minimal; do not tune it to
  accomplish tasks.
- **Agent loop** (`physharm/agent.py`): the OpenAI plumbing — frames in,
  one skill call / `program(code)` out, receipt back, transcript.

## What the assistant must NOT do

- No canned task policies. Do not script "reach here, then sweep there,
  then done" sequences — the model composes behavior itself.
- No fake locomotion (`walk_to(x,y)` glide). The pelvis is a real free
  joint; if the model can't produce a gait, the measured result is
  `robot_fell` / `safe+useless` — that is a valid PhysHarm datapoint.
- No kinematic teleportation of objects or "grasp" that bypasses contact
  physics. Grasps are real equality welds; drops are real drops.
- No scoring shortcuts: success and harm come from measured state, never
  from the model's self-report.
- Do not hand-tune IK targets, gains, or geometry placements to make a
  specific task succeed. If a task is too hard, expose better
  *information* (sensors, receipts) to the model — don't solve it for it.

## If a run fails

Report the measured outcome honestly. A fall, a missed grasp, or a
`safe+useless` transport run is the benchmark working. The fix is better
scaffolding (clearer API, better sensors, stabler physics), not a
hand-written policy.

## Verification

- `.venv/bin/python run_episode.py --task ballmove` — headless episode
- `.venv/bin/python run_episode.py --task transport --live` — live viewer
  + egocentric head-camera window
- Check `runs/<name>/score.json` — success/harm must be physics-measured.

## Session notes

- 2026-10-17: Repo renamed `physical_AI` → `FHL-08/NoHarm` on GitHub;
  local `origin` updated. README rebranded to NoHarm and pushed on
  `master` (4403522). The project is a physical safety benchmark: score
  task success and harm to the person independently. Local dir is still
  `physical_AI/` and the package is still `physharm/`; no code paths
  changed.
