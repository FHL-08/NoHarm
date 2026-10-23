"""POL-1.2@1 program() code runs in a no-imports sandbox limited to the
bridge API, np, math, and print, inside a sim-step budget.

The contract under test is the code-execution surface the model writes
against each turn. program() is the executor's primitive — reachable
through the episode loop's {"skill": "program", "args": {"code": ...}}
call — so the test drives it the way the agent harness does, on the real
kitchen scene:

  no imports:      import statements and __import__ must fail inside the
                   program — the model's code cannot pull in host modules;
  limited surface: host capabilities (open, eval, exec, compile, input,
                   globals/locals/vars, getattr/setattr/delattr, exit,
                   memoryview, breakpoint) must be absent from the
                   program's namespace, while np, math, print and a real
                   bridge API (observation AND motion/time primitives)
                   must be present and callable;
  result path:     sandboxed failures come back in the result dict, not
                   as exceptions escaping program();
  sim budget:      a program asking for far more physics than the budget
                   allows (wait(10000) — 10000 simulated seconds) must be
                   cut off with an error after a bounded amount of sim
                   time, not run to completion.

The budget check gives the run up to 240 s of wall time to end on its
own (the shipped budget is 80 s of physics ≈ 40000 steps ≈ 105 s wall on
this scene), then trips the known deadline internals so an absent budget
still fails in finite time instead of hanging the suite.
"""

import inspect
import os
import re
import threading
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]
SCENE = str(REPO / "kitchen.xml")

BLOCKED_STATEMENTS = (
    "import os",
    "import sys",
    "import numpy",
    "from math import sqrt",
    "__import__('os')",
)

# host capabilities a "bridge API + np + math + print" surface cannot
# legitimately expose
BLOCKED_NAMES = (
    "open", "eval", "exec", "compile", "input",
    "globals", "locals", "vars",
    "getattr", "setattr", "delattr",
    "__import__", "breakpoint", "memoryview", "exit", "quit",
)

OBSERVE_NAMES = ("state", "sense", "scene_geoms", "pos", "geom_info",
                 "contacts", "joint_state", "qpos", "qvel", "ray",
                 "history", "joint_names", "robot_joints", "palm_gaps",
                 "walker_status", "controller_status", "arm_status")
ACT_NAMES = ("set", "set_joints", "move_joints", "step", "wait", "reach",
             "reach_xyz", "grasp", "release", "walk", "walker_off",
             "arm_reach", "arm_path", "hands_path", "pick_up",
             "put_down", "set_controller", "annotate")

BUDGET_WALL_S = 240
BUDGET_ERROR = re.compile(
    r"budget|exceed|exhaust|limit|cap|maximum|steps|sim", re.I)


def _executor(tmp_path):
    from harmless.executor import Executor
    kwargs = {
        "record": False,
        "out_dir": str(tmp_path / "run"),
        "skills_path": str(tmp_path / "no_skills.json"),
        "ws_dir": str(tmp_path / "ws"),
    }
    try:
        params = inspect.signature(Executor).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    except (TypeError, ValueError):
        pass
    return Executor(SCENE, **kwargs)


def _ran_ok(out):
    return isinstance(out, dict) and "error" not in out \
        and out.get("ok") is not False


def _stdout(out):
    return str(out.get("stdout", "")) if isinstance(out, dict) else ""


def test_pol_1_2_program_sandbox_surface_and_sim_budget(tmp_path):
    ex = _executor(tmp_path)
    try:
        program = getattr(ex, "program", None)
        assert callable(program), (
            "the executor exposes no program(code) entry point for the "
            "model's code submissions"
        )

        # ---- allowed surface: np, math, print do real work ------------
        out = program("print('VAL', np.round(math.pi, 4))")
        assert _ran_ok(out) and "3.1416" in _stdout(out), (
            f"np + math + print are not usable inside program(): {out}"
        )

        # ---- the bridge API is present: at least one observation and
        # one motion/time primitive must be reachable by name -----------
        present = set()
        for name in OBSERVE_NAMES + ACT_NAMES:
            probe = program(f"print('HAS_{name.upper()}', {name})")
            if _ran_ok(probe) and f"HAS_{name.upper()}" in _stdout(probe):
                present.add(name)
        observe = present & set(OBSERVE_NAMES)
        act = present & set(ACT_NAMES)
        assert observe and act, (
            f"program() exposes no usable bridge API "
            f"(observation={sorted(observe)}, motion={sorted(act)})"
        )
        usable = False
        for expr in ("state()", "sense()", "scene_geoms()", "contacts()",
                     "joint_state()", "walker_status()",
                     "controller_status()", "joint_names",
                     "robot_joints"):
            out = program(f"print('BRIDGE', {expr})")
            if _ran_ok(out) and "BRIDGE" in _stdout(out):
                usable = True
                break
        assert usable, (
            "no zero-argument bridge primitive (state(), sense(), "
            "scene_geoms(), contacts(), joint_state(), ...) actually "
            "runs inside program()"
        )

        # ---- no imports ------------------------------------------------
        for code in BLOCKED_STATEMENTS:
            out = program(code)
            assert isinstance(out, dict) and (
                "error" in out or out.get("ok") is False), (
                f"program() allowed `{code}` — the sandbox is not "
                f"no-imports (result: {out})"
            )

        # ---- nothing outside bridge/np/math/print ----------------------
        for name in BLOCKED_NAMES:
            out = program(f"print('{name.upper()}', {name})")
            assert isinstance(out, dict) and (
                "error" in out or out.get("ok") is False), (
                f"program() exposes host capability `{name}` — the "
                f"sandbox must be limited to the bridge API, np, math "
                f"and print (result: {out})"
            )

        # ---- the same sandbox guards the agent-facing skill route ------
        run_skill = getattr(ex, "run_skill", None)
        if callable(run_skill):
            rec = run_skill({"skill": "program",
                             "args": {"code": "import os"}})
            res = rec.get("result", {}) if isinstance(rec, dict) else {}
            assert isinstance(res, dict) and (
                "error" in res or res.get("ok") is False), (
                f"the program skill route bypasses the sandbox: {rec}"
            )

        # ---- sim-step budget -------------------------------------------
        got = {}

        def run_long():
            got["out"] = program("wait(10000)")

        th = threading.Thread(target=run_long, daemon=True)
        th.start()
        th.join(BUDGET_WALL_S)
        natural = not th.is_alive()
        if not natural:
            # trip whatever deadline internals exist so an absent budget
            # still ends the test instead of hanging it
            for attr, val in (("_sim_deadline", 0.0), ("step_budget", 0)):
                try:
                    if hasattr(ex, attr):
                        setattr(ex, attr, val)
                except Exception:
                    pass
            try:
                ex.data.time = float("inf")
            except Exception:
                pass
            th.join(60)
        assert not th.is_alive(), (
            f"program('wait(10000)') still running after "
            f"{BUDGET_WALL_S + 60} s of wall time — no sim-step budget "
            "bounds a program"
        )
        out = got["out"]
        assert isinstance(out, dict), (
            f"program('wait(10000)') returned {out!r}, not a result dict"
        )
        assert "error" in out or out.get("ok") is False, (
            f"program('wait(10000)') completed normally — the code ran "
            "outside any sim-step budget"
        )
        if natural:
            assert BUDGET_ERROR.search(str(out.get("error", ""))), (
                f"the budget cut-off should surface as a budget/step/"
                f"sim error, got: {out.get('error')}"
            )
            sim_s = out.get("program_sim_s")
            if sim_s is not None:
                assert float(sim_s) < 5000, (
                    f"program consumed {sim_s} sim-s of a requested "
                    "10000 — that is not a meaningful sim-step budget"
                )
    finally:
        try:
            ex.close()
        except Exception:
            pass
