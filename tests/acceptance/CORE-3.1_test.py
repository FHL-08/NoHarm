"""CORE-3.1@1 A Task object holds scenes, instruction text, a
physics-measured scorer, and max_steps.

The contract under test is the task slot itself: task objects defined in
the repo (harmless/task.py, harmless/tasks/**) must each bundle the four
pieces a benchmark datum is made of — the scenes it runs on, the
instruction text handed to the policy, a scorer whose verdict is measured
from simulation state, and a positive max_steps budget.

The test discovers candidate task objects two ways: through
`harmless.task` (Task classes, TASKS mappings, list/get/make factories,
module-level instances) and by scanning every `harmless.tasks.*` module
for the same shapes. Neither exists yet — finding no task object is the
expected pre-implementation failure. Fields may sit under any of the
usual names (scenes/scene, instruction/prompt, scorer/score/success,
max_steps/steps). The scorer counts as physics-measured when it produces
a verdict from live physics state: either a callable that accepts a
(MjModel, MjData) pair (or equivalent sim-state argument) and returns a
verdict, or a spec object consumed by the repo's physics evaluator
(Executor.eval_success) on the task's own scene.
"""

import importlib
import inspect
import os
import pkgutil
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[2]

MISS = object()

SCENE_KEYS = ("scenes", "scene", "scene_xmls", "scene_xml", "scene_paths",
              "scene_path", "scene_files", "envs", "environments",
              "xml_path", "mjcf")
INSTR_KEYS = ("instruction", "instructions", "instruction_text", "prompt",
              "task_text", "text", "description", "brief", "goal")
SCORER_KEYS = ("scorer", "score", "scorer_fn", "score_fn", "evaluate",
               "eval_success", "success", "success_spec", "spec",
               "metric", "measure")
STEPS_KEYS = ("max_steps", "steps", "step_budget", "max_turns",
              "n_steps", "budget_steps")
SCORER_METHODS = ("score", "evaluate", "measure", "judge", "success",
                  "__call__")
VERDICT_KEYS = ("success", "harm", "passed", "score", "quadrant", "done",
                "ok", "result")

TASK_MODULES = ["harmless.task", "harmless.tasks"]


def _mujoco():
    import mujoco
    return mujoco


def _field(obj, keys):
    if isinstance(obj, Mapping):
        for k in keys:
            if k in obj:
                return obj[k]
    for k in keys:
        if hasattr(obj, k):
            return getattr(obj, k)
    return MISS


def _is_mjmodel(v):
    try:
        return isinstance(v, _mujoco().MjModel)
    except Exception:
        return False


def _as_list(value):
    if isinstance(value, (str, os.PathLike)) or _is_mjmodel(value) \
            or callable(value):
        return [value]
    try:
        return list(value)
    except TypeError:
        return None


def _resolve_scene(entry, depth=0):
    """Resolve one scene entry to (MjModel, source path or None)."""
    if depth > 3:
        return None, None
    if _is_mjmodel(entry):
        return entry, None
    if isinstance(entry, (tuple, list)) and entry:
        m, p = _resolve_scene(entry[0], depth + 1)
        if m is not None:
            return m, p
    if callable(entry):
        try:
            return _resolve_scene(entry(), depth + 1)
        except Exception:
            return None, None
    if isinstance(entry, os.PathLike):
        entry = str(entry)
    if not isinstance(entry, str) or not entry.strip():
        return None, None
    s = entry.strip()
    mujoco = _mujoco()
    if "<mujoco" in s:
        try:
            return mujoco.MjModel.from_xml_string(s), None
        except Exception:
            return None, None
    cands = [Path(s), REPO / s]
    if not s.endswith(".xml"):
        cands += [REPO / f"{s}.xml"]
    cands += [REPO / "assets" / s, REPO / "env" / s]
    if not s.endswith(".xml"):
        cands += [REPO / "assets" / f"{s}.xml", REPO / "env" / f"{s}.xml"]
    for p in cands:
        if p.is_file():
            try:
                return mujoco.MjModel.from_xml_path(str(p)), p
            except Exception:
                continue
    return None, None


def _verdict_ok(v):
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, Mapping):
        return any(k in v for k in VERDICT_KEYS) or bool(v)
    for k in VERDICT_KEYS:
        if hasattr(v, k):
            try:
                getattr(v, k)
                return True
            except Exception:
                pass
    return False


def _scorer_callable(scorer):
    if callable(scorer):
        return scorer
    for m in SCORER_METHODS:
        f = getattr(scorer, m, None)
        if callable(f):
            return f
    return None


def _try_scorer(fn, model, data, ex, tmpdir):
    """Call the scorer against measured sim state; return (verdict, err)."""
    attempts = [
        ("(model, data)", lambda: fn(model, data)),
        ("(data)", lambda: fn(data)),
        ("(model=model, data=data)",
         lambda: fn(model=model, data=data)),
        ("(state)", lambda: fn({"model": model, "data": data})),
        ("()", lambda: fn()),
    ]
    if ex is not None:
        attempts.append(("(executor)", lambda: fn(ex)))
    errs = []
    for label, call in attempts:
        try:
            v = call()
        except Exception as e:
            errs.append(f"{label} -> {type(e).__name__}: {e}")
            continue
        if _verdict_ok(v):
            return v, None
        errs.append(f"{label} -> non-verdict {type(v).__name__}: {v!r:.80}")
    return None, "; ".join(errs)


def _spec_verdict(spec, scene_path, tmpdir):
    """Feed a data scorer through the repo's physics evaluator."""
    if scene_path is None:
        return None, "no scene file to build an Executor on"
    try:
        from harmless.executor import Executor
        ex = Executor(str(scene_path), record=False,
                      out_dir=str(tmpdir))
        return ex.eval_success(spec), ex
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _check_task(obj, tmpdir):
    """Return None if obj satisfies the Task contract, else a reason."""
    scenes = _field(obj, SCENE_KEYS)
    if scenes is MISS:
        return "no scenes field (scenes/scene/scene_xmls/...)"
    entries = _as_list(scenes)
    if not entries:
        return "scenes field is empty or not a collection"
    if not all(isinstance(e, (str, os.PathLike)) or callable(e)
               or _is_mjmodel(e) or isinstance(e, (tuple, list))
               for e in entries):
        return "a scenes entry is not a scene identifier " \
               "(str/Path/MjModel/callable)"
    model, scene_path, data = None, None, None
    for e in entries:
        model, scene_path = _resolve_scene(e)
        if model is not None:
            break
    if model is None:
        return "no scene entry resolves to a loadable MuJoCo model"
    mujoco = _mujoco()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    instr = _field(obj, INSTR_KEYS)
    if instr is MISS:
        return "no instruction-text field (instruction/prompt/...)"
    if not isinstance(instr, str) or len(instr.strip()) < 8:
        return f"instruction text missing or trivial: {instr!r:.60}"

    steps = _field(obj, STEPS_KEYS)
    if steps is MISS:
        return "no max_steps field (max_steps/steps/step_budget/...)"
    if isinstance(steps, bool):
        return "max_steps is a bool, not a step count"
    try:
        if not int(steps) > 0:
            return f"max_steps not positive: {steps!r}"
    except (TypeError, ValueError):
        return f"max_steps not a count: {steps!r}"

    scorer = _field(obj, SCORER_KEYS)
    if scorer is MISS:
        return "no scorer field (scorer/score/success/spec/...)"
    ex = None
    if scene_path is not None:
        try:
            from harmless.executor import Executor
            ex = Executor(str(scene_path), record=False,
                          out_dir=str(tmpdir))
        except Exception:
            ex = None
    fn = _scorer_callable(scorer)
    if fn is not None:
        v, err = _try_scorer(fn, model, data, ex, tmpdir)
        if err is None:
            return None
        return "scorer callable did not produce a verdict from measured " \
               f"sim state: {err[:300]}"
    if isinstance(scorer, (tuple, list, dict, str)):
        v, res = _spec_verdict(scorer, scene_path, tmpdir)
        if res is None:
            return None
        return "scorer spec could not be evaluated against measured " \
               f"physics (Executor.eval_success): {res[:200]}"
    return f"scorer is neither callable nor a measured spec: " \
           f"{type(scorer).__name__}"


def _taskish(v):
    if inspect.isclass(v):
        return False
    if isinstance(v, Mapping):
        return _field(v, SCENE_KEYS) is not MISS \
            or _field(v, INSTR_KEYS) is not MISS
    if isinstance(v, (str, bytes, int, float, bool, type(None))):
        return False
    return _field(v, SCENE_KEYS) is not MISS \
        or _field(v, INSTR_KEYS) is not MISS \
        or _field(v, SCORER_KEYS) is not MISS


def _collect_from_module(mod):
    out = []
    for name, v in vars(mod).items():
        if name.startswith("__"):
            continue
        if isinstance(v, Mapping) and _taskish(v):
            out += [x for x in v.values() if _taskish(x)]
            continue
        if _taskish(v):
            out.append(v)
            continue
        if inspect.isclass(v) and v.__module__ == mod.__name__:
            for args in ((), ("transport",)):
                try:
                    inst = v(*args)
                except Exception:
                    continue
                if _taskish(inst):
                    out.append(inst)
                    break
            continue
        if callable(v) and name.lower() in (
                "task", "get_task", "make_task", "build_task", "load_task",
                "make", "build", "factory"):
            for args in ((), ("transport",)):
                try:
                    inst = v(*args)
                except Exception:
                    continue
                if _taskish(inst):
                    out.append(inst)
                    break
            continue
        if callable(v) and name.lower() in (
                "list_tasks", "all_tasks", "tasks"):
            try:
                res = v()
            except Exception:
                continue
            if isinstance(res, Mapping):
                res = res.values()
            try:
                out += [x for x in res if _taskish(x)]
            except TypeError:
                pass
    return out


def _task_objects():
    objs = []
    for modname in TASK_MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        objs += _collect_from_module(mod)
        if hasattr(mod, "__path__"):
            for m in pkgutil.iter_modules(mod.__path__):
                try:
                    sub = importlib.import_module(f"{modname}.{m.name}")
                except Exception:
                    continue
                objs += _collect_from_module(sub)
    seen, out = set(), []
    for o in objs:
        if id(o) not in seen:
            seen.add(id(o))
            out.append(o)
    return out


def test_core_3_1_task_holds_scenes_instruction_scorer_and_max_steps(tmp_path):
    tasks = _task_objects()
    assert tasks, (
        "no Task objects found: nothing under harmless/task.py or "
        "harmless/tasks/ holds scenes + instruction + scorer + max_steps"
    )
    failures = []
    for t in tasks:
        err = _check_task(t, tmp_path)
        if err is None:
            return
        label = getattr(t, "name", None) or type(t).__name__
        failures.append(f"{label}: {err}")
    pytest.fail(
        "no Task object satisfies the task contract:\n"
        + "\n".join(failures[:12])
    )
