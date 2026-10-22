"""eval(task, policy, embodiment): run the triple, return the eval log.

Resolves a task, a policy and an embodiment — registry names, classes or
live objects — then drives the policy's act(observation) loop against
the task's real MuJoCo scenes for its step budget, one control tick per
turn, and returns the schema-versioned eval log: resolved config, git
revision, per-turn transcript, per-episode measured metrics and the
aggregate success/harm scores. Every scored quantity is measured from
simulation state, never reported by the policy; the saved log re-scores
offline through harmless.eval_log.rescore_log.
"""
from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

from .eval_log import (SCHEMA_VERSION, aggregate_scores, episode_metrics,
                       save_log)
from .executor import Executor, ViewerClosed
from .task import Task, TASKS

REPO = Path(__file__).resolve().parents[1]


def _resolve_task(task):
    if isinstance(task, str):
        if task not in TASKS:
            raise KeyError(
                f"unknown task '{task}' (known: {sorted(TASKS)})")
        return TASKS[task]
    if isinstance(task, Task):
        return task
    if isinstance(task, Mapping):
        return Task(**dict(task))
    if getattr(task, "instruction", None) is not None:
        return task
    raise TypeError(f"cannot resolve task {task!r}")


def _registry_name(obj, registry):
    for k, v in registry.items():
        if obj is v:
            return k
        if isinstance(v, type) and isinstance(obj, v):
            return k
    n = getattr(obj, "name", None)
    return n if isinstance(n, str) and n else type(obj).__name__.lower()


def _resolve_policy(policy):
    from .registry import POLICIES
    if isinstance(policy, str):
        return POLICIES[policy], None, policy
    cls = policy if isinstance(policy, type) else type(policy)
    inst = None if isinstance(policy, type) else policy
    return cls, inst, _registry_name(policy, POLICIES)


def _resolve_embodiment(embodiment):
    from .registry import EMBODIMENTS
    if isinstance(embodiment, str):
        return EMBODIMENTS[embodiment], None, embodiment
    cls = embodiment if isinstance(embodiment, type) else type(embodiment)
    inst = None if isinstance(embodiment, type) else embodiment
    return cls, inst, _registry_name(embodiment, EMBODIMENTS)


def _scenes(task, scene, overrides, kwargs):
    pick = scene or overrides.get("scene") or overrides.get("scenes") \
        or kwargs.get("scenes") or kwargs.get("scene_xml") \
        or kwargs.get("xml") or kwargs.get("xml_path") \
        or getattr(task, "scenes", None) or ["kitchen.xml"]
    if isinstance(pick, (str, os.PathLike)):
        pick = [pick]
    return [str(s) for s in pick]


def _configure_avoid(ex, task):
    avoid = getattr(task, "avoid", None) or []
    if isinstance(avoid, str):
        avoid = [avoid]
    avoid = dict(avoid) if isinstance(avoid, Mapping) \
        else {"objects": list(avoid)}
    objs = set(avoid.get("objects") or [])
    if "*" in objs:
        objs = set(ex.scene_geoms()) - set(avoid.get("allow") or []) - {"*"}
        for o in list(objs):
            try:
                if ex.geom_info(o)["aabb_max"][2] < 0.05:
                    objs.discard(o)
            except (ValueError, KeyError):
                objs.discard(o)
    ex.body_avoid = set(avoid.get("body_only") or [])
    ex.avoid_objects = objs - ex.body_avoid


def _bind_embodiment(emb_cls, ex):
    try:
        return emb_cls(ex.model, ex.data)
    except TypeError:
        emb = emb_cls()
        emb.model, emb.data = ex.model, ex.data
        return emb


def _bind_policy(pol_cls, pol_inst, emb):
    if pol_inst is not None:
        pol = pol_inst
    else:
        try:
            pol = pol_cls(emb)
        except TypeError:
            pol = pol_cls()
    bind = getattr(pol, "bind", None)
    if callable(bind):
        try:
            bind(emb)
        except Exception:
            pass
    return pol


def _jsonable(v, depth=0):
    if depth > 6:
        return str(v)
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x, depth + 1) for x in v]
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    try:
        return float(v)
    except Exception:
        return str(v)


def _receipt(ex, result):
    try:
        return ex.receipt("act", result)
    except Exception as e:
        return {"skill": "act", "result": result,
                "measured": {"error": f"{type(e).__name__}: {e}"}}


def _measured(ex, receipt, turns, sim_steps, placed):
    st = ex.stats
    m = (receipt or {}).get("measured") or {}
    return {
        "turns": int(turns),
        "sim_steps": int(sim_steps),
        "sim_s": round(float(ex.data.time), 3),
        "placed": bool(placed),
        "person_peak_force_N": round(float(st.person_peak_force), 1),
        "person_contacts": int(st.person_contacts),
        "person_down": bool(st.person_down),
        "person_near_misses": int(st.person_near_misses),
        "person_dist_m": m.get("person_dist_m"),
        "person_hit_at_s": st.person_hit_at_s,
        "person_impact_speed_mps": st.person_impact_speed,
        "dropped_tool": st.dropped_tool,
        "robot_fell": bool(st.robot_fell),
        "furniture_contacts": int(st.furniture_contacts),
        "furniture_peak_N": {str(k): float(v)
                             for k, v in st.furniture_peak_N.items()},
    }


def _run_episode(ex, emb, pol, task, n_steps, index, scene, transcript):
    rate = float(getattr(getattr(emb, "action_space", None), "rate_hz", 0)
                 or 50.0)
    ctrl = max(1, int(round(1.0 / (rate * ex.model.opt.timestep))))
    turns, sim_steps = 0, 0
    for t in range(n_steps):
        try:
            obs = emb.observation()
            act = getattr(pol, "act", None)
            action = act(obs) if callable(act) else pol(obs)
        except ViewerClosed:
            raise
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            transcript.append({"episode": index, "turn": t,
                               "call": {"skill": "act", "error": err},
                               "receipt": _receipt(ex, {"error": err})})
            turns += 1
            break
        applied, rejected = 0, []
        if isinstance(action, Mapping):
            for j, v in action.items():
                try:
                    ex.set(j, float(v))
                    applied += 1
                except Exception:
                    rejected.append(str(j))
        else:
            rejected.append("non-mapping action")
        call = {"skill": "act", "args": _jsonable(action)}
        result = {"ok": not rejected, "applied": applied,
                  "rejected": rejected}
        try:
            ex.step(ctrl)
            sim_steps += ctrl
        except ViewerClosed:
            raise
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
            transcript.append({"episode": index, "turn": t, "call": call,
                               "receipt": _receipt(ex, result)})
            turns += 1
            break
        transcript.append({"episode": index, "turn": t, "call": call,
                           "receipt": _receipt(ex, result)})
        turns += 1
    try:
        placed = bool(ex.eval_success(getattr(task, "spec", None)))
    except Exception:
        placed = False
    measured = _measured(ex, _receipt(ex, {"done": True}),
                         turns, sim_steps, placed)
    return {"episode": index, "scene": scene, "measured": measured,
            "metrics": episode_metrics(measured)}


def _git_info():
    def run(*a):
        try:
            r = subprocess.run(["git", *a], cwd=REPO, capture_output=True,
                               text=True, timeout=20)
        except Exception:
            return None
        return r.stdout.strip() if r.returncode == 0 else None
    return {"revision": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def eval(task, policy, embodiment, episodes=1, max_steps=None, out_dir=None,
         record=False, live=False, headless=True, seed=None, scene=None,
         config=None, **kwargs):
    """Run one eval: task x policy x embodiment -> schema-versioned log.

    The triple resolves by name through the registries (a Task object, a
    policy class/instance and an embodiment class/instance are accepted
    the same way). Each episode runs on the task's scenes — the policy's
    act() is applied at the embodiment's declared control rate — and the
    log records the resolved config, this repo's git revision, the
    per-turn transcript, per-episode measured metrics and the scores.
    """
    overrides = dict(config or {})
    task_obj = _resolve_task(task)
    pol_cls, pol_inst, pol_name = _resolve_policy(policy)
    emb_cls, _emb, emb_name = _resolve_embodiment(embodiment)
    scenes = _scenes(task_obj, scene, overrides, kwargs)
    n_eps = max(1, int(overrides.get("episodes") or episodes or 1))
    n_steps = max(1, int(overrides.get("max_steps") or max_steps
                         or getattr(task_obj, "max_steps", 1) or 1))
    transcript, ep_records, ex, cur = [], [], None, None
    try:
        for i in range(n_eps):
            sc = scenes[i % len(scenes)]
            if ex is None or sc != cur:
                if ex is not None:
                    ex.close()
                ep_dir = (os.path.join(str(out_dir), f"episode_{i:03d}")
                          if out_dir else f"runs/eval/episode_{i:03d}")
                ex = Executor(xml_path=sc, record=bool(record),
                              out_dir=ep_dir)
                cur = sc
                _configure_avoid(ex, task_obj)
                if live and not headless:
                    try:
                        ex.attach_viewer()
                    except Exception as e:
                        print(f"[eval] viewer unavailable ({e}); "
                              "continuing headless", flush=True)
            elif i:
                ex.reset_trial()
            emb = _bind_embodiment(emb_cls, ex)
            pol = _bind_policy(pol_cls, pol_inst, emb)
            ep_records.append(_run_episode(
                ex, emb, pol, task_obj, n_steps, i, sc, transcript))
    finally:
        if ex is not None:
            ex.close()
    log = {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "task": getattr(task_obj, "name", str(task)),
            "policy": pol_name,
            "embodiment": emb_name,
            "scenes": scenes,
            "spec": list(getattr(task_obj, "spec", None) or []) or None,
            "instruction": getattr(task_obj, "instruction", None),
            "episodes": n_eps,
            "max_steps": n_steps,
            "seed": seed,
            "record": bool(record),
            "live": bool(live and not headless),
            "headless": bool(headless),
            "overrides": overrides,
        },
        "git": _git_info(),
        "transcript": transcript,
        "episodes": ep_records,
        "scores": aggregate_scores([e["metrics"] for e in ep_records]),
    }
    if out_dir:
        save_log(log, Path(str(out_dir)) / "eval_log.json")
    return log


evaluate = eval
run_eval = eval
run_eval_triple = eval
