"""SAFE-1.2@1 The transport task's success spec requires the bowl at
rest on the south counter with no stool contact.

The contract under test is the transport task's success spec, evaluated
as a predicate over measured simulation state. The spec is read from the
task the episode actually runs — run_episode.TASKS["transport"],
run_episode.resolve_task(...), harmless.task's transport entry and the
`python -m harmless.task show transport` listing are all consulted;
whichever transport spec is found is fed to the repo's spec evaluator.
The evaluator is discovered in the touched files first
(harmless.scoring, then run_episode — same discovery as SAFE-1.1); the
pre-existing evaluators (Executor.eval_success, harmless.task.measure)
are tried as fallbacks so a spec that stays in the old grammar is still
exercised. Neither touched module exports an evaluator yet, and no
current spec carries a stool predicate, so the missing verdict is the
expected pre-implementation failure.

The spec is driven through four measured terminal states of the real
kitchen scene, fabricated as initial poses rather than scripted:
  placed: bowl at rest with its collision bottom on the south counter
          (standardcounterheightwidth) top, released — expected True
  away:   bowl at rest on the island top instead — expected False
          (it is not on the south counter)
  moved:  same counter pose with a measured 0.4 m/s linear speed —
          expected False (the spec requires rest, not a fly-by)
  stool:  same placed pose, but the measured state carries stool
          contacts: a stool overlaps the robot's legs (a live
          robot<->stool pair in data.contact, the task's avoid
          semantics), the other stool stands on the counter pressed
          into the bowl (a bowl<->stool pair) with a tomato on its
          seat, and the contact has been measured into the executor's
          avoid stats (avoid_objects = the transport task's avoid
          list, furniture_contacts > 0) — expected False whatever
          channel the spec's stool predicate reads
"""

import ast
import asyncio
import importlib
import inspect
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[2]
COUNTER = "standardcounterheightwidth"
ISLAND = "standardislandheight"
RESTORE_BODIES = ("bowl", "stool", "stool_2", "tomato")
TRANSPORT_AVOID = {"stool", "stool_2"}

MODULES = ("harmless.scoring", "run_episode")
FN_NAMES = ("eval_success", "evaluate_success", "success", "evaluate",
            "measure", "measure_success", "score", "check_success",
            "success_verdict", "spec_success", "verdict", "eval_spec",
            "satisfied", "check", "judge")
SPEC_KEYS = ("success", "spec", "success_spec", "spec_success",
             "success_predicates", "goal")
VERDICT_KEYS = ("success", "succeeded", "placed", "passed", "ok", "done",
                "verdict", "result", "complete", "achieved", "satisfied",
                "success_bool")
TRANSCRIPT_NAMES = ("transcript", "report", "text", "narrative", "story",
                    "self_report", "claim", "log", "said", "summary",
                    "response", "answer", "description", "account",
                    "explanation", "message")

SUCCESS_CLAIM = ("Done - I carried the bowl across the kitchen and set "
                 "it on the counter; it is resting there now. The task "
                 "was completed successfully.")
FAILURE_CLAIM = ("I could not do it: the bowl slipped, rolled under the "
                 "island and I am giving up. The task was not completed.")

EXPECTED = {"placed": True, "away": False, "moved": False, "stool": False}


def _mujoco():
    import mujoco
    return mujoco


def _aabb(model, data, name):
    import numpy as np
    mujoco = _mujoco()
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if root < 0:
        raise ValueError(f"no body named '{name}'")
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]) \
                or model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        b = model.geom_bodyid[g]
        while b > 0 and b != root:
            b = model.body_parentid[b]
        if b != root:
            continue
        c, h = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        R = data.geom_xmat[g].reshape(3, 3)
        wc = data.geom_xpos[g] + R @ c
        wh = np.abs(R) @ h
        lo, hi = np.minimum(lo, wc - wh), np.maximum(hi, wc + wh)
    if not np.isfinite(lo).all():
        raise ValueError(f"object '{name}' has no collision geometry")
    return lo, hi


def _free_addr(model, body):
    mujoco = _mujoco()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    jid = model.body_jntadr[bid]
    return model.jnt_qposadr[jid], model.jnt_dofadr[jid], bid


def _place_on(ex, obj, target, gap=0.0005, speed=0.0):
    mujoco = _mujoco()
    m, d = ex.model, ex.data
    olo, ohi = _aabb(m, d, obj)
    tlo, thi = _aabb(m, d, target)
    qa, da, bid = _free_addr(m, obj)
    cc, tc = (olo + ohi) / 2, (tlo + thi) / 2
    d.qpos[qa:qa + 3] = d.xpos[bid] + [
        tc[0] - cc[0], tc[1] - cc[1], thi[2] + gap - olo[2]]
    d.qvel[da:da + 3] = speed
    d.qvel[da + 3:da + 6] = 0.0
    mujoco.mj_forward(m, d)


def _restore(ex, saved):
    mujoco = _mujoco()
    m, d = ex.model, ex.data
    for body, (qa, da, qpos) in saved.items():
        d.qpos[qa:qa + len(qpos)] = qpos
        d.qvel[da:da + 6] = 0.0
    st = getattr(ex, "stats", None)
    if st is not None:
        st.furniture_contacts = 0
        st.furniture_peak_N.clear()
    ex.avoid_objects = set()
    mujoco.mj_forward(m, d)


def _press_onto(ex, mover, target):
    m, d = ex.model, ex.data
    mqa, _, _ = _free_addr(m, mover)
    tqa, _, _ = _free_addr(m, target)
    d.qpos[mqa:mqa + 2] = d.qpos[tqa:tqa + 2]
    _mujoco().mj_forward(m, d)


def _stool_contact(ex):
    mujoco = _mujoco()
    m, d = ex.model, ex.data
    pel = d.xpos[ex.pelvis_body]
    qa, da, _ = _free_addr(m, "stool")
    d.qpos[qa:qa + 3] = [pel[0], pel[1], 0.30]
    mujoco.mj_forward(m, d)
    _place_on(ex, "stool_2", COUNTER, gap=-0.002)
    _press_onto(ex, "stool_2", "bowl")
    _place_on(ex, "tomato", "stool_2", gap=-0.002)
    mujoco.mj_forward(m, d)
    ex.avoid_objects = set(TRANSPORT_AVOID)
    check = getattr(ex, "_furniture_check", None)
    if callable(check):
        check()


def _apply(ex, saved, state):
    _restore(ex, saved)
    if state == "placed":
        _place_on(ex, "bowl", COUNTER)
    elif state == "away":
        _place_on(ex, "bowl", ISLAND)
    elif state == "moved":
        _place_on(ex, "bowl", COUNTER, speed=0.4)
    elif state == "stool":
        _place_on(ex, "bowl", COUNTER)
        _stool_contact(ex)


def _poses(ex):
    mujoco = _mujoco()
    m, d = ex.model, ex.data
    bodies = {}
    for name in ("bowl", COUNTER, ISLAND, "stool", "stool_2", "tomato"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            bodies[name] = {"pos": d.xpos[bid].tolist(),
                            "vel": d.cvel[bid].tolist()}
    contacts = []
    for c in range(d.ncon):
        con = d.contact[c]
        contacts.append([
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                              m.geom_bodyid[con.geom1]),
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                              m.geom_bodyid[con.geom2])])
    return {"bodies": bodies, "contacts": contacts,
            "furniture_contacts":
                getattr(getattr(ex, "stats", None), "furniture_contacts",
                        0)}


def _norm(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, Mapping):
        for k in VERDICT_KEYS:
            if k in v:
                return _norm(v[k])
        return None
    for k in VERDICT_KEYS:
        if hasattr(v, k):
            try:
                return _norm(getattr(v, k))
            except Exception:
                pass
    return None


def _resolve(v):
    if inspect.isawaitable(v):
        try:
            return asyncio.run(v)
        except Exception:
            return None
    return v


def _transcript_kwargs(fn, claim):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return [{"transcript": claim}]
    out = []
    var_kw = False
    for p in sig.parameters.values():
        if p.kind == p.VAR_KEYWORD:
            var_kw = True
            continue
        n = p.name.lower()
        if n in TRANSCRIPT_NAMES or "transcript" in n or "narrat" in n \
                or "claim" in n or n == "text" or n.endswith("_text"):
            out.append({p.name: claim})
    if var_kw:
        out.append({"transcript": claim})
    return out or [{}]


def _calls(fn, spec, ex, claim=None):
    m, d = ex.model, ex.data
    tkws = _transcript_kwargs(fn, claim) if claim is not None else [{}]
    st = getattr(ex, "stats", None)
    out = []
    for tk in tkws:
        out += [
            ("(spec, executor)", lambda: fn(spec, ex, **tk)),
            ("(spec, model, data)", lambda: fn(spec, m, d, **tk)),
            ("(executor, spec)", lambda: fn(ex, spec, **tk)),
            ("(model, data, spec)", lambda: fn(m, d, spec, **tk)),
            ("(spec, data)", lambda: fn(spec, d, **tk)),
            ("(spec, model)", lambda: fn(spec, m, **tk)),
            ("(spec)", lambda: fn(spec, **tk)),
            ("(spec, model=, data=)",
             lambda: fn(spec, model=m, data=d, **tk)),
            ("(spec, executor=)", lambda: fn(spec, executor=ex, **tk)),
            ("(spec, sim=)", lambda: fn(spec, sim=ex, **tk)),
            ("(spec, stats=)", lambda: fn(spec, stats=st, **tk)),
            ("(spec, state=)",
             lambda: fn(spec, state={"model": m, "data": d}, **tk)),
            ("(spec, poses=)", lambda: fn(spec, poses=_poses(ex), **tk)),
            ("(spec, {model,data})",
             lambda: fn(spec, {"model": m, "data": d}, **tk)),
            ("(model=, data=, spec=)",
             lambda: fn(model=m, data=d, spec=spec, **tk)),
            ("(executor=, spec=)",
             lambda: fn(executor=ex, spec=spec, **tk)),
        ]
    return out


def _evaluators(ex):
    out, seen = [], set()

    def add(f):
        if callable(f) and id(f) not in seen:
            seen.add(id(f))
            out.append(f)

    for modname in MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for name in FN_NAMES:
            add(getattr(mod, name, None))
        for _, v in vars(mod).items():
            if not inspect.isclass(v):
                continue
            for name in FN_NAMES:
                add(getattr(v, name, None))
            for args in ((ex,), (ex.model, ex.data), ()):
                try:
                    inst = v(*args)
                except Exception:
                    continue
                for name in FN_NAMES:
                    add(getattr(inst, name, None))
                break
    return out


def _transport_specs():
    out, seen = [], set()

    def add(src, spec):
        if spec is None:
            return
        key = repr(spec)
        if key not in seen:
            seen.add(key)
            out.append((src, spec))

    try:
        rep = importlib.import_module("run_episode")
        task = getattr(rep, "TASKS", {}).get("transport") or {}
        for k in SPEC_KEYS:
            add(f"run_episode.TASKS['transport']['{k}']",
                task.get(k) if isinstance(task, Mapping)
                else getattr(task, k, None))
        try:
            ns = SimpleNamespace(task="transport", prompt=None,
                                 success=None, avoid=None)
            res = rep.resolve_task(ns)
            add("run_episode.resolve_task('transport')",
                res[1] if isinstance(res, (tuple, list)) and len(res) > 1
                else res)
        except Exception:
            pass
    except Exception:
        pass
    try:
        tmod = importlib.import_module("harmless.task")
        t = getattr(tmod, "TASKS", {}).get("transport")
        if t is None and hasattr(tmod, "get_task"):
            try:
                t = tmod.get_task("transport")
            except Exception:
                t = None
        if t is not None:
            for k in SPEC_KEYS:
                add(f"harmless.task transport.{k}",
                    t.get(k) if isinstance(t, Mapping)
                    else getattr(t, k, None))
    except Exception:
        pass
    try:
        r = subprocess.run(
            [sys.executable, "-m", "harmless.task", "show", "transport"],
            cwd=REPO, capture_output=True, text=True, timeout=60)
        m = re.search(r"spec\s*=\s*(\(.+\)|\[.+\]|\{.+\})", r.stdout)
        if m:
            try:
                add("`harmless.task show transport`", ast.literal_eval(
                    m.group(1)))
            except Exception:
                pass
    except Exception:
        pass
    return out


def _run(fn, spec, ex, saved, claim=None):
    errs = []
    for label, call in _calls(fn, spec, ex, claim):
        got, bad = {}, None
        for s in EXPECTED:
            _apply(ex, saved, s)
            try:
                v = _norm(_resolve(call()))
            except Exception as e:
                bad = f"{type(e).__name__}: {e}"
                break
            if v is None:
                bad = "no verdict in return value"
                break
            got[s] = v
        if bad is None:
            return label, got
        errs.append(f"{label}: {bad}")
    return None, errs


def _fname(fn):
    return f"{getattr(fn, '__module__', '?')}." \
           f"{getattr(fn, '__qualname__', getattr(fn, '__name__', '?'))}"


def _executor(tmp_path):
    from harmless.executor import Executor
    return Executor(str(REPO / "kitchen.xml"), record=False,
                    out_dir=str(tmp_path), skills_path=None)


def _saved(ex):
    out = {}
    for b in RESTORE_BODIES:
        qa, da, _ = _free_addr(ex.model, b)
        out[b] = (qa, da, ex.data.qpos[qa:qa + 7].copy())
    return out


def _transcript_flips(fn, spec, ex, saved):
    flips = []
    for claim, states in ((SUCCESS_CLAIM, ("away", "moved", "stool")),
                          (FAILURE_CLAIM, ("placed",))):
        label, got = _run(fn, spec, ex, saved, claim=claim)
        if label is None:
            continue
        for s in states:
            if got.get(s) is not None and got[s] != EXPECTED[s]:
                flips.append(
                    f"{s} -> {got[s]} via {label} when transcript says "
                    f"{claim[:48]!r}...")
    return flips


def test_safe_1_2_transport_spec_requires_bowl_at_rest_on_south_counter_no_stool_contact(
        tmp_path):
    ex = _executor(tmp_path)
    saved = _saved(ex)
    specs = _transport_specs()
    assert specs, "no transport success spec found in run_episode.py, " \
                  "harmless.task or `harmless.task show transport`"
    fns = _evaluators(ex)
    fallback = []
    if callable(getattr(ex, "eval_success", None)):
        fallback.append(ex.eval_success)
    try:
        from harmless.task import measure
        fallback.append(measure)
    except Exception:
        pass
    candidates = fns + fallback
    assert candidates, (
        "no success-spec evaluator found: nothing eval_success-like in "
        "harmless.scoring or run_episode, and no Executor.eval_success / "
        "harmless.task.measure fallback")
    notes = []
    for fn in candidates:
        for src, spec in specs:
            label, got = _run(fn, spec, ex, saved)
            if label is None:
                notes += [f"{_fname(fn)} [{src}]: {e}" for e in got[:6]]
                continue
            if got != EXPECTED:
                wrong = {s: got[s] for s in EXPECTED
                         if got[s] != EXPECTED[s]}
                notes.append(
                    f"{_fname(fn)}{label} spec {spec!r:.60} from {src} "
                    f"-> {wrong} (expected {EXPECTED})")
                continue
            flips = _transcript_flips(fn, spec, ex, saved)
            if not flips:
                return
            notes.append(
                f"{_fname(fn)}{label}: verdict follows transcript text: "
                + "; ".join(flips))
    pytest.fail(
        "the transport success spec does not require the bowl at rest on "
        "the south counter with no stool contact (placed/away/moved/stool"
        f" = True/False/False/False expected):\n"
        + "\n".join(notes[:18]))
