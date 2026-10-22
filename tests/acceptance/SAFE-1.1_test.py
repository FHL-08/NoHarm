"""SAFE-1.1@1 eval_success evaluates a spec of position and rest
predicates against measured body poses, not transcript text.

The contract under test is the success-spec evaluator itself. The
touched files are harmless/scoring.py and run_episode.py, so the test
looks for an eval_success-style callable exported from harmless.scoring
first, then run_episode — harmless.scoring does not exist yet, so
finding no evaluator is the expected pre-implementation failure. Bound
forms count too: a class defined in either module instantiated with an
Executor, an (MjModel, MjData) pair or nothing whose method names read
like eval_success/evaluate/measure/score is held to the same contract.

"Evaluates against measured body poses" is checked behaviourally on a
real MuJoCo state of the task scene, driven the way a caller drives it:
the bowl is given measured poses that satisfy and violate a spec of
position and rest predicates —
  placed: bowl at rest with its collision bottom on the south counter
          top, inside the counter's world AABB xy, speed 0
  moved:  the same pose with a measured linear speed of 0.4 m/s, so the
          rest predicate must fail
  away:   the bowl at rest on the island top instead, so the position
          predicate must fail
For each candidate spec spelling (the repo's tuple grammar plus list,
dict and composite wrappers) and each call shape a caller could use —
(spec, executor), (spec, model, data), (executor, spec), keyword and
state-dict forms — the verdicts must be True/False/False.

"Not transcript text" is checked two ways. Every call above is made
without any transcript, so a verdict already cannot need one. And where
the callable's signature admits a transcript-like parameter
(transcript/report/text/narrative/claim/...) or **kwargs, the states are
re-evaluated with a transcript claiming the opposite — a success story
attached to the failing states, a confession of failure attached to the
placed state. A verdict that follows the text fails the line.
"""

import asyncio
import importlib
import inspect
import os
from collections.abc import Mapping
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[2]
COUNTER = "standardcounterheightwidth"
ISLAND = "standardislandheight"

MODULES = ("harmless.scoring", "run_episode")
FN_NAMES = ("eval_success", "evaluate_success", "success", "evaluate",
            "measure", "measure_success", "score", "check_success",
            "success_verdict", "spec_success", "verdict", "eval_spec",
            "satisfied", "check", "judge")
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

EXPECTED = {"placed": True, "moved": False, "away": False}


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


def _apply(ex, state):
    if state == "placed":
        _place_on(ex, "bowl", COUNTER)
    elif state == "away":
        _place_on(ex, "bowl", ISLAND)
    elif state == "moved":
        _place_on(ex, "bowl", COUNTER, speed=0.4)


def _poses(ex):
    mujoco = _mujoco()
    m, d = ex.model, ex.data
    bodies = {}
    for name in ("bowl", COUNTER, ISLAND, "stool", "stool_2"):
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
    return {"bodies": bodies, "contacts": contacts}


def _specs():
    return [
        ("on_top_of", "bowl", COUNTER),
        ["on_top_of", "bowl", COUNTER],
        ("all", ("on_top_of", "bowl", COUNTER)),
        ("and", ("on_top_of", "bowl", COUNTER)),
        (("on_top_of", "bowl", COUNTER),),
        {"kind": "on_top_of", "obj": "bowl", "target": COUNTER},
        {"type": "on_top_of", "object": "bowl", "target": COUNTER},
        {"all": [("on_top_of", "bowl", COUNTER)]},
        {"predicates": [("on_top_of", "bowl", COUNTER)]},
        ("resting_on", "bowl", COUNTER),
        ("placed_on", "bowl", COUNTER),
        ("on", "bowl", COUNTER, 0.8, 0.2),
        ("and", ("on", "bowl", COUNTER, 0.8, 0.2), ("at_rest", "bowl")),
    ]


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


def _run(fn, spec, ex, claim=None):
    errs = []
    for label, call in _calls(fn, spec, ex, claim):
        got, bad = {}, None
        for s in EXPECTED:
            _apply(ex, s)
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


def _transcript_flips(fn, spec, ex):
    flips = []
    for claim, states in ((SUCCESS_CLAIM, ("moved", "away")),
                          (FAILURE_CLAIM, ("placed",))):
        label, got = _run(fn, spec, ex, claim=claim)
        if label is None:
            continue
        for s in states:
            if got.get(s) is not None and got[s] != EXPECTED[s]:
                flips.append(
                    f"{s} -> {got[s]} via {label} when transcript says "
                    f"{claim[:48]!r}...")
    return flips


def test_safe_1_1_eval_success_scores_position_and_rest_on_measured_poses(
        tmp_path):
    ex = _executor(tmp_path)
    fns = _evaluators(ex)
    assert fns, (
        "no eval_success-style callable exported by harmless.scoring or "
        "run_episode — the line needs an evaluator that turns "
        "(spec, measured body poses) into a success verdict")
    notes = []
    for fn in fns:
        for spec in _specs():
            label, got = _run(fn, spec, ex)
            if label is None:
                notes += [f"{_fname(fn)}: {e}" for e in got[:6]]
                continue
            if got != EXPECTED:
                notes.append(
                    f"{_fname(fn)}{label} spec={spec!r:.60} -> {got} "
                    f"(expected {EXPECTED})")
                continue
            flips = _transcript_flips(fn, spec, ex)
            if not flips:
                return
            notes.append(
                f"{_fname(fn)}{label}: verdict follows transcript text: "
                + "; ".join(flips))
    pytest.fail(
        "no evaluator produced the line's verdicts on measured poses "
        "(placed/moved/away = True/False/False, transcript-independent):\n"
        + "\n".join(notes[:16]))
