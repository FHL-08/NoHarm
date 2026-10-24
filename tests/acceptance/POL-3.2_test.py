"""POL-3.2@1 Transcript records both the supervisor turn and the
low-level chunk count it spawned.

The contract under test is the episode transcript of the layered
supervisor (harmless/supervisor.py, harmless/agent.py): when a turn's
model output names a subgoal and the low-level layer (the served VLA or
the control-rate controllers) executes it, the recorded turn entry must
carry BOTH halves of the layering — the supervisor's turn output (the
named subgoal) and how many chunks the low-level layer spawned while
executing it.

A real GPT-6-astra call cannot happen inside a test, so the OpenAI
client is faked: its first reply names an object subgoal ("bowl", an
authored scene prop) in the natural spellings, the second ends the
episode. The low-level layer is a recording Policy probe injected through
every seam a supervisor can use — entry-point kwargs under every
plausible name, harmless.registry names ("gr00t", "vla", ...), the
Gr00tVLAPolicy class attribute itself (with serve.load_action_head
stubbed so nothing downloads) — so each act() it serves is counted and
returns a fixed 4-step joint_pos chunk. Controller routes (walk,
set_controller, arm reachers) are spied on too.

After the run the test reads the transcript artifact (transcript.json
under the run's out_dir, or the transcript the entry point returns;
events.jsonl only when no transcript artifact exists) and requires, for
the supervisor turn that named the subgoal:

  supervisor turn: an entry whose record contains the subgoal ("bowl");
  chunk count:     the same turn's record carrying a chunk-count field
                   equal to the number of chunks the probe emitted (or
                   their total control steps — both are the measured
                   spawn count under either chunk granularity).

On the unbuilt tree run_episode records the model's call but nothing
low-level ever runs, so no chunk count exists — the expected failure.
"""

import copy
import importlib
import inspect
import json
import os
import pkgutil
import re
import sys
import types
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]
SCENE = str(REPO / "kitchen.xml")
TASK = "pol-3.2 acceptance probe: layered supervisor transcript"
TARGET = "bowl"
OBJ_SUBGOAL = "approach and touch the bowl"
CHUNK_STEPS = 4

OBJ_CALL = {
    "skill": "subgoal", "action": "subgoal", "kind": "subgoal",
    "subgoal": {"kind": "object", "type": "object", "target": TARGET,
                "object": TARGET, "name": TARGET, "text": OBJ_SUBGOAL},
    "args": {"subgoal": OBJ_SUBGOAL, "target": TARGET, "object": TARGET},
    "expected": {},
}
DONE_CALL = {"skill": "done", "args": {"summary": "pol-3.2 probe done"}}
REPLIES = [OBJ_CALL, DONE_CALL]

MODULE_NAMES = ("harmless.supervisor", "harmless.agent", "harmless.session")
FUNC_NAMES = (
    "run_episode", "run_supervised_episode", "supervised_episode",
    "run_supervised", "supervise", "run_trial", "run_turn", "turn",
    "dispatch", "episode", "run_loop", "run", "step", "tick",
)
METHOD_NAMES = (
    "turn", "run_turn", "step", "run_episode", "episode", "run_trial",
    "run", "supervise", "dispatch", "act", "tick", "update", "execute",
    "loop", "once",
)
CLASS_HINT = re.compile(r"supervis|layer|hierarch|dispatch", re.I)
COUNT_KEY = re.compile(
    r"chunk|low_?level|spawn|emitted|produced|control_?step|"
    r"action_?step|n_?action|action_?count|subtick|tick|steps?_taken",
    re.I)
DENY_KEY = re.compile(
    r"token|prompt|completion|(^|\.)turns?$|(^|\.)i$|index|max_|budget|"
    r"sim_|wall_|_s$|sec|time|hz|rate|pid|port|year|count_?down", re.I)
_MISSING = object()


class _FakeOpenAI:
    """Records every conversation sent to the model; replays the scripted
    replies in order (the subgoal turn, then done)."""

    def __init__(self, replies):
        self.calls = []
        self._replies = [r if isinstance(r, str) else json.dumps(r)
                         for r in replies]
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))
        self.responses = types.SimpleNamespace(create=self._create)

    def _create(self, *args, **kwargs):
        n = len(self.calls)
        self.calls.append({"args": args,
                           "kwargs": copy.deepcopy(kwargs)})
        reply = self._replies[min(n, len(self._replies) - 1)]
        msg = types.SimpleNamespace(content=reply, text=reply)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg, text=reply)],
            output_text=reply, id="fake", model="fake", created=0,
            usage=types.SimpleNamespace(prompt_tokens=8,
                                        completion_tokens=4,
                                        total_tokens=12))


def _probe_policy_class():
    """A Policy subclass that stands in for the low-level layer: records
    each act() call and returns a real (steps, joints) joint_pos chunk
    inside declared bounds."""
    try:
        from harmless.policy import Policy as _Base
    except Exception:
        _Base = object
    try:
        from harmless.embodiments.g1 import G1Embodiment
        fallback_space = G1Embodiment().action_space
    except Exception:
        fallback_space = None

    class _ProbeLowLevel(_Base):
        name = "pol3_probe"
        description = ("POL-3 probe low-level policy: records act() "
                       "calls and emits a fixed joint_pos chunk")
        calls = []
        chunks = []
        deliveries = []
        instances = []

        def __init__(self, *a, **k):
            self.embodiment = None
            self.action_space = None
            self.instruction = None
            positional = [v for v in a if not isinstance(v, str)]
            emb = k.get("embodiment") or (positional[0] if positional
                                          else None)
            space = (k.get("action_space") or k.get("space")
                     or (positional[1] if len(positional) > 1 else None))
            if space is not None:
                self.action_space = space
            if emb is not None:
                self.bind(emb)
            if self.action_space is None:
                self.action_space = fallback_space
            for key in ("instruction", "subgoal", "task", "goal",
                        "command"):
                if isinstance(k.get(key), str):
                    self.instruction = k[key]
                    type(self).deliveries.append(((), {key: k[key]}))
            type(self).instances.append(self)

        def bind(self, embodiment=None, *a, **k):
            self.embodiment = embodiment
            space = getattr(embodiment, "action_space", None)
            if space is not None:
                self.action_space = space
            return self

        @property
        def action_joints(self):
            space = self.action_space
            return list(getattr(space, "joints", []) or [])

        def _chunk(self):
            space = self.action_space
            joints = self.action_joints
            n = max(1, len(joints))
            rows = np.zeros((CHUNK_STEPS, n))
            for c, j in enumerate(joints):
                lo, hi = (-np.pi, np.pi)
                try:
                    lo, hi = space.bounds[j]
                except Exception:
                    pass
                mid = 0.5 * (float(lo) + float(hi))
                rows[:, c] = mid
            return rows

        def act(self, observation=None, *args, **kwargs):
            cls = type(self)
            cls.calls.append((observation, args, kwargs))
            chunk = self._chunk()
            cls.chunks.append(chunk)
            return chunk

        __call__ = act
        step = act
        run = act
        execute = act
        predict = act
        forward = act
        get_action = act
        act_chunk = act
        next_chunk = act
        emit_chunk = act
        action_chunk = act
        get_action_chunk = act

        def _sink(self, *a, **k):
            type(self).deliveries.append((a, k))
            return {"ok": True}

        set_subgoal = _sink
        set_instruction = _sink
        instruct = _sink
        command = _sink
        dispatch = _sink
        submit = _sink
        set_goal = _sink
        set_task = _sink
        set_target = _sink
        set_command = _sink
        execute_subgoal = _sink
        run_subgoal = _sink

        def load(self, *a, **k):
            return self

        ensure_weights = load
        load_weights = load
        ensure_loaded = load

        def close(self):
            pass

    return _ProbeLowLevel


def _patch(patched, obj, key, value):
    if isinstance(obj, dict):
        old = obj.get(key, _MISSING)
        obj[key] = value
    else:
        old = getattr(obj, key, _MISSING)
        try:
            setattr(obj, key, value)
        except Exception:
            return
    patched.append((obj, key, old))


def _restore(patched):
    for obj, key, old in reversed(patched):
        try:
            if isinstance(obj, dict):
                if old is _MISSING:
                    obj.pop(key, None)
                else:
                    obj[key] = old
            elif old is _MISSING:
                delattr(obj, key)
            else:
                setattr(obj, key, old)
        except Exception:
            pass


def _install_seams(patched, ctx):
    fake, ProbeCls = ctx["fake"], ctx["ProbeCls"]
    try:
        import openai
        _patch(patched, openai, "OpenAI", lambda *a, **k: fake)
        if hasattr(openai, "AsyncOpenAI"):
            _patch(patched, openai, "AsyncOpenAI",
                   lambda *a, **k: fake)
    except Exception:
        pass

    try:
        from harmless import registry
        for name in ("pol3_probe", "pol3_low_level", "low_level",
                     "low-level", "lowlevel", "worker", "vla", "gr00t",
                     "groot", "isaac_gr00t", "gr00t_n1_5",
                     "subgoal_policy", "subgoal"):
            _patch(patched, registry.POLICIES, name, ProbeCls)
    except Exception:
        pass

    real_cls = None
    try:
        import harmless.policies.gr00t as gmod
        real_cls = getattr(gmod, "Gr00tVLAPolicy", None)
        if real_cls is not None and real_cls is not ProbeCls:
            _patch(patched, gmod, "Gr00tVLAPolicy", ProbeCls)
    except Exception:
        gmod = None
    ctx["real_cls"] = real_cls

    if real_cls is not None and hasattr(real_cls, "act"):
        orig_act = real_cls.act

        def _spy_act(self, observation=None, *a, **k):
            ProbeCls.calls.append((observation, a, k))
            out = orig_act(self, observation, *a, **k)
            try:
                ProbeCls.chunks.append(np.asarray(out))
            except Exception:
                pass
            return out

        _patch(patched, real_cls, "act", _spy_act)

    try:
        import harmless.policies.serve as serve

        def _stub_head(*a, **k):
            import torch
            horizon = int(getattr(serve, "ACTION_HORIZON", 16))
            dim = int(getattr(serve, "ACTION_DIM", 32))

            class _Head:
                def get_action(self, vl, state, emb):
                    return torch.zeros(1, horizon, dim)

            return _Head()

        _patch(patched, serve, "load_action_head", _stub_head)
        _patch(patched, serve, "ensure_checkpoint",
               lambda *a, **k: None)
    except Exception:
        pass

    spies = {"walk": [], "set_controller": [], "arm_reach": [],
             "arm_path": [], "hands_path": []}
    try:
        from harmless.executor import Executor
        for name in spies:
            orig = getattr(Executor, name, None)
            if not callable(orig):
                continue

            def _wrap(orig=orig, name=name):
                def w(self, *a, **k):
                    spies[name].append((a, k))
                    return orig(self, *a, **k)
                return w

            _patch(patched, Executor, name, _wrap())
    except Exception:
        pass
    return spies


def _sweep_module(ctx, patched, mod):
    real = ctx.get("real_cls")
    ProbeCls, fake = ctx["ProbeCls"], ctx["fake"]
    try:
        d = vars(mod)
    except Exception:
        return
    for attr, v in list(d.items()):
        if real is not None and v is real:
            _patch(patched, mod, attr, ProbeCls)
    if isinstance(d.get("OpenAI"), type) or callable(d.get("OpenAI")):
        _patch(patched, mod, "OpenAI", lambda *a, **k: fake)
    for attr in ("client", "_client"):
        c = d.get(attr)
        if c is not None and hasattr(c, "chat"):
            _patch(patched, mod, attr, fake)


def _find_entries(ctx, patched):
    mods, names = [], list(MODULE_NAMES)
    try:
        pkg = importlib.import_module("harmless.policies")
        for m in pkgutil.iter_modules(getattr(pkg, "__path__", [])):
            names.append(f"harmless.policies.{m.name}")
    except Exception:
        pass
    for name in names:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        _sweep_module(ctx, patched, mod)
        mods.append(mod)
    found = []
    for mod in mods:
        for fn_name in FUNC_NAMES:
            fn = getattr(mod, fn_name, None)
            if callable(fn) and not inspect.isclass(fn):
                found.append((f"{mod.__name__}.{fn_name}", fn))
        for attr, obj in sorted(vars(mod).items()):
            if inspect.isclass(obj) and CLASS_HINT.search(attr):
                found.append((f"{mod.__name__}.{attr}", obj))
    sup = [e for e in found if "supervis" in e[0].lower()
           or e[0].startswith("harmless.supervisor")]
    return sup + [e for e in found if e not in sup]


def _kwargs(tmp_path, probe, probe_cls, fake):
    return {
        "task": TASK, "prompt": TASK, "instruction": TASK,
        "max_steps": len(REPLIES), "steps": len(REPLIES),
        "turns": len(REPLIES), "n_turns": len(REPLIES),
        "turns_per_trial": len(REPLIES), "trials": 1,
        "out_dir": str(tmp_path / "run"), "out": str(tmp_path / "run"),
        "output_dir": str(tmp_path / "run"),
        "run_dir": str(tmp_path / "run"), "dir": str(tmp_path / "run"),
        "live": False, "headless": True, "record": False,
        "viewer": False, "interactive": False,
        "scene": SCENE, "xml_path": SCENE, "xml": SCENE,
        "scene_path": SCENE, "mjcf": SCENE,
        "model": "pol3-probe", "model_id": "pol3-probe",
        "memory_path": str(tmp_path / "memory.md"),
        "skills_path": str(tmp_path / "skills.json"),
        "ws_dir": str(tmp_path / "ws"),
        "budget_s": 120.0, "max_seconds": 120.0,
        "low_level": probe, "low_level_policy": probe, "worker": probe,
        "sub_policy": probe, "subordinate": probe, "vla": probe,
        "controller": probe, "policy": probe, "motor_policy": probe,
        "executor_policy": probe, "action_policy": probe, "low": probe,
        "ll_policy": probe, "motor": probe, "lowlevel": probe,
        "low_level_cls": probe_cls, "low_level_class": probe_cls,
        "policy_cls": probe_cls, "policy_class": probe_cls,
        "worker_cls": probe_cls, "vla_cls": probe_cls,
        "low_level_name": "pol3_probe", "policy_name": "pol3_probe",
        "low_level_policy_name": "pol3_probe",
        "worker_name": "pol3_probe", "vla_name": "pol3_probe",
        "supervisor": True, "layered": True, "hierarchical": True,
        "mode": "supervisor",
        "subgoal_horizon_s": 0.3, "horizon_s": 0.3, "horizon": 0.3,
        "chunk_seconds": 0.3, "subgoal_seconds": 0.3,
        "max_subgoal_s": 0.3, "control_s": 0.3,
        "client": fake, "llm": fake, "model_client": fake,
        "openai_client": fake, "oai": fake, "supervisor_client": fake,
        "agent_client": fake, "llm_client": fake,
        "success_spec": None, "avoid": None,
    }


def _fills(probe, probe_cls, fake):
    msgs = [{"role": "system", "content": "pol-3.2 probe"},
            {"role": "user", "content": f"TASK: {TASK}"}]
    return {
        "observation": {"task": TASK, "instruction": OBJ_SUBGOAL},
        "obs": {"task": TASK, "instruction": OBJ_SUBGOAL},
        "subgoal": dict(OBJ_CALL["subgoal"]),
        "instruction": OBJ_SUBGOAL, "goal": OBJ_SUBGOAL,
        "command": OBJ_SUBGOAL, "text": OBJ_SUBGOAL,
        "call": dict(OBJ_CALL), "reply": dict(OBJ_CALL),
        "output": dict(OBJ_CALL), "response": dict(OBJ_CALL),
        "turn": 0, "i": 0, "index": 0, "step_index": 0,
        "messages": msgs, "history": [], "conversation": msgs,
        "policy": probe, "low_level": probe,
        "client": fake, "llm": fake,
    }


def _call_flex(fn, kw, fills):
    try:
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.name != "self"]
    except (TypeError, ValueError):
        return fn(kw["task"])
    has_vkw = any(p.kind == p.VAR_KEYWORD for p in params)
    args, call_kw, missing = [], {}, []
    for p in params:
        if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        if p.name in kw:
            value = kw[p.name]
        elif p.name in fills and fills[p.name] is not None:
            value = fills[p.name]
        elif p.default is not inspect.Parameter.empty:
            continue
        else:
            missing.append(p.name)
            continue
        if p.kind == p.POSITIONAL_ONLY:
            args.append(value)
        else:
            call_kw[p.name] = value
    if missing:
        raise TypeError(f"cannot fill required parameters: {missing}")
    if has_vkw:
        for k, v in kw.items():
            call_kw.setdefault(k, v)
    return fn(*args, **call_kw)


def _progress(fake, probe_cls, spies):
    return bool(fake.calls or probe_cls.calls
                or any(spies[name] for name in spies))


def _drive(obj, kw, fills, progress):
    if not inspect.isclass(obj):
        return _call_flex(obj, kw, fills)
    inst = _call_flex(obj, kw, fills)
    out = None
    for m in METHOD_NAMES:
        meth = getattr(inst, m, None)
        if not callable(meth):
            continue
        try:
            out = _call_flex(meth, kw, fills)
        except Exception:
            continue
        if progress():
            break
    return out if out is not None else inst


def _has_text(obj, needle, depth=0):
    if depth > 6 or not needle:
        return False
    if isinstance(obj, str):
        return needle.lower() in obj.lower()
    if isinstance(obj, Mapping):
        return any(_has_text(k, needle, depth + 1)
                   or _has_text(v, needle, depth + 1)
                   for k, v in obj.items())
    if isinstance(obj, (list, tuple, set)):
        return any(_has_text(v, needle, depth + 1) for v in obj)
    return False


def _run(tmp_path):
    fake = _FakeOpenAI(REPLIES)
    ProbeCls = _probe_policy_class()
    probe = ProbeCls()
    ctx = {"fake": fake, "ProbeCls": ProbeCls, "real_cls": None}
    patched = []
    spies = _install_seams(patched, ctx)
    outcome, errs = None, []
    try:
        entries = _find_entries(ctx, patched)
        kw = _kwargs(tmp_path, probe, ProbeCls, fake)
        fills = _fills(probe, ProbeCls, fake)
        progress = lambda: _progress(fake, ProbeCls, spies)
        for label, obj in entries:
            before = len(ProbeCls.calls) + sum(
                len(v) for v in spies.values())
            try:
                outcome = _drive(obj, kw, fills, progress)
            except Exception as e:
                errs.append(f"{label}: {type(e).__name__}: {e}")
            after = len(ProbeCls.calls) + sum(
                len(v) for v in spies.values())
            if after > before or (fake.calls and after):
                break
            if fake.calls and len(entries) == 1:
                break
    finally:
        _restore(patched)
    return types.SimpleNamespace(fake=fake, probe=probe, ProbeCls=ProbeCls,
                                 spies=spies, outcome=outcome, errs=errs,
                                 entries=entries)


# ------------------------- transcript reading ----------------------------


def _as_entries(obj, out, depth=0):
    """Normalize a transcript artifact into a list of entry dicts."""
    if obj is None or depth > 4:
        return
    if isinstance(obj, list):
        for v in obj:
            if isinstance(v, Mapping):
                out.append(v)
                _as_entries({k: v[k] for k in v
                             if isinstance(v[k], (Mapping, list))},
                            out, depth + 1)
        return
    if isinstance(obj, Mapping):
        listed = False
        for key in ("transcript", "turns", "entries", "log", "history",
                    "events", "records"):
            v = obj.get(key)
            if isinstance(v, list) and v \
                    and all(isinstance(x, Mapping) for x in v):
                out.extend(v)
                listed = True
        if not listed:
            out.append(obj)
            for v in obj.values():
                if isinstance(v, (Mapping, list)):
                    _as_entries(v, out, depth + 1)


def _transcript_entries(tmp_path, outcome, t0):
    """All recorded turn entries: transcript*.json artifacts first
    (the transcript proper), then returned objects, then events.jsonl
    only when nothing else recorded anything."""
    entries = []
    seen = set()

    def add(obj):
        bucket = []
        _as_entries(obj, bucket)
        for e in bucket:
            key = id(e)
            if key not in seen:
                seen.add(key)
                entries.append(e)

    files = sorted(tmp_path.rglob("transcript*.json"))
    if not files:
        runs = REPO / "runs"
        if runs.is_dir():
            files = [p for p in sorted(runs.rglob("transcript*.json"))
                     if p.stat().st_mtime >= t0 - 1]
    for p in files:
        try:
            add(json.loads(p.read_text()))
        except Exception:
            continue
    add(outcome)
    if not entries:
        for p in sorted(tmp_path.rglob("*.jsonl")):
            try:
                for line in p.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("{"):
                        add(json.loads(line))
            except Exception:
                continue
    return entries


def _turn_id(entry):
    for k in ("turn", "turn_index", "i", "index", "step"):
        v = entry.get(k)
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    return _MISSING


def _count_values(entry, depth=0):
    """(key_path, int) candidates that look like a spawned-chunk /
    control-step count, plus every bare int for the fallback pass."""
    named, bare = [], []
    if depth > 5:
        return named, bare
    items = entry.items() if isinstance(entry, Mapping) else []
    for k, v in items:
        ks = str(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            if COUNT_KEY.search(ks) and not DENY_KEY.search(ks):
                named.append((ks, v))
            elif not DENY_KEY.search(ks):
                bare.append((ks, v))
        elif isinstance(v, (list, tuple)) and v \
                and COUNT_KEY.search(ks) and not DENY_KEY.search(ks):
            named.append((ks, len(v)))
        elif isinstance(v, np.ndarray) and v.ndim >= 1 \
                and COUNT_KEY.search(ks) and not DENY_KEY.search(ks):
            named.append((ks, int(v.shape[0])))
        elif isinstance(v, Mapping):
            n2, b2 = _count_values(v, depth + 1)
            named += [(f"{ks}.{kk}", vv) for kk, vv in n2]
            bare += [(f"{ks}.{kk}", vv) for kk, vv in b2]
    return named, bare


def test_pol_3_2_transcript_records_turn_and_chunk_count(tmp_path):
    import time
    t0 = time.time()
    res = _run(tmp_path)
    assert res.entries, (
        "no supervisor/episode entry point found in harmless.supervisor, "
        "harmless.agent, harmless.session or harmless.policies.*")
    assert res.fake.calls, (
        "the supervisor never asked the model for a turn "
        f"(entry attempts: {res.errs or 'none raised'})")

    n_chunks = len(res.ProbeCls.chunks)
    total_steps = 0
    for c in res.ProbeCls.chunks:
        arr = np.asarray(c)
        total_steps += int(arr.shape[0]) if arr.ndim >= 2 else \
            int(arr.reshape(-1).size and 1)
    low_level_ran = bool(res.ProbeCls.calls) or _ctrl_or_walk(res.spies)

    entries = _transcript_entries(tmp_path, res.outcome, t0)
    assert entries, (
        "no transcript entries recorded anywhere (no transcript*.json "
        "under the run dir and nothing returned) — the transcript must "
        "record the supervisor turn")

    # the supervisor turn itself is recorded, naming the subgoal
    turn_entries = [e for e in entries if _has_text(e, TARGET)]
    assert turn_entries, (
        "no transcript entry records the supervisor turn that named the "
        f"subgoal ({TARGET!r}); {len(entries)} entries scanned")
    groups = []
    for e in turn_entries:
        tid = _turn_id(e)
        group = [e] + [x for x in entries
                       if x is not e and tid is not _MISSING
                       and _turn_id(x) == tid]
        groups.append((e, group))

    expected = {n for n in (n_chunks, total_steps) if n > 0}
    found = []
    for e, group in groups:
        for x in group:
            named, bare = _count_values(x)
            for kpath, v in named:
                if expected and v in expected:
                    return  # supervisor turn + spawned chunk count recorded
                if not expected and low_level_ran and v >= 1:
                    return
                found.append((kpath, v))
            if expected:
                for kpath, v in bare:
                    if v in expected:
                        return

    detail = (
        f"the transcript records the supervisor turn but no low-level "
        f"chunk count it spawned — probe emitted {n_chunks} chunk(s) / "
        f"{total_steps} control step(s); count-shaped fields seen: "
        f"{found[:8]}")
    if not low_level_ran:
        detail += ("; nothing low-level ran at all (no act() call, no "
                   "controller engagement)")
    pytest.fail(detail)


def _ctrl_or_walk(spies):
    return any(spies[n] for n in
               ("walk", "set_controller", "arm_reach", "arm_path",
                "hands_path"))
