"""POL-3.1@1 The supervisor's turn output names a subgoal (target object,
destination, or walk command) that the low-level layer executes.

The contract under test is the layered control path this requirement
touches (harmless/supervisor.py, harmless/agent.py): the LLM is the
supervisor — each turn its output names a subgoal, and the low-level
layer (the served VLA policy or the control-rate controllers: walker,
set_controller, arm reachers) executes that subgoal at control rate.

A real GPT-6-astra call cannot happen inside a test, so the OpenAI client
is replaced with a fake whose replies name subgoals in the natural
spellings ("subgoal" key, subgoal skill, target/object/destination
fields, a walk command dict). The low-level layer is observed through
every seam a layering can use:

  - a probe Policy subclass (harmless.policy.Policy) that records every
    act(observation, ...) invocation and the subgoal text delivered to
    it, and returns a real joint_pos action chunk (a (steps, joints)
    array inside declared bounds, the same shape Gr00tVLAPolicy.act
    emits). It is offered to the entry point under every plausible kwarg
    name, registered in harmless.registry under the names a supervisor
    would resolve ("gr00t", "vla", ...), and bound over the
    Gr00tVLAPolicy class attribute so a default-constructed low level is
    the probe — with serve.load_action_head stubbed so even a real
    adapter instance can never hit the network;
  - spies on Executor.walk / set_controller / arm_reach / arm_path /
    hands_path — the control-rate controller route a subgoal may take
    instead of a VLA.

The supervisor entry point is discovered by name across the touched
modules (harmless.supervisor first, then harmless.agent / session /
policies.*): episode-loop functions and Supervisor-like classes are both
driven. The line holds when the model's turn output naming a subgoal
leads to a low-level execution: the probe policy ran with the named
target delivered to it, or the named walk command reached the walker, or
a control-rate controller was engaged for the subgoal. On the unbuilt
tree the only loop is harmless.agent.run_episode, which passes the
subgoal call to the executor skill table and back — nothing low-level
ever runs, which is the expected failure.
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
TASK = "pol-3.1 acceptance probe: layered supervisor control"
TARGET = "bowl"                     # an authored prop in the iTHOR scene
OBJ_SUBGOAL = "approach and touch the bowl"
WALK_VX = 0.25
CHUNK_STEPS = 4

OBJ_CALL = {
    "skill": "subgoal", "action": "subgoal", "kind": "subgoal",
    "subgoal": {"kind": "object", "type": "object", "target": TARGET,
                "object": TARGET, "name": TARGET, "text": OBJ_SUBGOAL},
    "args": {"subgoal": OBJ_SUBGOAL, "target": TARGET, "object": TARGET},
    "expected": {},
}
WALK_CALL = {
    "skill": "walk", "action": "subgoal",
    "subgoal": {"kind": "walk", "type": "walk", "command": "walk",
                "vx": WALK_VX, "vy": 0.0, "yaw_rate": 0.0,
                "seconds": 0.5, "text": "walk forward"},
    "walk": {"vx": WALK_VX, "vy": 0.0, "yaw_rate": 0.0},
    "args": {"subgoal": "walk forward", "vx": WALK_VX, "vy": 0.0,
             "yaw_rate": 0.0, "walk": {"vx": WALK_VX}},
    "expected": {},
}
DONE_CALL = {"skill": "done", "args": {"summary": "pol-3.1 probe done"}}
REPLIES = [OBJ_CALL, WALK_CALL, DONE_CALL]

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
_MISSING = object()


class _FakeOpenAI:
    """Records every conversation sent to the model; replays the scripted
    replies in order (subgoal turns, then done)."""

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
    each act() call and what text was delivered to it, and returns a real
    (steps, joints) joint_pos chunk inside declared bounds."""
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
        calls = []        # (observation, args, kwargs) per act() call
        chunks = []       # emitted chunk arrays
        deliveries = []   # text handed over via set_*/instruct sinks
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
                if c == 0:
                    rows[:, c] = mid + 0.5 * (float(hi) - mid)
                elif c == 1:
                    rows[:, c] = mid - 0.5 * (mid - float(lo))
                else:
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
    """OpenAI -> fake; registry names + Gr00tVLAPolicy -> probe; controller
    entry points on the Executor -> spies; the served-weights loader ->
    stub so nothing in a test ever downloads."""
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
    """A module imported after the seams were installed may have bound the
    real VLA class or a real OpenAI constructor — rebind them."""
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
    """Discover the supervisor turn entry point across the touched
    modules: episode-loop functions and Supervisor-like classes."""
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
    msgs = [{"role": "system", "content": "pol-3.1 probe"},
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
        "embodiment": None, "client": fake, "llm": fake,
    }


def _call_flex(fn, kw, fills):
    """Invoke fn with the kwargs its signature accepts, filling required
    parameters from the probe superset."""
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
    """Drive one candidate entry point until a supervisor turn runs."""
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


def _delivered(probe_cls, needle):
    """The named subgoal reached the low-level layer: inside an act()
    observation/instruction, a delivery sink call, a kwarg, or an
    attribute the supervisor set on the bound policy."""
    for obs, a, k in probe_cls.calls:
        if _has_text(obs, needle) or _has_text(a, needle) \
                or _has_text(k, needle):
            return True
    for a, k in probe_cls.deliveries:
        if _has_text(a, needle) or _has_text(k, needle):
            return True
    for inst in probe_cls.instances:
        try:
            d = {n: v for n, v in vars(inst).items()
                 if n not in ("embodiment", "action_space")}
        except Exception:
            continue
        if _has_text(d, needle):
            return True
    return False


def _walk_hit(spies):
    for a, k in spies["walk"]:
        vx = k.get("vx", a[0] if a else None)
        try:
            if vx is not None and abs(float(vx) - WALK_VX) < 0.06:
                return True
        except Exception:
            continue
    return False


def _ctrl_hit(spies):
    return any(spies[n] for n in
               ("set_controller", "arm_reach", "arm_path", "hands_path"))


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


def test_pol_3_1_supervisor_turn_names_subgoal_low_level_executes(tmp_path):
    res = _run(tmp_path)
    assert res.entries, (
        "no supervisor/episode entry point found in harmless.supervisor, "
        "harmless.agent, harmless.session or harmless.policies.* — "
        "nothing drives a supervisor turn")
    assert res.fake.calls, (
        "the supervisor never asked the model for a turn "
        f"(entry attempts: {res.errs or 'none raised'})")

    vla_ran = bool(res.ProbeCls.calls)
    obj_executed = vla_ran and _delivered(res.ProbeCls, TARGET)
    walk_executed = _walk_hit(res.spies)
    ctrl_executed = _ctrl_hit(res.spies)

    assert obj_executed or walk_executed or ctrl_executed, (
        "the supervisor's turns named subgoals (object 'bowl', then a "
        "walk command) but the low-level layer never executed one: "
        f"policy act() calls={len(res.ProbeCls.calls)} "
        f"(subgoal text delivered: {_delivered(res.ProbeCls, TARGET)}), "
        f"walk() calls={res.spies['walk']}, controller engagements="
        f"{ {n: len(v) for n, v in res.spies.items() if v} }; "
        f"entry attempts: {res.errs[:4]}")

    if vla_ran and not obj_executed:
        pytest.fail(
            "the low-level policy ran "
            f"({len(res.ProbeCls.calls)} act() call(s)) but the named "
            f"subgoal ({OBJ_SUBGOAL!r} / target {TARGET!r}) was never "
            "delivered to it — the low-level layer must execute the "
            "supervisor's subgoal, not a blind one")
