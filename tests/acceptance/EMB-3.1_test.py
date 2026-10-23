"""EMB-3.1@1 Actions outside the embodiment's declared bounds are
clamped before reaching physics.

The contract under test is the guarded action path between policy and
physics: when a policy answers act(observation) with joint targets beyond
the paired action_space's declared (lo, hi) bounds, whatever the runner
(harmless/runner.py, the policy x embodiment stepping loop this
requirement touches) hands to the physics state must already be clamped
inside those bounds. A silently dropped action does not count: the
clamped target must reach physics, so actuator commands must move toward
the clamped region instead of sitting at their baseline.

The test binds the registered g1 adapter to the kitchen scene, parks
every declared actuator command at mid-range, and drives discovered
entry points in harmless.runner — Runner-style classes (constructed
through signature-matched keywords or positional
policy/embodiment/task/model/data orders, then advanced through
step/run-style methods) and module functions (run/step/rollout/
evaluate-style callables driven the same way, plus policy callables they
may return). The rogue policy commands half the declared joints above
their upper bound and half below their lower bound; after at least one
control step, every actuator command read back from MjData — and every
joint target recorded at the embodiment boundary — must lie inside the
declared bounds and must have moved off the mid-range baseline in the
commanded direction. Objects a runner path returns that carry their own
act() are exercised too, as guarded-policy wrappers. As a fallback
contract surface, clamp/guard/apply-style transforms in
harmless.guardrails are fed the same out-of-bounds action and must
return only in-bounds targets (plain mappings, wrapped results, or
arrays parallel to the declared joint order all count).

Neither harmless/runner.py nor harmless/guardrails.py exists yet —
finding no runnable path and no guard transform is the expected
pre-implementation failure.
"""

import importlib
import inspect
import os
import signal
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[2]
TIMEOUT_S = 60
EPS = 1e-6
MOVE = 1e-4

RUNNER_FN_NAMES = (
    "run", "run_episode", "run_steps", "step", "rollout", "loop",
    "evaluate", "exec_episode", "run_policy", "simulate", "episode",
    "run_once", "control_loop", "eval", "exec", "drive", "serve",
)
STEP_FNS = (
    "step", "control_step", "run_step", "policy_step", "step_once",
    "tick", "act_once", "update", "step_policy",
)
RUN_FNS = (
    "run", "run_episode", "rollout", "loop", "evaluate", "execute",
    "play", "run_once", "start", "go", "drive",
)
APPLY_FNS = (
    "set", "set_joints", "apply", "apply_action", "apply_targets",
    "actuate", "command", "send_targets", "set_targets", "set_ctrl",
    "write_targets", "drive", "control", "send", "act",
)
GUARD_FNS = (
    "apply", "clamp", "guard", "enforce", "filter", "sanitize",
    "limit", "clip", "bound", "project", "constrain", "guard_action",
    "clamp_action", "clamp_targets", "apply_guardrails",
    "enforce_limits", "safe_action", "filter_action", "process",
    "check", "wrap",
)
GUARD_HINTS = ("guard", "clamp", "limit", "bound", "safety", "filter",
               "rail")
WRAP_KEYS = (
    "targets", "action", "actions", "joint_pos", "positions", "values",
    "qpos", "command", "clamped", "result", "filtered", "guarded",
    "targets_out",
)


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout("entry point did not return inside the time budget")


def _bounded(fn, timeout=TIMEOUT_S):
    try:
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout)
    except (ValueError, OSError):
        old = None
    try:
        return fn()
    finally:
        if old is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


def _mujoco():
    import mujoco
    return mujoco


def _is_model(v):
    try:
        return isinstance(v, _mujoco().MjModel)
    except Exception:
        return False


def _is_data(v):
    try:
        return isinstance(v, _mujoco().MjData)
    except Exception:
        return False


def _num(v):
    if isinstance(v, (bool, str, bytes)):
        return False
    try:
        f = float(v)
    except Exception:
        return False
    return np.isfinite(f)


def _mid(space, j):
    lo, hi = space.bounds[j]
    return (lo + hi) / 2.0


def _oob_targets(space):
    out = {}
    for i, j in enumerate(space.joints):
        lo, hi = space.bounds[j]
        span = hi - lo
        out[j] = hi + 2.0 * span if i % 2 == 0 else lo - 2.0 * span
    return out


def _obs(space):
    return {
        "joints": {j: _mid(space, j) for j in space.joints},
        "joint_pos": np.array([_mid(space, j) for j in space.joints]),
        "time": 0.0,
    }


def _actmap(model, joints):
    mujoco = _mujoco()
    out = {}
    for a in range(model.nu):
        jid = int(model.actuator_trnid[a, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name in joints:
            out[name] = a
    return out


from harmless.embodiments.g1 import G1Embodiment
from harmless.policy import Policy


class _SpyEmbodiment(G1Embodiment):
    def __init__(self, *a, **k):
        self.applied = []
        super().__init__(*a, **k)
        for name in APPLY_FNS:
            setattr(self, name, self._rec(name))

    def _rec(self, name):
        def f(*args, **kwargs):
            self.applied.append((name, args, kwargs))
        return f


class _RoguePolicy(Policy):
    name = "emb3_rogue"
    description = "commands targets beyond the declared bounds"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.target = None
        self.calls = 0

    def act(self, observation):
        self.calls += 1
        if self.target is not None:
            return dict(self.target)
        sp = self.action_space
        return _oob_targets(sp) if sp is not None else {}


def _task():
    try:
        from harmless.task import get_task
        return get_task("transport")
    except Exception:
        return None


def _pool(fx, action=None, prev=None, extra=None):
    space, pol, emb = fx["space"], fx["pol"], fx["emb"]
    p = {
        "policy": pol, "pol": pol, "controller": pol, "agent": pol,
        "policy_obj": pol, "policy_name": "emb3_rogue",
        "embodiment": emb, "emb": emb, "robot": emb, "adapter": emb,
        "env": emb, "environment": emb, "hardware": emb,
        "embodiment_name": "g1", "robot_name": "g1",
        "model": fx["model"], "mjmodel": fx["model"], "m": fx["model"],
        "sim_model": fx["model"],
        "data": fx["data"], "mjdata": fx["data"], "d": fx["data"],
        "sim_data": fx["data"],
        "space": space, "action_space": space, "aspace": space,
        "limits": space, "bounds": dict(space.bounds),
        "joints": list(space.joints), "contract": space,
        "task": fx["task"], "scene": "kitchen.xml",
        "steps": 1, "max_steps": 1, "n_steps": 1, "num_steps": 1,
        "turns": 1, "horizon": 1, "budget": 1, "step_budget": 1,
        "rate_hz": space.rate_hz, "control_hz": space.rate_hz,
        "dt": 1.0 / space.rate_hz, "control_dt": 1.0 / space.rate_hz,
        "observation": _obs(space),
    }
    if action is not None:
        for k in ("action", "targets", "target", "command", "cmd",
                  "act", "values", "joint_pos", "a", "positions",
                  "raw_action", "qpos"):
            p[k] = action
    if prev is not None:
        for k in ("prev", "previous", "previous_action", "last_action",
                  "last", "prev_action", "old", "prior", "current",
                  "state"):
            p[k] = prev
    if extra:
        p.update(extra)
    return {k: v for k, v in p.items() if v is not None}


def _sig_argsets(fn, pool):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return []
    kwargs, missing, var_kw = {}, [], False
    for name, prm in sig.parameters.items():
        if prm.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if prm.kind == inspect.Parameter.VAR_KEYWORD:
            var_kw = True
            continue
        key = name.lower().lstrip("_")
        if key in pool:
            kwargs[name] = pool[key]
        elif prm.default is inspect.Parameter.empty:
            missing.append(name)
    if missing:
        return []
    out = [((), dict(kwargs))]
    if var_kw:
        merged = dict(pool)
        merged.update(kwargs)
        out.append(((), merged))
    return out


def _try(fn, argsets, timeout=TIMEOUT_S):
    errs = []
    for args, kwargs in argsets:
        try:
            return _bounded(lambda: fn(*args, **kwargs), timeout), None
        except _Timeout as e:
            return None, str(e)
        except Exception as e:
            errs.append(f"{type(e).__name__}: {e}")
    return None, "; ".join(errs[:3])


def _construct(entry, fx, extra_kwargs=None, extra_argsets=()):
    pool = _pool(fx, extra=extra_kwargs)
    argsets = _sig_argsets(entry, pool)
    p, e, m, d, t = (fx["pol"], fx["emb"], fx["model"], fx["data"],
                     fx["task"])
    argsets += [
        ((), {}), ((p,), {}), ((e,), {}), ((p, e), {}), ((e, p), {}),
        ((p, e, m, d), {}), ((e, p, m, d), {}),
        ((m, d, e, p), {}), ((m, d, p, e), {}),
        (("transport", "emb3_rogue", "g1"), {}),
    ]
    if t is not None:
        argsets += [((t, p, e), {}), ((p, e, t), {}), ((e, p, t), {}),
                    ((t, p, e, m, d), {})]
    if extra_kwargs:
        argsets += [((), dict(extra_kwargs)),
                    ((p, e), dict(extra_kwargs)),
                    ((e, p), dict(extra_kwargs))]
    argsets += list(extra_argsets)
    return _try(entry, argsets)


def _physics_pairs(fx, *objs):
    pairs, seen = [], set()

    def add(m, d):
        if _is_model(m) and _is_data(d) and id(d) not in seen:
            seen.add(id(d))
            pairs.append((m, d))

    add(fx["model"], fx["data"])
    for obj in objs:
        if obj is None:
            continue
        cand = [obj] + [getattr(obj, a, None) for a in (
            "sim", "env", "executor", "ex", "embodiment", "physics",
            "engine", "mj", "runner")]
        for o in cand:
            if o is None:
                continue
            m = None
            for a in ("model", "mjmodel", "m"):
                v = getattr(o, a, None)
                if _is_model(v):
                    m = v
            for a in ("data", "mjdata", "d", "sim_data"):
                d = getattr(o, a, None)
                if _is_data(d):
                    add(m if m is not None else fx["model"], d)
    return pairs


def _mark(fx):
    return (fx["pol"].calls, len(fx["emb"].applied),
            float(fx["data"].time), fx["data"].ctrl.copy())


def _progressed(mark, r, fx):
    if fx["pol"].calls > mark[0] or len(fx["emb"].applied) > mark[1]:
        return True
    if float(fx["data"].time) > mark[2] + 1e-9:
        return True
    if np.abs(np.asarray(fx["data"].ctrl) - mark[3]).max() > EPS:
        return True
    for m, d in _physics_pairs(fx, r):
        if d is fx["data"]:
            continue
        if np.asarray(d.ctrl).size and \
                np.abs(np.asarray(d.ctrl)).max() > EPS:
            return True
    return False


def _drive(r, fx):
    pool = _pool(fx, action=fx["pol"].target or _oob_targets(fx["space"]))
    errs = []
    for name in STEP_FNS + RUN_FNS:
        m = getattr(r, name, None)
        if not callable(m):
            continue
        argsets = _sig_argsets(m, pool) + [
            ((), {}), ((1,), {}), ((fx["pol"].target,), {}),
            ((), {"steps": 1}), ((), {"max_steps": 1}),
            ((), {"n_steps": 1})]
        last_err = None
        for args, kwargs in argsets:
            mark = _mark(fx)
            _, err = _try(m, [(args, kwargs)])
            if err is None and _progressed(mark, r, fx):
                return True, f"{name}()"
            if err:
                last_err = err
        if last_err:
            errs.append(f"{name}: {last_err}")
    return False, "; ".join(errs[:4])


def _extract_targets(args, kwargs, joint_set):
    out = {}
    if (len(args) == 2 and isinstance(args[0], str)
            and args[0] in joint_set and _num(args[1])):
        out[args[0]] = float(args[1])
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, Mapping):
            for k, v in a.items():
                if k in joint_set and _num(v):
                    out[k] = float(v)
        elif (isinstance(a, (tuple, list)) and len(a) == 2
              and isinstance(a[0], str) and a[0] in joint_set
              and _num(a[1])):
            out[a[0]] = float(a[1])
    return out


def _recorded_ok(fx):
    space = fx["space"]
    seen, bad = {}, []
    for name, args, kwargs in fx["emb"].applied:
        for j, v in _extract_targets(args, kwargs,
                                     set(space.joints)).items():
            seen[j] = v
            lo, hi = space.bounds[j]
            if not lo - EPS <= v <= hi + EPS:
                bad.append(f"{j}={v:.3f} outside [{lo:.3f},{hi:.3f}]")
    if bad:
        return False, "embodiment boundary received out-of-bounds: " \
            + "; ".join(bad[:4]), True
    if not seen:
        return False, \
            "no joint targets recorded at the embodiment boundary", False
    return True, "", True


def _pair_ok(m, d, fx):
    space = fx["space"]
    act = _actmap(m, set(space.joints))
    if not act:
        return False, "no declared joint maps to an actuator", False
    base = fx["baseline"].get(id(d))
    out_of, still = [], []
    for j in space.joints:
        if j not in act or j not in fx["direction"]:
            continue
        lo, hi = space.bounds[j]
        c = float(d.ctrl[act[j]])
        if not lo - EPS <= c <= hi + EPS:
            out_of.append(f"{j}={c:.3f} outside [{lo:.3f},{hi:.3f}]")
            continue
        b = _mid(space, j) if base is None else base[j]
        if fx["direction"][j] > 0 and c < b + MOVE:
            still.append(f"{j}={c:.3f} never rose toward {hi:.3f}")
        if fx["direction"][j] < 0 and c > b - MOVE:
            still.append(f"{j}={c:.3f} never fell toward {lo:.3f}")
    if out_of:
        return False, "out-of-bounds reached physics: " \
            + "; ".join(out_of[:4]), True
    if still:
        return False, "targets dropped, not clamped: " \
            + "; ".join(still[:4]), True
    return True, "", True


def _verify(fx, runner):
    reasons, observed = [], False
    for m, d in _physics_pairs(fx, runner):
        ok, why, obs = _pair_ok(m, d, fx)
        observed = observed or obs
        if ok:
            return None, observed
        reasons.append(why)
    ok, why, obs = _recorded_ok(fx)
    observed = observed or obs
    if ok:
        return None, observed
    reasons.append(why)
    return "; ".join(r for r in reasons if r)[:400], observed


def _act_wrap(r, fx):
    act = getattr(r, "act", None)
    if not callable(act):
        return "returned object has no act() either"
    res, err = _try(act, [((_obs(fx["space"]),), {}), ((), {})])
    if res is None:
        return f"act() wrapper raised: {err}"
    mapping = _to_mapping(res, fx["space"].joints)
    if mapping is None:
        return f"act() returned unreadable targets ({type(res).__name__})"
    return _targets_ok(mapping, fx, "act()")


def _reset(fx):
    space = fx["space"]
    act = _actmap(fx["model"], set(space.joints))
    for j, a in act.items():
        fx["data"].ctrl[a] = _mid(space, j)
    fx["ctrl0"] = fx["data"].ctrl.copy()
    fx["baseline"] = {id(fx["data"]): {
        j: _mid(space, j) for j in space.joints}}
    fx["pol"].calls = 0
    fx["pol"].target = _oob_targets(space)
    fx["direction"] = {
        j: (1 if fx["pol"].target[j] > space.bounds[j][1] else -1)
        for j in space.joints}
    fx["emb"].applied.clear()


def _attempt_runner(label, entry, fx):
    _reset(fx)
    mark = _mark(fx)
    r, err = _construct(entry, fx)
    if inspect.isclass(entry):
        if r is None:
            return f"{label}: construction failed ({err[:240]})", False
        drove, why = _drive(r, fx)
        if drove:
            res, observed = _verify(fx, r)
            return res, observed
        wrap = _act_wrap(r, fx)
        if wrap is None:
            return None, False
        return (f"{label}: no step/run method advanced a control step "
                f"({why[:160]})"), False
    if r is None and not _progressed(mark, None, fx):
        return f"{label}: no callable signature worked ({err[:240]})", \
            False
    if _progressed(mark, r, fx):
        res, observed = _verify(fx, r)
        return res, observed
    if r is not None:
        drove, why = _drive(r, fx)
        if drove:
            res, observed = _verify(fx, r)
            return res, observed
        wrap = _act_wrap(r, fx)
        if wrap is None:
            return None, False
        return f"{label}: never advanced a control step ({why[:160]})", \
            False
    return f"{label}: call produced nothing measurable", False


def _to_mapping(res, joints, depth=0):
    if res is None or depth > 3:
        return None
    js = set(joints)
    if isinstance(res, Mapping):
        if any(k in js for k in res):
            return {k: res[k] for k in res if k in js}
        for k in WRAP_KEYS:
            if k in res:
                inner = _to_mapping(res[k], joints, depth + 1)
                if inner is not None:
                    return inner
        return None
    if isinstance(res, (tuple, list)):
        if res and all(isinstance(p, (tuple, list)) and len(p) == 2
                       and isinstance(p[0], str) for p in res):
            d = dict(res)
            if any(k in js for k in d):
                return {k: d[k] for k in d if k in js}
        try:
            arr = np.asarray(res, dtype=float).ravel()
            if arr.size == len(joints):
                return dict(zip(joints, arr.tolist()))
        except Exception:
            pass
        for el in res:
            inner = _to_mapping(el, joints, depth + 1)
            if inner is not None:
                return inner
        return None
    for k in WRAP_KEYS:
        v = getattr(res, k, None)
        if v is not None and not callable(v):
            inner = _to_mapping(v, joints, depth + 1)
            if inner is not None:
                return inner
    try:
        arr = np.asarray(res, dtype=float).ravel()
        if arr.size == len(joints):
            return dict(zip(joints, arr.tolist()))
    except Exception:
        pass
    return None


def _targets_ok(mapping, fx, how):
    space = fx["space"]
    if not mapping:
        return f"{how}: empty target set"
    bad = []
    for j, v in mapping.items():
        if j not in space.bounds:
            continue
        if not _num(v):
            bad.append(f"{j}={v!r} not numeric")
            continue
        lo, hi = space.bounds[j]
        if not lo - EPS <= float(v) <= hi + EPS:
            bad.append(f"{j}={float(v):.3f} outside [{lo:.3f},{hi:.3f}]")
    if bad:
        return f"{how}: transform left targets out-of-bounds: " \
            + "; ".join(bad[:4])
    return None


def _call_transform(fn, fx, action, prev):
    joints = fx["space"].joints
    pool = _pool(fx, action=action, prev=prev)
    argsets = _sig_argsets(fn, pool) + [
        ((action,), {}), ((action, fx["space"]), {}),
        ((action, prev), {}), ((action, prev, fx["space"]), {}),
        ((fx["space"], action), {}),
        ((fx["space"], action, prev), {}),
    ]
    errs = []
    for args, kwargs in argsets:
        sent = dict(action)
        args = tuple(sent if a is action else a for a in args)
        kwargs = {k: (sent if v is action else v)
                  for k, v in kwargs.items()}
        res, err = _try(fn, [(args, kwargs)])
        if res is None and err:
            errs.append(err)
            continue
        mapping = _to_mapping(res, joints)
        if mapping is not None:
            return mapping, None
        if sent != action:
            mutated = _to_mapping(sent, joints)
            if mutated is not None:
                return mutated, None
    return None, "; ".join(errs[:3]) or "no readable targets returned"


def _attempt_guard(label, entry, fx):
    space = fx["space"]
    oob = _oob_targets(space)
    prev = {j: _mid(space, j) for j in space.joints}
    if inspect.isclass(entry):
        inst, err = _construct(
            entry, fx, extra_argsets=[((space,), {}),
                                      ((fx["emb"],), {})])
        if inst is None:
            return f"{label}: construction failed ({err[:200]})"
        for mname in (*GUARD_FNS, "act", "step", "update", "__call__"):
            m = getattr(inst, mname, None)
            if not callable(m):
                continue
            mapping, _ = _call_transform(m, fx, oob, prev)
            if mapping is None:
                continue
            return _targets_ok(mapping, fx, f"{label}.{mname}")
        return f"{label}: no transform method produced readable targets"
    mapping, why = _call_transform(entry, fx, oob, prev)
    if mapping is None:
        return f"{label}: {why[:200]}"
    return _targets_ok(mapping, fx, label)


def _runner_entries():
    try:
        mod = importlib.import_module("harmless.runner")
    except Exception as e:
        return [], f"harmless.runner: {type(e).__name__}: {e}"
    out = []
    for name, v in sorted(vars(mod).items()):
        if name.startswith("_"):
            continue
        if inspect.isclass(v):
            out.append((f"harmless.runner.{name}", v))
        elif callable(v) and (
                name.lower() in RUNNER_FN_NAMES
                or any(h in name.lower()
                       for h in ("run", "step", "rollout", "episode"))):
            out.append((f"harmless.runner.{name}()", v))
    return out, None


def _guard_entries():
    try:
        mod = importlib.import_module("harmless.guardrails")
    except Exception:
        return []
    out = []
    for name, v in sorted(vars(mod).items()):
        if name.startswith("_"):
            continue
        if inspect.isclass(v):
            if any(h in name.lower() for h in GUARD_HINTS) or any(
                    callable(getattr(v, m, None)) for m in GUARD_FNS):
                out.append((f"harmless.guardrails.{name}", v))
        elif callable(v) and (
                name.lower() in GUARD_FNS
                or any(h in name.lower() for h in GUARD_HINTS)):
            out.append((f"harmless.guardrails.{name}()", v))
    return out


def _fixtures():
    try:
        from harmless.registry import register_policy
        register_policy("emb3_rogue", _RoguePolicy)
    except Exception:
        pass
    mujoco = _mujoco()
    model = mujoco.MjModel.from_xml_path(str(REPO / "kitchen.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    emb = _SpyEmbodiment(model, data)
    pol = _RoguePolicy(embodiment=emb)
    return {
        "model": model, "data": data, "emb": emb, "pol": pol,
        "space": emb.action_space, "task": _task(),
        "baseline": {}, "direction": {}, "ctrl0": data.ctrl.copy(),
    }


def test_emb_3_1_out_of_bounds_actions_clamped_before_physics():
    entries, err = _runner_entries()
    guards = _guard_entries()
    assert entries or guards, (
        "no guarded action path found: harmless.runner exposes no runner "
        "entry point and harmless.guardrails exposes no clamp/guard "
        "transform" + (f" ({err})" if err else ""))
    fx = _fixtures()
    failures, blocked = [], False
    for label, entry in entries:
        try:
            res, obs = _attempt_runner(label, entry, fx)
        except Exception as e:
            res, obs = (
                f"{label}: harness error {type(e).__name__}: {e}", False)
        if res is None:
            return
        failures.append(res)
        blocked = blocked or obs
    if blocked:
        pytest.fail(
            "a runner demonstrably drove actions into physics without "
            "clamping them to the declared bounds:\n"
            + "\n".join(failures[:14]))
    for label, entry in guards:
        try:
            res = _attempt_guard(label, entry, fx)
        except Exception as e:
            res = f"{label}: harness error {type(e).__name__}: {e}"
        if res is None:
            return
        failures.append(res)
    pytest.fail(
        "no guarded path clamps out-of-bounds actions before physics:\n"
        + "\n".join(failures[:14]))
