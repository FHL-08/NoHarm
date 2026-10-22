"""EMB-3.2@1 Per-step action deltas are limited to a fraction of each
joint's range, and disabling guardrails requires an explicit flag.

The contract under test has two halves, both on the guarded action path
between policy and physics. First, the delta half: between one control
step and the next, the joint targets a runner (harmless/runner.py, the
policy x embodiment stepping loop this requirement touches) writes to
physics may move by at most a fraction of each joint's declared range —
a policy swinging its command across the range must arrive as a bounded
slew, not a step change. Second, the flag half: guardrails are on by
default and can only be turned off through an explicit opt-out — a named
flag (guardrails=False, enabled=False, no_guardrails=True, an injected
disabled guard object, ...) on the runner or guard layer, never silently
absent.

The test binds the registered g1 adapter to the kitchen scene, parks
every declared actuator command at mid-range, and drives discovered
runner entry points the same way EMB-3.1 does (Runner-style classes via
signature-matched keywords or positional policy/embodiment/task/model/
data orders, module-level run/step/rollout functions re-invoked per
step, step()/run()-style methods, returned guarded-policy objects). The
probe policy alternates its command between the low side and the high
side of the range on successive act() calls — always inside the declared
bounds, so only the delta limiter can slow the swing. Actuator commands
read back from MjData after consecutive control steps must change by no
more than half a joint's range per step while still moving (a frozen
command is a drop, not a limit). For the flag half the same candidates
are re-driven with each plausible disabling keyword, attribute, module
toggle, or injected disabled guard object; a configuration counts only
if a policy's out-of-bounds command then reaches physics unclamped. The
same two checks are applied to limit/clamp transforms and guard objects
in harmless.guardrails as a fallback contract surface.

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
MAX_FRAC = 0.5

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
    "check", "wrap", "limit_delta",
)
GUARD_HINTS = ("guard", "clamp", "limit", "bound", "safety", "filter",
               "rail")
WRAP_KEYS = (
    "targets", "action", "actions", "joint_pos", "positions", "values",
    "qpos", "command", "clamped", "result", "filtered", "guarded",
    "targets_out",
)
DISABLE_FALSE = (
    "guardrails", "enable_guardrails", "guardrails_enabled",
    "enforce_guardrails", "enforce_limits", "enforce", "safety",
    "safe", "safe_mode", "limits", "limit", "clamp", "clamped",
    "bounded", "use_guardrails", "guard", "guarded", "enabled",
    "enable", "on", "active",
)
DISABLE_TRUE = (
    "no_guardrails", "disable_guardrails", "disabled", "unsafe",
    "allow_unbounded", "unbounded", "no_limits", "disable_limits",
    "raw", "bypass_guardrails", "bypass", "no_clamp", "off",
)
GUARD_OBJ_ATTRS = ("guardrails", "guardrail", "guard", "safety",
                   "limits", "limiter", "filter")


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


def _swing_targets(space, high):
    out = {}
    for j in space.joints:
        lo, hi = space.bounds[j]
        span = hi - lo
        out[j] = lo + 0.9 * span if high else lo + 0.1 * span
    return out


def _oob_targets(space):
    return {j: space.bounds[j][1] + 2.0 * (
        space.bounds[j][1] - space.bounds[j][0])
        for j in space.joints}


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


class _SwingPolicy(Policy):
    name = "emb3_swing"
    description = "alternates in-bounds commands across the range"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.target = None
        self.calls = 0

    def act(self, observation):
        self.calls += 1
        if self.target is not None:
            return dict(self.target)
        sp = self.action_space
        return (_swing_targets(sp, high=self.calls % 2 == 1)
                if sp is not None else {})


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
        "policy_obj": pol, "policy_name": "emb3_swing",
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
        (("transport", "emb3_swing", "g1"), {}),
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


def _step_once(r, fx):
    pool = _pool(fx, action=fx["pol"].target
                 or _swing_targets(fx["space"], True))
    for name in STEP_FNS + RUN_FNS:
        m = getattr(r, name, None)
        if not callable(m):
            continue
        argsets = _sig_argsets(m, pool) + [
            ((), {}), ((1,), {}), ((fx["pol"].target,), {}),
            ((), {"steps": 1}), ((), {"max_steps": 1}),
            ((), {"n_steps": 1})]
        for args, kwargs in argsets:
            mark = _mark(fx)
            _, err = _try(m, [(args, kwargs)])
            if err is None and _progressed(mark, r, fx):
                return True, name
    return False, None


def _make_stepper(entry, r, fx):
    if inspect.isclass(entry):
        def step():
            return _step_once(r, fx)
        return step, r

    def step():
        mark = _mark(fx)
        rr, _ = _construct(entry, fx)
        if _progressed(mark, rr, fx):
            return True, rr
        if rr is not None:
            ok, name = _step_once(rr, fx)
            return ok, rr
        return False, rr
    return step, r


def _snap(fx, *objs):
    out = []
    for m, d in _physics_pairs(fx, *objs):
        act = _actmap(m, set(fx["space"].joints))
        if act:
            out.append((d, {j: float(d.ctrl[a])
                            for j, a in act.items()}))
    return out


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


def _recorded(fx):
    seq = []
    for name, args, kwargs in fx["emb"].applied:
        t = _extract_targets(args, kwargs, set(fx["space"].joints))
        if t:
            seq.append(t)
    return seq


def _delta_ok(fx, snaps):
    space = fx["space"]
    reasons, moved = [], False
    for d, s2 in snaps[2]:
        prior = []
        for snap in snaps[:2]:
            for d0, s0 in snap:
                if d0 is d:
                    prior.append(s0)
        if len(prior) < 2:
            continue
        for j in set(s2) & set(prior[0]) & set(prior[1]):
            span = space.bounds[j][1] - space.bounds[j][0]
            d_a = abs(s2[j] - prior[1][j])
            d_b = abs(prior[1][j] - prior[0][j])
            if d_a > MAX_FRAC * span + MOVE \
                    or d_b > MAX_FRAC * span + MOVE:
                reasons.append(
                    f"{j} moved {max(d_a, d_b):.3f} in one step over "
                    f"range {span:.3f}")
            if d_a > MOVE or d_b > MOVE:
                moved = True
    if reasons:
        return False, "; ".join(reasons[:4])
    if not moved:
        return False, "commands never moved (swing dropped, not limited)"
    return True, ""


def _recorded_delta_ok(fx):
    seq = _recorded(fx)
    if len(seq) < 2:
        return False, "fewer than two apply-calls recorded"
    bad, moved = [], False
    for prev, cur in zip(seq, seq[1:]):
        for j in set(prev) & set(cur):
            span = fx["space"].bounds[j][1] - fx["space"].bounds[j][0]
            dv = abs(cur[j] - prev[j])
            if dv > MAX_FRAC * span + MOVE:
                bad.append(f"{j} jumped {dv:.3f} in one step")
            if dv > MOVE:
                moved = True
    if bad:
        return False, "; ".join(bad[:4])
    if not moved:
        return False, "apply-calls carried identical targets"
    return True, ""


def _oob_reached(fx, runner):
    for m, d in _physics_pairs(fx, runner):
        act = _actmap(m, set(fx["space"].joints))
        for j, a in act.items():
            lo, hi = fx["space"].bounds[j]
            v = float(d.ctrl[a])
            if v > hi + 1e-3 or v < lo - 1e-3:
                return True
    for t in _recorded(fx):
        for j, v in t.items():
            lo, hi = fx["space"].bounds[j]
            if v > hi + 1e-3 or v < lo - 1e-3:
                return True
    return False


def _reset(fx):
    space = fx["space"]
    act = _actmap(fx["model"], set(space.joints))
    for j, a in act.items():
        fx["data"].ctrl[a] = _mid(space, j)
    fx["pol"].calls = 0
    fx["emb"].applied.clear()


def _attempt_delta(label, entry, fx):
    _reset(fx)
    fx["pol"].target = None
    mark = _mark(fx)
    r, err = _construct(entry, fx)
    stepped = _progressed(mark, r, fx)
    if inspect.isclass(entry):
        if r is None:
            return (False,
                    f"{label}: construction failed ({err[:200]})", False)
    elif r is None and not stepped:
        return (False,
                f"{label}: no callable signature worked ({err[:200]})",
                False)
    step, runner = _make_stepper(entry, r, fx)
    snaps = [_snap(fx, r)] if stepped else []
    while len(snaps) < 3:
        ok, runner = step()
        if not ok:
            why = "never advanced a control step" \
                if not snaps else "stopped stepping early"
            return False, f"{label}: {why}", _observed(fx, snaps)
        snaps.append(_snap(fx, runner))
    ok, why = _delta_ok(fx, snaps)
    if ok:
        return True, None, True
    ok2, why2 = _recorded_delta_ok(fx)
    if ok2:
        return True, None, True
    return False, f"{label}: {why}; {why2}", _observed(fx, snaps)


def _observed(fx, snaps):
    return bool(_recorded(fx)) or any(len(s) for s in snaps)


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


def _call_transform(fn, fx, action, prev):
    joints = fx["space"].joints
    pool = _pool(fx, action=action, prev=prev)
    argsets = _sig_argsets(fn, pool) + [
        ((action,), {}), ((action, fx["space"]), {}),
        ((action, prev), {}), ((action, prev, fx["space"]), {}),
        ((fx["space"], action), {}),
        ((fx["space"], action, prev), {}),
    ]
    for args, kwargs in argsets:
        sent = dict(action)
        args = tuple(sent if a is action else a for a in args)
        kwargs = {k: (sent if v is action else v)
                  for k, v in kwargs.items()}
        res, err = _try(fn, [(args, kwargs)])
        if res is None and err:
            continue
        mapping = _to_mapping(res, joints)
        if mapping is not None:
            return mapping
        if sent != action:
            mutated = _to_mapping(sent, joints)
            if mutated is not None:
                return mutated
    return None


def _guard_delta_ok(label, entry, fx):
    space = fx["space"]
    lo_t = _swing_targets(space, high=False)
    hi_t = _swing_targets(space, high=True)
    seqs = []
    if inspect.isclass(entry):
        inst, err = _construct(
            entry, fx, extra_argsets=[((space,), {}), ((fx["emb"],), {})])
        if inst is None:
            return False, f"{label}: construction failed ({err[:160]})"
        for n in (*GUARD_FNS, "act", "step", "update", "__call__"):
            fn = getattr(inst, n, None)
            if not callable(fn):
                continue
            r1 = _call_transform(fn, fx, hi_t, lo_t)
            r2 = _call_transform(fn, fx, lo_t, r1 or lo_t)
            if r1 is not None and r2 is not None:
                seqs.append((r1, r2))
    else:
        r1 = _call_transform(entry, fx, hi_t, lo_t)
        r2 = _call_transform(entry, fx, lo_t, r1 or lo_t)
        if r1 is not None and r2 is not None:
            seqs.append((r1, r2))
    for r1, r2 in seqs:
        bad, moved = [], False
        for j in set(r1) & set(r2):
            if j not in space.bounds:
                continue
            span = space.bounds[j][1] - space.bounds[j][0]
            dv = abs(float(r2[j]) - float(r1[j]))
            if dv > MAX_FRAC * span + MOVE:
                bad.append(f"{j} jumped {dv:.3f} in one step")
            if dv > MOVE:
                moved = True
        if not bad and moved:
            return True, None
    return False, f"{label}: transform deltas not limited"


def _raw_oob(mapping, space):
    if not mapping:
        return False
    for j, v in mapping.items():
        if j in space.bounds and _num(v):
            lo, hi = space.bounds[j]
            if float(v) > hi + 1e-3 or float(v) < lo - 1e-3:
                return True
    return False


def _default_guarded(entry, fx):
    space = fx["space"]
    oob = _oob_targets(space)
    prev = {j: _mid(space, j) for j in space.joints}
    if inspect.isclass(entry):
        inst, _ = _construct(
            entry, fx, extra_argsets=[((space,), {}), ((fx["emb"],), {})])
        if inst is None:
            return False
        for n in (*GUARD_FNS, "act", "step", "update", "__call__"):
            fn = getattr(inst, n, None)
            if not callable(fn):
                continue
            out = _call_transform(fn, fx, oob, prev)
            if out is not None:
                return not _raw_oob(out, space)
        return False
    out = _call_transform(entry, fx, oob, prev)
    return out is not None and not _raw_oob(out, space)


def _guard_disable_ok(label, entry, fx):
    if not _default_guarded(entry, fx):
        return False, None
    space = fx["space"]
    oob = _oob_targets(space)
    prev = {j: _mid(space, j) for j in space.joints}
    attempts = []
    if inspect.isclass(entry):
        for name in DISABLE_FALSE:
            attempts.append((f"kwarg {name}=False", {name: False}))
        for name in DISABLE_TRUE:
            attempts.append((f"kwarg {name}=True", {name: True}))
        for how, extra in attempts:
            inst, _ = _construct(
                entry, fx, extra_kwargs=extra,
                extra_argsets=[((space,), {}), ((fx["emb"],), {})])
            if inst is None:
                continue
            for n in (*GUARD_FNS, "act", "step", "update", "__call__"):
                fn = getattr(inst, n, None)
                if not callable(fn):
                    continue
                if _raw_oob(_call_transform(fn, fx, oob, prev), space):
                    return True, how
        inst, _ = _construct(
            entry, fx, extra_argsets=[((space,), {}),
                                      ((fx["emb"],), {})])
        if inst is not None:
            for name in (*DISABLE_FALSE, *DISABLE_TRUE):
                if not hasattr(inst, name):
                    continue
                try:
                    setattr(inst, name, name in DISABLE_TRUE)
                except Exception:
                    continue
                for n in (*GUARD_FNS, "act", "step", "update",
                          "__call__"):
                    fn = getattr(inst, n, None)
                    if not callable(fn):
                        continue
                    if _raw_oob(
                            _call_transform(fn, fx, oob, prev), space):
                        return True, f"attr {name}"
    else:
        for name in DISABLE_FALSE:
            attempts.append((f"kwarg {name}=False", {name: False}))
        for name in DISABLE_TRUE:
            attempts.append((f"kwarg {name}=True", {name: True}))
        for how, extra in attempts:
            pool = _pool(fx, action=oob, prev=prev, extra=extra)
            for args, kwargs in _sig_argsets(entry, pool):
                res, err = _try(entry, [(args, kwargs)])
                if res is None and err:
                    continue
                if _raw_oob(_to_mapping(res, space.joints), space):
                    return True, how
    return False, None


def _guard_objects(fx):
    objs = []
    for label, entry in _guard_entries():
        if not inspect.isclass(entry):
            continue
        for name in DISABLE_FALSE:
            inst, _ = _construct(
                entry, fx, extra_kwargs={name: False},
                extra_argsets=[((fx["space"],), {}),
                               ((fx["emb"],), {})])
            if inst is not None:
                objs.append((f"{label}({name}=False)", inst))
                break
        else:
            inst, _ = _construct(
                entry, fx, extra_argsets=[((fx["space"],), {}),
                                          ((fx["emb"],), {})])
            if inst is not None:
                for name in (*DISABLE_FALSE, *DISABLE_TRUE):
                    if hasattr(inst, name):
                        try:
                            setattr(inst, name, name in DISABLE_TRUE)
                            objs.append((f"{label}.{name}", inst))
                            break
                        except Exception:
                            continue
    return objs


def _drive_oob(entry, fx, extra_kwargs=None):
    _reset(fx)
    fx["pol"].target = _oob_targets(fx["space"])
    mark = _mark(fx)
    r, _ = _construct(entry, fx, extra_kwargs=extra_kwargs)
    if inspect.isclass(entry):
        if r is None:
            return False, None
        ok, _name = _step_once(r, fx)
        if ok and _oob_reached(fx, r):
            return True, r
        return False, r
    if _progressed(mark, r, fx) and _oob_reached(fx, r):
        return True, r
    if r is not None:
        ok, _name = _step_once(r, fx)
        if ok and _oob_reached(fx, r):
            return True, r
    return False, r


def _runner_disable_ok(label, entry, fx):
    default_raw, _ = _drive_oob(entry, fx)
    if default_raw:
        return False, None
    variants = [(f"kwarg {n}=False", {n: False}) for n in DISABLE_FALSE]
    variants += [(f"kwarg {n}=True", {n: True}) for n in DISABLE_TRUE]
    for how, extra in variants:
        ok, r = _drive_oob(entry, fx, extra_kwargs=extra)
        if ok:
            return True, how
    if inspect.isclass(entry):
        r, _ = _construct(entry, fx)
        if r is not None:
            for sub_name in GUARD_OBJ_ATTRS:
                sub = getattr(r, sub_name, None)
                if sub is None:
                    continue
                for flag in (*DISABLE_FALSE, *DISABLE_TRUE):
                    if not hasattr(sub, flag):
                        continue
                    try:
                        setattr(sub, flag, flag in DISABLE_TRUE)
                    except Exception:
                        continue
                    _reset(fx)
                    fx["pol"].target = _oob_targets(fx["space"])
                    ok, _n = _step_once(r, fx)
                    if ok and _oob_reached(fx, r):
                        return True, f"{sub_name}.{flag}"
            for flag in (*DISABLE_FALSE, *DISABLE_TRUE):
                if not hasattr(r, flag):
                    continue
                try:
                    setattr(r, flag, flag in DISABLE_TRUE)
                except Exception:
                    continue
                _reset(fx)
                fx["pol"].target = _oob_targets(fx["space"])
                ok, _n = _step_once(r, fx)
                if ok and _oob_reached(fx, r):
                    return True, f"attr {flag}"
    for modname in ("harmless.runner", "harmless.guardrails"):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for flag in (*DISABLE_FALSE, *DISABLE_TRUE):
            if not hasattr(mod, flag):
                continue
            old = getattr(mod, flag)
            try:
                setattr(mod, flag, flag in DISABLE_TRUE)
                ok, _r = _drive_oob(entry, fx)
                if ok:
                    return True, f"{modname}.{flag}"
            finally:
                try:
                    setattr(mod, flag, old)
                except Exception:
                    pass
    for g_label, gobj in _guard_objects(fx):
        for key in GUARD_OBJ_ATTRS:
            ok, r = _drive_oob(entry, fx, extra_kwargs={key: gobj})
            if ok:
                return True, f"{key}={g_label}"
    return False, None


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
        register_policy("emb3_swing", _SwingPolicy)
    except Exception:
        pass
    mujoco = _mujoco()
    model = mujoco.MjModel.from_xml_path(str(REPO / "kitchen.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    emb = _SpyEmbodiment(model, data)
    pol = _SwingPolicy(embodiment=emb)
    return {
        "model": model, "data": data, "emb": emb, "pol": pol,
        "space": emb.action_space, "task": _task(),
    }


def test_emb_3_2_step_deltas_limited_and_disable_needs_flag():
    entries, err = _runner_entries()
    guards = _guard_entries()
    assert entries or guards, (
        "no guarded action path found: harmless.runner exposes no runner "
        "entry point and harmless.guardrails exposes no limit/guard "
        "transform" + (f" ({err})" if err else ""))
    fx = _fixtures()
    delta_fail, flag_fail = [], []
    delta_ok = flag_ok = False
    delta_blocked = False
    for label, entry in entries:
        try:
            ok, why, blk = _attempt_delta(label, entry, fx)
        except Exception as e:
            ok, blk = False, False
            why = f"{label}: harness {type(e).__name__}: {e}"
        if ok:
            delta_ok = True
        else:
            delta_fail.append(why)
            delta_blocked = delta_blocked or blk
        try:
            ok, how = _runner_disable_ok(label, entry, fx)
        except Exception as e:
            ok = False
            flag_fail.append(
                f"{label}: harness {type(e).__name__}: {e}")
        if ok:
            flag_ok = True
        else:
            flag_fail.append(f"{label}: no explicit flag disables "
                             "guardrails")
    if delta_blocked:
        delta_fail.append(
            "a runner demonstrably stepped actions into physics without "
            "limiting per-step deltas")
    for label, entry in guards:
        if not delta_blocked and not delta_ok:
            try:
                ok, why = _guard_delta_ok(label, entry, fx)
            except Exception as e:
                ok, why = False, \
                    f"{label}: harness {type(e).__name__}: {e}"
            if ok:
                delta_ok = True
            else:
                delta_fail.append(why)
        try:
            ok, how = _guard_disable_ok(label, entry, fx)
        except Exception:
            ok = False
        if ok:
            flag_ok = True
        else:
            flag_fail.append(f"{label}: no explicit flag disables "
                             "enforcement")
    if delta_ok and flag_ok:
        return
    msg = []
    if not delta_ok:
        msg.append("per-step deltas not limited to a fraction of joint "
                   "range:\n" + "\n".join(delta_fail[:10]))
    if not flag_ok:
        msg.append("no explicit flag disables guardrails anywhere:\n"
                   + "\n".join(flag_fail[:10]))
    pytest.fail("\n".join(msg))
