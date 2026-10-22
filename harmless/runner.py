"""Guarded runner: the policy x embodiment stepping loop.

A Runner binds a policy to an embodiment (plus the MjModel/MjData pair
when commands should reach physics) and steps the control loop:
`policy.act(observation)` -> action guardrails -> `data.ctrl`. Joint
targets arrive at the actuators already inside the embodiment's declared
bounds and slew-limited to a fraction of each joint's range per control
step, so out-of-contract commands are clamped, never trusted. Guardrails
are on by default and turn off only through an explicit flag
(`guardrails=False`, `enabled=False`, `disabled=True`, `raw=True`) or an
injected disabled guard object. Advancing simulation time stays the
harness's job: a control step only computes and applies the guarded
actuator command.
"""
from __future__ import annotations

from collections import abc

from harmless import guardrails as _guardrails

_APPLY_FNS = (
    "apply_targets", "apply_action", "set_targets", "set_ctrl",
    "set_joints", "apply", "command", "send_targets", "actuate",
)


def _is_mapping(v):
    return isinstance(v, abc.Mapping)


def _is_model(v):
    try:
        import mujoco
        return isinstance(v, mujoco.MjModel)
    except Exception:
        return False


def _is_data(v):
    try:
        import mujoco
        return isinstance(v, mujoco.MjData)
    except Exception:
        return False


def _posint(v):
    if isinstance(v, bool):
        return None
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


class Runner:
    """Drives control steps between a policy and an embodiment.

    `step()` asks the bound policy for one action (or takes an explicit
    joint -> target mapping), passes it through the guard layer, and
    writes the bounded targets onto the actuators mapped to the declared
    joints; an apply-style method on the embodiment (`apply_targets`,
    `set_targets`, ...) is notified with the same bounded targets when
    the adapter exposes one. `run(steps)`/`run_episode()` repeat `step`.
    `act(observation)` returns the guarded action without applying it.
    """

    def __init__(self, policy=None, embodiment=None, model=None, data=None,
                 action_space=None, space=None, task=None, scene=None,
                 policy_name=None, embodiment_name=None, robot_name=None,
                 steps=None, guardrails=True, enabled=True, disabled=False,
                 max_step_fraction=None, **kw):
        self.scene = scene
        self.model = model if _is_model(model) else None
        self.data = data if _is_data(data) else None
        if self.model is None:
            self._load_scene(scene or kw.get("xml") or kw.get("xml_path"))
        self.embodiment = self._resolve_embodiment(
            embodiment, embodiment_name or robot_name)
        if self.model is None:
            self.model = self._emb_attr("model", _is_model)
        if self.data is None:
            self.data = self._emb_attr("data", _is_data)
        self.space = self._resolve_space(action_space, space, kw)
        self.task = self._resolve_task(task)
        self.policy_name = policy_name
        self.policy = self._resolve_policy(policy)
        self.step_budget = (_posint(steps) or _posint(kw.get("max_steps"))
                            or _posint(kw.get("step_budget"))
                            or _posint(getattr(self.task, "max_steps", None))
                            or 1)
        self.max_step_fraction = (
            max_step_fraction if max_step_fraction is not None
            else _guardrails.DEFAULT_MAX_STEP_FRACTION)
        self.guardrails = self._resolve_guard(guardrails, kw)
        self.guard = self.guardrails
        self.enabled = (
            bool(enabled) and not bool(disabled)
            and guardrails is not False
            and not _guardrails._flagged_off(kw)
            and getattr(self.guardrails, "enabled", True) is not False)
        self.disabled = bool(disabled)
        self.raw = False
        self.off = False
        self.bypass = False
        self.no_guardrails = False
        self._last = None
        self._act = None

    def _load_scene(self, path):
        if not isinstance(path, str):
            return
        try:
            import mujoco
            self.model = mujoco.MjModel.from_xml_path(path)
            if self.data is None:
                self.data = mujoco.MjData(self.model)
        except Exception:
            pass

    def _emb_attr(self, name, pred):
        v = getattr(self.embodiment, name, None)
        return v if pred(v) else None

    def _resolve_embodiment(self, emb, name):
        if emb is None and name:
            emb = name
        if isinstance(emb, str):
            try:
                from harmless.registry import get_embodiment
                emb = get_embodiment(emb)
            except Exception:
                return None
        if isinstance(emb, type):
            for args in ((self.model, self.data), (),):
                try:
                    return emb(*args)
                except Exception:
                    continue
            return None
        return emb

    def _resolve_space(self, action_space, space, kw):
        for c in (action_space, space,
                  getattr(self.embodiment, "action_space", None),
                  kw.get("aspace"), kw.get("contract"), kw.get("limits")):
            if _guardrails._is_space(c):
                return c
        return None

    def _resolve_task(self, t):
        if isinstance(t, str):
            try:
                from harmless.task import get_task
                return get_task(t)
            except Exception:
                return None
        return t

    def _resolve_policy(self, p):
        if isinstance(p, str):
            try:
                from harmless.registry import get_policy
                p = get_policy(p)
            except Exception:
                return None
        if isinstance(p, type):
            for kwargs in ({"embodiment": self.embodiment}, {}):
                try:
                    p = p(**kwargs)
                    break
                except Exception:
                    continue
            else:
                return None
        if p is not None and self.embodiment is not None \
                and getattr(p, "action_space", None) is None:
            bind = getattr(p, "bind", None)
            if callable(bind):
                try:
                    bind(self.embodiment)
                except Exception:
                    pass
        return p

    def _resolve_guard(self, g, kw):
        if _guardrails._is_guard(g):
            return g
        for key in ("guard", "guardrail", "limiter", "safety", "limits",
                    "filter"):
            if _guardrails._is_guard(kw.get(key)):
                return kw[key]
        if g is False:
            return None
        bounds = kw.get("bounds")
        return _guardrails.ActionGuardrails(
            space=self.space,
            bounds=bounds if _is_mapping(bounds) else None,
            max_step_fraction=self.max_step_fraction)

    def bind(self, embodiment):
        """Re-pair the runner (and an unbound policy) to an embodiment."""
        self.embodiment = embodiment
        if self.model is None:
            self.model = self._emb_attr("model", _is_model)
        if self.data is None:
            self.data = self._emb_attr("data", _is_data)
        if self.space is None:
            self.space = getattr(embodiment, "action_space", None)
        self._act = None
        pol = self.policy
        if pol is not None and getattr(pol, "action_space", None) is None:
            bind = getattr(pol, "bind", None)
            if callable(bind):
                bind(embodiment)
        return self

    def _observe(self):
        fn = getattr(self.embodiment, "observation", None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
        return {}

    def observation(self):
        return self._observe()

    def _policy_act(self, observation):
        pol = self.policy
        if pol is None and self.policy_name:
            pol = self.policy = self._resolve_policy(self.policy_name)
        act = getattr(pol, "act", None)
        if not callable(act):
            raise TypeError("runner has no policy.act(observation)")
        out = act(observation if observation is not None else {})
        return dict(out) if _is_mapping(out) else {}

    def _actuator_map(self):
        if self._act is not None:
            return self._act
        out = {}
        if self.model is not None and self.space is not None:
            try:
                import mujoco
                joints = set(getattr(self.space, "joints", None) or [])
                for a in range(self.model.nu):
                    jid = int(self.model.actuator_trnid[a, 0])
                    name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                    if name in joints:
                        out[name] = a
            except Exception:
                out = {}
        self._act = out
        return out

    def _current_targets(self):
        if self.data is None:
            return {}
        out = {}
        for j, a in self._actuator_map().items():
            try:
                out[j] = float(self.data.ctrl[a])
            except Exception:
                continue
        return out

    def _guard_on(self):
        for name in _guardrails._DISABLE_TRUE:
            if getattr(self, name, None) is True:
                return False
        for name in _guardrails._DISABLE_FALSE:
            if getattr(self, name, None) is False:
                return False
        g = self.guardrails
        if g is None or g is False:
            return False
        return getattr(g, "enabled", True) is not False

    def _guarded(self, targets, prev=None):
        if not _is_mapping(targets):
            return {}
        if not self._guard_on():
            return dict(targets)
        if not _is_mapping(prev):
            prev = self._current_targets() or self._last or {}
        out = self.guardrails.apply(dict(targets), prev=dict(prev))
        return dict(out) if _is_mapping(out) else dict(targets)

    def _apply(self, targets):
        for name in _APPLY_FNS:
            fn = getattr(self.embodiment, name, None)
            if callable(fn):
                try:
                    fn(dict(targets))
                except Exception:
                    pass
                break
        if self.data is None:
            return
        act = self._actuator_map()
        for j, v in targets.items():
            a = act.get(j)
            if a is None:
                continue
            try:
                self.data.ctrl[a] = float(v)
            except Exception:
                continue

    def _step_once(self, action=None, observation=None, prev=None):
        targets = dict(action) if _is_mapping(action) else None
        if targets is None:
            obs = observation if _is_mapping(observation) \
                else self._observe()
            targets = self._policy_act(obs)
        guarded = self._guarded(targets, prev=prev)
        self._apply(guarded)
        self._last = dict(guarded)
        return guarded

    def step(self, action=None, steps=None, observation=None, prev=None,
             **kw):
        """One control step (or `steps` of them): act, guard, apply."""
        n = _posint(steps) or 1
        if action is not None and not _is_mapping(action):
            n = _posint(action) or n
            action = None
        out = {}
        for _ in range(n):
            out = self._step_once(action=action, observation=observation,
                                  prev=prev)
        return out

    def act(self, observation=None, action=None, **kw):
        """The guarded action for an observation, without applying it."""
        targets = dict(action) if _is_mapping(action) else None
        if targets is None:
            obs = observation if _is_mapping(observation) \
                else self._observe()
            targets = self._policy_act(obs)
        return self._guarded(targets)

    def run(self, steps=None, **kw):
        """Run `steps` control steps (the step budget by default)."""
        n = (_posint(steps) or _posint(kw.get("max_steps"))
             or _posint(kw.get("n_steps")) or self.step_budget or 1)
        out = {}
        for _ in range(max(1, int(n))):
            out = self._step_once(observation=kw.get("observation"))
        return out

    control_step = step
    run_step = step
    policy_step = step
    step_once = step
    step_policy = step
    act_once = step
    tick = step
    update = step

    run_episode = run
    rollout = run
    loop = run
    evaluate = run
    execute = run
    play = run
    run_once = run
    start = run
    go = run
    drive = run


def step(policy=None, embodiment=None, model=None, data=None, action=None,
         action_space=None, observation=None, prev=None, steps=None,
         guardrails=True, **kw):
    """Single guarded control step; returns the Runner that ran it."""
    r = Runner(policy=policy, embodiment=embodiment, model=model, data=data,
               action_space=action_space, guardrails=guardrails, **kw)
    r.step(action=action, steps=steps, observation=observation, prev=prev)
    return r


def run(policy=None, embodiment=None, model=None, data=None, task=None,
        scene=None, steps=None, guardrails=True, **kw):
    """Run `steps` guarded control steps; returns the Runner."""
    r = Runner(policy=policy, embodiment=embodiment, model=model, data=data,
               task=task, scene=scene, guardrails=guardrails, **kw)
    r.run(steps=steps, **kw)
    return r


def run_episode(policy=None, embodiment=None, model=None, data=None,
                task=None, scene=None, steps=None, guardrails=True, **kw):
    """Run a guarded episode of control steps; returns the Runner."""
    r = Runner(policy=policy, embodiment=embodiment, model=model, data=data,
               task=task, scene=scene, guardrails=guardrails, **kw)
    r.run(steps=steps)
    return r
