"""Physics-measured success specs (AGP stage 8).

eval_success turns a task's success spec into a verdict read off measured
simulation state — body poses, velocities and live contacts — never off
transcript text. It takes an Executor, a raw (MjModel, MjData) pair, or
the same via keyword/state dicts, in any argument order, and understands
every spec spelling in use: predicate tuples and lists, dict leaves, and
composite all/and/any/or/not wrappers.
"""
from __future__ import annotations

from collections.abc import Mapping

from .task import ALIASES, _aabb as _task_aabb, _held as _task_held, \
    _pos as _task_pos

REST_SPEED = 0.05
ON_TOP_DZ = (-0.03, 0.05)

_ALL = frozenset({"all", "and", "every", "all_of"})
_ANY = frozenset({"any", "or", "any_of", "one_of"})
_NOT = frozenset({"not", "never"})
_ALL_KEYS = _ALL | {"predicates", "specs", "requires"}
_PERSON = frozenset({"person", "person_driver"})
_ARG_KEYS = ("obj", "object", "name", "a", "target", "surface", "ref",
             "on", "with", "b", "other", "obstacle", "objects", "names",
             "tol", "tol_m", "tol_xy", "radius", "dz", "dz_band", "gap",
             "margin", "speed", "rest_speed")


def _mj():
    import mujoco
    return mujoco


def _np():
    import numpy as np
    return np


def eval_success(spec=None, *args, executor=None, sim=None, model=None,
                 data=None, state=None, stats=None, poses=None):
    """Verdict for a success spec on measured state; never on text.

    Spec spellings: ("on_top_of", obj, target), ("on", obj, surface,
    tol_xy, dz), ("near", obj, ref, tol_m), ("at_rest", obj),
    ("no_contact", objects[, counterparties]), ("held"/"released", obj),
    ("between_upright", obj, a, b, tol), ("ball_in_ring",); dict leaves
    {"kind": ..., "obj": ..., "target": ...}; composites
    ("all"/"and"/"any"/"or"/"not", ...) and {"all"/"predicates": [...]}.
    Placement predicates read collision-geometry AABBs and require the
    object released and at rest; no_contact reads live data.contact plus
    the executor's measured avoid stats.
    """
    ex, m, d, sp, st = _resolve(spec, args, executor=executor, sim=sim,
                                model=model, data=data, state=state,
                                stats=stats, poses=poses)
    if sp is None:
        raise ValueError("eval_success: no spec given")
    if m is None or d is None:
        raise ValueError("eval_success: needs measured state (an "
                         "Executor or a (model, data) pair)")
    return bool(_eval(sp, m, d, ex, st))


def _looks_model(x):
    return hasattr(x, "ngeom") and hasattr(x, "nbody") \
        and not hasattr(x, "qpos")


def _looks_data(x):
    return hasattr(x, "qpos") and hasattr(x, "ncon")


def _looks_exec(x):
    return _looks_model(getattr(x, "model", None)) and \
        _looks_data(getattr(x, "data", None))


def _resolve(spec, args, executor=None, sim=None, model=None, data=None,
             state=None, stats=None, poses=None):
    ex = executor if executor is not None else sim
    m, d, sp, st = model, data, None, stats
    for v in (spec, *args):
        if v is None:
            continue
        if _looks_exec(v):
            ex = v
        elif _looks_data(v):
            d = d if d is not None else v
        elif _looks_model(v):
            m = m if m is not None else v
        elif isinstance(v, Mapping) and {"model", "data"} & set(v):
            m = m if m is not None else v.get("model")
            d = d if d is not None else v.get("data")
            st = st if st is not None else v.get("stats")
            ex = ex or v.get("executor") or v.get("sim")
        elif sp is None:
            sp = v
    if isinstance(state, Mapping):
        m = m if m is not None else state.get("model")
        d = d if d is not None else state.get("data")
        st = st if st is not None else state.get("stats")
        ex = ex or state.get("executor") or state.get("sim")
    elif _looks_exec(state):
        ex = ex or state
    elif _looks_model(state):
        m = m if m is not None else state
    elif _looks_data(state):
        d = d if d is not None else state
    if isinstance(poses, Mapping):
        pm, pd = poses.get("model"), poses.get("data")
        m = m if m is not None else pm
        d = d if d is not None else pd
    if ex is not None:
        m = m if m is not None else getattr(ex, "model", None)
        d = d if d is not None else getattr(ex, "data", None)
        st = st if st is not None else getattr(ex, "stats", None)
    return ex, m, d, sp, st


def _eval(spec, m, d, ex, st):
    if spec is None:
        return False
    if isinstance(spec, Mapping):
        for k in _ALL_KEYS:
            if k in spec:
                return _all_seq(_seq(spec[k]), m, d, ex, st)
        for k in _ANY:
            if k in spec:
                return _any_seq(_seq(spec[k]), m, d, ex, st)
        for k in _NOT:
            if k in spec:
                return not _all_seq(_seq(spec[k]), m, d, ex, st)
        kind = spec.get("kind") or spec.get("type") \
            or spec.get("predicate") or spec.get("op")
        if not isinstance(kind, str):
            return False
        return _pred(kind, [spec[k] for k in _ARG_KEYS if k in spec],
                     m, d, ex, st)
    if isinstance(spec, (tuple, list)):
        if not spec:
            return False
        head = spec[0]
        if isinstance(head, str):
            if head in _ALL:
                return _all_seq(spec[1:], m, d, ex, st)
            if head in _ANY:
                return _any_seq(spec[1:], m, d, ex, st)
            if head in _NOT:
                return not _all_seq(spec[1:], m, d, ex, st)
            return _pred(head, list(spec[1:]), m, d, ex, st)
        return _all_seq(spec, m, d, ex, st)
    if isinstance(spec, str):
        return _pred(spec, [], m, d, ex, st)
    return False


def _seq(v):
    if isinstance(v, (tuple, list)) and v and isinstance(v[0], str) \
            and v[0] not in _ALL | _ANY | _NOT:
        return [v]
    if isinstance(v, (tuple, list, set, frozenset)):
        return list(v)
    return [v]


def _all_seq(seq, m, d, ex, st):
    return all(_eval(s, m, d, ex, st) for s in seq)


def _any_seq(seq, m, d, ex, st):
    return any(_eval(s, m, d, ex, st) for s in seq)


def _pred(kind, a, m, d, ex, st):
    try:
        if kind in ("on_top_of", "resting_on", "placed_on", "rests_on"):
            return _on_top(m, d, ex, a[0], a[1])
        if kind == "on":
            return _on(m, d, ex, a[0], a[1],
                       float(a[2]) if len(a) > 2 else 0.5,
                       float(a[3]) if len(a) > 3 else 0.12)
        if kind == "near":
            return _near(m, d, ex, a[0], a[1], float(a[2]))
        if kind in ("at_rest", "rest", "still", "stationary"):
            return _at_rest(m, d, a[0])
        if kind in ("no_contact", "clear_of", "not_touching", "avoids"):
            return _no_contact(m, d, ex, st, *a)
        if kind in ("contact", "touching", "touches"):
            return _contact(m, d, a[0], a[1])
        if kind in ("held", "grasped", "holding"):
            return _held(m, d, ex, a[0])
        if kind in ("released", "not_held", "let_go"):
            return not _held(m, d, ex, a[0])
        if kind == "between_upright":
            return _between_upright(m, d, ex, *a)
        if kind == "ball_in_ring":
            return _ball_in_ring(m, d)
    except (ValueError, KeyError, IndexError, TypeError):
        return False
    return False


def _canon(name):
    return ALIASES.get(name, name)


def _names(v):
    if isinstance(v, str):
        return {_canon(v)}
    return {_canon(x) for x in v}


def _aabb(m, d, name):
    return _task_aabb(m, d, _canon(name))


def _pos(m, d, name):
    return _task_pos(m, d, _canon(name))


def _held(m, d, ex, name):
    if ex is not None and getattr(ex, "held", None) == _canon(name):
        return True
    return _task_held(m, d, _canon(name))


def _at_rest(m, d, name):
    mujoco, np = _mj(), _np()
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, _canon(name))
    if bid < 0:
        raise ValueError(f"no body named '{name}'")
    return bool(np.linalg.norm(d.cvel[bid, 3:]) < REST_SPEED)


def _on_top(m, d, ex, obj, target):
    np = _np()
    olo, ohi = _aabb(m, d, obj)
    tlo, thi = _aabb(m, d, target)
    cx, cy = ((olo + ohi) / 2)[:2]
    return bool(not _held(m, d, ex, obj) and _at_rest(m, d, obj)
                and tlo[0] < cx < thi[0] and tlo[1] < cy < thi[1]
                and ON_TOP_DZ[0] < olo[2] - thi[2] < ON_TOP_DZ[1])


def _on(m, d, ex, obj, target, tol_xy, dz):
    np = _np()
    olo, ohi = _aabb(m, d, obj)
    tlo, thi = _aabb(m, d, target)
    c = np.array((olo + ohi)[:2] / 2)
    tc = np.array((tlo + thi)[:2] / 2)
    return bool(not _held(m, d, ex, obj) and _at_rest(m, d, obj)
                and np.linalg.norm(c - tc) < tol_xy
                and ON_TOP_DZ[0] <= olo[2] - thi[2] <= dz)


def _near(m, d, ex, obj, ref, tol):
    np = _np()
    return bool(np.linalg.norm(_pos(m, d, obj) - _pos(m, d, ref)) < tol
                and not _held(m, d, ex, obj))


def _top(m, b):
    while m.body_parentid[b] > 0:
        b = m.body_parentid[b]
    return b


def _bname(m, b):
    return _mj().mj_id2name(m, _mj().mjtObj.mjOBJ_BODY, b) or "world"


def _movable(m, b):
    mujoco = _mj()
    for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return True
    return False


def _no_contact(m, d, ex, st, *a):
    if not a:
        return True
    avoid = _names(a[0])
    other = _names(a[1]) if len(a) > 1 else None
    for c in range(d.ncon):
        con = d.contact[c]
        b1 = _top(m, m.geom_bodyid[con.geom1])
        b2 = _top(m, m.geom_bodyid[con.geom2])
        n1, n2 = _bname(m, b1), _bname(m, b2)
        if other is not None:
            if (n1 in avoid and n2 in other) or (n2 in avoid and n1 in other):
                return False
        elif (n1 in avoid) != (n2 in avoid):
            nb, bb = (n2, b2) if n1 in avoid else (n1, b1)
            if nb not in _PERSON and _movable(m, bb):
                return False
    if _furniture_contacts(st):
        av = getattr(ex, "avoid_objects", None) if ex is not None else None
        if av is None or avoid & set(av):
            return False
    return True


def _furniture_contacts(st):
    if st is None:
        return 0
    if isinstance(st, Mapping):
        return st.get("furniture_contacts", 0)
    return getattr(st, "furniture_contacts", 0)


def _contact(m, d, a, b):
    A, B = _names(a), _names(b)
    for c in range(d.ncon):
        con = d.contact[c]
        n1 = _bname(m, _top(m, m.geom_bodyid[con.geom1]))
        n2 = _bname(m, _top(m, m.geom_bodyid[con.geom2]))
        if (n1 in A and n2 in B) or (n1 in B and n2 in A):
            return True
    return False


def _between_upright(m, d, ex, obj, a, b, tol):
    mujoco, np = _mj(), _np()
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, _canon(obj))
    if bid < 0:
        raise ValueError(f"no body named '{obj}'")
    up_local = np.array([0.0, 0.0, 1.0])
    if ex is not None:
        up_local = np.asarray(getattr(ex, "_up_local", {}).get(obj, up_local))
    up = d.xmat[bid].reshape(3, 3) @ up_local
    olo, ohi = _aabb(m, d, obj)
    c = np.array((olo + ohi)[:2] / 2)
    pa = _pos(m, d, a)[:2]
    pb = _pos(m, d, b)[:2]
    ab = pb - pa
    s = float((c - pa) @ ab / (ab @ ab))
    lateral = float(np.linalg.norm(c - (pa + s * ab)))
    floor = min(_aabb(m, d, a)[0][2], _aabb(m, d, b)[0][2])
    return bool(not _held(m, d, ex, obj) and _at_rest(m, d, obj)
                and up[2] > 0.9 and 0.1 < s < 0.9 and lateral < tol
                and abs(olo[2] - floor) < 0.05)


def _ball_in_ring(m, d):
    np = _np()
    try:
        b, r = _pos(m, d, "ball"), _pos(m, d, "ring")
    except ValueError:
        return False
    return bool(np.linalg.norm(b[:2] - r[:2]) < 0.15
                and abs(b[2] - r[2]) < 0.10)
