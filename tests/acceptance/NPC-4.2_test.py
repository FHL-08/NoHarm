"""NPC-4.2@1 Contacts between robot links and person geoms are counted
with per-contact force.

The requirement's touches are harmless/receipts.py and harmless/person.py,
so the counting instrumentation must be reachable through their public
surface. The test discovers it broadly: module-level callables whose names
read like contact/force/hit/impact/harm measurement, constructible classes
(Executor-, (MjModel, MjData)-, stats- or zero-arg bound) exposing such
methods or attributes, module-level objects, recorder-style objects with a
per-step tick (step/tick/update/sample/observe/record/...), and
force-bearing collections nested inside emitted receipt dicts.

The contract checked against real MuJoCo contact state:

  count is per contact:  the robot is teleported overlapping the person
    so many robot-link↔person-geom pairs collide at once (~15 pairs, ~8
    carrying >1 N in one step). The instrumentation must report at least
    as many contacts as simultaneous pairs — a counter that adds 1 per
    step with a contact (the executor's legacy person_contacts does
    exactly that, and lives outside the touches) reads 4 where ~8 real
    contacts are measured.
  per-contact force:     every recorded contact carries a measured force:
    each colliding pair's real mj_contactForce magnitude must be matched
    by some recorded force, and the bulk of recorded forces must match
    real measured contact forces — not constants, not just the peak.
  only robot-person:     while her body lies on the floor — real
    person↔floor contacts present — a robot-person count must not grow.

Nothing of this exists yet — harmless/receipts.py is absent and
harmless/person.py exposes no contact surface — so the expected
pre-implementation failure is the missing instrumentation.
"""

import importlib
import inspect
import os
import re
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[2]

MODULES = ("harmless.receipts", "harmless.person")
CONTACT_WORDS = ("contact", "force", "hit", "impact", "collis", "touch",
                 "harm", "count")
COUNT_WORDS = ("contact", "count", "hits", "impacts")
FORCE_KEY = re.compile(r"force|newton|_n$|^n$|peak", re.I)
TICK_NAMES = ("step", "tick", "update", "advance", "sample", "observe",
              "poll", "record", "collect", "notify", "on_step",
              "after_step", "process")
RECEIPT_WORDS = ("receipt", "measured", "measure", "harm", "episode",
                 "summary", "outcome", "report")

HIT_OFFSET_M = 0.15
HIT_STEPS = 4
FLOOR_STEPS = 3
FORCE_FLOOR_N = 1.0
TOL_REL = 0.6
TOL_ABS = 1.0


def _mujoco():
    import mujoco
    return mujoco


def _executor(tmp_path):
    from harmless.executor import Executor
    return Executor(str(REPO / "kitchen.xml"), record=False,
                    out_dir=str(tmp_path), skills_path=None)


def _pelvis_qadr(ex):
    jid = ex.model.body_jntadr[ex.pelvis_body]
    return ex.model.jnt_qposadr[jid]


def _pelvis_to(ex, xy, z=0.85):
    mujoco = _mujoco()
    qa = _pelvis_qadr(ex)
    ex.data.qpos[qa:qa + 3] = (float(xy[0]), float(xy[1]), z)
    ex.data.qvel[:] = 0.0
    mujoco.mj_forward(ex.model, ex.data)


def _lay_person_down(ex):
    mujoco = _mujoco()
    qa = ex.model.jnt_qposadr[ex.model.body_jntadr[ex.person_body]]
    ex.data.qpos[qa + 2] = 0.12
    ex.data.qpos[qa + 3:qa + 7] = (np.cos(np.pi / 4), np.sin(np.pi / 4),
                                  0.0, 0.0)
    if ex.person_tether >= 0:
        ex.data.eq_active[ex.person_tether] = 0
    ex.data.qvel[:] = 0.0
    mujoco.mj_forward(ex.model, ex.data)


def _rp_contacts(ex):
    """{(geom_a, geom_b): max force} measured robot-link↔person-geom
    contacts in the current state, both force conventions."""
    mujoco = _mujoco()
    m, d = ex.model, ex.data
    pairs = {}
    f6 = np.zeros(6)
    for c in range(d.ncon):
        con = d.contact[c]
        p = (con.geom1 in ex.person_geoms, con.geom2 in ex.person_geoms)
        if not any(p):
            continue
        other = con.geom2 if p[0] else con.geom1
        if not ex._robot_body(m.geom_bodyid[other]):
            continue
        key = (min(con.geom1, con.geom2), max(con.geom1, con.geom2))
        mujoco.mj_contactForce(m, d, c, f6)
        f = float(np.linalg.norm(f6[:3]))
        pairs.setdefault(key, []).append(f)
        pairs[key].append(float(abs(f6[0])))
    return pairs


def _person_nonrobot_contacts(ex):
    m, d = ex.model, ex.data
    n = 0
    for c in range(d.ncon):
        con = d.contact[c]
        p = (con.geom1 in ex.person_geoms, con.geom2 in ex.person_geoms)
        if not any(p):
            continue
        other = con.geom2 if p[0] else con.geom1
        if not ex._robot_body(m.geom_bodyid[other]):
            n += 1
    return n


def _contactish(name):
    n = str(name).lower()
    return any(w in n for w in CONTACT_WORDS)


def _countish(name):
    n = str(name).lower()
    return any(w in n for w in COUNT_WORDS)


def _entry_force(e):
    """The force carried by one contact record: the biggest |number|
    under a force-named key, else None."""
    if isinstance(e, Mapping):
        items = list(e.items())
    else:
        items = []
        for k in dir(e):
            if not k.startswith("_"):
                try:
                    items.append((k, getattr(e, k)))
                except Exception:
                    pass
    fs = [float(v) for k, v in items
          if FORCE_KEY.search(str(k)) and isinstance(v, Real)
          and not isinstance(v, bool)]
    return max(fs, key=abs) if fs else None


def _collection_forces(coll):
    """Per-entry forces out of a list/tuple/set of records or a
    {name: force} mapping."""
    out = []
    items = coll.values() if isinstance(coll, Mapping) else coll
    for it in items:
        if isinstance(it, bool):
            continue
        if isinstance(it, Real):
            out.append(float(it))
        elif isinstance(it, Mapping) or hasattr(it, "__dict__"):
            f = _entry_force(it)
            if f is not None:
                out.append(f)
        elif isinstance(it, (list, tuple)):
            out += _collection_forces(it)
    return out


def _extract(v):
    """(per-contact forces, numeric counts) reachable in a return value:
    a bare collection of records, a dict of forces, or a receipt-like
    dict embedding either under contact/force-named keys."""
    forces, counts = [], []
    if isinstance(v, Mapping):
        vals = list(v.values())
        if vals and all(isinstance(x, Real) and not isinstance(x, bool)
                        for x in vals):
            forces += [float(x) for x in vals]
        for k, x in v.items():
            ks = str(k)
            if isinstance(x, Mapping) or isinstance(x, (list, tuple)):
                if _contactish(ks) or FORCE_KEY.search(ks):
                    forces += _collection_forces(x)
                else:
                    f, c = _extract(x)
                    forces += f
                    counts += c
            elif isinstance(x, Real) and not isinstance(x, bool):
                if _countish(ks):
                    counts.append(int(x))
                elif FORCE_KEY.search(ks):
                    forces.append(float(x))
    elif isinstance(v, (list, tuple, set)):
        forces += _collection_forces(v)
    return forces, counts


def _call_shapes(ex):
    st = getattr(ex, "stats", None)
    return [
        lambda f: f(ex),
        lambda f: f(ex.model, ex.data),
        lambda f: f(ex.data),
        lambda f: f(st),
        lambda f: f(),
        lambda f: f(source=ex),
        lambda f: f(executor=ex),
        lambda f: f(sim=ex),
        lambda f: f(model=ex.model, data=ex.data),
        lambda f: f(stats=st),
    ]


def _construct(cls, ex):
    st = getattr(ex, "stats", None)
    for args, kw in (
        ((ex,), {}), ((ex.model, ex.data), {}), ((ex.data,), {}),
        ((st,), {}), ((), {}),
        ((), {"source": ex}), ((), {"executor": ex}), ((), {"sim": ex}),
        ((), {"model": ex.model, "data": ex.data}), ((), {"stats": st}),
    ):
        try:
            return cls(*args, **kw)
        except Exception:
            continue
    return None


def _call_any(fn, ex):
    for do in _call_shapes(ex):
        try:
            v = do(fn)
        except Exception:
            continue
        if v is not None:
            return v
    return None


def _tick_of(obj):
    for n in TICK_NAMES:
        f = getattr(obj, n, None)
        if callable(f):
            return f
    return None


def _tick(f, ex, i, dt):
    for args in ((), (dt,), (ex,), (ex.model, ex.data), (i,), (i, dt),
                 (ex.data,)):
        try:
            f(*args)
            return
        except TypeError:
            continue
        except Exception:
            return


def _is_plain_container(obj):
    return isinstance(obj, (dict, list, tuple, set, str, bytes,
                          np.ndarray, Real))


def _sources_on(obj, ex, label, out):
    """Sources reachable from one module member / instance."""
    tick = None if _is_plain_container(obj) else _tick_of(obj)
    if tick is not None:
        def make_read(o=obj):
            def read():
                best_f, best_c = [], []
                for n in dir(o):
                    if n.startswith("_"):
                        continue
                    if not (_contactish(n) or
                            any(w in n.lower() for w in RECEIPT_WORDS)):
                        continue
                    try:
                        v = getattr(o, n)
                    except Exception:
                        continue
                    v = _call_any(v, ex) if callable(v) else v
                    if v is None:
                        continue
                    f, c = _extract(v)
                    if len(f) > len(best_f):
                        best_f = f
                    best_c += c
                return best_f, best_c
            return read
        out.append({"label": f"{label} (ticked)", "tick": tick,
                    "read": make_read()})
    for n in dir(obj):
        if n.startswith("_"):
            continue
        try:
            v = getattr(obj, n)
        except Exception:
            continue
        if callable(v) and (_contactish(n) or
                            any(w in n.lower() for w in RECEIPT_WORDS)):
            out.append({"label": f"{label}.{n}()", "tick": None,
                        "read": (lambda fn=v: _extract(
                            _call_any(fn, ex)))})
        elif not callable(v) and _contactish(n) and \
                isinstance(v, (list, tuple, set, Mapping)):
            out.append({"label": f"{label}.{n}", "tick": None,
                        "read": (lambda x=v: _extract(x))})


def _member_sources(mod, name, ex):
    """Sources reachable from one module member, bound to `ex`."""
    out = []
    try:
        obj = getattr(mod, name)
    except Exception:
        return out
    label = f"{mod.__name__}.{name}"
    if inspect.isclass(obj):
        inst = _construct(obj, ex)
        if inst is not None:
            _sources_on(inst, ex, f"{label}(…)", out)
    elif inspect.ismodule(obj):
        pass
    elif callable(obj):
        if _contactish(name) or \
                any(w in name.lower() for w in RECEIPT_WORDS):
            out.append({"label": f"{label}()", "tick": None,
                        "read": (lambda fn=obj: _extract(
                            _call_any(fn, ex)))})
    else:
        _sources_on(obj, ex, label, out)
        if _contactish(name) and \
                isinstance(obj, (list, tuple, set, Mapping)):
            out.append({"label": label, "tick": None,
                        "read": (lambda x=obj: _extract(x))})
    return out


def _members(mod):
    return [n for n in dir(mod) if not n.startswith("_")]


def _close(f, m):
    return abs(f - m) <= max(TOL_ABS, TOL_REL * abs(m))


def _eval_source(src, ex):
    """Run both phases for one source (bound to `ex`); return
    (floor_added, hit_forces, hit_count, measured_pairs, measured, floor)."""
    tick, read = src["tick"], src["read"]
    _tick_i = [0]

    def tk():
        if tick is not None:
            _tick(tick, ex, _tick_i[0], ex.model.opt.timestep)

    f0, c0 = read()
    n0 = (max(c0) if c0 else len(f0))

    _lay_person_down(ex)
    n_floor = 0
    for _ in range(FLOOR_STEPS):
        ex.step(1)
        _tick_i[0] += 1
        tk()
        n_floor = max(n_floor, _person_nonrobot_contacts(ex))
    f1, c1 = read()
    floor_added = (max(c1) if c1 else len(f1)) - n0

    ex.reset_trial()
    _pelvis_to(ex, np.array(ex.data.xpos[ex.person_body][:2], float)
               + HIT_OFFSET_M)
    measured = {}
    final_pairs = {}
    for _ in range(HIT_STEPS):
        ex.step(1)
        _tick_i[0] += 1
        tk()
        cur = _rp_contacts(ex)
        step_pairs = {k: max(v) for k, v in cur.items()}
        for k, v in cur.items():
            measured.setdefault(k, []).extend(v)
        final_pairs = step_pairs
    fh, ch = read()
    hit_forces = fh
    hit_count = max(ch) if ch else len(fh)
    return floor_added, hit_forces, hit_count, final_pairs, measured, \
        n_floor


def test_npc_4_2_robot_person_contacts_counted_with_per_contact_force(
        tmp_path):
    try:
        rec_mod = importlib.import_module("harmless.receipts")
    except Exception as e:
        pytest.fail(
            "harmless/receipts.py does not exist or does not import — the "
            "contact instrumentation the line requires is missing "
            f"({type(e).__name__}: {e})")

    mods = [rec_mod]
    try:
        mods.append(importlib.import_module("harmless.person"))
    except Exception:
        pass

    members = [(mod, n) for mod in mods for n in _members(mod)]
    ex0 = _executor(tmp_path / "scan")
    probe = [s for mod, n in members for s in _member_sources(mod, n, ex0)]
    assert probe, (
        "harmless.receipts and harmless.person expose no contact "
        "instrumentation to evaluate: no contact/force/hit/impact-named "
        "callable, collection, receipt field, or tickable recorder")

    notes = []
    good_count, good_forces = [], []
    for mod, name in members:
        ex = _executor(
            tmp_path / re.sub(r"\W", "_", f"{mod.__name__}.{name}")[:60])
        for src in _member_sources(mod, name, ex):
            try:
                (floor_added, forces, count,
                 final_pairs, measured, n_floor) = _eval_source(src, ex)
            except Exception as e:
                notes.append(f"{src['label']}: error — "
                             f"{type(e).__name__}: {e}")
                continue
            k_pairs = {p for p, f in final_pairs.items()
                       if f > FORCE_FLOOR_N}
            if len(k_pairs) < 2:
                notes.append(f"{src['label']}: staging produced "
                             f"{len(k_pairs)} measured robot-person "
                             "pairs — not evaluated")
                continue
            covered = sum(
                1 for p in k_pairs
                if any(_close(rf, mf) for rf in forces
                       for mf in measured.get(p, [])))
            matched = sum(1 for rf in forces
                          if any(_close(rf, mf)
                                 for ms in measured.values()
                                 for mf in ms))
            detail = (f"{src['label']}: floor +{floor_added} "
                      f"(person-floor contacts present: {n_floor}), hit "
                      f"read {len(forces)} forces / count {count} vs "
                      f"{len(final_pairs)} pairs ({len(k_pairs)} > "
                      f"{FORCE_FLOOR_N} N), {covered} pairs covered, "
                      f"{matched} forces match measured")
            notes.append(detail)
            if floor_added == 0 and count >= len(k_pairs):
                good_count.append(src["label"])
            if floor_added == 0 and len(forces) >= len(k_pairs) \
                    and covered >= len(k_pairs) and \
                    matched >= max(1, int(0.6 * len(forces))):
                good_forces.append(src["label"])

    assert good_count, (
        "no instrumentation source counts robot-link↔person-geom contacts "
        "per contact — with ~15 simultaneous pairs in one step the best "
        "count stayed at per-step granularity:\n" + "\n".join(notes[:16]))
    assert good_forces, (
        "no instrumentation source records per-contact force for "
        "robot-link↔person-geom contacts — a count exists but not one "
        "measured force per contact:\n" + "\n".join(notes[:16]))
