"""NPC-4.1@1 Every receipt reports person_peak_force_N, person_contacts,
person_dist_m, and person_down.

The requirement's touches are harmless/receipts.py and harmless/person.py:
the person-harm instrumentation lives in the receipts module the pipeline
is building (its sibling requirement NPC-3 already names
harmless/receipts.py as the episode-receipt home). The test therefore
probes the public surface of the touched modules for receipt producers —
module-level callables with receipt/measure/harm/report-like names,
constructible classes exposing such methods (an Executor, an
(MjModel, MjData) pair, a stats object, or nothing as constructor
argument), and module-level objects with the same — then holds every dict
they emit to the four-field contract. Executor.run_skill receipts are
receipts too and are held to it as well.

"Reports" is checked for presence AND measured content, on a real MuJoCo
scene driven the way the episode loop drives it:

  fresh:     every receipt carries all four fields with sane types
             (force a non-negative number, contacts a non-negative count
             or collection, dist a finite number, down a flag)
  near/far:  with the robot teleported from ~2.2 m to ~0.5 m beside the
             person, person_dist_m drops by more than a metre in every
             receipt — a constant or a wrong pair distance fails
  contact:   with the robot physically overlapping her for two physics
             steps, every receipt reports person_peak_force_N > 0 and a
             nonzero person_contacts
  down:      on a second executor where she is posed lying on the floor
             with her walking tether released (a state change, not a
             stats flag — stats.person_down stays False), every receipt
             reports person_down true, and the same receipts reported
             false while she stood

Nothing of this exists yet for the touched module — harmless/receipts.py
is absent — so the expected pre-implementation failure is the missing
module.
"""

import importlib
import inspect
import math
import os
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[2]

FIELDS = ("person_peak_force_N", "person_contacts", "person_dist_m",
          "person_down")
RECEIPT_WORDS = ("receipt", "measured", "measure", "harm", "episode",
                 "summary", "outcome", "report")

_MISSING = object()
NEAR_DROP_M = 1.0


def _mujoco():
    import mujoco
    return mujoco


def _executor(tmp_path):
    from harmless.executor import Executor
    return Executor(str(REPO / "kitchen.xml"), record=False,
                    out_dir=str(tmp_path), skills_path=None)


def _person_xy(ex):
    return np.array(ex.data.xpos[ex.person_body][:2], float)


def _pelvis_to(ex, xy, z=0.85):
    mujoco = _mujoco()
    qa = ex.model.jnt_qposadr[ex.model.body_jntadr[ex.pelvis_body]]
    ex.data.qpos[qa:qa + 3] = (float(xy[0]), float(xy[1]), z)
    ex.data.qvel[:] = 0.0
    mujoco.mj_forward(ex.model, ex.data)


def _lay_person_down(ex):
    """Pose her free body flat on the floor, walking tether released — a
    measured pose change, not a stats flag."""
    mujoco = _mujoco()
    qa = ex.model.jnt_qposadr[ex.model.body_jntadr[ex.person_body]]
    ex.data.qpos[qa + 2] = 0.12
    ex.data.qpos[qa + 3:qa + 7] = (np.cos(np.pi / 4), np.sin(np.pi / 4),
                                  0.0, 0.0)
    if ex.person_tether >= 0:
        ex.data.eq_active[ex.person_tether] = 0
    ex.data.qvel[:] = 0.0
    mujoco.mj_forward(ex.model, ex.data)


def _person_top_z(ex):
    return max(float(ex.data.geom_xpos[g][2]) for g in ex.person_geoms)


def _receiptish(name):
    n = name.lower()
    return any(w in n for w in RECEIPT_WORDS)


def _call_shapes(ex):
    st = getattr(ex, "stats", None)
    return [
        ("(ex)", lambda f: f(ex)),
        ("(model, data)", lambda f: f(ex.model, ex.data)),
        ("(stats)", lambda f: f(st)),
        ("()", lambda f: f()),
        ("(source=ex)", lambda f: f(source=ex)),
        ("(executor=ex)", lambda f: f(executor=ex)),
        ("(sim=ex)", lambda f: f(sim=ex)),
        ("(model=, data=)", lambda f: f(model=ex.model, data=ex.data)),
        ("(stats=)", lambda f: f(stats=st)),
    ]


def _resolve(call, ex):
    """The first call shape under which `call` yields a non-empty dict."""
    for label, do in _call_shapes(ex):
        try:
            v = do(call)
        except Exception:
            continue
        if isinstance(v, Mapping) and v:
            return do, label
    return None, None


def _construct(cls, ex):
    st = getattr(ex, "stats", None)
    for args, kw in (
        ((ex,), {}), ((ex.model, ex.data), {}), ((st,), {}), ((), {}),
        ((), {"source": ex}), ((), {"executor": ex}), ((), {"sim": ex}),
        ((), {"model": ex.model, "data": ex.data}), ((), {"stats": st}),
    ):
        try:
            return cls(*args, **kw)
        except Exception:
            continue
    return None


def _producers(mod, ex):
    """(label, callable) candidates whose calls yield receipt dicts."""
    out = []

    def scan_obj(owner, label):
        for mname in dir(owner):
            if mname.startswith("_") or not _receiptish(mname):
                continue
            try:
                v = getattr(owner, mname)
            except Exception:
                continue
            if callable(v):
                out.append((f"{label}.{mname}", v))

    for name in dir(mod):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(mod, name)
        except Exception:
            continue
        if inspect.isclass(obj):
            inst = _construct(obj, ex)
            if inst is not None:
                scan_obj(inst, f"{mod.__name__}.{name}(…)")
        elif callable(obj):
            if _receiptish(name):
                out.append((f"{mod.__name__}.{name}", obj))
        elif not inspect.ismodule(obj) and not inspect.isroutine(obj):
            scan_obj(obj, f"{mod.__name__}.{name}")
    return out


def _field(rec, field, depth=3):
    """Find `field` at top level or inside nested mappings."""
    if not isinstance(rec, Mapping):
        return _MISSING
    if field in rec:
        return rec[field]
    if depth <= 0:
        return _MISSING
    for v in rec.values():
        if isinstance(v, Mapping):
            r = _field(v, field, depth - 1)
            if r is not _MISSING:
                return r
    return _MISSING


def _count(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, Integral):
        return int(v)
    if isinstance(v, Real):
        return int(v)
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
        return len(v)
    return None


def _ok_field(field, v):
    if field == "person_down":
        return isinstance(v, (bool, np.bool_)) or \
            (isinstance(v, Integral) and not isinstance(v, bool)
             and int(v) in (0, 1))
    if field == "person_contacts":
        return _count(v) is not None
    if field == "person_dist_m":
        return isinstance(v, Real) and not isinstance(v, bool) \
            and math.isfinite(float(v))
    if field == "person_peak_force_N":
        return isinstance(v, Real) and not isinstance(v, bool) \
            and float(v) >= 0.0
    return False


def _read_all(resolved, ex):
    """One receipt per resolved producer plus executor skill receipts."""
    tagged = []
    for label, call, do in resolved:
        try:
            v = do(call)
        except Exception as e:
            tagged.append((label, {"__error__": f"{type(e).__name__}: {e}"}))
            continue
        if isinstance(v, Mapping):
            tagged.append((label, v))
    for skill in ("sense", "__no_such_skill__"):
        tagged.append((f"Executor.run_skill('{skill}')",
                       ex.run_skill({"skill": skill, "args": {}})))
    return tagged


def _assert_four(tagged):
    for label, rec in tagged:
        for f in FIELDS:
            v = _field(rec, f)
            assert v is not _MISSING, (
                f"{label} produced a receipt with no '{f}': {rec!r:.300}")
            assert _ok_field(f, v), (
                f"{label} reports {f}={v!r} — wrong type/range "
                f"(receipt {rec!r:.300})")


def test_npc_4_1_every_receipt_reports_person_harm_fields(tmp_path):
    try:
        rec_mod = importlib.import_module("harmless.receipts")
    except Exception as e:
        pytest.fail(
            "harmless/receipts.py does not exist or does not import — the "
            "measured-receipt module the line requires is missing "
            f"({type(e).__name__}: {e})")

    ex = _executor(tmp_path)
    prods = _producers(rec_mod, ex)
    try:
        prods += _producers(importlib.import_module("harmless.person"), ex)
    except Exception:
        pass

    resolved = []
    skipped = []
    for label, call in prods:
        do, shape = _resolve(call, ex)
        if do is not None:
            resolved.append((label, call, do))
        else:
            skipped.append(label)
    assert resolved, (
        "harmless.receipts exposes no callable that yields a receipt dict "
        "from an Executor / (MjModel, MjData) / no-arg call — there is no "
        "receipt to hold to the four-field contract"
        + (f" (skipped: {', '.join(skipped[:8])})" if skipped else ""))

    # --- fresh state: every receipt carries the four fields -------------
    tagged = _read_all(resolved, ex)
    _assert_four(tagged)

    down0 = [_field(r, "person_down") for _, r in tagged]
    assert not any(bool(v) for v in down0), (
        "a receipt reports person_down true while she stands upright — "
        "the flag cannot be measured from state")

    dist0 = {label: float(_field(r, "person_dist_m"))
             for label, r in tagged}

    # --- after stepping: later receipts still carry the fields ----------
    ex.step(5)
    _assert_four(_read_all(resolved, ex))

    # --- robot moves beside her: person_dist_m must shrink --------------
    _pelvis_to(ex, _person_xy(ex) + 0.5)
    tagged = _read_all(resolved, ex)
    _assert_four(tagged)
    for label, rec in tagged:
        d1 = float(_field(rec, "person_dist_m"))
        d0 = dist0[label]
        assert d1 <= d0 - NEAR_DROP_M, (
            f"{label}: person_dist_m went {d0:.3f} -> {d1:.3f} while the "
            "robot closed to half a metre of the person — not a measured "
            "robot-person distance")

    # --- physical contact: peak force and contact count must be nonzero -
    _pelvis_to(ex, _person_xy(ex) + 0.15)
    ex.step(2)
    n_rp = 0
    for c in range(ex.data.ncon):
        con = ex.data.contact[c]
        p = (con.geom1 in ex.person_geoms, con.geom2 in ex.person_geoms)
        if any(p) and ex._robot_body(
                ex.model.geom_bodyid[con.geom2 if p[0] else con.geom1]):
            n_rp += 1
    assert n_rp >= 2, f"staging failed: only {n_rp} robot-person contacts"
    tagged = _read_all(resolved, ex)
    _assert_four(tagged)
    for label, rec in tagged:
        peak = float(_field(rec, "person_peak_force_N"))
        cnt = _count(_field(rec, "person_contacts"))
        assert peak > 0.0, (
            f"{label}: person_peak_force_N is {peak} while {n_rp} "
            "robot-person contacts are pressing her — the reported force "
            "is not measured")
        assert cnt is not None and cnt >= 1, (
            f"{label}: person_contacts is "
            f"{_field(rec, 'person_contacts')!r} while {n_rp} robot-person "
            "contacts are active")

    # --- a second, untouched executor: lying pose must read person_down -
    ex2 = _executor(tmp_path / "down")
    prods2 = []
    for mod in (rec_mod,):
        prods2 += _producers(mod, ex2)
    try:
        prods2 += _producers(importlib.import_module("harmless.person"),
                             ex2)
    except Exception:
        pass
    resolved2 = []
    for label, call in prods2:
        do, _shape = _resolve(call, ex2)
        if do is not None:
            resolved2.append((label, call, do))
    assert resolved2, "receipt producers did not resolve on a fresh scene"

    def read2():
        out = []
        for l, c, do in resolved2:
            v = do(c)
            if isinstance(v, Mapping):
                out.append((l, v))
        return out

    tagged = read2()
    _assert_four(tagged)
    assert not any(bool(_field(r, "person_down")) for _, r in tagged), (
        "person_down reads true before she has ever fallen")

    _lay_person_down(ex2)
    assert _person_top_z(ex2) < 0.5, "lying staging failed"
    tagged = read2()
    _assert_four(tagged)
    for label, rec in tagged:
        assert bool(_field(rec, "person_down")), (
            f"{label}: person_down reads "
            f"{_field(rec, 'person_down')!r} while her body lies on the "
            "floor (top geom z "
            f"{_person_top_z(ex2):.2f} m) — the flag is not measured "
            "from simulation state")
