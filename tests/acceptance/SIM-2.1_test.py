"""SIM-2.1@1 Executor loads kitchen.xml and steps the G1, props, and person
with contacts measured.

The contract under test is the MuJoCo reference backend living under
harmless/sim/** (the only sim code this requirement may touch). The line
names an "Executor", so the test resolves one from the backend package:
`harmless.sim.Executor`, a class named Executor in any harmless.sim
submodule, then any executor-named class or executor-named factory in
those modules. `harmless.sim` does not exist yet — failing to import it
is the expected pre-implementation failure.

The Executor is driven the way a caller drives it: constructed on
kitchen.xml headless (NOHARM_HEADLESS=1, record off, outputs to a tmp
dir), then stepped about two seconds of sim through its own stepping
entry point (run_seconds/step and the usual synonyms). The three actors
the line names are checked against the live MjData:

  - G1: the pelvis-rooted subtree exists, finishes stepping at standing
    height, and carries measured contact force against the scene — the
    feet bear the robot's weight, so a stepped G1 shows real newtons.
  - props: the authored free props of kitchen.xml rest in the scene and
    participate in the contact set while stepped.
  - person: the free-body person is a stepped actor — she moves under
    the scene's driver or is already in contact; a welded statue that
    the step never touches does not count.

"Contacts measured" is checked twice: raw physics produces contacts whose
normal force mj_contactForce can measure on a robot contact, and the
Executor exposes its own measured-contact surface — a contacts-style
call returning entries, a stats object carrying contact/force fields, or
a sense/state reading carrying touch or force values.
"""

import importlib
import inspect
import os
import pkgutil
import re
from pathlib import Path

import mujoco
import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]
KITCHEN_XML = REPO / "kitchen.xml"

# the authored free props of kitchen.xml (named in its grasp welds)
PROP_NAMES = ("mug", "plate", "knife", "butterknife", "vase", "apple",
              "bowl", "tomato", "bread")

_PATH_PARAMS = ("xml_path", "scene", "scene_path", "scene_xml", "xml",
                "path", "model_path", "mjcf", "xml_file", "mjcf_path")
_HEADLESS_KW = {"record": False, "recording": False, "record_video": False,
                "headless": True, "viewer": False, "show_viewer": False,
                "gui": False, "render": False, "rendering": False,
                "attach_viewer": False, "window": False, "visible": False}
_OUT_PARAMS = ("out_dir", "output_dir", "run_dir", "out", "log_dir",
               "outdir", "output")
_WS_PARAMS = ("skills_path", "learned_skills", "skills", "ws_dir",
              "workspace", "workspace_dir")
_STEP_SECONDS_FNS = ("run_seconds", "step_seconds", "advance_seconds",
                     "simulate_seconds", "run_for", "forward_seconds")
_STEP_FNS = ("step", "advance", "simulate", "forward", "tick")


def _sim_modules():
    """Every importable module under harmless/sim — the backend package."""
    sim = importlib.import_module("harmless.sim")
    modules = [sim]
    path = getattr(sim, "__path__", None)
    if path is not None:
        for info in pkgutil.walk_packages(path, prefix="harmless.sim."):
            try:
                modules.append(importlib.import_module(info.name))
            except Exception:
                continue
    return modules


def _find_executor(modules):
    """The Executor the line names, wherever under harmless.sim it lives."""
    named = []
    factories = []
    for mod in modules:
        ex = getattr(mod, "Executor", None)
        if inspect.isclass(ex) or callable(ex):
            return ex
        for attr, value in vars(mod).items():
            low = attr.lower()
            if "executor" in low and inspect.isclass(value):
                named.append(value)
            elif "executor" in low and callable(value):
                factories.append(value)
            elif low in ("make_executor", "create_executor",
                         "load_executor", "get_executor") and callable(value):
                factories.append(value)
    if named:
        return named[0]
    if factories:
        return factories[0]
    return None


def _make_executor(factory, tmp_path):
    """Construct the Executor on kitchen.xml with headless-friendly kwargs
    wherever its signature accepts them."""
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        params = {}
    kw = {}
    for name in params:
        if name in _PATH_PARAMS:
            kw[name] = str(KITCHEN_XML)
        elif name in _HEADLESS_KW:
            kw[name] = _HEADLESS_KW[name]
        elif name in _OUT_PARAMS:
            kw[name] = str(tmp_path)
        elif name in _WS_PARAMS:
            kw[name] = str(tmp_path / name)
    errors = []
    for args in ((), (str(KITCHEN_XML),)):
        try:
            return factory(*args, **kw)
        except Exception as e:  # noqa: BLE001 - reported if all spellings fail
            errors.append(f"{factory}{args}+{sorted(kw)}: {e}")
    pytest.fail("could not construct the sim Executor on kitchen.xml:\n"
                + "\n".join(errors))


def _body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _run_seconds(ex, seconds):
    for name in _STEP_SECONDS_FNS:
        fn = getattr(ex, name, None)
        if callable(fn):
            fn(seconds)
            return
    for name in _STEP_FNS:
        fn = getattr(ex, name, None)
        if callable(fn):
            fn(int(round(seconds / float(ex.model.opt.timestep))))
            return
    pytest.fail("the sim Executor exposes no way to step sim time "
                "(no run_seconds/step-style method)")


def _contact_participants(ex):
    """Body-root id sets of every active contact pair."""
    pairs = []
    for c in range(ex.data.ncon):
        con = ex.data.contact[c]
        pairs.append((
            ex.model.body_rootid[ex.model.geom_bodyid[con.geom1]],
            ex.model.body_rootid[ex.model.geom_bodyid[con.geom2]]))
    return pairs


def _robot_body(ex, bid):
    pelvis = _body_id(ex.model, "pelvis")
    while bid > 0:
        if bid == pelvis:
            return True
        bid = ex.model.body_parentid[bid]
    return False


def _measured_surface(ex):
    """The Executor's own measured-contact surface, whichever it ships."""
    for name in ("contacts", "contact_forces", "measured_contacts",
                 "get_contacts", "contact_state", "contacts_measured"):
        fn = getattr(ex, name, None)
        if callable(fn):
            for args in ((), (False,)):
                try:
                    out = fn(*args)
                except TypeError:
                    continue
                if out:
                    return f"{name}() -> {len(out)} entries"
                break
    for name in ("stats", "episode_stats", "receipts", "measurements"):
        obj = getattr(ex, name, None)
        fields = vars(obj) if obj is not None and hasattr(obj, "__dict__") \
            else (obj if isinstance(obj, dict) else {})
        hits = [k for k in fields
                if re.search(r"contact|force|impact|touch|hit|near", str(k), re.I)]
        if hits:
            return f"{name}.{hits[0]}"
    for name in ("sense", "state"):
        fn = getattr(ex, name, None)
        if callable(fn):
            try:
                out = fn()
            except Exception:  # noqa: BLE001
                continue
            if isinstance(out, dict):
                hits = [k for k in out
                        if re.search(r"touch|contact|force", str(k), re.I)]
                if hits:
                    return f"{name}()['{hits[0]}']"
    return None


def test_sim_2_1_executor_loads_kitchen_and_steps_with_measured_contacts(tmp_path):
    ex = _make_executor(_require_executor(), tmp_path)
    try:
        model, data = getattr(ex, "model", None), getattr(ex, "data", None)
        assert isinstance(model, mujoco.MjModel) and isinstance(
            data, mujoco.MjData), \
            "the sim Executor does not expose the loaded MjModel/MjData pair"

        pelvis = _body_id(model, "pelvis")
        person = _body_id(model, "person")
        props = {n: _body_id(model, n) for n in PROP_NAMES}
        props = {n: b for n, b in props.items() if b >= 0}
        assert pelvis >= 0, "kitchen.xml loaded without the G1 (no pelvis body)"
        assert person >= 0, "kitchen.xml loaded without the person body"
        assert len(props) >= 5, (
            f"only {sorted(props)} of the authored props are in the scene")

        person_pos0 = np.array(data.xpos[person])
        _run_seconds(ex, 2.0)

        assert data.time >= 1.9, (
            f"stepping ran {data.time:.2f}s of sim, expected ~2s")
        assert np.isfinite(data.qpos).all(), "qpos went non-finite"
        assert data.xpos[pelvis][2] > 0.5, (
            f"the G1 does not stay at standing height while stepped "
            f"(pelvis z={data.xpos[pelvis][2]:.3f})")

        pairs = _contact_participants(ex)
        assert pairs, "no contacts generated while stepping the scene"

        # the G1 is stepped for real: a robot-vs-scene contact carries
        # measurable normal force (the feet bear its weight)
        f6 = np.zeros(6)
        robot_forces = []
        for c in range(data.ncon):
            con = data.contact[c]
            b1 = model.body_rootid[model.geom_bodyid[con.geom1]]
            b2 = model.body_rootid[model.geom_bodyid[con.geom2]]
            if _robot_body(ex, b1) != _robot_body(ex, b2):
                mujoco.mj_contactForce(model, data, c, f6)
                robot_forces.append(abs(float(f6[0])))
        assert robot_forces and max(robot_forces) > 1.0, (
            "no measurable contact force on the stepped G1 — the feet "
            "must bear weight")

        # the props are stepped: authored props rest in the contact set
        participants = {root for pair in pairs for root in pair}
        touched = {n for n, b in props.items()
                   if model.body_rootid[b] in participants}
        assert len(touched) >= 3, (
            f"only {sorted(touched)} props participate in contacts — "
            "the props are not being stepped in the scene")

        # the person is stepped: she walks the scene's route (mocap
        # driver) or is already interacting — a statue does not count
        person_moved = float(np.linalg.norm(
            np.array(data.xpos[person])[:2] - person_pos0[:2]))
        person_root = model.body_rootid[person]
        person_in_contact = any(
            person_root in pair for pair in pairs)
        assert person_moved > 0.05 or person_in_contact, (
            "the person is not stepped — she neither moves along the "
            "scene route nor participates in contacts")

        # measured, not just generated: the Executor reports contacts
        surface = _measured_surface(ex)
        assert surface, (
            "contacts exist but the Executor measures none — no "
            "contacts()/stats/sense surface reports them")
    finally:
        close = getattr(ex, "close", None)
        if callable(close):
            close()


def _require_executor():
    modules = _sim_modules()
    factory = _find_executor(modules)
    assert factory is not None, (
        "harmless.sim exposes no Executor (no Executor class, "
        "executor-named class, or executor factory)")
    return factory
