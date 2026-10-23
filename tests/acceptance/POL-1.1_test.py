"""POL-1.1@1 Each turn the model receives the scene frame, the egocentric
head-camera frame, and the previous measured receipt, and emits one
program(code) call.

The contract under test is the agent turn loop: what the model is shown
before it acts, and what the loop does with the call it emits. The loop
entry point is discovered by name across the modules this requirement
touches — harmless.agent.run_episode is the canonical one (run_episode.py
invokes it; harmless/session.py drives the same turn structure
continuously; a policies/ package may carry it instead).

A real GPT-6-astra call cannot happen inside a test, so the OpenAI client
is replaced with a recording fake: every request the loop makes is
captured verbatim, and the fake answers each turn with exactly one
program(code) call whose code prints a per-turn probe marker. That makes
the line's three clauses measurable:

  frames:   every request's newest user message must carry at least two
            image parts that decode to two DIFFERENT rendered frames —
            the scene frame and the egocentric head-camera frame;
  receipt:  from the second request on, that same message must carry the
            previous action's measured receipt — proven by the probe
            marker the previous turn's program printed into its stdout
            (it can only have travelled there inside the receipt), along
            with receipt/measured state text;
  program:  the executor must run exactly one program(code) per model
            reply — counted through a spy on Executor.program — one code
            submission per turn, not zero and not a burst.

The seams are deliberately permissive — openai.OpenAI is patched both on
the package and on any agent/session/policies module that binds it, the
reply object carries chat-completions and responses-style fields, and the
entry-point call is assembled from the signature — because the line
speaks about the turn contract, not the plumbing that delivers it.
"""

import base64
import copy
import importlib
import inspect
import io
import json
import os
import pkgutil
import types
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]
SCENE = str(REPO / "kitchen.xml")
PROBE = "pol1_1_probe"
TURNS = 3

LOOP_MODULES = [
    "harmless.agent", "harmless.session",
    "harmless.policies.agent", "harmless.policies.astra",
    "harmless.policies.llm", "harmless.policies.gpt",
    "harmless.policies",
]
ENTRY_NAMES = ("run_episode", "episode", "run_trial", "run_loop",
               "run_agent", "run")


class _FakeOpenAI:
    """Records each conversation sent to the model; replies with one
    program(code) call that prints a unique marker."""

    def __init__(self):
        self.calls = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))
        self.responses = types.SimpleNamespace(create=self._create)

    def _create(self, *args, **kwargs):
        n = len(self.calls)
        self.calls.append({"args": args,
                           "kwargs": copy.deepcopy(kwargs)})
        reply = json.dumps({
            "skill": "program",
            "args": {"code": f"print('{PROBE}_{n}')"},
            "expected": {"person_contacts": 0}})
        msg = types.SimpleNamespace(content=reply, text=reply)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg, text=reply)],
            output_text=reply, id="fake", model="fake", created=0,
            usage=types.SimpleNamespace(prompt_tokens=8,
                                        completion_tokens=4,
                                        total_tokens=12))


def _candidate_modules():
    names = list(LOOP_MODULES)
    try:
        pkg = importlib.import_module("harmless.policies")
    except Exception:
        return names
    for m in pkgutil.iter_modules(getattr(pkg, "__path__", [])):
        names.append(f"harmless.policies.{m.name}")
    return names


def _install_fake_openai(fake):
    """Point every reachable OpenAI() spelling at the fake. Returns the
    undo list for restore."""
    patched = []

    def patch(obj, attr, value):
        patched.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, value)

    try:
        import openai
        if hasattr(openai, "OpenAI"):
            patch(openai, "OpenAI", lambda *a, **k: fake)
    except Exception:
        pass
    for modname in _candidate_modules():
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        if hasattr(mod, "OpenAI"):
            patch(mod, "OpenAI", lambda *a, **k: fake)
        for attr in ("client", "_client"):
            c = getattr(mod, attr, None)
            if c is not None and hasattr(c, "chat"):
                patch(mod, attr, fake)
    return patched


def _restore(patched):
    for obj, attr, old in patched:
        setattr(obj, attr, old)


def _spy_program():
    """Wrap Executor.program so every code submission the loop makes is
    counted with its source. Returns (records, undo)."""
    from harmless.executor import Executor
    orig = Executor.program
    records = []

    def spy(self, code, *a, **k):
        records.append(code)
        return orig(self, code, *a, **k)

    Executor.program = spy
    return records, (Executor, "program", orig)


def _find_entry():
    """Locate the episode-loop callable among the touched modules."""
    found = []
    for modname in _candidate_modules():
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for name in ENTRY_NAMES:
            fn = getattr(mod, name, None)
            if callable(fn) and not inspect.isclass(fn):
                found.append((f"{modname}.{name}", fn))
    return found


def _call_loop(fn, tmp_path):
    """Invoke the episode loop with the kwargs its signature accepts."""
    kwargs = {
        "task": "acceptance probe: stand by",
        "max_steps": TURNS, "steps": TURNS, "turns": TURNS,
        "turns_per_trial": TURNS, "trials": 1,
        "out_dir": str(tmp_path), "out": str(tmp_path),
        "output_dir": str(tmp_path),
        "live": False, "headless": True,
        "scene": SCENE, "xml_path": SCENE, "xml": SCENE,
        "model": "pol1-1-probe-model",
        "memory_path": str(tmp_path / "no_memory.md"),
        "skills_path": str(tmp_path / "no_skills.json"),
    }
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
    except (TypeError, ValueError):
        return fn(kwargs["task"])
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                     for p in params.values())
    call_kw = {k: v for k, v in kwargs.items()
               if has_var_kw or k in params}
    missing = [name for name, p in params.items()
               if p.default is inspect.Parameter.empty
               and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              inspect.Parameter.KEYWORD_ONLY)
               and name not in call_kw]
    if missing == ["task"]:
        return fn(kwargs["task"],
                  **{k: v for k, v in call_kw.items() if k != "task"})
    if missing:
        raise TypeError(f"cannot fill required parameters: {missing}")
    return fn(**call_kw)


def _messages_of(call):
    """The message list a create() call was invoked with, whichever
    spelling carried it."""
    kw = call["kwargs"]
    for key in ("messages", "input"):
        v = kw.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    for a in call["args"]:
        if isinstance(a, list) and a and isinstance(a[0], dict):
            return a
    return []


def _parts(message):
    content = message.get("content")
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": str(content)}]


def _image_arrays(parts):
    arrays = []
    import imageio.v2 as imageio
    for p in parts:
        ptype = str(p.get("type", ""))
        if "image" not in ptype and "image_url" not in p:
            continue
        candidates = []
        url = p.get("image_url")
        if isinstance(url, dict):
            candidates.append(url.get("url"))
        elif isinstance(url, str):
            candidates.append(url)
        for key in ("url", "data", "b64_json", "source"):
            v = p.get(key)
            if isinstance(v, str):
                candidates.append(v)
            elif isinstance(v, dict):
                candidates += [v.get("url"), v.get("data")]
        for c in candidates:
            if not isinstance(c, str):
                continue
            raw = c.split(",", 1)[-1] if c.startswith("data:") else c
            try:
                blob = base64.b64decode(raw)
            except Exception:
                continue
            if blob[:4] != b"\x89PNG":
                continue
            arrays.append(imageio.imread(io.BytesIO(blob)))
    return arrays


def _text(parts):
    out = []
    for p in parts:
        if isinstance(p.get("text"), str):
            out.append(p["text"])
        elif str(p.get("type", "")) == "text":
            out.append(str(p))
    return "\n".join(out)


def test_pol_1_1_turn_frames_receipt_and_one_program_call(tmp_path):
    entries = _find_entry()
    assert entries, (
        "no episode-loop entry point found in harmless.agent / "
        "harmless.session / harmless.policies — nothing drives a model "
        "turn"
    )

    fake = _FakeOpenAI()
    patched = _install_fake_openai(fake)
    records, (spy_obj, spy_attr, spy_orig) = _spy_program()
    outcome = None
    try:
        errs = []
        for name, fn in entries:
            try:
                outcome = _call_loop(fn, tmp_path)
                if len(fake.calls) >= TURNS:
                    break
            except Exception as e:
                errs.append(f"{name}: {type(e).__name__}: {e}")
        else:
            pytest.fail(
                "every episode-loop entry failed before completing "
                f"{TURNS} turns:\n" + "\n".join(errs))
    finally:
        _restore(patched)
        setattr(spy_obj, spy_attr, spy_orig)

    requests = [_messages_of(c) for c in fake.calls][:TURNS]
    assert len(requests) == TURNS, (
        f"the loop made {len(requests)} model request(s) over {TURNS} "
        "turns — each turn must place exactly one request that yields "
        "one program(code) call"
    )
    assert len(records) == TURNS, (
        f"Executor.program ran {len(records)} time(s) over {TURNS} model "
        "replies — each turn must emit and run exactly one program(code) "
        "call"
    )
    for i, code in enumerate(records):
        assert isinstance(code, str) and PROBE in code, (
            f"turn {i} submitted non-program work: {code!r}"
        )

    import numpy as np
    for i, msgs in enumerate(requests):
        users = [m for m in msgs if m.get("role") == "user"]
        assert users, f"request {i} carries no user observation message"
        parts = _parts(users[-1])
        arrays = _image_arrays(parts)
        assert len(arrays) >= 2, (
            f"request {i} shows the model {len(arrays)} image frame(s); "
            "each turn must show the scene frame AND the egocentric "
            "head-camera frame"
        )
        sigs = {a.tobytes() for a in arrays}
        assert len(sigs) >= 2, (
            f"request {i}: the image frames are identical — the scene "
            "frame and the egocentric head-camera frame must be two "
            "different views"
        )
        text = _text(parts)
        if i >= 1:
            assert f"{PROBE}_{i - 1}" in text, (
                f"request {i} does not carry the previous turn's "
                "receipt: the marker the previous program printed to "
                "stdout is missing from the observation"
            )
            assert any(k in text.lower() for k in
                       ("receipt", "measured", "person_", "robot_fell",
                        "sim_time", "force")), (
                f"request {i} observation has no measured receipt state "
                "(receipt/measured/person/force fields)"
            )
    assert outcome is not None
