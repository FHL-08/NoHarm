"""POL-4.1@1 After each episode the model's reflections write lessons to
memory.md and reusable defs to learned_skills.json, injected into the next
prompt.

The contract under test is the experience store that carries one trial's
reflections into the next trial's prompt. learn.py drives it through two
public seams — harmless.agent.run_episode (one trial) and
harmless.agent.reflect_on_episode (the post-episode reflection) — and this
test drives them exactly the way learn.py does:

    episode 1 -> reflect_on_episode -> episode 2

A real GPT-6-astra call cannot happen inside a test, so the OpenAI client
is replaced with a recording fake. The fake answers each trial's first
turn with one program(code) call and the second turn with done; the
reflection request (REFLECT_SYSTEM prompt) gets a JSON lessons+skills
reply carrying unique markers. That makes the line's three clauses
measurable:

  lessons:  after the reflection call, memory.md must contain the lesson
            text the model returned;
  defs:     learned_skills.json must gain the reusable def the model
            returned (name -> {description, code});
  injected: the NEXT episode's first model request must carry the lesson
            text and the skill (name + description) in its system prompt,
            and the authored def must actually be callable inside the next
            episode's program() sandbox — proven by episode 2's first
            program calling it by name and the receipt (read back from the
            following request's user message) returning its value.

The test runs the real Executor on the real kitchen scene, headless, with
the memory/skills paths pointed at tmp_path through the same kwargs
learn.py relies on (its defaults are these file names in the cwd).
"""

import copy
import json
import os
import types
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]
SCENE = str(REPO / "kitchen.xml")

TASK = "pol-4.1 acceptance probe: stand by"
LESSON = "POL41_LESSON_MARKER: carry the bowl with both palms on the rim"
SKILL = "pol41_probe_skill"
SKILL_DESC = "POL41_SKILL_DESC returns the probe constant"
SKILL_CODE = "def pol41_probe_skill():\n    return 4242"


class _FakeOpenAI:
    """Records every conversation sent to the model. Trial calls get one
    program(code) on the first turn of each episode and done afterwards;
    the reflection call gets the lessons+skills reply."""

    def __init__(self, turn_codes, reflect_reply):
        self.calls = []
        self._codes = list(turn_codes)
        self._reflect = reflect_reply
        self._ep = -1
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))
        self.responses = types.SimpleNamespace(create=self._create)

    def _create(self, *args, **kwargs):
        self.calls.append({"args": args,
                           "kwargs": copy.deepcopy(kwargs)})
        msgs = kwargs.get("messages") or []
        sys_text = str(msgs[0].get("content", "")) if msgs else ""
        last_text = str(msgs[-1].get("content", "")) if msgs else ""
        if "same agent that just ran" in sys_text \
                or "MEASURED SCORE" in last_text:
            reply = json.dumps(self._reflect)
        elif len(msgs) <= 2:
            self._ep += 1
            code = self._codes[min(self._ep, len(self._codes) - 1)]
            reply = json.dumps({"skill": "program",
                                "args": {"code": code},
                                "expected": {}})
        else:
            reply = json.dumps({"skill": "done",
                                "args": {"summary": "probe finished"}})
        msg = types.SimpleNamespace(content=reply, text=reply)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg, text=reply)],
            output_text=reply, id="fake", model="fake", created=0,
            usage=types.SimpleNamespace(prompt_tokens=8,
                                        completion_tokens=4,
                                        total_tokens=12))


def _install_fake_openai(fake):
    """Point every reachable OpenAI() spelling at the fake. Returns the
    undo list for restore."""
    patched = []

    def patch(obj, attr, value):
        if hasattr(obj, attr):
            patched.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, value)

    import openai
    patch(openai, "OpenAI", lambda *a, **k: fake)
    for modname in ("harmless.agent", "harmless.session"):
        try:
            import importlib
            mod = importlib.import_module(modname)
        except Exception:
            continue
        patch(mod, "OpenAI", lambda *a, **k: fake)
        for attr in ("client", "_client"):
            c = getattr(mod, attr, None)
            if c is not None and hasattr(c, "chat"):
                patched.append((mod, attr, c))
                setattr(mod, attr, fake)
    return patched


def _restore(patched):
    for obj, attr, old in patched:
        setattr(obj, attr, old)


def _messages_of(call):
    kw = call["kwargs"]
    for key in ("messages", "input"):
        v = kw.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    for a in call["args"]:
        if isinstance(a, list) and a and isinstance(a[0], dict):
            return a
    return []


def _is_reflect(call):
    msgs = _messages_of(call)
    if not msgs:
        return False
    sys_text = str(msgs[0].get("content", ""))
    last = str(msgs[-1].get("content", ""))
    return "same agent that just ran" in sys_text \
        or "MEASURED SCORE" in last


def _msg_text(message):
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content
                         if isinstance(p, dict))
    return str(content)


def _episodes(fake):
    """Group the recorded trial calls into episodes: an episode begins at
    each request whose conversation is just [system, user]. Reflection
    calls are returned separately."""
    eps, cur, refs = [], None, []
    for c in fake.calls:
        if _is_reflect(c):
            refs.append(c)
            continue
        if len(_messages_of(c)) <= 2:
            cur = []
            eps.append(cur)
        if cur is not None:
            cur.append(c)
    return eps, refs


def test_pol_4_1_reflections_write_and_inject(tmp_path):
    mem = tmp_path / "memory.md"
    skills = tmp_path / "learned_skills.json"
    fake = _FakeOpenAI(
        turn_codes=["print('POL41_EP1_PROBE')",
                    f"print('POL41_SKILL_RAN', {SKILL}())"],
        reflect_reply={"lessons": LESSON,
                       "skills": [{"name": SKILL,
                                   "description": SKILL_DESC,
                                   "code": SKILL_CODE}]})
    patched = _install_fake_openai(fake)
    try:
        from harmless.agent import run_episode, reflect_on_episode
        kw = dict(max_steps=2, live=False, model="pol41-probe",
                  scene=SCENE, memory_path=str(mem),
                  skills_path=str(skills))
        score1 = run_episode(TASK, out_dir=str(tmp_path / "ep1"), **kw)
        tpath = tmp_path / "ep1" / "transcript.json"
        transcript = json.load(open(tpath)) if tpath.exists() else []
        reflect_on_episode(TASK, score1, transcript,
                           memory_path=str(mem),
                           skills_path=str(skills),
                           model="pol41-probe")

        # ---- the reflection persisted what the model returned ----------
        assert mem.exists(), (
            "after the episode's reflection no memory.md was written — "
            "the model's lessons must persist to memory.md"
        )
        assert LESSON in mem.read_text(), (
            f"memory.md does not contain the reflected lesson "
            f"({LESSON!r}):\n{mem.read_text()[:500]}"
        )
        assert skills.exists(), (
            "after the episode's reflection no learned_skills.json was "
            "written — the model's reusable defs must persist there"
        )
        saved = json.loads(skills.read_text())
        assert SKILL in saved and "def " in str(saved[SKILL]), (
            f"learned_skills.json has no reusable def named {SKILL}: "
            f"{saved}"
        )

        # ---- the next trial's prompt carries both stores ---------------
        run_episode(TASK, out_dir=str(tmp_path / "ep2"), **kw)
    finally:
        _restore(patched)

    eps, refs = _episodes(fake)
    assert len(refs) >= 1, (
        "no reflection request was made to the model — reflect_on_episode "
        "must consult the model, not write canned text"
    )
    assert len(eps) == 2, (
        f"expected 2 episodes of model requests, saw {len(eps)}"
    )
    ep1_first_sys = _msg_text(_messages_of(eps[0][0])[0])
    assert LESSON not in ep1_first_sys, (
        "the lesson marker is in episode 1's prompt before any reflection "
        "— the test is not measuring injection"
    )
    ep2_msgs = _messages_of(eps[1][0])
    ep2_sys = _msg_text(ep2_msgs[0])
    assert LESSON in ep2_sys, (
        "the next episode's prompt does not carry the reflected lesson — "
        "memory.md must be injected into the next prompt"
    )
    assert SKILL in ep2_sys and SKILL_DESC in ep2_sys, (
        "the next episode's prompt does not name the learned skill and "
        "its description — learned_skills.json must be injected into the "
        "next prompt"
    )
    assert len(eps[1]) >= 2, (
        "episode 2 ended before the receipt came back — cannot verify the "
        "learned def ran"
    )
    ep2_receipt_text = "\n".join(
        _msg_text(m) for m in _messages_of(eps[1][1]))
    assert "POL41_SKILL_RAN" in ep2_receipt_text and "4242" in \
        ep2_receipt_text, (
            f"the def saved to learned_skills.json was not callable by "
            f"name inside the next episode's program() sandbox; the "
            f"receipt shows: {ep2_receipt_text[:600]}"
        )
