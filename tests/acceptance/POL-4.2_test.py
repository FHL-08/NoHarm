"""POL-4.2@1 Each trial starts a fresh conversation; only memory.md,
learned_skills.json, and workspace/ persist.

The contract under test is what carries over between two trials of the
learning loop and what does not. learn.py runs one run_episode per trial;
this test drives that same public seam (harmless.agent.run_episode) twice
with the loop's default store names — memory.md, learned_skills.json and
workspace/ under the run directory — by chdir-ing into a tmp_path, exactly
as learn.py uses them from the repo root.

A real GPT-6-astra call cannot happen inside a test, so the OpenAI client
is replaced with a recording fake: each trial's first turn emits one
program(code) call, the second turn emits done. Before trial 1 the named
stores are seeded (a lesson line in memory.md, a seed skill in
learned_skills.json); during trial 1 the program writes a workspace key
and authors a skill — both through the sandbox's own save()/save_skill().
Then:

  fresh:  trial 2's FIRST model request must be exactly [system, user] —
          a new conversation with no assistant turns and no receipt text
          from trial 1, proven by a unique stdout marker that did appear
          in trial 1's requests being absent from every trial-2 message;
  memory: trial 2's system prompt must carry the seeded memory.md text;
  skills: trial 2's system prompt must name both the seeded skill and the
          skill authored mid-trial-1 (learned_skills.json persists);
  worksp: trial 2's system prompt must list the workspace key trial 1
          wrote, and trial 2's program() must read that value back through
          load() — visible in the next request's receipt text;
  only:   anything else — above all the trial-1 transcript itself — must
          not show up in trial 2's conversation.

The test runs the real Executor on the real kitchen scene, headless.
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

TASK = "pol-4.2 acceptance probe: stand by"
MEM_SEED = "POL42_MEMORY_SEED: the walker drifts left at vx>0.5"
SEED_SKILL = "pol42_seed_skill"
SEED_CODE = "def pol42_seed_skill():\n    return 7"
EP1_SKILL = "pol42_ep1_skill"
EP1_SKILL_CODE = "def pol42_ep1_skill():\\n    return 5"
WS_KEY = "pol42_ws_key"
WS_VALUE = "POL42_WS_VALUE"
EP1_MARKER = "POL42_EP1_MARKER"

EP1_CODE = (
    f"save('{WS_KEY}', {{'probe': '{WS_VALUE}'}})\n"
    f"save_skill('{EP1_SKILL}', \"{EP1_SKILL_CODE}\", 'written in ep1')\n"
    f"print('{EP1_MARKER}')\n"
)
EP2_CODE = f"print('POL42_WS_READBACK', load('{WS_KEY}'))"


class _FakeOpenAI:
    """Records every conversation sent to the model. Each trial's first
    turn gets one program(code) call; later turns get done."""

    def __init__(self, turn_codes):
        self.calls = []
        self._codes = list(turn_codes)
        self._ep = -1
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))
        self.responses = types.SimpleNamespace(create=self._create)

    def _create(self, *args, **kwargs):
        self.calls.append({"args": args,
                           "kwargs": copy.deepcopy(kwargs)})
        msgs = kwargs.get("messages") or []
        if len(msgs) <= 2:
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


def _msg_text(message):
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content
                         if isinstance(p, dict))
    return str(content)


def _episodes(fake):
    """Group the recorded calls into episodes: an episode begins at each
    request whose conversation is just [system, user]."""
    eps, cur = [], None
    for c in fake.calls:
        if len(_messages_of(c)) <= 2:
            cur = []
            eps.append(cur)
        if cur is not None:
            cur.append(c)
    return eps


def test_pol_4_2_fresh_conversation_named_stores_persist(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        Path("memory.md").write_text(f"- {MEM_SEED}\n")
        Path("learned_skills.json").write_text(json.dumps(
            {SEED_SKILL: {"description": "seeded skill",
                          "code": SEED_CODE}}))
        fake = _FakeOpenAI([EP1_CODE, EP2_CODE])
        patched = _install_fake_openai(fake)
        try:
            from harmless.agent import run_episode
            run_episode(TASK, max_steps=2, out_dir=str(tmp_path / "ep1"),
                        live=False, model="pol42-probe", scene=SCENE)
            run_episode(TASK, max_steps=2, out_dir=str(tmp_path / "ep2"),
                        live=False, model="pol42-probe", scene=SCENE)
        finally:
            _restore(patched)
    finally:
        os.chdir(cwd)

    eps = _episodes(fake)
    assert len(eps) == 2, (
        f"expected 2 trials of model requests, saw {len(eps)}"
    )
    assert all(len(e) == 2 for e in eps), (
        f"each trial should take exactly 2 model turns "
        f"(program, done): {[len(e) for e in eps]}"
    )

    # ---- trial 1 really did contain the marker (so its absence in trial
    # 2 proves a fresh conversation, not a weak probe) --------------------
    ep1_all_text = "\n".join(
        _msg_text(m) for c in eps[0] for m in _messages_of(c))
    assert EP1_MARKER in ep1_all_text, (
        "trial 1's program output never reached its own conversation — "
        "the freshness probe is not measurable"
    )

    # ---- trial 2 starts a FRESH conversation ----------------------------
    ep2_first = _messages_of(eps[1][0])
    assert [m.get("role") for m in ep2_first] == ["system", "user"], (
        f"trial 2's first request is not a fresh [system, user] "
        f"conversation: {[m.get('role') for m in ep2_first]}"
    )
    ep2_all_text = "\n".join(
        _msg_text(m) for c in eps[1] for m in _messages_of(c))
    assert EP1_MARKER not in ep2_all_text, (
        "trial 1's conversation content leaked into trial 2 — each trial "
        "must start a fresh conversation; only memory.md, "
        "learned_skills.json and workspace/ persist"
    )

    # ---- only the named stores carry forward ----------------------------
    ep2_sys = _msg_text(ep2_first[0])
    assert MEM_SEED in ep2_sys, (
        "trial 2's prompt lacks the memory.md lesson — the experience "
        "store did not persist across trials"
    )
    assert SEED_SKILL in ep2_sys, (
        "trial 2's prompt lacks the seeded learned_skills.json skill"
    )
    assert EP1_SKILL in ep2_sys, (
        "the skill authored mid-trial-1 via save_skill() is missing from "
        "trial 2's prompt — learned_skills.json must persist across "
        "trials"
    )
    assert WS_KEY in ep2_sys, (
        "trial 2's prompt does not list the workspace key written during "
        "trial 1 — workspace/ must persist across trials"
    )
    ep2_receipt_text = "\n".join(
        _msg_text(m) for m in _messages_of(eps[1][1]))
    assert WS_VALUE in ep2_receipt_text, (
        f"trial 2's program() could not load() the workspace entry "
        f"written in trial 1; the receipt shows: "
        f"{ep2_receipt_text[:600]}"
    )
    assert (tmp_path / "workspace" / f"{WS_KEY}.json").exists(), (
        f"the workspace entry trial 1 wrote is not on disk at "
        f"workspace/{WS_KEY}.json"
    )
