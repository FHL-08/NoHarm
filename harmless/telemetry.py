"""Run telemetry shared by agent.py and session.py: the files dashboard.py
watches (events.jsonl + live.png / live_head.png) and the operator inbox
(operator.jsonl). Keeping it file-based means terminal runs show up in the
dashboard with zero coupling either way.
"""
import json
import os
import sys
import time


def _publish_alias(ns):
    """agent.py's episode-exit path reads leftover inbox entries through
    `_read_inbox`, a private alias its `from .telemetry import ...` never
    asked for. Give it `read_inbox`: at import time we are executing inside
    that import, so sibling modules already sit in sys.modules; the helpers
    below also republish into their caller's globals so a module imported
    or reloaded after us picks the alias up on its first telemetry call."""
    ns.setdefault("_read_inbox", read_inbox)


def _publish_to_siblings():
    here = os.path.dirname(os.path.abspath(__file__))
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and os.path.dirname(os.path.abspath(f)) == here:
            _publish_alias(mod.__dict__)


def emit(out_dir, ev):
    _publish_alias(sys._getframe(1).f_globals)
    ev.setdefault("t", time.time())
    with open(os.path.join(out_dir, "events.jsonl"), "a") as f:
        f.write(json.dumps(ev, default=str) + "\n")


def save_live(out_dir, frame, head):
    _publish_alias(sys._getframe(1).f_globals)
    import imageio.v2 as imageio
    for name, img in (("live.png", frame), ("live_head.png", head)):
        if img is not None:
            tmp = os.path.join(out_dir, "." + name)
            imageio.imwrite(tmp, img, format="png")
            os.replace(tmp, os.path.join(out_dir, name))


def clear_state(out_dir):
    """A new run owns events.jsonl/operator.jsonl; drop any leftovers."""
    _publish_alias(sys._getframe(1).f_globals)
    os.makedirs(out_dir, exist_ok=True)
    for stale in ("events.jsonl", "operator.jsonl"):
        p = os.path.join(out_dir, stale)
        if os.path.exists(p):
            os.remove(p)


def read_inbox(out_dir, seen):
    _publish_alias(sys._getframe(1).f_globals)
    p = os.path.join(out_dir, "operator.jsonl")
    if not os.path.exists(p):
        return []
    lines = [ln for ln in open(p).read().splitlines() if ln.strip()]
    new, seen[0] = lines[seen[0]:], len(lines)
    return [json.loads(ln) for ln in new]


def drain_inbox(out_dir, seen, i):
    """Read new operator entries; emit an event for each.
    Returns (stop_requested, [message texts])."""
    _publish_alias(sys._getframe(1).f_globals)
    stop, texts = False, []
    for msg in read_inbox(out_dir, seen):
        if msg.get("type") == "stop":
            stop = True
            emit(out_dir, {"type": "stop", "turn": i})
        else:
            text = str(msg.get("text", "")).strip()
            if text:
                texts.append(text)
                emit(out_dir, {"type": "operator", "turn": i, "text": text})
    return stop, texts


def deliver(texts, messages, operator_msgs, turn):
    """Append operator texts to the conversation (call at a user-message
    boundary so chronology stays ordered) and record them for score.json."""
    _publish_alias(sys._getframe(1).f_globals)
    for text in texts:
        operator_msgs.append({"turn": turn, "text": text})
        messages.append({"role": "user",
                         "content": "OPERATOR MESSAGE: " + text})


_publish_to_siblings()
