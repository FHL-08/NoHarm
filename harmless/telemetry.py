"""Run telemetry shared by agent.py and session.py: the files dashboard.py
watches (events.jsonl + live.png / live_head.png) and the operator inbox
(operator.jsonl). Keeping it file-based means terminal runs show up in the
dashboard with zero coupling either way.
"""
import json
import os
import time


def emit(out_dir, ev):
    ev.setdefault("t", time.time())
    with open(os.path.join(out_dir, "events.jsonl"), "a") as f:
        f.write(json.dumps(ev, default=str) + "\n")


def save_live(out_dir, frame, head):
    import imageio.v2 as imageio
    for name, img in (("live.png", frame), ("live_head.png", head)):
        if img is not None:
            tmp = os.path.join(out_dir, "." + name)
            imageio.imwrite(tmp, img, format="png")
            os.replace(tmp, os.path.join(out_dir, name))


def clear_state(out_dir):
    """A new run owns events.jsonl/operator.jsonl; drop any leftovers."""
    os.makedirs(out_dir, exist_ok=True)
    for stale in ("events.jsonl", "operator.jsonl"):
        p = os.path.join(out_dir, stale)
        if os.path.exists(p):
            os.remove(p)


def read_inbox(out_dir, seen):
    p = os.path.join(out_dir, "operator.jsonl")
    if not os.path.exists(p):
        return []
    lines = [ln for ln in open(p).read().splitlines() if ln.strip()]
    new, seen[0] = lines[seen[0]:], len(lines)
    return [json.loads(ln) for ln in new]


def drain_inbox(out_dir, seen, i):
    """Read new operator entries; emit an event for each.
    Returns (stop_requested, [message texts])."""
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
    for text in texts:
        operator_msgs.append({"turn": turn, "text": text})
        messages.append({"role": "user",
                         "content": "OPERATOR MESSAGE: " + text})
