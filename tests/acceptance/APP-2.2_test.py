"""APP-2.2@1 A message written to a run's operator.jsonl inbox reaches
the model before its next turn, and a stop entry ends the episode.

Contract under test: while an episode runs, the operator.jsonl file in
the run's output directory is the human's inbox to the policy. Writing
entries to it changes the run two ways — a {"type": "message", "text"}
entry is delivered to the model before it is asked for its next turn,
and a {"type": "stop"} entry ends the episode gracefully (the run
finishes, still writing its measured score and closing the turn feed).

The test drives the real public entry point: a run_episode.py
subprocess (--task free --headless, no viewer, no session lock) whose
model calls are served by a stub chat-completions endpoint
(OPENAI_BASE_URL), so no model account is needed and every request the
agent loop makes is recorded verbatim. The test plays the human by
appending lines to the run's operator.jsonl between model calls:

  message:  written while turn 0's think is in flight — the messages
            payload of the NEXT request (turn 1) must carry the text in
            an operator-message user entry: the inbox reached the model
            before its next turn
  stop:     written while turn 1's think is in flight — the episode
            must then end: the process exits 0, events.jsonl carries
            the "stop" event followed by the "end" event, and
            score.json is written recording the operator stop

Expected failure today: the inbox half works — the drain marks the
message, the operator event is emitted, and the text is delivered into
the conversation — but no episode can end gracefully. The loop's exit
path calls _read_inbox, a name agent.py never imports from
harmless/telemetry.py (a touched module), so the process dies with
NameError before score.json, transcript.json and the "end" event are
written. A stop entry ends the turn loop and then kills the run: no
end event, no score, a crashed subprocess.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
SERVER_DEADLINE_S = 360   # executor + scene + first model call
TURN_DEADLINE_S = 180
END_DEADLINE_S = 180
MSG_TEXT = "operator probe: hold position and report what you see"


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Stub:
    """A minimal chat-completions endpoint. Request bodies are recorded
    verbatim; responses are held on per-request events so the test can
    write to the run's operator inbox while a think is in flight."""

    def __init__(self):
        self.requests = []
        self.arrived = [threading.Event(), threading.Event()]
        self.release = [threading.Event(), threading.Event()]
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), self._handler())
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _handler(self):
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                n = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(n)
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    body = {"_unparsed": raw.decode("utf-8", "replace")}
                idx = len(outer.requests)
                outer.requests.append(body)
                if idx < 2:
                    outer.arrived[idx].set()
                    outer.release[idx].wait(END_DEADLINE_S)
                if idx == 0:
                    call = {"skill": "program",
                            "args": {"code": "print('turn0 probe')"},
                            "expected": {}}
                else:
                    call = {"skill": "done",
                            "args": {"summary": "stub ends the episode"}}
                out = {"id": "chatcmpl-stub", "object": "chat.completion",
                       "choices": [{"index": 0, "finish_reason": "stop",
                                    "message": {"role": "assistant",
                                                "content": json.dumps(call)}}],
                       "usage": {"prompt_tokens": 50, "completion_tokens": 10}}
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        return H

    @property
    def port(self):
        return self.server.server_address[1]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _operator_write(out_dir, entry):
    path = Path(out_dir) / "operator.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _events(out_dir):
    p = Path(out_dir) / "events.jsonl"
    out = []
    if p.exists():
        for ln in p.read_text().splitlines():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def _message_texts(request_body):
    """Every text fragment in a chat-completions request's messages."""
    out = []
    for m in (request_body or {}).get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            out += [p.get("text", "") for p in c if isinstance(p, dict)]
    return out


def test_app_2_2_operator_inbox_reaches_model_and_stop_ends_episode():
    stub = _Stub()
    out_rel = f"runs/app22_{int(time.time())}"
    out_dir = REPO / out_rel
    shutil.rmtree(out_dir, ignore_errors=True)
    log = open(RUNS / "app22_episode.log", "wb")
    env = {**os.environ,
           "OPENAI_API_KEY": "app22-dummy",
           "OPENAI_BASE_URL": f"http://127.0.0.1:{stub.port}/v1",
           "PYTHONUNBUFFERED": "1",
           "NOHARM_HEADLESS": "1",
           "MUJOCO_GL": "egl"}
    proc = subprocess.Popen(
        [sys.executable, "run_episode.py", "--task", "free",
         "--steps", "5", "--headless", "--out", out_rel],
        cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    try:
        assert stub.arrived[0].wait(SERVER_DEADLINE_S), (
            f"the episode never made its first model call within "
            f"{SERVER_DEADLINE_S}s — see runs/app22_episode.log")

        # the human writes a message to the run's operator inbox while
        # turn 0's think is still in flight
        _operator_write(out_dir, {"type": "message", "text": MSG_TEXT})
        stub.release[0].set()

        assert stub.arrived[1].wait(TURN_DEADLINE_S), (
            "the episode never made a second model call after turn 0")

        texts = _message_texts(stub.requests[1])
        assert any("OPERATOR MESSAGE" in t and MSG_TEXT in t
                   for t in texts), (
            "the operator.jsonl message never reached the model: the "
            "request payload for turn 1 contains no operator-message "
            f"entry carrying {MSG_TEXT!r}")

        # a stop entry while turn 1 thinks must end the episode
        _operator_write(out_dir, {"type": "stop"})
        # the inbox drain marks the entry on the turn feed when it is
        # consumed — wait for that before releasing the held reply, so
        # the stop lands mid-think instead of racing the response
        deadline = time.time() + TURN_DEADLINE_S
        seen_stop = False
        while time.time() < deadline:
            if any(e.get("type") == "stop" for e in _events(out_dir)):
                seen_stop = True
                break
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        stub.release[1].set()
        assert seen_stop, (
            "the operator.jsonl stop entry was never consumed — no "
            "'stop' event appeared on the run's turn feed while the "
            "episode kept thinking")

        try:
            rc = proc.wait(timeout=END_DEADLINE_S)
        except subprocess.TimeoutExpired:
            rc = None
        ev = _events(out_dir)
        assert rc is not None, (
            "the episode did not end within "
            f"{END_DEADLINE_S}s of the stop entry")
        assert any(e.get("type") == "end" for e in ev), (
            "the stop entry did not end the episode: events.jsonl has no "
            f"closing 'end' event (events: {[e.get('type') for e in ev]}; "
            f"exit code {rc})")
        score = None
        sp = out_dir / "score.json"
        if sp.exists():
            score = json.loads(sp.read_text())
        assert score is not None, (
            "the stop entry did not end the episode gracefully: the run "
            f"crashed or exited without writing score.json (exit code "
            f"{rc} — see runs/app22_episode.log)")
        assert score.get("stopped_by_operator") or "operator" in str(
            score.get("done_summary", "")).lower(), (
            "score.json does not record the operator stop: "
            f"stopped_by_operator={score.get('stopped_by_operator')!r}, "
            f"done_summary={score.get('done_summary')!r}")
        assert rc == 0, (
            f"the episode subprocess exited {rc}, expected a graceful "
            "end (see runs/app22_episode.log)")
    finally:
        stub.release[0].set()
        stub.release[1].set()
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        log.close()
        stub.close()
        (REPO / "MUJOCO_LOG.TXT").unlink(missing_ok=True)
