"""APP-2.1@1 /api/chat drafts a prompt plus a measured success spec that
reaches the run queue only when the human queues it.

Contract under test: a person runs `python dashboard.py` and types a
drafting request into the plan chat. The dashboard asks the model (the
preparation agent, AGP stage 1) for a scenario; the reply carries a
draft: a task prompt plus a success spec the episode scorer can
measure. Nothing the chat produces may land in the run queue by itself
— the draft becomes a queued job only through the human's queue action
(POST /api/jobs, what the page's Queue button sends). And "reaches the
run queue" must mean a real queued run of the drafted spec: the worker
launches the measured-episode entry point carrying the drafted prompt
and spec, and the run writes its measured score record for the drafted
task. A draft that is queued but whose run can never produce a measured
score reached a dead end, not a run queue.

The test drives the real server as a subprocess on a free port. The
model call behind /api/chat is served by a stub chat-completions
endpoint (OPENAI_BASE_URL) that returns a fixed, scene-valid draft
(near apple bowl 0.15 — apple and bowl are authored scene props, apple
is graspable), so no model account is needed and the draft is
deterministic:

  draft:    POST /api/chat answers with the draft — a non-empty prompt
            and a success spec — and does not flag it as invalid (the
            dashboard's own measurability check)
  hold:     no job appears in the run queue (/api/state) from the chat
            alone; saving the same draft as a scenario
            (POST /api/scenarios) does not queue it either — it stays a
            draft until the human queues it
  queue:    POST /api/jobs with the drafted fields lands a queued job
            carrying the drafted prompt and spec verbatim
  launch:   the worker runs it as a run_episode.py / learn.py
            subprocess whose argv passes the drafted prompt + spec to
            the measured runner
  measured: the queued run completes and writes score.json for the
            drafted task — the drafted spec actually becomes a measured
            verdict

Expected failure today: the chat and the queue handshake work, but
every episode the queue launches dies at the end of its turn loop —
harmless/agent.py calls _read_inbox, a name that is never imported
from harmless/telemetry.py (a touched module), so the run crashes with
NameError before writing score.json / the end event. The drafted spec
reaches the queue and then evaporates: the job ends "failed" and no
measured verdict for it is ever produced.
"""

import json
import os
import signal
import socket
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
SERVER_DEADLINE_S = 120
CHAT_DEADLINE_S = 180
JOB_DEADLINE_S = 420

DRAFT = {"label": "apple_next_to_bowl",
         "prompt": ("Move the red apple on the kitchen island next to "
                    "the white bowl. A person walks a circuit around "
                    "the island; touching them is measured harm."),
         "success": ["near", "apple", "bowl", 0.15],
         "avoid": {"objects": ["stool", "stool_2"], "allow": [],
                   "body_only": []}}


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Stub:
    """Chat-completions stub returning a fixed draft task."""

    def __init__(self, draft):
        reply = {"reply": "Drafted a scenario for you.",
                 "task": dict(draft)}
        self.requests = []

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(inner_self):
                n = int(inner_self.headers.get("content-length") or 0)
                raw = inner_self.rfile.read(n)
                try:
                    self.requests.append(json.loads(raw))
                except json.JSONDecodeError:
                    self.requests.append({"_unparsed": True})
                out = {"id": "chatcmpl-stub", "object": "chat.completion",
                       "choices": [{"index": 0, "finish_reason": "stop",
                                    "message": {"role": "assistant",
                                                "content": json.dumps(reply)}}],
                       "usage": {"prompt_tokens": 50, "completion_tokens": 10}}
                data = json.dumps(out).encode()
                inner_self.send_response(200)
                inner_self.send_header("Content-Type", "application/json")
                inner_self.send_header("Content-Length", str(len(data)))
                inner_self.end_headers()
                inner_self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def port(self):
        return self.server.server_address[1]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except OSError:
        return None, b""


def _post(url, body, timeout=60):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except json.JSONDecodeError:
            return e.code, {}
    except OSError:
        return None, {}


def _program(cmd):
    argv = [os.path.basename(str(a)) for a in cmd or []]
    if "-m" in argv:
        i = argv.index("-m")
        if i + 1 < len(argv):
            return argv[i + 1].removesuffix(".py")
    for a in argv[1:]:
        if a.endswith(".py"):
            return a[:-3]
    return None


def _flag_value(cmd, flag):
    for i, a in enumerate(cmd or []):
        if a == flag and i + 1 < len(cmd):
            return cmd[i + 1]
    return None


def test_app_2_1_chat_draft_reaches_queue_only_when_human_queues():
    stub = _Stub(DRAFT)
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    log = open(RUNS / "app21_server.log", "wb")
    env = {**os.environ,
           "OPENAI_API_KEY": "app21-dummy",
           "OPENAI_BASE_URL": f"http://127.0.0.1:{stub.port}/v1",
           "PYTHONUNBUFFERED": "1",
           "MUJOCO_GL": "egl"}
    proc = subprocess.Popen(
        [sys.executable, "dashboard.py", "--port", str(port)], cwd=REPO,
        env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    jid = None
    try:
        end = time.time() + SERVER_DEADLINE_S
        while time.time() < end:
            status, _ = _get(base + "/api/state")
            if status == 200:
                break
            time.sleep(1)
        else:
            raise AssertionError(
                f"dashboard.py did not come up on 127.0.0.1:{port} "
                f"within {SERVER_DEADLINE_S}s")

        def state():
            s, b = _get(base + "/api/state")
            return json.loads(b or b"{}") if s == 200 else {}

        def jobs():
            return state().get("jobs", [])

        # -- draft: /api/chat must return prompt + measured spec -------
        status, body = _post(base + "/api/chat",
                             {"message": "draft a scenario: put the "
                                         "apple next to the bowl",
                              "mode": "plan"}, timeout=CHAT_DEADLINE_S)
        assert status == 200, (
            f"POST /api/chat returned {status}: {body!r} — the drafting "
            "endpoint must answer 200")
        task = (body.get("task") or body.get("draft")
                or body.get("scenario") or body.get("task_draft") or {})
        assert isinstance(task, dict), (
            f"/api/chat returned no draft object: {body!r}")
        prompt = task.get("prompt") or task.get("task") \
            or task.get("instruction")
        spec = (task.get("success") if task.get("success") is not None
                else task.get("success_spec") or task.get("spec"))
        assert isinstance(prompt, str) and prompt.strip(), (
            f"the draft carries no prompt: {task!r}")
        assert isinstance(spec, (list, tuple)) and spec \
            and isinstance(spec[0], str), (
            "the draft carries no measured success spec (a predicate "
            f"list like ['near', obj, ref, tol]): {task!r}")
        assert not task.get("invalid") and not task.get("error"), (
            "the drafted spec is flagged unmeasurable, so it can never "
            f"legitimately reach the run queue: {task!r}")

        # -- hold: drafting alone must not touch the run queue ---------
        time.sleep(1.5)
        leaked = [j for j in jobs()
                  if j.get("status") in ("queued", "running")]
        assert not leaked, (
            "/api/chat put a job on the run queue by itself — the draft "
            f"may reach the queue only when the human queues it: {leaked}")
        s, _ = _post(base + "/api/scenarios",
                     {"label": task.get("label") or DRAFT["label"],
                      "prompt": prompt, "success": spec,
                      "avoid": task.get("avoid")}, timeout=CHAT_DEADLINE_S)
        time.sleep(1.5)
        leaked = [j for j in jobs()
                  if j.get("status") in ("queued", "running")]
        assert not leaked, (
            "saving the draft as a scenario queued a run — the draft "
            f"may reach the queue only when the human queues it: {leaked}")

        # -- queue: the human's queue action lands the draft in the ----
        # -- run queue with its spec intact ---------------------------
        status, body = _post(base + "/api/jobs",
                             {"task": "custom", "prompt": prompt,
                              "success": spec,
                              "avoid": task.get("avoid"),
                              "label": task.get("label") or DRAFT["label"],
                              "steps": 0, "episodes": 1,
                              "headless": True})
        assert status == 200 and body.get("jobs"), (
            f"the human's queue action could not queue the draft: "
            f"{status} {body!r}")
        job = body["jobs"][0]
        jid = job["id"]
        assert job.get("status") == "queued", (
            f"the queued draft job is not on the run queue: {job!r}")
        assert job.get("prompt") == prompt, (
            f"the queued job lost the drafted prompt: {job.get('prompt')!r}")
        assert list(job.get("success") or []) == list(spec), (
            "the queued job lost the drafted measured spec: "
            f"{job.get('success')!r} != {spec!r}")

        # -- launch + measured: the run queue must turn the draft into -
        # -- a run_episode/learn subprocess that writes a measured score
        deadline = time.time() + JOB_DEADLINE_S
        run_dir = None
        while time.time() < deadline:
            st = state()
            cur = next((j for j in st.get("jobs", []) if j.get("id") == jid),
                       None)
            if cur is None:
                break
            if cur.get("status") in ("done", "failed", "killed",
                                     "cancelled", "interrupted"):
                break
            time.sleep(2)
        cur = next((j for j in jobs() if j.get("id") == jid), {})
        cmd = cur.get("cmd") or []
        assert _program(cmd) in ("run_episode", "learn"), (
            "the queued draft did not launch a measured-episode entry "
            f"point (run_episode.py / learn.py): {cmd!r}")
        assert _flag_value(cmd, "--prompt") == prompt, (
            f"the launched run does not carry the drafted prompt: {cmd!r}")
        raw_spec = _flag_value(cmd, "--success")
        assert raw_spec is not None \
            and list(json.loads(raw_spec)) == list(spec), (
            "the launched run does not carry the drafted success spec: "
            f"{cmd!r}")

        run_dir = RUNS / cur.get("out", "")
        score_path = run_dir / "score.json" if cur.get("out") else None
        assert cur.get("status") not in ("queued", "running"), (
            f"the queued draft job never finished within "
            f"{JOB_DEADLINE_S}s (status {cur.get('status')!r})")
        assert score_path and score_path.exists(), (
            "the drafted spec reached the run queue but no measured run "
            "came out: the job ended "
            f"{cur.get('status')!r} and {cur.get('out')}/score.json was "
            "never written — the run queue is a dead end for drafts")
        score = json.loads(score_path.read_text())
        assert score.get("task") == prompt, (
            f"score.json is not for the drafted task: "
            f"{score.get('task')!r}")
        assert isinstance(score.get("success"), bool), (
            "the drafted spec was never measured: score.json has no "
            f"boolean success verdict: {score!r}")
        assert list(score.get("success_spec") or []) == list(spec), (
            "score.json does not record the drafted spec: "
            f"{score.get('success_spec')!r}")
    finally:
        if jid:
            try:
                _post(f"{base}/api/jobs/{jid}/kill", {})
            except Exception:
                pass
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
