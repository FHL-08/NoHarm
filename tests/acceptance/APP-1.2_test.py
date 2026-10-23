"""APP-1.2@1 Jobs queue and run one at a time as subprocesses of
run_episode.py or learn.py.

Contract under test: every way the dashboard offers to schedule work lands
in the job queue; the worker executes the queue strictly serially — one
subprocess at a time — and each job that runs does so as a subprocess of
run_episode.py or learn.py, the measured-episode entry points.

The test starts the real server (jobs run --steps 0 --headless and each is
killed through POST /api/jobs/<id>/kill as soon as its command is captured,
so nothing reaches a simulator session or the model) and checks:

  queue:    while the first queued job runs, a second stays "queued"
  serial:   across every executed job, started/finished timestamps never
            overlap — job N+1 may only start after job N finished
  entrypt:  for each enqueue route — POST /api/jobs modes episode/learn/
            session, POST /api/demo (both variants) — the command the
            worker records for the running job must invoke run_episode.py
            or learn.py (script argument or -m module form). Routes the
            dashboard no longer offers are skipped, not failed.

Expected failure today: session jobs spawn live.py and demo jobs spawn
bowl_policy.py / demo_replay.py — queued dashboard jobs running as
subprocesses of neither run_episode.py nor learn.py.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
ALLOWED_PROGRAMS = {"run_episode", "learn"}
SERVER_DEADLINE_S = 120
RUNNING_DEADLINE_S = 90


def _get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except OSError:
        return None, b""


def _post(url, body, timeout=15):
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


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _job(st, jid):
    return next((j for j in st.get("jobs", []) if j.get("id") == jid), None)


def _program(cmd):
    """The entry point a job's command line invokes: the script argument,
    or the module after -m."""
    if not cmd:
        return None
    argv = [os.path.basename(str(a)) for a in cmd]
    if "-m" in argv:
        i = argv.index("-m")
        if i + 1 < len(argv):
            return argv[i + 1].removesuffix(".py")
    for a in argv[1:]:
        if a.endswith(".py"):
            return a[:-3]
    return None


def test_app_1_2_jobs_queue_serial_and_run_as_episode_subprocesses():
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    log = open(RUNS / "app12_server.log", "wb")
    env = {**os.environ, "OPENAI_API_KEY": "app12-dummy-no-model-calls",
           "PYTHONUNBUFFERED": "1", "MUJOCO_GL": "egl"}
    proc = subprocess.Popen(
        [sys.executable, "dashboard.py", "--port", str(port)], cwd=REPO,
        env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    problems = []
    ran = []
    try:
        end = time.time() + SERVER_DEADLINE_S
        while time.time() < end:
            status, _ = _get(base + "/api/state")
            if status == 200:
                break
            time.sleep(1)
        else:
            raise AssertionError(
                f"dashboard.py did not come up on 127.0.0.1:{port} within "
                f"{SERVER_DEADLINE_S}s")

        def state():
            s, b = _get(base + "/api/state")
            return json.loads(b or b"{}") if s == 200 else {}

        def wait_running(jid):
            e = time.time() + RUNNING_DEADLINE_S
            while time.time() < e:
                j = _job(state(), jid)
                if j and j.get("status") == "running" and j.get("cmd"):
                    return j
                if j and j.get("status") not in ("queued", "running"):
                    return j
                time.sleep(0.5)
            return _job(state(), jid)

        def kill(jid):
            e = time.time() + RUNNING_DEADLINE_S
            while time.time() < e:
                s, b = _post(f"{base}/api/jobs/{jid}/kill", {})
                if s == 200 or (_job(state(), jid) or {}).get("status") \
                        not in ("queued", "running"):
                    return
                time.sleep(0.5)

        # -- queue + serial: two episode jobs; the second must wait -------
        made = []
        for _ in range(2):
            s, b = _post(base + "/api/jobs",
                         {"task": "free", "steps": 0, "episodes": 1,
                          "headless": True})
            assert s == 200 and b.get("jobs"), (
                f"POST /api/jobs rejected an episode job: {s} {b}")
            made.append(b["jobs"][0]["id"])
        ja, jb = made

        first = wait_running(ja)
        assert first and first.get("status") == "running", (
            f"queued episode job {ja} never ran: {first}")
        prog = _program(first.get("cmd"))
        assert prog in ALLOWED_PROGRAMS, (
            f"episode job ran {prog or first.get('cmd')!r}, not "
            "run_episode.py or learn.py")
        other = _job(state(), jb)
        assert other and other.get("status") == "queued", (
            f"while job {ja} ran, queued job {jb} showed status "
            f"{other and other.get('status')!r} — jobs are not running one "
            "at a time")
        kill(ja)

        second = wait_running(jb)
        assert second and second.get("status") == "running", (
            f"job {jb} was not picked up after {ja} finished: {second}")
        kill(jb)
        ran += [first, second]

        # -- every enqueue route: whatever runs is a run_episode/learn ----
        legs = [
            ("learn", "jobs", {"task": "free", "mode": "learn",
                               "episodes": 1, "steps": 0, "headless": True}),
            ("session", "jobs", {"task": "free", "mode": "session",
                                 "episodes": 1, "steps": 1}),
            ("demo-policy", "demo", {"run": "policy"}),
            ("demo-replay", "demo", {"run": "replay"}),
        ]
        for label, route, payload in legs:
            s, b = _post(f"{base}/api/{route}", payload)
            if s != 200 or not b.get("jobs"):
                continue   # route gone or rejected: not a queueable job
            job = wait_running(b["jobs"][0]["id"])
            if job is None:
                problems.append(f"{label} job never reached a state")
                continue
            if job.get("status") == "running":
                prog = _program(job.get("cmd"))
                if prog not in ALLOWED_PROGRAMS:
                    problems.append(
                        f"{label} job {job['id']} runs as a subprocess of "
                        f"{prog or job.get('cmd')!r}, not run_episode.py or "
                        "learn.py")
                ran.append(job)
            kill(job["id"])

        # -- serial execution, asserted on recorded timestamps -----------
        st = state()
        final = [j for j in (_job(st, r["id"]) for r in ran)
                 if j and j.get("started") and j.get("finished")]
        for i in range(len(final) - 1):
            if not (final[i]["finished"] <= final[i + 1]["started"]):
                problems.append(
                    f"jobs {final[i]['id']} and {final[i + 1]['id']} "
                    f"overlapped: finished={final[i]['finished']} "
                    f"started={final[i + 1]['started']} — not one at a time")
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        log.close()
        (REPO / "MUJOCO_LOG.TXT").unlink(missing_ok=True)

    assert not problems, (
        "queue/subprocess violations:\n" + "\n".join(problems))
