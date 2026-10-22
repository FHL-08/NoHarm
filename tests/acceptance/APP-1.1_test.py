"""APP-1.1@1 The dashboard at 127.0.0.1:8050 shows the live turn feed and
last-seen frames for the active run.

Contract under test: a person runs `python dashboard.py` and opens
http://127.0.0.1:8050. Whenever a run is active — launched from the
dashboard's own queue or from a terminal — the page has (a) the live turn
feed: the run's turn events as they land in events.jsonl, and (b) the
last-seen frames: the scene and head-camera images the model last saw
(live.png / live_head.png), refetched as they update.

The test drives the real server as a subprocess exactly as a user starts
it — no --port flag, since the line names the address: the default
invocation must bind 127.0.0.1:8050. It then exercises the public HTTP
surface three ways:

  display:  a fabricated active run (events.jsonl whose start event names a
            live pid, one turn event, live.png + live_head.png) must
            surface in /api/state as live.run with the turn visible in
            live.turns and live.frames_t set, and both frames must be
            served at /runs/<run>/live.png and /runs/<run>/live_head.png
  episode:  an episode queued through POST /api/jobs must become live-
            observable while it executes — /api/state.live resolves to its
            run and the run dir carries events.jsonl + both live frames
  coverage: the same must hold for every kind of run the dashboard
            schedules — the Demo button's run included

OPENAI_API_KEY is set to a dummy value so the queued episode subprocess
gets past client construction and emits telemetry; the job runs --steps 0
so no model call is ever made.

Expected failure today: the dashboard also schedules demo jobs
(POST /api/demo -> bowl_policy.py / demo_replay.py). Those subprocesses
never write events.jsonl or live frames, so while a demo run executes the
dashboard reports a running job with no live run, no turn feed and no
last-seen frames — a scheduled run that cannot be watched from the page.
"""

import json
import os
import signal
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
HOST = "http://127.0.0.1:8050"
SERVER_DEADLINE_S = 120
LIVE_DEADLINE_S = 480
JOB_DEADLINE_S = 240


def _png(path):
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))
    raw = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    payload = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", raw)
               + chunk(b"IDAT", zlib.compress(b"\x00\x11\x22\x33"))
               + chunk(b"IEND", b""))
    Path(path).write_bytes(payload)


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


def _start_dashboard():
    log = open(RUNS / "app11_server.log", "wb")
    env = {**os.environ, "OPENAI_API_KEY": "app11-dummy-no-model-calls",
           "PYTHONUNBUFFERED": "1", "MUJOCO_GL": "egl"}
    proc = subprocess.Popen(
        [sys.executable, "dashboard.py"], cwd=REPO, env=env,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return proc, log


def _wait_server(deadline_s=SERVER_DEADLINE_S):
    end = time.time() + deadline_s
    while time.time() < end:
        status, _ = _get(HOST + "/api/state")
        if status == 200:
            return True
        time.sleep(1)
    return False


def _state():
    status, body = _get(HOST + "/api/state")
    if status != 200:
        return {}
    return json.loads(body or b"{}")


def _job(st, jid):
    return next((j for j in st.get("jobs", []) if j.get("id") == jid), None)


def _under(run, out):
    return run == out or run.startswith(out + "_") or run.startswith(out + "/")


def _feed_and_frames_served(st, job):
    """True while the dashboard's live view resolves to this job's run with
    a turn feed and fetchable last-seen frames."""
    live = st.get("live") or {}
    run = live.get("run")
    if not run or not _under(run, job["out"]):
        return False
    if not isinstance(live.get("turns"), list) or live.get("frames_t") is None:
        return False
    for fname in ("live.png", "live_head.png"):
        status, body = _get(f"{HOST}/runs/{urllib.parse.quote(run)}/{fname}")
        if status != 200 or not body.startswith(b"\x89PNG"):
            return False
    return True


def _artifacts(out):
    """Feed + frame files the dashboard needs, anywhere under runs/<out>."""
    base = RUNS / out
    found = {"events.jsonl": False, "live.png": False, "live_head.png": False}
    if base.is_dir():
        for p in base.rglob("*"):
            if p.name in found:
                found[p.name] = True
    return found


def _kill_job(jid):
    for _ in range(60):
        status, body = _post(f"{HOST}/api/jobs/{jid}/kill", {})
        if status == 200 and body.get("ok"):
            return
        st = _state()
        j = _job(st, jid)
        if j and j.get("status") not in ("queued", "running"):
            return
        time.sleep(0.5)


def _watch_job(jid, deadline_s):
    """Whether the dashboard ever showed this job's run live: turn feed +
    last-seen frames while the job runs, or the artifacts they come from."""
    end = time.time() + deadline_s
    while time.time() < end:
        st = _state()
        job = _job(st, jid)
        if job is None:
            return False
        if job.get("status") == "running" and _feed_and_frames_served(st, job):
            return True
        if job.get("status") not in ("queued", "running"):
            break
        time.sleep(1)
    return all(_artifacts(_job(_state(), jid)["out"]).values())


def test_app_1_1_dashboard_shows_live_feed_and_frames_for_active_run():
    proc, log = _start_dashboard()
    problems = []
    try:
        assert _wait_server(), (
            "dashboard.py did not come up at http://127.0.0.1:8050 within "
            f"{SERVER_DEADLINE_S}s — the line names this address, so the "
            "default invocation must serve there")

        status, body = _get(HOST + "/")
        assert status == 200 and body, (
            f"GET http://127.0.0.1:8050/ returned {status}, expected the "
            "control-room page")

        probe = "app11_probe"
        pdir = RUNS / probe
        pdir.mkdir(parents=True, exist_ok=True)
        with open(pdir / "events.jsonl", "w") as f:
            f.write(json.dumps({"type": "start", "task": "probe",
                                "max_steps": 5, "t": time.time(),
                                "pid": os.getpid()}) + "\n")
            f.write(json.dumps({"type": "turn", "turn": 0,
                                "call": {"skill": "look"},
                                "measured": {"person_peak_force": 0.0}}) + "\n")
        _png(pdir / "live.png")
        _png(pdir / "live_head.png")

        shown = False
        end = time.time() + 60
        while time.time() < end:
            st = _state()
            live = st.get("live") or {}
            if live.get("run") == probe:
                turns = live.get("turns")
                assert isinstance(turns, list) and any(
                    e.get("type") == "turn" for e in turns), (
                    f"/api/state exposes no live turn feed for the active "
                    f"run {probe}: turns={turns!r}")
                assert live.get("frames_t") is not None, (
                    f"/api/state names no last-seen frames for the active "
                    f"run {probe} (frames_t missing)")
                for fname in ("live.png", "live_head.png"):
                    s, b = _get(f"{HOST}/runs/{probe}/{fname}")
                    assert s == 200 and b.startswith(b"\x89PNG"), (
                        f"last-seen frame /runs/{probe}/{fname} not served "
                        f"(status {s})")
                shown = True
                break
            time.sleep(0.5)
        assert shown, (
            f"the fabricated active run {probe} never became the live run "
            "in /api/state — the dashboard does not show an active run's "
            "feed and frames")

        # the probe is over; only dashboard-scheduled runs stay observable
        with open(pdir / "events.jsonl", "a") as f:
            f.write(json.dumps({"type": "end", "t": time.time()}) + "\n")

        legs = [
            ("episode", "jobs", {"task": "free", "steps": 0, "episodes": 1,
                                 "headless": True}),
            ("demo", "demo", {"run": "policy"}),
        ]
        for label, route, payload in legs:
            status, body = _post(f"{HOST}/api/{route}", payload)
            if status != 200 or not body.get("jobs"):
                continue   # a route the dashboard no longer offers is out
            job = body["jobs"][0]
            deadline = LIVE_DEADLINE_S if label == "episode" else JOB_DEADLINE_S
            if not _watch_job(job["id"], deadline):
                problems.append(
                    f"{label} job {job['id']} (out={job['out']}) ran but the "
                    "dashboard never showed a live turn feed or last-seen "
                    f"frames for it: artifacts={_artifacts(job['out'])}")
            _kill_job(job["id"])
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
        "scheduled runs the dashboard cannot show live:\n"
        + "\n".join(problems))
