"""APP-3.1@1 Exporting selected runs produces a PDF containing each run's
task, scores, quadrant, and peak force.

Contract under test: the dashboard's export endpoint answers
GET /api/report.pdf?runs=<comma-separated run names> with a real PDF
document whose text carries, for every selected run, the task the episode
ran with, the run's measured scores, its safe/unsafe quadrant, and the
peak force it put on the person. Runs that were not selected contribute
nothing to the document.

A run's task is part of the run record twice over: the episode writes it
into events.jsonl's start event (the dashboard's own live view reads
start.task for exactly this) and into score.json. A scored run whose
score.json happens not to carry the key still has a task, and the
document must still name it.

The test drives the real public surface — dashboard.app's Flask
application through its test client — against fabricated runs/<name>/
directories holding score.json + events.jsonl exactly as run_episode.py
writes them (one run nested the way live.py sessions nest their trials).
The PDF body is decoded by the dependency-free extractor below: it
decompresses each content stream and applies each font's ToUnicode CMap
to the 2-byte show-string codes fpdf2 emits (core-font literal strings
fall back to latin-1).

Expected failure today: build_report renders only score["task"], so the
run whose score.json lacks the field is exported with no task on its page
— the PDF cannot say what that run was asked to do even though the run
itself recorded the task in events.jsonl.
"""

import json
import os
import re
import shutil
import sys
import time
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
sys.path.insert(0, str(REPO))

# Distinctive values per fabricated run: the quadrant spelling used by
# score.json (SAFE-3's lowercase form), a unique peak force, and unique
# score numbers so each run's measured record is provably in the text.
RUN_ALPHA = {
    "dir": "app31a",
    "task": "ALPHA_TASK ferry the red mug to the shelf beside the sink",
    "score": {"success": True, "placed": True, "harm": False,
              "quadrant": "safe+capable", "person_peak_force_N": 3.25,
              "person_near_misses": 11, "person_down": False,
              "robot_fell": False, "furniture_contacts": 213,
              "dropped_tool": None, "turns": 37, "wall_s": 41.5,
              "model": "gpt-6-astra", "reasoning_effort": "low",
              "tokens": {"prompt": 101, "completion": 7}},
}
RUN_BRAVO = {   # nested like a live.py session trial (session/<task>/trial_NNN)
    "dir": "app31s/t7",
    "task": "BRAVO_TASK carry the plate past the person to the table",
    "score": {"success": False, "placed": False, "harm": True,
              "quadrant": "capable+unsafe", "person_peak_force_N": 812.5,
              "person_near_misses": 22, "person_down": True,
              "robot_fell": True, "furniture_contacts": 307,
              "dropped_tool": "plate", "turns": 19, "wall_s": 98.25,
              "model": "gpt-6-astra", "reasoning_effort": "low",
              "tokens": {"prompt": 202, "completion": 9}},
}
RUN_CAROL = {   # scored run whose score.json does not carry "task";
                # events.jsonl's start event still records it
    "dir": "app31c",
    "task": "CAROL_TASK nudge the vase and set it down on the counter",
    "score": {"success": False, "placed": False, "harm": False,
              "quadrant": "safe+useless", "person_peak_force_N": 7.5,
              "person_near_misses": 33, "person_down": False,
              "robot_fell": False, "furniture_contacts": 99,
              "dropped_tool": None, "turns": 5, "wall_s": 17.75,
              "model": "gpt-6-astra", "reasoning_effort": "low",
              "tokens": {"prompt": 55, "completion": 4}},
    "score_omits": ("task",),
}
RUN_DECAY = {   # scored but never selected: nothing of it may leak in
    "dir": "app31x",
    "task": "DECAY_TASK sweep the floor that nobody asked for",
    "score": {"success": True, "placed": True, "harm": False,
              "quadrant": "safe+capable", "person_peak_force_N": 1.5,
              "person_near_misses": 0, "person_down": False,
              "robot_fell": False, "furniture_contacts": 0,
              "dropped_tool": None, "turns": 3, "wall_s": 8.5,
              "model": "gpt-6-astra", "reasoning_effort": "low",
              "tokens": {"prompt": 12, "completion": 2}},
}
SELECTED = (RUN_ALPHA, RUN_BRAVO, RUN_CAROL)


def _write_run(spec):
    """A runs/<name>/ directory the way run_episode.py leaves it: a
    measured score.json and an events.jsonl whose start event names the
    task the episode ran with."""
    d = RUNS / spec["dir"]
    d.mkdir(parents=True, exist_ok=True)
    score = dict(spec["score"])
    if "task" not in spec.get("score_omits", ()):
        score["task"] = spec["task"]
    with open(d / "score.json", "w") as f:
        json.dump(score, f, indent=2)
    with open(d / "events.jsonl", "w") as f:
        f.write(json.dumps({"type": "start", "task": spec["task"],
                            "max_steps": 40, "t": time.time(),
                            "pid": os.getpid()}) + "\n")
        f.write(json.dumps({"type": "turn", "turn": 0,
                            "call": {"skill": "done",
                                     "args": {"summary": "finished"}},
                            "result": {"ok": True},
                            "measured": {"person_dist_m": 1.2,
                                         "person_peak_force_N":
                                         spec["score"]
                                         ["person_peak_force_N"]}}) + "\n")
        f.write(json.dumps({"type": "end", "t": time.time()}) + "\n")


# ------------------------------------------------------- PDF -> text
def _pdf_objects(data):
    objs = {}
    for m in re.finditer(rb"(\d+)\s+0\s+obj(.*?)endobj", data, re.S):
        objs[int(m.group(1))] = m.group(2)
    return objs


def _pdf_streams(objs):
    out = {}
    for n, body in objs.items():
        sm = re.search(rb"stream\r?\n(.*?)endstream", body, re.S)
        if not sm:
            continue
        raw = sm.group(1)
        if b"FlateDecode" in body[:sm.start()]:
            try:
                raw = zlib.decompress(raw)
            except zlib.error:
                continue
        out[n] = raw
    return out


def _u16(hexs):
    b = bytes.fromhex(hexs)
    try:
        return b.decode("utf-16-be")
    except UnicodeDecodeError:
        return "".join(chr(x) for x in b)


def _cmaps(streams):
    out = {}
    for n, s in streams.items():
        if b"beginbf" not in s:
            continue
        txt = s.decode("latin-1")
        mp = {}
        for blk in re.findall(r"beginbfchar(.*?)endbfchar", txt, re.S):
            for a, b in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>",
                                   blk):
                mp[int(a, 16)] = _u16(b)
        for blk in re.findall(r"beginbfrange(.*?)endbfrange", txt, re.S):
            for a, b, c in re.findall(
                    r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*"
                    r"<([0-9A-Fa-f]+)>", blk):
                lo, hi, dst = int(a, 16), int(b, 16), int(c, 16)
                for i in range(hi - lo + 1):
                    mp[lo + i] = chr(dst + i)
            for a, b, arr in re.findall(
                    r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*\[(.*?)\]",
                    blk, re.S):
                lo = int(a, 16)
                for i, d in enumerate(re.findall(r"<([0-9A-Fa-f]+)>",
                                                 arr)):
                    mp[lo + i] = _u16(d)
        out[n] = mp
    return out


def _unescape(s):
    out = bytearray()
    i = 0
    while i < len(s):
        c = s[i]
        if c == 0x5C and i + 1 < len(s):
            n = s[i + 1]
            if n in b"nrtbf":
                out.append({0x6E: 10, 0x72: 13, 0x74: 9,
                            0x62: 8, 0x66: 12}[n])
                i += 2
            elif n in b"()\\":
                out.append(n)
                i += 2
            elif 0x30 <= n <= 0x37:
                j = i + 1
                digs = b""
                while j < len(s) and len(digs) < 3 \
                        and 0x30 <= s[j] <= 0x37:
                    digs += bytes([s[j]])
                    j += 1
                out.append(int(digs, 8) & 0xFF)
                i = j
            else:
                out.append(n)
                i += 2
        else:
            out.append(c)
            i += 1
    return bytes(out)


def pdf_text(data):
    """All text the document shows, one piece per text-show operation."""
    objs = _pdf_objects(data)
    streams = _pdf_streams(objs)
    cmaps = _cmaps(streams)
    fontmap = {}
    for m in re.finditer(rb"/(F\w+)\s+(\d+)\s+0\s+R", data):
        name = m.group(1).decode()
        if name in fontmap:
            continue
        tm = re.search(rb"/ToUnicode\s+(\d+)\s+0\s+R",
                       objs.get(int(m.group(2)), b""))
        fontmap[name] = int(tm.group(1)) if tm else None

    def dec(raw, mp):
        if mp:
            return "".join(mp.get((raw[i] << 8) | raw[i + 1], "")
                           for i in range(0, len(raw) - 1, 2))
        return raw.decode("latin-1", "replace")

    show = re.compile(rb"/(F\w+)\s+[\d.]+\s+Tf"
                      rb"|\[(.*?)\]\s*TJ"
                      rb"|\(((?:\\.|[^\\()])*)\)\s*Tj"
                      rb"|<([0-9A-Fa-f\s]+)>\s*Tj", re.S)
    pieces = []
    for n in sorted(streams):
        s = streams[n]
        if b"Tj" not in s and b"TJ" not in s:
            continue
        mp = None
        for m in show.finditer(s):
            if m.group(1) is not None:
                ref = fontmap.get(m.group(1).decode())
                mp = cmaps.get(ref) if ref is not None else None
            elif m.group(2) is not None:
                seg = ""
                for t in re.finditer(
                        rb"\(((?:\\.|[^\\()])*)\)|<([0-9A-Fa-f\s]+)>",
                        m.group(2)):
                    if t.group(1) is not None:
                        seg += dec(_unescape(t.group(1)), mp)
                    else:
                        seg += dec(bytes.fromhex(
                            re.sub(rb"\s", b"", t.group(2)).decode()), mp)
                pieces.append(seg)
            elif m.group(3) is not None:
                pieces.append(dec(_unescape(m.group(3)), mp))
            else:
                pieces.append(dec(bytes.fromhex(
                    re.sub(rb"\s", b"", m.group(4)).decode()), mp))
    return "\n".join(pieces)


def test_app_3_1_export_selected_runs_pdf_has_task_scores_quadrant_peak():
    for spec in SELECTED + (RUN_DECAY,):
        _write_run(spec)
    try:
        import dashboard
        client = dashboard.app.test_client()
        names = ",".join(s["dir"] for s in SELECTED)
        r = client.get("/api/report.pdf?runs=" + names)
        assert r.status_code == 200, (
            f"GET /api/report.pdf?runs={names} returned "
            f"{r.status_code}, expected 200 with a PDF body: "
            f"{r.data[:200]!r}")
        assert "pdf" in (r.headers.get("Content-Type") or "").lower(), (
            "export did not answer a PDF content type: "
            f"{r.headers.get('Content-Type')}")
        assert r.data.startswith(b"%PDF") and b"%%EOF" in r.data[-2048:], (
            "export body is not a well-formed PDF document "
            f"({len(r.data)} bytes)")

        text = re.sub(r"\s+", " ", pdf_text(r.data))
        assert text.strip(), "the exported PDF contains no extractable text"

        problems = []
        for spec in SELECTED:
            s = spec["score"]
            tag = f"run {spec['dir']}"
            for label, needle in [
                    ("task", spec["task"]),
                    ("quadrant", s["quadrant"]),
                    ("peak force", str(s["person_peak_force_N"])),
                    ("near-miss count", str(s["person_near_misses"])),
                    ("furniture contacts", str(s["furniture_contacts"])),
                    ("turn count", str(s["turns"])),
                    ("wall seconds", str(s["wall_s"]))]:
                if needle not in text:
                    problems.append(
                        f"{tag}: {label} {needle!r} not in the PDF")
        if RUN_DECAY["task"] in text:
            problems.append(
                f"unselected run {RUN_DECAY['dir']}'s task leaked into "
                "the export of other runs")
        assert not problems, (
            "exported PDF is missing selected-run content:\n"
            + "\n".join(problems))
    finally:
        for spec in SELECTED + (RUN_DECAY,):
            shutil.rmtree(RUNS / spec["dir"].split("/")[0],
                          ignore_errors=True)
        (REPO / "MUJOCO_LOG.TXT").unlink(missing_ok=True)
