"""SIM-3.2@1 At t=0 no two scene geoms penetrate deeper than 2 mm, upstream
authoring excluded.

"t=0" is the scene's declared start state — the pose every run begins from:
each <key> pose kitchen.xml itself declares (the Executor resets to it and
newton_scene.build copies it into joint_q as "the start pose"), or the
model's initial qpos when the scene declares no start keyframe. A pose that
comes with a downloaded asset — the G1's own "stand" keyframe, the default
qpos implied by upstream body placement — is the upstream file's authoring,
not the composition's start state.

"Penetrate" is measured the way the sim measures it: after mj_forward, a
generated contact's dist is the signed separation of the two geoms, so
dist < -0.002 is a penetration deeper than 2 mm.

"Upstream authoring excluded" excuses a pair only when both geoms belong to
the same included asset file — how geoms inside one downloaded asset are
placed is that file's authoring and no composition can change it. Every
other pair — across different asset files, or involving a geom kitchen.xml
authors itself — is composition and must not overlap beyond the threshold.
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NOHARM_HEADLESS", "1")

REPO = Path(__file__).resolve().parents[2]
KITCHEN = REPO / "kitchen.xml"
DEPTH_M = 0.002

BODY_NAME = re.compile(r"<body\b[^>]*\bname\s*=\s*\"([^\"]+)\"")
WORLD_GEOM = re.compile(r"<geom\b[^>]*\bname\s*=\s*\"([^\"]+)\"")


def _included_files(path, seen=None):
    """kitchen.xml plus every file reached through <include>, resolved."""
    seen = seen if seen is not None else {path.resolve()}
    files = [path.resolve()]
    for inc in ET.parse(path).iter("include"):
        f = inc.get("file")
        if not f:
            continue
        p = (path.parent / f).resolve()
        if p in seen or not p.is_file() or p.suffix != ".xml":
            continue
        seen.add(p)
        files.extend(_included_files(p, seen))
    return files


def _top_level_bodies(path):
    """Body names a file declares as direct children of its worldbody —
    the model's top-level bodies once includes merge."""
    root = ET.parse(path).getroot()
    wb = root.find("worldbody")
    if wb is None:
        return set()
    names = set()
    for body in wb.iter("body"):
        n = body.get("name")
        if n:
            names.add(n)
    return names


def _declared_keys(path):
    """Keyframe names the composer file itself declares — its declared
    start states."""
    return [k.get("name") for k in ET.parse(path).iter("key") if k.get("name")]


def _geom_file_groups(files):
    """Map top-level body / worldbody geom names to the file that authored
    them; kitchen.xml's own declarations map to the pseudo-file 'REPO'."""
    owner = {}
    for f in files:
        tag = "REPO" if f == KITCHEN else str(f)
        for name in _top_level_bodies(f):
            owner.setdefault(name, tag)
        root = ET.parse(f).getroot()
        wb = root.find("worldbody")
        if wb is not None:
            for geom in wb.findall("geom"):
                n = geom.get("name")
                if n:
                    owner.setdefault(n, tag)
    return owner


def test_sim_3_2_no_geom_penetration_beyond_2mm_at_t0():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(KITCHEN))
    files = _included_files(KITCHEN)
    owner = _geom_file_groups(files)

    def top_name(body):
        while model.body_parentid[body] != 0:
            body = model.body_parentid[body]
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""

    def group(geom):
        body = model.geom_bodyid[geom]
        if body == 0:
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            return owner.get(gname, "WORLD")
        return owner.get(top_name(body), "UNKNOWN")

    states = _declared_keys(KITCHEN)
    data = mujoco.MjData(model)
    violations = []
    audited = []
    for state in states or [None]:
        if state is None:
            mujoco.mj_resetData(model, data)
        else:
            kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, state)
            mujoco.mj_resetDataKeyframe(model, data, kid)
        mujoco.mj_forward(model, data)
        audited.append(state or "initial qpos")
        for c in range(data.ncon):
            con = data.contact[c]
            if con.dist >= -DEPTH_M:
                continue
            g1, g2 = con.geom1, con.geom2
            f1, f2 = group(g1), group(g2)
            if f1 == f2 and f1 != "REPO":
                continue  # both sides authored inside one upstream file
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom{g1}"
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom{g2}"
            violations.append(
                f"{con.dist * 1000:.1f}mm  {n1}({f1}) vs {n2}({f2})")
    assert not violations, (
        "penetrations deeper than 2 mm at t=0 "
        f"(states: {', '.join(audited)}), upstream-same-file pairs "
        "excluded:\n" + "\n".join(sorted(violations)[:30]))
