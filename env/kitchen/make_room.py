"""Vendor the AI2-THOR "FloorPlan1" kitchen (MolmoSpaces MJCF conversion,
allenai/molmospaces on Hugging Face, CC BY 4.0) into assets/ithor_kitchen.xml
+ assets/ithor_kitchen/.

The room, fixtures (cabinets, drawers, fridge, oven, dishwasher, microwave,
sink) and free props (mug, knife, vase, apple, bowl, pot, ...) are the
upstream sim-ready assets: visual meshes + convex-decomposed colliders,
articulated doors/drawers, masses from per-object densities. Nothing is
hand-modelled. This script only:
  - downloads the scene package + the THOR object packs it references
    (HTTP range reads out of the HF tar shards, cached in --cache),
  - drops the few bodies whose source pack we don't fetch (bedroom/living
    room packs are ~300 MB for one book and one houseplant),
  - renames hashed names to short ones (`mug_8caf1bb3..._1_0_0` -> `mug`),
  - moves the ceiling to a hidden geom group so the viewer can look in,
  - puts structural wall colliders in group 4 with the other colliders,
  - copies only the referenced files and rewrites paths for kitchen.xml.

Why the room XML sits directly in assets/: MuJoCo resolves an included
file's mesh paths against the (global, last-wins) compiler meshdir, which
is "assets"; Newton's add_mjcf resolves them against the included file's
own directory. Both are assets/ only if the room file lives there.

Usage: .venv/bin/python env/kitchen/make_room.py [--cache /tmp/molmospaces]
"""
import argparse
import json
import re
import shutil
import subprocess
import urllib.request
from collections import defaultdict
from pathlib import Path

HF = "https://huggingface.co/datasets/allenai/molmospaces/resolve/main/mujoco"
SCENE_PKG = ("scenes/ithor/20251217", "ithor_FloorPlan1_physics.tar.zst")
OBJ_PKGS = ["Kitchen Objects", "Bathroom Objects", "Common Objects",
            "Miscellaneous Objects"]
OBJ_DIR = "objects/thor/20251117"

here = Path(__file__).parent
assets = here.parent.parent / "assets"
out_dir = assets / "ithor_kitchen"
ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/tmp/molmospaces")
cache = Path(ap.parse_args().cache)


def fetch(subdir, pkg, dest):
    """Range-read one package out of its HF shard and untar it into dest."""
    marker = dest / f".{pkg}.done"
    if marker.exists():
        return
    table = json.load(urllib.request.urlopen(f"{HF}/{subdir}/arrow_table.json"))
    e = next(x for x in table if x["path"] == pkg)
    url = f"{HF}/{subdir}/shards/{e['shard_id']:05d}.tar"
    dest.mkdir(parents=True, exist_ok=True)
    blob = cache / pkg.replace(" ", "_")
    subprocess.run(["curl", "-sL", "--retry", "5", "-r",
                    f"{e['offset']}-{e['offset'] + e['size'] - 1}", url,
                    "-o", str(blob)], check=True)
    subprocess.run(f"zstd -dc '{blob}' | tar -x -C '{dest}'", shell=True,
                   check=True)
    blob.unlink()
    marker.touch()


fetch(*SCENE_PKG, cache / "scenes/ithor")
for p in OBJ_PKGS:
    fetch(OBJ_DIR, f"thor_{p}.tar.zst", cache / "objects/thor")

src_xml = cache / "scenes/ithor/FloorPlan1_physics.xml"
s = src_xml.read_text()
base = src_xml.parent

# 1. drop assets whose file we don't have, then everything that uses them
missing = {m.group(1) for m in re.finditer(r'<(?:mesh|texture) [^>]*?name="([^"]+)"[^>]*file="([^"]+)"', s)
           if not (base / m.group(2)).exists()}
s = re.sub(r'\s*<(?:mesh|texture) [^>]*?name="(' + "|".join(map(re.escape, missing)) + r')"[^>]*/>', "", s)
s = re.sub(r' texture="(' + "|".join(map(re.escape, missing)) + r')"', "", s)
dead = {m.group(1) for m in re.finditer(r'<geom name="([a-z0-9]+_[0-9a-f]{32})_1_[^"]*"[^>]*mesh="(' + "|".join(map(re.escape, missing)) + r')"', s)}
for obj in dead:   # remove the object's whole top-level body + its excludes
    s = re.sub(r'\n(\s*)<body name="' + obj + r'_1_0_0".*?\n\1</body>', "", s, flags=re.S)
    s = re.sub(r'\s*<exclude [^>]*' + obj + r'[^>]*/>', "", s)
print("dropped (pack not fetched):", sorted(o.split("_")[0] for o in dead))

# 2. short names: <type>_<hash>_1_0_0 -> <type>[_n]; sub-parts keep a suffix
count = defaultdict(int)
for typ, h, k in dict.fromkeys(re.findall(r'"([A-Za-z0-9()]+)_([0-9a-f]{32})_(\d+)_0_0"', s)):
    short = re.sub(r"\(\d+\)", "", typ).lower()
    count[short] += 1
    short = short if count[short] == 1 else f"{short}_{count[short]}"
    s = s.replace(f"{typ}_{h}_{k}_0_0", short).replace(f"{typ}_{h}_{k}_", short + "_")

# 3. ceiling visuals hidden (group 3) so cameras/viewer can see in from
#    above; walls collide in group 4 like every other collider
s = re.sub(r'(<geom name="ceiling_[^"]*visual[^"]*" )', r'\1group="3" ', s)
s = s.replace('<default class="__STRUCTURAL_WALL_MJT__">\n          <geom group="3" />',
              '<default class="__STRUCTURAL_WALL_MJT__">\n          <geom group="4" />')

# 4. vendor only referenced files; paths are relative to kitchen.xml's
#    meshdir/texturedir ("assets"), hence the ../env prefix
if out_dir.exists():
    shutil.rmtree(out_dir)


def vendor(m):
    rel = Path(m.group(1))
    srcf = (base / rel).resolve()
    tail = srcf.relative_to(cache.resolve())
    dst = out_dir / tail
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(srcf, dst)
    return f'file="ithor_kitchen/{tail.as_posix()}"'


s = re.sub(r'file="([^"]+)"', vendor, s)
s = s.replace('<mujoco model="FloorPlan1_physics">',
              '<mujoco model="ithor_floorplan1_kitchen">\n  <!-- GENERATED by env/kitchen/make_room.py from MolmoSpaces '
              'ithor/FloorPlan1_physics.xml (Ai2, CC BY 4.0). Do not edit by hand. -->', 1)
(assets / "ithor_kitchen.xml").write_text(s)
size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
print(f"wrote {assets / 'ithor_kitchen.xml'} ({size:.1f} MB vendored in {out_dir})")

# 5. splice the rigged person (NPC-1 Rocketbox glTF baked to a Bip01-boned
#    MJCF subtree) into the generated file — her meshes only resolve for
#    relative and absolute loads from inside the meshdir
import sys
sys.path.insert(0, str(here))
import make_person
make_person.inject(assets / "ithor_kitchen.xml")
