"""Rig-load the upstream Microsoft Rocketbox FBX and export a skinned glTF.

Source: Assets/Avatars/Adults/Female_Adult_01/Export/Female_Adult_01.fbx in
github.com/microsoft/Microsoft-Rocketbox (MIT). The FBX carries the Rocketbox
biped (Bip01 skeleton) and its skinned mesh; this script imports it into an
empty Blender scene, rebinds the texture images the FBX references to the
PNGs already vendored in assets/person/ (the upstream TGAs are not committed;
MuJoCo cannot read TGA), and writes a self-contained GLB.

Headless usage:

    blender --background --python env/person/rig_person.py -- [fbx] [out]

Defaults: fbx = env/person/Female_Adult_01.fbx (downloaded when absent),
out = assets/person/female_adult_01.glb.
"""
import sys
import urllib.request
from pathlib import Path

import bpy

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FBX_URL = ("https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/"
           "master/Assets/Avatars/Adults/Female_Adult_01/Export/"
           "Female_Adult_01.fbx")

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
fbx_path = Path(argv[0]) if len(argv) > 0 else HERE / "Female_Adult_01.fbx"
out_path = (Path(argv[1]) if len(argv) > 1
            else REPO / "assets" / "person" / "female_adult_01.glb")

if not fbx_path.is_file():
    print(f"rig_person: fetching {FBX_URL} -> {fbx_path}")
    fbx_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(FBX_URL, fbx_path)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=str(fbx_path))

tex_dir = REPO / "assets" / "person"
for img in list(bpy.data.images):
    if img.packed_file or not img.filepath:
        continue
    if Path(bpy.path.abspath(img.filepath)).is_file():
        continue
    png = tex_dir / f"{Path(img.filepath).stem}.png"
    if png.is_file():
        img.filepath = str(png)
        img.reload()
    else:
        bpy.data.images.remove(img)

armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
if not armatures:
    raise RuntimeError(f"rig_person: no armature in {fbx_path}")
bones = [b.name for a in armatures for b in a.data.bones]
if not all(n.startswith("Bip01") for n in bones):
    raise RuntimeError("rig_person: non-Bip01 bones in rig: "
                       + ", ".join(n for n in bones
                                   if not n.startswith("Bip01")))
print(f"rig_person: {len(armatures)} armature(s), {len(bones)} bones, "
      f"{len(bpy.data.objects)} objects")

out_path.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=str(out_path),
    export_format="GLB",
    export_skins=True,
    export_def_bones=True,
    export_animations=True,
    export_image_format="AUTO",
    export_yup=True,
)
print(f"rig_person: wrote {out_path}")
