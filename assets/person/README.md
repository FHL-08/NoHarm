Person model: Microsoft Rocketbox avatar `Adults/Female_Adult_01`
(https://github.com/microsoft/Microsoft-Rocketbox, MIT licence).

Rigged asset: `env/person/rig_person.py` imports the upstream
`Export/Female_Adult_01.fbx` (vendored at `env/person/Female_Adult_01.fbx`,
re-downloaded if absent) in headless Blender and exports
`female_adult_01.glb` — a skinned glTF on the Rocketbox Bip01 skeleton
(80 bones), with the vendored `f001_{body,head,opacity}_color.png`
textures bound into its materials:

    blender --background --python env/person/rig_person.py

Static scene meshes: `body.obj`, `head.obj`, `hair.obj` are the same
avatar baked to z-up OBJ (upper arms lowered from the authoring A-pose
to rest) for the current mocap person body in kitchen.xml — produced by
`bake_glb.py` from an FBX2glTF 0.9.7 conversion:

    POSE="Bip01 L UpperArm:-38,Bip01 R UpperArm:38" \
      python bake_glb.py fa01.glb none out 0

Textures are the upstream `f001_{body,head,opacity}_color.tga`, resized
to 1024 px PNG (MuJoCo does not read TGA).
