Person model: Microsoft Rocketbox avatar `Adults/Female_Adult_01`
(https://github.com/microsoft/Microsoft-Rocketbox, MIT licence).

Converted, not modelled: `Export/Female_Adult_01.fbx` -> glTF with
FBX2glTF 0.9.7, then baked to static OBJ (z-up, metres) with
`bake_glb.py`, upper arms lowered from the authoring A-pose to rest:

    POSE="Bip01 L UpperArm:-38,Bip01 R UpperArm:38" \
      python bake_glb.py fa01.glb none out 0

Textures are the upstream `f001_{body,head,opacity}_color.tga`, resized
to 1024 px PNG (MuJoCo does not read TGA).
