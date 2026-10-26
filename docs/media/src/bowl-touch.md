# Bowl-touch recording

The README clip is the unedited head-camera MP4 from one scripted
demonstration on base `fae5ad65134710afc2408914640017c7e254b531`.
It is training data, not a learned-policy evaluation or a harm benchmark.

From the repository root, with requirements installed, record into a new
directory in an unattended context:

```sh
NOHARM_HEADLESS=1 .venv/bin/python -m harmless.policies.demos --episodes 1 --out /tmp/harmless-readme-pass/g1_bowl_touch --seed 0 --headless
cp /tmp/harmless-readme-pass/g1_bowl_touch/videos/chunk-000/observation.images.head_cam/episode_000000.mp4 docs/media/bowl-touch.mp4
```

The captured export contains 203 frames at 50 Hz, each 224 × 224 pixels,
and 203 Parquet rows of joint positions, velocities and actuator targets
for 29 joints. The recorder saves the episode after its physics contact
check succeeds. This pass checked export consistency and viewed the
camera output; it did not independently replay or score the trajectory.

Render the GIF from the retained MP4:

```sh
ffmpeg_bin="$(.venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
"$ffmpeg_bin" -y -loglevel error -i docs/media/bowl-touch.mp4 \
  -vf 'fps=15,split[a][b];[a]palettegen[p];[b][p]paletteuse' docs/media/bowl-touch.gif
```
