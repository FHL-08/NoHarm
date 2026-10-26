#!/usr/bin/env bash
# Renders architecture.html to ../architecture.gif and ../architecture.mp4.
# Needs npx (Playwright) and the ffmpeg bundled with imageio-ffmpeg in .venv.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
ffmpeg="$("$repo/.venv/bin/python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
frames="$(mktemp -d)"
n=40
for i in $(seq 0 $((n - 1))); do
  t=$(python3 -c "print($i/$n)")
  npx --yes playwright screenshot --viewport-size=960,590 \
    "file://$here/architecture.html?t=$t" "$frames/$(printf %03d "$i").png" >/dev/null
done
"$ffmpeg" -y -loglevel error -framerate 5 -i "$frames/%03d.png" \
  -pix_fmt yuv420p -vf "scale=960:-2" "$here/../architecture.mp4"
"$ffmpeg" -y -loglevel error -framerate 5 -i "$frames/%03d.png" \
  -vf "split[a][b];[a]palettegen[p];[b][p]paletteuse" "$here/../architecture.gif"
rm -rf "$frames"
