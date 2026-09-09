#!/usr/bin/env bash
set -euo pipefail
exec "$HOME/.local/bin/mflux-generate-qwen" \
  --model mflux-community/qwen-image-2512-mflux-q4 --base-model qwen-image \
  --low-ram --width 768 --height 768 --steps 30 --seed 42 \
  --prompt "${1:-A red wooden cabin beside a peaceful Scandinavian lake at sunrise, photographic detail, soft mist, a small wooden sign reading WELCOME}" \
  --output "${2:-/private/tmp/qwen-image.png}"
