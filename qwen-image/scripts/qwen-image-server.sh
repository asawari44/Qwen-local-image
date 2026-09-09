#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "$$" > local_image/server.pid
echo 'Qwen image API: http://127.0.0.1:8001/docs (logs: local_image/server.log)'
exec "$HOME/.local/share/uv/tools/mflux/bin/python" -m uvicorn \
  local_image.server:app --host 127.0.0.1 --port 8001 --workers 1 --no-access-log >> local_image/server.log 2>&1
