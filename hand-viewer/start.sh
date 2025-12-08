#!/bin/bash
# Start the Hand Viewer server

set -e

cd "$(dirname "$0")"

echo "🎴 Starting Hand Viewer..."

PORT_ARG="${1:-}"
if [[ -n "$PORT_ARG" ]]; then
  shift
fi

BASE_PORT="${PORT_ARG:-${PORT:-5052}}"
PORT_SCAN_RANGE="${PORT_SCAN_RANGE:-15}"

is_port_free() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    ! lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi

  # Fallback: try to bind the port with Python (via uv, per repo conventions).
  uv run python - <<PY >/dev/null 2>&1
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(${port})))
except OSError:
    sys.exit(1)
finally:
    s.close()
sys.exit(0)
PY
}

PORT_TO_USE=""
MAX_PORT=$((BASE_PORT + PORT_SCAN_RANGE))
for ((p=BASE_PORT; p<=MAX_PORT; p++)); do
  if is_port_free "$p"; then
    PORT_TO_USE="$p"
    break
  fi
done

if [[ -z "$PORT_TO_USE" ]]; then
  echo "❌ No free port found in range ${BASE_PORT}-${MAX_PORT}."
  echo "Tip: set PORT=#### or pass a port: ./start.sh 5055"
  exit 1
fi

export PORT="$PORT_TO_USE"

echo "Open http://localhost:${PORT} in your browser"
echo ""

uv run python server.py
