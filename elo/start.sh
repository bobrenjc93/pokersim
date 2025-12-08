#!/usr/bin/env bash
# Start the Poker ELO Rating Server

set -euo pipefail
trap 'echo "Error: start.sh failed on line ${LINENO}. Re-run with: bash -x ./start.sh" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

# Add project root to PYTHONPATH so we import common/ directly from source,
# not from an installed package. This ensures /elo and /training always share
# the same MODEL_VERSION without manual re-installs.
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Allow overriding the port via env vars.
# - ELO_PORT is specific to this server
# - PORT matches common deployment conventions (and is what server.py reads)
PORT="${ELO_PORT:-${PORT:-5051}}"

is_port_free() {
  local port="$1"
  # Use uv to run python per workspace rules.
  # Exit codes:
  # - 0: free (no listener on localhost)
  # - 1: busy/unavailable
  # - 2: port-check not permitted (e.g., restricted sandbox) -> treat as "unknown"
  uv run python - "$port" >/dev/null 2>&1 <<'PY'
import errno
import socket
import sys

port = int(sys.argv[1])
try:
    # If we can connect, something is already listening => busy.
    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
        raise SystemExit(1)
except OSError as e:
    if e.errno in (errno.EPERM, errno.EACCES):
        raise SystemExit(2)
    # Connection failed (refused/timeout/etc.) => no listener => free.
    raise SystemExit(0)
PY
}

# If requested port is busy, find the next available one.
if is_port_free "$PORT"; then
  probe_rc=0
else
  probe_rc="$?"
fi
if [[ "$probe_rc" -eq 2 ]]; then
  echo "Warning: unable to probe local ports in this environment; starting on PORT=${PORT}." >&2
elif [[ "$probe_rc" -ne 0 ]]; then
  base="$PORT"
  found=""
  for p in $(seq "$base" 5099); do
    if is_port_free "$p"; then
      rc=0
    else
      rc="$?"
    fi
    if [[ "$rc" -eq 2 ]]; then
      echo "Warning: unable to probe local ports in this environment; starting on PORT=${PORT}." >&2
      found="$PORT"
      break
    fi
    if [[ "$rc" -eq 0 ]]; then
      found="$p"
      break
    fi
  done
  if [[ -z "${found}" ]]; then
    echo "Error: no free port found in range ${base}-5099. Set PORT or ELO_PORT to an available port." >&2
    exit 1
  fi
  PORT="$found"
fi

export PORT

echo "Starting Poker ELO Rating Server..."
echo "Open http://localhost:${PORT} in your browser"
echo ""

exec uv run python server.py
