#!/bin/bash
# Start the Poker ELO Rating Server

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Poker ELO Rating Server..."
echo "Open http://localhost:5051 in your browser"
echo ""

# Install common package (required dependency)
COMMON_DIR="$(dirname "$SCRIPT_DIR")/common"
if [ -d "$COMMON_DIR" ]; then
    uv pip install -e "$COMMON_DIR" --quiet 2>/dev/null || true
fi

uv run python server.py
