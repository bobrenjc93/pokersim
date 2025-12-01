#!/bin/bash
# Start the Hand Viewer server

set -e

cd "$(dirname "$0")"

echo "🎴 Starting Hand Viewer..."
echo "Open http://localhost:5052 in your browser"
echo ""

uv run python server.py
