#!/bin/bash
# Start the Poker ELO Rating Server

cd "$(dirname "$0")"

echo "Starting Poker ELO Rating Server..."
echo "Open http://localhost:5051 in your browser"
echo ""

uv run python server.py

