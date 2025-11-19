#!/bin/bash
# Start the Poker Playground server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure we're in the playground directory
echo "🃏 Starting Poker Playground..."
echo "Directory: $SCRIPT_DIR"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first."
    exit 1
fi

# Sync dependencies
echo "📦 Syncing dependencies..."
uv sync

# Copy the poker_api_binding if it exists in training
TRAINING_DIR="$(dirname "$SCRIPT_DIR")/training"
BINDING_FILE="$TRAINING_DIR/poker_api_binding.cpython-312-darwin.so"

if [ -f "$BINDING_FILE" ]; then
    cp "$BINDING_FILE" "$SCRIPT_DIR/"
    echo "✓ Copied poker_api_binding from training directory"
else
    echo "Warning: poker_api_binding not found at $BINDING_FILE"
    echo "You may need to build it first by running training tests"
fi

# Start the server
PORT="${PORT:-5001}"
echo "🚀 Starting server on port $PORT..."
echo "Open http://localhost:$PORT in your browser"
echo ""

uv run python app.py

