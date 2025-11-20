#!/bin/bash

# Test script for pokersim project
# Runs all C++ tests and Python API tests

set -e  # Exit on error

SCRIPT_DIR="$(dirname "$0")"

echo "🃏 Running Poker Simulator Tests"
echo "================================"
echo ""

# Run C++ unit tests
echo "📋 Part 1: C++ Unit Tests"
echo "-------------------------"
cd "$SCRIPT_DIR/api"
make test
echo ""

# Run Python API integration tests
echo "📋 Part 2: Python API Integration Tests"
echo "----------------------------------------"

# Check if uv is installed, otherwise try python3
if command -v uv &> /dev/null; then
    # Run Python API tests with uv
    uv run tests/test_stateless_api.py "$@"
    echo ""
elif command -v python3 &> /dev/null; then
    # Run Python API tests with python3
    python3 tests/test_stateless_api.py "$@"
    echo ""
else
    echo "⚠️  Neither uv nor python3 found - skipping Python API tests"
    echo ""
    echo "To run these tests, either:"
    echo "  1. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  2. Or ensure python3 is available in your PATH"
    echo ""
fi

echo "✅ All tests completed successfully!"

