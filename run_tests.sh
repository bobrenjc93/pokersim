#!/bin/bash

# Test script for pokersim project
# Runs all C++ tests and Python API tests

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "🃏 Running Poker Simulator Tests"
echo "================================"
echo ""

# Run C++ unit tests
echo "📋 Part 1: C++ Unit Tests"
echo "-------------------------"
cd "$SCRIPT_DIR/api"
make test
# Build python binding for RL tests
echo "🔨 Building Python bindings..."
make module
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

echo "📋 Part 3: RL Training Tests"
echo "---------------------------"
cd "$SCRIPT_DIR/training"

if command -v uv &> /dev/null; then
    # Run training tests (installing pytest temporarily if needed)
    uv run --with pytest pytest tests/
elif command -v pytest &> /dev/null; then
    # Fallback to system pytest if available
    echo "⚠️  uv not found, trying system pytest..."
    pytest tests/
else
    echo "⚠️  Skipping RL tests: uv not found and pytest not in PATH"
    echo "   Please install uv to run these tests: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi
echo ""

echo "✅ All tests completed successfully!"

