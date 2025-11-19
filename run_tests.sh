#!/bin/bash

# Test script for pokersim project
# Runs all C++ tests in the api directory

set -e  # Exit on error

echo "🃏 Running Poker Simulator Tests"
echo "================================"
echo ""

# Navigate to api directory and run tests
cd "$(dirname "$0")/api"

# Build and run all tests
make test

echo ""
echo "✅ All tests completed successfully!"

