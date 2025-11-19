#!/bin/bash
# Quick start script for the poker simulation website

set -e

echo "🎯 Poker Simulation Website - Quick Start"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "⚠️  uv is not installed. Installing now..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✓ uv installed successfully!"
    echo ""
    echo "⚠️  Please restart your terminal and run this script again."
    exit 0
fi

echo "✓ uv is installed"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
uv venv --allow-existing
uv pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Check if API server is running
echo "🔍 Checking if C++ API server is running on port 8080..."
if curl -s http://localhost:8080 > /dev/null 2>&1; then
    echo "✓ API server is running"
else
    echo "⚠️  API server doesn't seem to be running on port 8080"
    echo "   Start it with: cd ../api && make && ./poker_api"
    echo ""
    echo "   Continuing anyway (you can start the API server later)..."
fi
echo ""

# Start the Flask server
echo "🚀 Starting Flask web server..."
echo "   Server will be available at: http://localhost:5000"
echo ""

# Activate venv and run
source .venv/bin/activate
python app.py

