# Poker Simulation Website

A Python Flask-based web interface for the poker simulation system.

## Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager

## Installation

1. Install uv if you haven't already:
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

2. Navigate to the website directory:
```bash
cd website
```

3. Install dependencies:

```bash
uv sync
```

## Running the Server

### Quick Start (Easiest)

Use the included start script that handles everything:

```bash
./start.sh
```

This script will:
- Install uv if needed
- Install all dependencies with `uv sync`
- Check if the API server is running
- Start the Flask server

### Manual Start

Start the server with default settings (port 5000):

```bash
# First time setup and run
uv sync
uv run python app.py
```

The server will start at `http://localhost:5000`

### Custom Port

To run on a different port:

```bash
PORT=3000 python app.py
```

### Configure API Server Location

By default, the website connects to the C++ API server at `localhost:8080`. To change this:

```bash
API_HOST=your-api-host API_PORT=8080 python app.py
```

### Production Mode

For production deployment, use a WSGI server like Gunicorn:

```bash
# Add gunicorn to pyproject.toml dev dependencies first, then:
uv sync
uv run gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Using the Web Interface

1. Open your browser and navigate to `http://localhost:5000`
2. Enter a game state in JSON format (e.g., `{"players": 2, "pot": 100, "cards": ["AS", "KD"]}`)
3. Enter a random seed (any integer)
4. Click "Simulate Next State" to get the next game state

## API Endpoints

### `GET /`
Returns the main web interface.

### `POST /api/simulate`
Proxies requests to the C++ API server.

**Request body:**
```json
{
  "gameState": {
    "players": 2,
    "pot": 100,
    "cards": ["AS", "KD"]
  },
  "seed": 42
}
```

**Response:**
```json
{
  "success": true,
  "nextGameState": {
    "players": 2,
    "pot": 130,
    "cards": ["AS", "KD", "QH"],
    "lastAction": "bet",
    "lastBetAmount": 30,
    "simulated": true,
    "timestamp": 1700000000
  }
}
```

## Project Structure

```
website/
├── app.py              # Flask application
├── pyproject.toml      # Python dependencies
├── start.sh            # Quick start script
├── templates/
│   └── index.html     # Web interface
└── README.md          # This file
```

## Why uv?

`uv` is **10-100x faster** than pip for installing packages:
- ⚡ Written in Rust for speed
- 🔒 Better dependency resolution
- 🎯 Drop-in replacement for pip
- 🚀 Automatic virtual environment management

## Troubleshooting

### Connection Error to API Server

If you see "Could not connect to API server", make sure:
1. The C++ API server is running (see `/api/README.md`)
2. The `API_HOST` and `API_PORT` environment variables are set correctly
3. No firewall is blocking the connection

### Port Already in Use

If port 5000 is already in use, specify a different port:
```bash
PORT=5001 python app.py
```

## Development

To run in development mode with auto-reload:

```bash
FLASK_ENV=development python app.py
```

This enables debug mode and auto-reloads the server when you make changes.

