# Poker Simulation Website

A Python Flask-based web interface for the poker simulation system.

For full documentation, see the [main project README](../README.md).

## Quick Start

```bash
./start.sh
```

Or manually:

```bash
uv sync
uv run python app.py
```

Open http://localhost:5000 in your browser.

## Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) for package management
- C++ API server running (see `../api/README.md`)

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PORT` | 5000 | Web server port |
| `API_HOST` | localhost | C++ API hostname |
| `API_PORT` | 8080 | C++ API port |

Example:
```bash
PORT=3000 API_HOST=localhost API_PORT=8080 uv run python app.py
```

## Troubleshooting

**Connection Error to API Server**: Ensure the C++ API server is running (`cd ../api && make && ./poker_api`)

**Port Already in Use**: Use a different port: `PORT=5001 uv run python app.py`
