# PokerSim Arena

Real-time evaluation arena for PokerSim agents. Compare model checkpoints, track ELO ratings, and visualize training progress with a live web interface.

For full documentation, see the [main project README](../README.md#advanced-evaluation-arena).

## Files

- `engine.py` - Core arena logic (hand simulation, ELO calculations)
- `server.py` - Flask web server with Server-Sent Events for real-time updates

## Quick Start

Start the arena server:

```bash
uv run python server.py
```

Open `http://localhost:5000` in your browser.

## Features

- **Real-time hand streaming** - Each hand result appears immediately in the UI
- **Live statistics** - Watch win rate, BB/100, and hand count update in real-time
- **Multiple evaluation modes**:
  - **Ladder** - Compare each checkpoint against its predecessor
  - **vs Random** - Evaluate all checkpoints against a random baseline
  - **ELO Rating** - Compute ELO ratings across all checkpoints
- **Interactive charts** - Visualize performance progression over training iterations
- **Hand feed** - See individual hand outcomes as they happen

## Configuration

- **Hands per Match** - Number of hands to play per matchup (default: 100)
- **Compute Device** - CPU, CUDA (NVIDIA), or MPS (Apple Silicon)

## Environment Variables

- `PORT` - Server port (default: 5000)
