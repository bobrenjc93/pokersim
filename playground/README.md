# Poker Playground

An interactive web interface to play poker against trained AI models.

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

Open http://localhost:5001 in your browser.

## Prerequisites

- Python 3.10+
- Trained model checkpoints (see `../training/config.py` for path)
- The `poker_api_binding` shared library (copied automatically from training)

## Gameplay

1. Select a trained model checkpoint as your opponent
2. Click "Start Game" to begin
3. Make betting decisions using the action buttons
4. After each hand completes, you'll see the result and can deal a new hand
5. Track your progress with the bankroll display

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PORT` | 5001 | Web server port |
