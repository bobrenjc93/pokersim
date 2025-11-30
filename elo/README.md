# Poker ELO Rating System (Go/Chess Scale)

A web-based ELO rating simulation system for evaluating poker AI models against each other and heuristic bots.

## Overview

This system runs poker matches between different AI checkpoints and heuristic bots, updating their ELO ratings based on match outcomes using the classic ELO formula with a Go/Chess-like scale where top players can reach 4000-5000 rating.

### Rating Scale

- **Initial Rating**: 2500 (similar to professional Go/Chess systems)
- **Target Top Rating**: 4000-5000 for the strongest players
- **K-Factor**: 40 (allows faster rating spread)

### ELO Formula

**Expected Score:**
```
E = 1 / (1 + 10^((Rb - Ra) / 400))
```

**New Rating:**
```
Ra' = Ra + K × (S - E)
```

Where:
- `E` = Expected score (probability of winning)
- `Ra`, `Rb` = Current ratings of players A and B
- `K` = K-factor (default 40 for faster differentiation)
- `S` = Actual score (1 = win, 0.5 = draw, 0 = loss)

## Features

- **Real-time ELO updates**: Watch ratings change live as matches complete
- **Model checkpoints**: Automatically loads trained model checkpoints
- **Heuristic bots**: Includes multiple bot types for comparison:
  - Heuristic, Tight, Aggressive, LoosePassive
  - CallingStation, HeroCaller, Random
- **Interactive UI**: Modern web interface with charts and leaderboards
- **Configurable parameters**: Adjust rounds, hands per match, and K-factor

## Usage

### Start the server

```bash
cd elo
uv run python server.py
```

Or use the start script:

```bash
./start.sh
```

The server runs on port 5051 by default. Open http://localhost:5051 in your browser.

### Configuration Options

- **Rounds**: Number of rounds of matches (default: 100)
- **Hands/Match**: Hands played per match to determine winner (default: 50)
- **K-Factor**: Rating change sensitivity (default: 40)
  - Higher K = faster rating spread (use 40-60 to reach 4-5k top ratings quickly)
  - Lower K = more stable ratings (use 20-32 for slower convergence)

## Files

- `engine.py` - Core ELO calculation and poker game logic
- `server.py` - Flask web server with SSE streaming
- `templates/index.html` - Web UI
- `pyproject.toml` - Python dependencies

## Architecture

The system uses:
- Flask for the web server
- Server-Sent Events (SSE) for real-time updates
- Chart.js for visualizations
- The same poker API binding as the training system

## Interpreting Results

- **Starting ELO**: All players begin at 2500 (Go/Chess scale)
- **Target Range**: Top players should reach 4000-5000 after many rounds
- **Match Winner**: Determined by total big blinds won across all hands
- **ELO Change**: Higher-rated players gain less for beating lower-rated opponents
- **Leaderboard**: Sorted by current ELO rating

### Rating Tiers (approximate)

| Rating | Skill Level |
|--------|-------------|
| 4500+  | Elite / Professional |
| 4000-4500 | Expert |
| 3500-4000 | Advanced |
| 3000-3500 | Intermediate |
| 2500-3000 | Beginner |
| < 2500 | Below Average |

