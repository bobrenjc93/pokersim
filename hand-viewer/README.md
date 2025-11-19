# Hand Viewer

A web interface for viewing poker hand histories logged during ELO evaluation.

## Features

- Browse logged hands with filtering options
- View detailed hand replays with:
  - All actions taken by each player
  - Model probability distributions for each decision
  - Community cards progression
  - Pot sizes and chip stacks
- Search by player name, hand outcome, etc.

## Usage

```bash
cd hand-viewer
./start.sh
```

Then open http://localhost:5052 in your browser.

## Hand Log Location

Hands are logged to `/tmp/pokersim/hand_logs/` by the ELO evaluation system.
1 out of every 100 hands is logged for analysis.

## Hand Log Format

Each hand is stored as a JSON file with:
- `hand_id`: Unique identifier
- `timestamp`: When the hand was played
- `players`: Player info (name, model path, starting stack)
- `actions`: Sequence of actions with model predictions
- `community_cards`: Cards dealt at each stage
- `result`: Final outcome (winner, profits)

