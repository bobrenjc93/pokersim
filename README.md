# Poker Simulator Monorepo

A full-stack poker simulation system with a Python web interface, high-performance C++ API backend, and Reinforcement Learning training system.

## Project Overview

This monorepo contains:
- **Website**: Python Flask web server with modern UI for poker simulations
- **API**: Pure C++ API server for fast, stateless game simulations
- **Training**: Reinforcement learning system (PPO) for training poker AI agents
- **Playground**: Interactive web UI to play poker against trained AI models
- **ELO**: ELO rating system for comparing model checkpoints

## Architecture

```
┌─────────────────────┐
│   Web Browser       │
└──────────┬──────────┘
           │ HTTP
           ▼
┌─────────────────────┐
│  Python Flask       │  Port 5000
│  Web Server         │
└──────────┬──────────┘
           │ HTTP/JSON
           ▼
┌─────────────────────┐
│  C++ Poker Engine   │  Port 8080
│  (Stateless API)    │
└─────────────────────┘
```

The Python website serves the user interface and proxies requests to the C++ backend, which handles the computational work of simulating poker game states.

## Quick Start

### 1. Start the C++ API Server

```bash
cd api
make
./poker_api
```

You should see: `🚀 C++ API Server started on http://localhost:8080`

### 2. Start the Python Web Server

In a new terminal:

```bash
cd website
./start.sh
```

Or manually:
```bash
cd website
uv sync
uv run python app.py
```

You should see: `Running on http://0.0.0.0:5000`

### 3. Open the Website

Navigate to `http://localhost:5000` in your browser.

## Project Structure

```
pokersim/
├── README.md              # This file
├── run_tests.sh           # Master test script
├── api/                   # C++ API server and shared Python package
│   ├── src/               # C++ source files
│   ├── tests/             # Test files
│   ├── python/pokersim/   # Shared Python package (see below)
│   ├── CMakeLists.txt     # CMake configuration
│   ├── Makefile           # Make configuration
│   └── README.md          # API & engine reference
├── training/              # RL training system
│   ├── train.py           # RL training script (PPO)
│   ├── eval.py            # Model evaluation script
│   ├── ppo.py             # PPO algorithm
│   ├── rl_model.py        # Re-exports from pokersim
│   ├── rl_state_encoder.py # Re-exports from pokersim
│   └── start_rl_training_optimized.sh # Training script
├── elo/                   # ELO Rating Arena
│   ├── engine.py          # ELO calculation and match logic
│   ├── server.py          # Web server with real-time UI
│   └── start.sh           # Quick start script
├── playground/            # Interactive poker playground
│   ├── app.py             # Flask application
│   ├── start.sh           # Quick start script
│   └── templates/
│       └── index.html     # Poker table UI
└── hand-viewer/           # Hand history viewer
    ├── server.py          # Flask application
    └── start.sh           # Quick start script
```

### Shared Python Package (`api/python/pokersim/`)

The core Python components are centralized in `api/python/pokersim/`:

- `config.py` - Shared configuration constants (action space, model version)
- `model.py` - Neural network architecture (PokerActorCritic)
- `state_encoder.py` - State encoding for RL (RLStateEncoder)
- `agents.py` - Agent implementations (ModelAgent, HeuristicAgent, etc.)
- `utils.py` - Utility functions (action conversion, state extraction)
- `gameplay.py` - Game runner for matches (PokerGameRunner)
- `reward_shaping.py` - Reward shaping for training
- `checkpoint_utils.py` - Model checkpoint management
- `hand_logger.py` - Hand history logging

## Features

- ✅ **Complete Texas Hold'em Engine**: Hand evaluation, side pots, game flow (C++)
- ✅ **Stateless API**: Deterministic simulation based on seed + history
- ✅ **Modern Web UI**: Flask-based interface for interaction
- ✅ **Reinforcement Learning**: PPO agent training with self-play
- ✅ **High Performance**: C++ backend for sub-millisecond simulations

## Reinforcement Learning Training

Train a neural network to learn poker strategy using Proximal Policy Optimization (PPO) and self-play.

### Quick Start (Optimized)

```bash
cd training
./start_rl_training_optimized.sh
```

This uses recommended hyperparameters:
- **Heads-up play** (2 players) for simpler learning
- **Learning rate** 3e-4 with cosine annealing
- **Entropy bonus** 0.02 for exploration
- **Normalized rewards** for stability

### Manual Training

```bash
cd training
uv sync
uv run python train.py --iterations 5000 --episodes-per-iter 50 --num-players 2
```

### Evaluation

To evaluate a trained model against random agents:

```bash
uv run python eval.py --model /path/to/model.pt --num-hands 100
```

### ELO Rating System

The ELO system provides a real-time web interface for evaluating model checkpoints:

```bash
cd elo
./start.sh
```

Open `http://localhost:5051` in your browser to access the ELO rating UI.

#### Features

- **Real-time ELO updates** - Watch ratings change as matches complete
- **Live statistics** - Win rate, rating history, and match counts
- **Round-robin matches** - All checkpoints play against each other
- **Baseline comparisons** - Compare against Random and Heuristic agents
- **Interactive charts** - Visualize rating progression over matches

### Playground (Play Against AI)

The Playground provides an interactive web interface to play poker against trained AI models:

```bash
cd playground
./start.sh
```

Open `http://localhost:5001` in your browser to play heads-up poker against any trained checkpoint.

### Training Opponent Pool

The RL model trains against a diverse pool of opponents to learn robust strategies:

| Agent Type | Strategy | Purpose |
|------------|----------|---------|
| `CallingStation` | Calls most bets, especially all-ins | Punishes weak all-in plays |
| `HeroCaller` | Calls down suspected bluffs | Teaches that bluffing has limits |
| `TightAgent` | Only plays premium hands | Punishes over-aggression |
| `HeuristicAgent` | Rule-based strategic play | Provides a baseline opponent |
| `AggressiveAgent` | Frequent bets and raises | Teaches calling down bluffs |
| `LoosePassive` | Calls too much, rarely raises | Rewards value betting |
| `AlwaysCall` | Always checks or calls | Tests thin value bets |
| `AlwaysRaise` | Always bets/raises max | Tests handling hyper-aggression |
| `AlwaysFold` | Folds to any bet | Tests blind stealing |
| `RandomAgent` | Random valid actions | Baseline exploration |

Self-play against past model checkpoints (15%) and the current model (10%) is also used.

### Convergence Notes
- **Short-term (<500 iters)**: Win rate fluctuates but avoids 0%.
- **Medium-term (500-2000 iters)**: Win rate stabilizes around 45-55%.
- **Long-term (>2000 iters)**: Consistent performance against random opponents (EV > 0.7).

## Website Configuration

### Environment Variables
- `PORT`: Web server port (default: 5000)
- `API_HOST`: C++ API hostname (default: localhost)
- `API_PORT`: C++ API port (default: 8080)

Example:
```bash
PORT=3000 API_HOST=localhost API_PORT=8080 uv run python app.py
```

### Production Mode
For production, use a WSGI server like Gunicorn:
```bash
uv run gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Testing

The project includes comprehensive test suites. Run all tests with:

```bash
./run_tests.sh
```

This runs:
1. **C++ Unit Tests**: Card, Deck, Hand, Player, Game
2. **API Snapshot Tests**: Validates complex scenarios (side pots, all-ins) via Python
3. **RL Training Tests**: Unit tests for PPO and state encoder
4. **ELO Tests**: Tests for the ELO rating system

## Performance

The C++ backend provides sub-millisecond response times. 

**Note on RL Training**: The training loop uses direct C++ bindings (pybind11) for high-performance communication with the poker engine, eliminating network overhead.

## Development

### Prerequisites
- **Python 3.9+** with [uv](https://github.com/astral-sh/uv)
- **C++17 Compiler** (g++ 7+ or clang 5+)
- **Make** or **CMake**

### Why uv?
We use `uv` for Python dependency management because it is significantly faster than pip, handles virtual environments automatically, and provides deterministic builds via `uv.lock`.

## License

This project is provided as-is for poker simulation purposes.
