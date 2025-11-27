# Poker Simulator Monorepo

A full-stack poker simulation system with a Python web interface, high-performance C++ API backend, and Reinforcement Learning training system.

## Project Overview

This monorepo contains:
- **Website**: Python Flask web server with modern UI for poker simulations
- **API**: Pure C++ API server for fast, stateless game simulations
- **Training**: Reinforcement learning system (PPO) for training poker AI agents

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
├── website/               # Python Flask web server
│   ├── app.py             # Flask application
│   ├── pyproject.toml     # Python dependencies
│   ├── start.sh           # Quick start script
│   └── templates/
│       └── index.html     # Web UI
├── api/                   # C++ API server
│   ├── src/               # Source files
│   ├── tests/             # Test files
│   ├── CMakeLists.txt     # CMake configuration
│   ├── Makefile           # Make configuration
│   └── README.md          # API & engine reference
└── training/              # RL training system
    ├── train.py           # RL training script (PPO)
    ├── eval.py            # Model evaluation script
    ├── ppo.py             # PPO algorithm
    ├── rl_model.py        # Neural network model
    ├── rl_state_encoder.py # State encoding
    └── start_rl_training_optimized.sh # Training script
└── arena/                 # AI vs AI Evaluation Arena
    ├── engine.py          # Core arena logic
    └── server.py          # Web server with real-time UI
```

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

### Advanced Evaluation (Arena)

The Arena provides a real-time web interface for evaluating model checkpoints:

```bash
cd arena
uv run python server.py
```

Open `http://localhost:5000` in your browser to access the evaluation UI.

#### Features

- **Real-time hand streaming** - Watch each hand play out live
- **Live statistics** - Win rate, BB/100, and hand count update in real-time
- **Round-robin tournaments** - All checkpoints play against each other
- **Baseline comparisons** - Compare against Random and Heuristic agents
- **Interactive charts** - Visualize performance progression over training

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
4. **Arena Tests**: Tests for the evaluation arena

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
