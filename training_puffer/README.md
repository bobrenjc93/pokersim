# Vectorized Poker Training (CleanRL-style)

This directory contains a CleanRL-style implementation of reinforcement learning training for the poker AI using vectorized environments.

## Overview

This implementation provides efficient vectorized environment training with PPO (Proximal Policy Optimization). Based on the clean, readable CleanRL style, it:

- Uses the C++ `poker_api_binding` for fast poker game simulation
- Provides a Gymnasium-compatible environment interface
- Supports parallel training with sync and async vectorized environments
- Uses a transformer-based policy network
- Includes TensorBoard logging for monitoring
- No external RL library dependencies (just PyTorch + Gymnasium)

## Prerequisites

1. **Build the C++ poker bindings**:
   ```bash
   cd ../api
   make module
   cp build/poker_api_binding.cpython-*.so ../training_puffer/
   ```

2. **Install dependencies**:
   ```bash
   cd ../training_puffer
   uv sync
   ```

## Usage

### Quick Start

Run training with default settings:

```bash
./start_training.sh
```

### Custom Training

Use command-line arguments to customize training:

```bash
./start_training.sh \
    --total-timesteps 10000000 \
    --num-envs 64 \
    --learning-rate 0.0003 \
    --hidden-dim 256 \
    --num-layers 4
```

### Resume from Checkpoint

```bash
./start_training.sh --checkpoint /tmp/pokersim/puffer_models_v1/latest.pt
```

### Run Training Directly

```bash
uv run python train.py --total-timesteps 1000000 --num-envs 32
```

## Architecture

### Environment (`poker_env.py`)

- `PokerEnv`: Gymnasium-compatible poker environment
- Uses C++ bindings for fast game simulation
- Handles opponent moves internally (single-agent view)
- Includes diverse opponent pool (Random, CallingStation, Tight, Aggressive)

### Policy (`policy.py`)

- `PokerPolicy`: Transformer-based actor-critic network
- Token-based input processing (cards, pot, position, etc.)
- Action masking for legal action enforcement
- Separate actor and critic heads

### Training (`train.py`)

- CleanRL-style PPO implementation
- Supports learning rate annealing
- GAE (Generalized Advantage Estimation)
- TensorBoard logging
- Automatic checkpointing

## Configuration

Key hyperparameters can be adjusted in `config.py` or via command-line:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_timesteps` | 10,000,000 | Total training timesteps |
| `num_envs` | 64 | Parallel environments |
| `learning_rate` | 3e-4 | Adam learning rate |
| `hidden_dim` | 256 | Transformer hidden dimension |
| `num_layers` | 4 | Transformer layers |
| `num_heads` | 8 | Attention heads |
| `clip_coef` | 0.2 | PPO clipping coefficient |
| `ent_coef` | 0.01 | Entropy coefficient |

## Monitoring

### TensorBoard

Start TensorBoard to monitor training:

```bash
tensorboard --logdir=/tmp/pokersim/tensorboard_puffer_v1
```

Then open http://localhost:6006 in your browser.

### Metrics

Key metrics tracked:
- `charts/avg_reward`: Average episode reward
- `charts/win_rate`: Win rate against opponents
- `losses/policy_loss`: PPO policy loss
- `losses/value_loss`: Value function loss
- `losses/entropy`: Policy entropy

## Comparison with Original Training

| Feature | Original (`training/`) | Vectorized (`training_puffer/`) |
|---------|----------------------|-------------------------------|
| RL Library | Custom PPO | CleanRL-style PPO |
| Vectorization | Custom `VecPokerEnv` | Sync/Async VecEnv |
| Environment | API-based | Direct Gymnasium interface |
| Opponent Pool | Extensive (10+ types) | Core types (4) |
| Features | PopArt, Adaptive Entropy | Standard PPO |
| Dependencies | torch, accelerate | torch, gymnasium only |
| Code Complexity | High | Low |

## File Structure

```
training_puffer/
├── config.py          # Configuration constants
├── poker_env.py       # Gymnasium environment
├── policy.py          # Neural network policy
├── train.py           # Training script
├── start_training.sh  # Shell entry point
├── pyproject.toml     # Dependencies
└── README.md          # This file
```

## Extending

### Adding New Opponents

Add new agent classes to `poker_env.py`:

```python
class MyAgent:
    def __init__(self):
        pass
    
    def reset_hand(self):
        pass
    
    def select_action(self, state, legal_actions):
        # Return (action_type, amount)
        return 'call', 0
```

Then add to `create_opponent_pool()`.

### Modifying the Policy

Edit `policy.py` to change the neural network architecture. The key methods are:
- `encode()`: Processes observations
- `get_action_and_value()`: Main inference method
- `get_value()`: Value estimation only

## Troubleshooting

### "poker_api_binding not found"

Build and copy the C++ bindings:
```bash
cd ../api && make module
cp build/poker_api_binding.cpython-*.so ../training_puffer/
```

### "ModuleNotFoundError" for dependencies

Install dependencies:
```bash
uv sync
```

### Training is slow

- Increase `--num-envs` (more parallel environments)
- Reduce `--hidden-dim` (smaller model)
- Use GPU with `--cuda`

### Training diverges

- Reduce `--learning-rate`
- Increase `--ent-coef` for more exploration
- Check for illegal action masking issues

