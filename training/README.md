# Poker RL Training

Generate reinforcement learning training data by simulating complete poker hands (rollouts), then train and evaluate neural network models.

## Table of Contents

- [Quick Start](#quick-start)
- [Data Generation](#data-generation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Continuous Training System](#continuous-training-system)
- [Troubleshooting](#troubleshooting)

## Quick Start

**Prerequisites:** Python 3.9+, uv, and C++ API server (see [`../api/README.md`](../api/README.md))

```bash
# Install dependencies
uv pip install -r requirements.txt

# Start API server
cd ../api && make && ./build/poker_api 8080

# Generate training data
cd ../training
uv run python generate_rollouts.py --num-rollouts 1000 --agent-type mixed

# Train a model
uv run python train.py --data /tmp/pokersim/data/rollouts.json --epochs 100

# Evaluate the model
uv run python eval.py --model /tmp/pokersim/models/poker_model.pt --data /tmp/pokersim/data/rollouts.json
```

## Complete Workflow

1. **Generate Data** → 2. **Train Model** → 3. **Evaluate Model** → 4. **Use for Inference**

## Data Generation

### What is Generated

Each **rollout** is a complete poker hand containing:
- **States**: Game observations (hole cards, community cards, pot, chips, position)
- **Actions**: Agent decisions (fold, check, call, bet, raise, all-in)
- **Rewards**: Final outcomes (chips won/lost)

Use with RL algorithms: Policy Gradient (REINFORCE, PPO), DQN, Actor-Critic, or CFR.

### Generate Training Data

**Custom settings:**

```bash
uv run python generate_rollouts.py \
  --num-rollouts 1000 \
  --num-players 3 \
  --small-blind 25 \
  --big-blind 50 \
  --starting-chips 2000 \
  --seed 42
```

**Agent types:** `random`, `call`, `tight`, `aggressive`, `mixed`

```bash
uv run python generate_rollouts.py --agent-type tight --num-rollouts 500
```

**Configuration presets:**

```python
from config import get_preset
config = get_preset('quick_test')  # Also: heads_up, full_ring, tournament, high_stakes
```

### Output Format

```json
{
  "rollout_id": 0,
  "game_seed": 42,
  "states": [{"player_id": "p1", "hole_cards": ["AS", "KH"], ...}],
  "actions": [{"type": "playerAction", "action": "raise", "amount": 60}],
  "rewards": {"p1": 50, "p2": -50}
}
```

### Custom Agents

```python
class MyAgent(Agent):
    def select_action(self, state, legal_actions):
        return ('raise', 100) if 'raise' in legal_actions else ('call', 0)
```

## Model Training

### Train a Model

Train a neural network to predict actions from game states:

```bash
uv run python train.py --data /tmp/pokersim/data/rollouts.json --epochs 100
```

This will:
- Load and encode poker states into feature vectors
- Create a neural network (PokerNet)
- Train it to predict actions
- Save the best model to `/tmp/pokersim/models/poker_model.pt`

**Key hyperparameters:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--hidden-dim`: Hidden layer size (default: 256)

### Feature Encoding

States are encoded into ~150-dimensional feature vectors:

- **Hole cards** (34 features): 2 cards × (13 ranks + 4 suits)
- **Community cards** (85 features): 5 cards × 17 features
- **Chip values** (5 features): pot, current bet, player chips (normalized)
- **Stage** (5 features): Preflop/Flop/Turn/River/Complete (one-hot)
- **Position** (6 features): num players, position, dealer/blinds

### Model Architecture

The `PokerNet` is a feedforward neural network:

```
Input (150) → FC(256) → ReLU → Dropout 
           → FC(128) → ReLU → Dropout
           → FC(64) → ReLU
           → FC(6) → Softmax
```

Outputs probability distribution over 6 actions:
- fold, check, call, bet, raise, all_in

### Training Process

1. **Data Loading**: Load rollouts from JSON
2. **Encoding**: Convert states to feature vectors
3. **Batching**: Create mini-batches for efficient training
4. **Optimization**: Use Adam optimizer with cross-entropy loss
5. **Validation**: Track performance on held-out data
6. **Checkpointing**: Save best model

## Model Evaluation

Evaluate the trained model on test data:

```bash
uv run python eval.py --model /tmp/pokersim/models/poker_model.pt --data /tmp/pokersim/data/rollouts.json
```

This will show:
- Overall accuracy
- Per-action performance
- Confusion matrix
- Analysis and recommendations

## Advanced Usage

### Custom Training

```python
from train import PokerNet, PokerDataset, encode_state
import torch

# Load your data
with open('/tmp/pokersim/data/rollouts.json') as f:
    rollouts = json.load(f)

# Create dataset
dataset = PokerDataset(rollouts)

# Custom model
model = PokerNet(input_dim=dataset.get_feature_dim(), hidden_dim=512)

# Train with your own loop
# ...
```

### Tips for Better Models

#### More and Better Data

- Generate 5,000-10,000 rollouts for better performance
- Use `--agent-type mixed` for diverse training data
- Include multiple player counts (2, 3, 6 players)

```bash
uv run python generate_rollouts.py --num-rollouts 5000 --agent-type mixed
```

#### Tune Hyperparameters

- Increase `--hidden-dim` to 512 or 1024 for more capacity
- Train for more epochs (100-200)
- Adjust learning rate (`--lr 0.0001` for slower, stabler training)

```bash
uv run python train.py --epochs 200 --hidden-dim 512 --lr 0.0001
```

#### Improve Features

Edit `encode_state()` in `train.py` to add:
- Hand strength indicators
- Pot odds calculations
- Opponent modeling features
- Betting history

#### Better Evaluation

- Test against multiple opponent types
- Track performance over many games (1000+)
- Analyze specific situations (preflop vs postflop)

## Continuous Training System

For long-running training that automatically generates data and trains models in an infinite loop:

### Quick Start

```bash
./start_continuous_training.sh
```

Press `Ctrl+C` to stop gracefully.

### What It Does

The system runs three concurrent processes:

1. **Data Generation** - Generates poker game rollouts in batches
2. **Model Training** - Trains neural networks on accumulated data  
3. **Periodic Evaluation** - Tests model performance vs baseline agents

By default, the system **accumulates all data** and trains on increasingly large datasets for better learning.

### Configuration

**Common options:**

```bash
# Evaluate more frequently with more hands
./start_continuous_training.sh --eval-interval 3 --num-eval-hands 500

# Skip evaluation (faster training)
./start_continuous_training.sh --skip-eval

# Custom data directory
./start_continuous_training.sh --data-dir ./my_data

# Limit data files to save disk space
./start_continuous_training.sh --max-data-files 5
```

**Advanced configuration:**

```bash
uv run python continuous_trainer.py \
    --batch-size 5000 \
    --epochs-per-cycle 50 \
    --max-data-files 10 \
    --cpu-threshold 0.80 \
    --memory-threshold 0.85
```

### Resource Management

The system automatically:
- Throttles data generation when CPU/memory is high
- Rotates old data files to limit disk usage
- Sets low priority for data generation to avoid impacting training
- Monitors and auto-restarts the API server if needed

### Model Outputs

Models are saved to `/tmp/pokersim/models/`:
- `poker_model.pt` - Latest model
- `poker_model_YYYYMMDD_HHMMSS.pt` - Timestamped checkpoints

### Evaluate Trained Models

```bash
cd training
uv run python eval.py \
    --model /tmp/pokersim/models/poker_model.pt \
    --num-hands 1000 \
    --num-players 6
```

## Troubleshooting

**Cannot connect to API server?**
```bash
cd ../api && ./build/poker_api 8080
```

**Port in use?**
```bash
./build/poker_api 8081
# Then: uv run python generate_rollouts.py --api-url http://localhost:8081/simulate
```

**ImportError: No module named 'torch'**
```bash
uv pip install torch
```

**Model file not found**
- Make sure you've run `train.py` first
- Check the path: default is `/tmp/pokersim/models/poker_model.pt`

**Low accuracy (<30%)**
- Generate more training data
- Train for more epochs
- Check data quality (diverse agents)

**Out of memory**
- Reduce batch size: `--batch-size 16`
- Reduce model size: `--hidden-dim 128`

## Performance

Data generation: ~20 rollouts/sec • 1K rollouts in ~50sec • 10K rollouts in ~8min

## Next Steps

1. **Implement better RL algorithms**: PPO, DQN, Actor-Critic
2. **Add opponent modeling**: Track and predict opponent behavior
3. **Multi-agent training**: Self-play and population-based training
4. **Advanced features**: Hand strength, pot odds, position play
5. **Hyperparameter tuning**: Grid search or Bayesian optimization

See the main project README for more information.
