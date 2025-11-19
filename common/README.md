# Pokersim Common

Shared utilities for poker AI training and evaluation systems.

This package consolidates simulation logic that is shared between:
- **`/training`**: RL training system for poker AI agents
- **`/elo`**: ELO rating and evaluation system

## Architecture

```
/common
├── config.py          # Configuration constants (MODEL_VERSION, paths)
├── model_agent.py     # Agent classes and action conversion
├── rl_state_encoder.py # State encoding for RL
├── rl_model.py        # Neural network architecture (PokerActorCritic)
└── simulation.py      # Core simulation engine (SHARED)
```

The key consolidation is in `simulation.py`, which provides the core game
simulation logic used by both training and ELO evaluation.

## Components

### config.py
Configuration constants used across the project:
- `MODEL_VERSION`: Current model version for checkpoints
- `DEFAULT_MODELS_DIR`: Default directory for model storage
- `LOG_LEVEL`: Logging verbosity level

### model_agent.py
Agent classes and action conversion:
- Action space definitions (`ACTION_MAP`, `ACTION_NAMES`, etc.)
- `convert_action_label()`: Convert action labels to game actions
- `create_legal_actions_mask()`: Create legal action masks for model
- `extract_state()`: Extract player state from game state
- Agent classes: `ModelAgent`, `RandomAgent`, `HeuristicAgent`, `TightAgent`,
  `AggressiveAgent`, `LoosePassiveAgent`, `CallingStationAgent`, `HeroCallerAgent`,
  `AlwaysRaiseAgent`, `AlwaysCallAgent`, `AlwaysFoldAgent`

### rl_state_encoder.py
State encoding for reinforcement learning:
- Card/suit/stage mappings (`RANK_MAP`, `SUIT_MAP`, `STAGE_MAP`)
- `estimate_hand_strength()`: Hand strength estimation
- `estimate_preflop_strength()`: Preflop-specific strength
- `RLStateEncoder`: Full state encoder with opponent modeling

### rl_model.py
Neural network architecture:
- `PokerActorCritic`: Transformer-based actor-critic model
- `create_actor_critic()`: Factory function for model creation

### simulation.py
**Core poker game simulation engine shared between training and ELO evaluation:**

- `GameConfig`: Configuration dataclass for poker games
  - Supports both snake_case and camelCase configuration
  - `to_dict()`: Convert to API format
  - `from_dict()`: Create from dict (supports both formats)
  - `to_binding_config()`: Create C++ binding config

- `PokerSimulator`: Stateless JSON-API based simulator
  - Used by ELO evaluation for hand simulation
  - `play_hand()`: Play a single hand with agents
  - `play_match()`: Play multiple hands between agents

- `DirectGameSimulator`: Direct C++ binding wrapper
  - Used by training for high-performance simulation
  - `create_game()`: Create new game instance
  - `process_action()`: Process player action

- `HandLogger`: Logging hands with full action history
  - Used for debugging and hand review

- Helper functions:
  - `call_poker_api()`: Canonical way to call poker API
  - `create_binding_config()`: Create C++ binding config from GameConfig
  - `check_binding_available()`: Check if binding is installed

## Usage

### Basic Simulation (ELO-style)

```python
from common import (
    ModelAgent,
    RandomAgent,
    GameConfig,
    PokerSimulator,
)

# Create a simulator
config = GameConfig(
    num_players=2,
    small_blind=10,
    big_blind=20,
    starting_chips=1000
)
simulator = PokerSimulator(config)

# Play a hand between agents
agents = {
    'p0': ModelAgent('p0', 'Model', model_path='/path/to/model.pt'),
    'p1': RandomAgent('p1', 'Random')
}
result = simulator.play_hand(agents)
print(f"Profits: {result['profits']}")
```

### High-Performance Training

```python
from common import (
    GameConfig,
    create_binding_config,
    DirectGameSimulator,
)
import poker_api_binding

# Use direct binding for maximum performance
config = GameConfig(small_blind=10, big_blind=20, starting_chips=1000)
binding_config = create_binding_config(config, seed=12345)

game = poker_api_binding.Game(binding_config)
game.add_player('p0', 'Agent', config.starting_chips)
game.add_player('p1', 'Opponent', config.starting_chips)
game.start_hand()

# Fast state access (no JSON serialization)
state = game.get_state_dict()
game.process_action('p0', 'raise', 50)
```

### With Hand Logging

```python
from common import (
    PokerSimulator,
    GameConfig,
    HandLogger,
    DEFAULT_HAND_LOGS_DIR,
)

config = GameConfig(num_players=2, small_blind=10, big_blind=20)
simulator = PokerSimulator(config)
hand_logger = HandLogger(logs_dir=DEFAULT_HAND_LOGS_DIR, log_frequency=10)

result = simulator.play_hand(
    agents=agents,
    hand_logger=hand_logger,
    agent_configs={'p0': {'type': 'model', 'path': '...'}, 'p1': {'type': 'random'}}
)
```

## Dependencies

This package requires:
- `torch>=2.0.0`: For neural network operations
- `numpy>=1.24.0`: For numerical operations
- `poker_api_binding`: C++ binding (compile with `cd api && make module`)

