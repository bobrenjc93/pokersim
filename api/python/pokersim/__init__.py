"""
pokersim - Python package for poker simulation and AI training.

This package provides:
- Configuration constants (action space, feature dimensions)
- Utility functions (state extraction, action conversion)
- State encoding (RLStateEncoder)
- Neural network models (PokerActorCritic)
- Agent implementations (ModelAgent, HeuristicAgent, etc.)
- Gameplay utilities (PokerGameRunner for running matches)
- Hand logging (HandLogger for saving hand histories)
"""

# Configuration constants
from .config import (
    MODEL_VERSION,
    DEFAULT_MODELS_DIR,
    ACTION_MAP,
    ACTION_NAMES,
    NUM_ACTIONS,
    RAISE_SIZE_MAP,
    BET_SIZE_MAP,
    FEATURE_DIM,
    RANK_MAP,
    SUIT_MAP,
    STAGE_MAP,
    ACTION_TYPE_MAP,
    LOG_LEVEL,
    # Game configuration defaults
    DEFAULT_SMALL_BLIND,
    DEFAULT_BIG_BLIND,
    DEFAULT_STARTING_CHIPS,
    DEFAULT_MAX_HANDS_PER_ROUND,
    DEFAULT_ROUNDS_PER_MATCH,
    DEFAULT_MAX_RAISES_PER_ROUND,
)

# Utility functions
from .utils import (
    convert_action_label,
    extract_state,
    create_legal_actions_mask,
)

# State encoding
from .state_encoder import (
    encode_card,
    estimate_preflop_strength,
    estimate_hand_strength,
    get_hand_category,
    ActionHistory,
    RLStateEncoder,
)

# Model architecture
from .model import (
    PokerActorCritic,
    create_actor_critic,
)

# Agents
from .agents import (
    ModelAgent,
    RandomAgent,
    HeuristicAgent,
    TightAgent,
    LoosePassiveAgent,
    AggressiveAgent,
    CallingStationAgent,
    HeroCallerAgent,
    SimpleAgent,
    AlwaysRaiseAgent,
    AlwaysCallAgent,
    AlwaysFoldAgent,
    load_model_agent,
)

# Gameplay utilities
from .gameplay import (
    PokerAgent,
    GameConfig,
    HandResult,
    RoundResult,
    MatchResult,
    PokerGameRunner,
)

# Hand logging
from .hand_logger import (
    HandLogger,
    HAND_LOGS_DIR,
    HAND_LOG_FREQUENCY,
)

# Checkpoint utilities
from .checkpoint_utils import (
    parse_checkpoints,
    select_spread_checkpoints,
    get_latest_checkpoint,
    get_checkpoint_iteration,
)

# Reward shaping
from .reward_shaping import (
    compute_action_shaping_reward,
)


__all__ = [
    # Config
    'MODEL_VERSION',
    'DEFAULT_MODELS_DIR',
    'ACTION_MAP',
    'ACTION_NAMES',
    'NUM_ACTIONS',
    'RAISE_SIZE_MAP',
    'BET_SIZE_MAP',
    'FEATURE_DIM',
    'RANK_MAP',
    'SUIT_MAP',
    'STAGE_MAP',
    'ACTION_TYPE_MAP',
    'LOG_LEVEL',
    # Game configuration defaults
    'DEFAULT_SMALL_BLIND',
    'DEFAULT_BIG_BLIND',
    'DEFAULT_STARTING_CHIPS',
    'DEFAULT_MAX_HANDS_PER_ROUND',
    'DEFAULT_ROUNDS_PER_MATCH',
    'DEFAULT_MAX_RAISES_PER_ROUND',
    # Utils
    'convert_action_label',
    'extract_state',
    'create_legal_actions_mask',
    # State encoding
    'encode_card',
    'estimate_preflop_strength',
    'estimate_hand_strength',
    'get_hand_category',
    'ActionHistory',
    'RLStateEncoder',
    # Model
    'PokerActorCritic',
    'create_actor_critic',
    # Agents
    'ModelAgent',
    'RandomAgent',
    'HeuristicAgent',
    'TightAgent',
    'LoosePassiveAgent',
    'AggressiveAgent',
    'CallingStationAgent',
    'HeroCallerAgent',
    'SimpleAgent',
    'AlwaysRaiseAgent',
    'AlwaysCallAgent',
    'AlwaysFoldAgent',
    'load_model_agent',
    # Gameplay
    'PokerAgent',
    'GameConfig',
    'HandResult',
    'RoundResult',
    'MatchResult',
    'PokerGameRunner',
    # Hand logging
    'HandLogger',
    'HAND_LOGS_DIR',
    'HAND_LOG_FREQUENCY',
    # Checkpoint utilities
    'parse_checkpoints',
    'select_spread_checkpoints',
    'get_latest_checkpoint',
    'get_checkpoint_iteration',
    # Reward shaping
    'compute_action_shaping_reward',
]
