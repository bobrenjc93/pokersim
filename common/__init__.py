"""
Common utilities shared between training and ELO evaluation systems.

This package provides:
- config: Configuration constants (MODEL_VERSION, DEFAULT_MODELS_DIR, etc.)
- model_agent: Agent classes and action conversion logic
- rl_state_encoder: State encoding for RL
- rl_model: Neural network architecture (PokerActorCritic)
- simulation: Core poker game simulation engine
"""

from .config import MODEL_VERSION, DEFAULT_MODELS_DIR, LOG_LEVEL

from .model_agent import (
    ACTION_MAP,
    ACTION_NAMES,
    NUM_ACTIONS,
    RAISE_SIZE_MAP,
    BET_SIZE_MAP,
    convert_action_label,
    create_legal_actions_mask,
    extract_state,
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

from .rl_state_encoder import (
    RANK_MAP,
    SUIT_MAP,
    STAGE_MAP,
    ACTION_TYPE_MAP,
    encode_card,
    estimate_preflop_strength,
    estimate_hand_strength,
    get_hand_category,
    ActionHistory,
    RLStateEncoder,
)

from .rl_model import (
    PokerActorCritic,
    create_actor_critic,
)

from .simulation import (
    GameConfig,
    PokerSimulator,
    DirectGameSimulator,
    HandLogger,
    DEFAULT_HAND_LOGS_DIR,
    DEFAULT_HAND_LOG_FREQUENCY,
    check_binding_available,
    create_binding_config,
    call_poker_api,
)

__all__ = [
    # config
    "MODEL_VERSION",
    "DEFAULT_MODELS_DIR",
    "LOG_LEVEL",
    # model_agent
    "ACTION_MAP",
    "ACTION_NAMES",
    "NUM_ACTIONS",
    "RAISE_SIZE_MAP",
    "BET_SIZE_MAP",
    "convert_action_label",
    "create_legal_actions_mask",
    "extract_state",
    "ModelAgent",
    "RandomAgent",
    "HeuristicAgent",
    "TightAgent",
    "LoosePassiveAgent",
    "AggressiveAgent",
    "CallingStationAgent",
    "HeroCallerAgent",
    "SimpleAgent",
    "AlwaysRaiseAgent",
    "AlwaysCallAgent",
    "AlwaysFoldAgent",
    "load_model_agent",
    # rl_state_encoder
    "RANK_MAP",
    "SUIT_MAP",
    "STAGE_MAP",
    "ACTION_TYPE_MAP",
    "encode_card",
    "estimate_preflop_strength",
    "estimate_hand_strength",
    "get_hand_category",
    "ActionHistory",
    "RLStateEncoder",
    # rl_model
    "PokerActorCritic",
    "create_actor_critic",
    # simulation
    "GameConfig",
    "PokerSimulator",
    "DirectGameSimulator",
    "HandLogger",
    "DEFAULT_HAND_LOGS_DIR",
    "DEFAULT_HAND_LOG_FREQUENCY",
    "check_binding_available",
    "create_binding_config",
    "call_poker_api",
]

