"""
Configuration for RL training.

This module re-exports shared constants from the pokersim package for backwards compatibility.
All constants are now centrally defined in /api/python/pokersim/config.py.
"""

# Add api/python to path for pokersim package
import sys
from pathlib import Path
_API_PYTHON_DIR = Path(__file__).parent.parent / "api" / "python"
if str(_API_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_API_PYTHON_DIR))

# Re-export everything from pokersim.config for backwards compatibility
from pokersim.config import (
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

__all__ = [
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
]
