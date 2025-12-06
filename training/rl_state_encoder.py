"""
State Encoding for RL Poker Agent.

This module re-exports state encoding classes and functions from the pokersim package
for backwards compatibility. All implementations are now centrally defined in
/api/python/pokersim/state_encoder.py.
"""

# Add api/python to path for pokersim package
import sys
from pathlib import Path
_API_PYTHON_DIR = Path(__file__).parent.parent / "api" / "python"
if str(_API_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_API_PYTHON_DIR))

# Re-export everything from pokersim.state_encoder for backwards compatibility
from pokersim.state_encoder import (
    encode_card,
    estimate_preflop_strength,
    estimate_hand_strength,
    get_hand_category,
    ActionHistory,
    RLStateEncoder,
)

# Also export the config constants that were previously in this module
from pokersim.config import (
    FEATURE_DIM,
    RANK_MAP,
    SUIT_MAP,
    STAGE_MAP,
    ACTION_TYPE_MAP,
)

__all__ = [
    # Functions
    'encode_card',
    'estimate_preflop_strength',
    'estimate_hand_strength',
    'get_hand_category',
    # Classes
    'ActionHistory',
    'RLStateEncoder',
    # Constants
    'FEATURE_DIM',
    'RANK_MAP',
    'SUIT_MAP',
    'STAGE_MAP',
    'ACTION_TYPE_MAP',
]
