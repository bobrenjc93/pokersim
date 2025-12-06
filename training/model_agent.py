"""
Model-based Agent for Poker.

This module re-exports agent classes and utilities from the pokersim package
for backwards compatibility. All implementations are now centrally defined in
/api/python/pokersim/.
"""

# Add api/python to path for pokersim package
import sys
from pathlib import Path
_API_PYTHON_DIR = Path(__file__).parent.parent / "api" / "python"
if str(_API_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_API_PYTHON_DIR))

# Re-export everything from pokersim for backwards compatibility
from pokersim.config import (
    ACTION_MAP,
    ACTION_NAMES,
    NUM_ACTIONS,
    RAISE_SIZE_MAP,
    BET_SIZE_MAP,
    FEATURE_DIM,
    DEFAULT_MODELS_DIR,
)

from pokersim.utils import (
    convert_action_label,
    extract_state,
    create_legal_actions_mask,
)

from pokersim.agents import (
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

from pokersim.checkpoint_utils import (
    parse_checkpoints,
    select_spread_checkpoints,
    get_latest_checkpoint,
    get_checkpoint_iteration,
)

__all__ = [
    # Constants
    'ACTION_MAP',
    'ACTION_NAMES',
    'NUM_ACTIONS',
    'RAISE_SIZE_MAP',
    'BET_SIZE_MAP',
    'FEATURE_DIM',
    'DEFAULT_MODELS_DIR',
    # Functions
    'convert_action_label',
    'extract_state',
    'create_legal_actions_mask',
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
    # Checkpoint utilities
    'parse_checkpoints',
    'select_spread_checkpoints',
    'get_latest_checkpoint',
    'get_checkpoint_iteration',
]
