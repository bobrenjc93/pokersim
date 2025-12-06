"""
Actor-Critic Neural Network Architecture for Poker RL.

This module re-exports the model architecture from the pokersim package
for backwards compatibility. All implementations are now centrally defined in
/api/python/pokersim/model.py.
"""

# Add api/python to path for pokersim package
import sys
from pathlib import Path
_API_PYTHON_DIR = Path(__file__).parent.parent / "api" / "python"
if str(_API_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_API_PYTHON_DIR))

# Re-export everything from pokersim.model for backwards compatibility
from pokersim.model import (
    PokerActorCritic,
    create_actor_critic,
)

__all__ = [
    'PokerActorCritic',
    'create_actor_critic',
]
