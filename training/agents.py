#!/usr/bin/env python3
"""Opponent Selection and Agent Caching for Training."""

import random
from typing import Any, Dict, List, Optional

import torch

from common import create_agent, ModelCache

# Training opponent distribution (must sum to ~1.0)
# IMPORTANT: Avoid opponents that enable degenerate strategies:
# - No always_fold (teaches "raise = instant win" → fold collapse)
# - Limited always_raise (can lead to raise-only spirals)
# - Higher weight on calling_station/hero_caller (forces model to play hands out)
OPPONENT_WEIGHTS = {
    'calling_station': 0.28,  # Forces showdowns - critical for learning hand value
    'hero_caller': 0.18,      # Similar - calls with decent hands
    'tight': 0.15,            # Folds bad hands, plays strong ones normally
    'heuristic': 0.12,        # Balanced play
    'aggressive': 0.08,       # Teaches defense against aggression
    'past_model': 0.08,       # Reduced: avoid self-play spiral early in training
    'model': 0.05,            # Reduced: same reason
    'loose_passive': 0.03,
    'random': 0.03,
    # Removed: always_fold, always_call, always_raise - degenerate opponents
}


def select_opponent_type(
    has_pool: bool = False,
    rng: Optional[random.Random] = None,
    num_checkpoints: int = 0,
) -> str:
    """Select opponent type using weighted random selection.

    If `rng` is provided, it is used for deterministic selection (useful for
    reproducible training runs).
    
    `num_checkpoints` controls self-play probability: we reduce self-play
    early in training when checkpoints may have degenerate policies.
    """
    # Make a copy of weights to modify
    weights_dict = dict(OPPONENT_WEIGHTS)
    
    # Reduce self-play early in training to avoid reinforcing bad policies
    # Gradually increase as we accumulate more (hopefully better) checkpoints
    if num_checkpoints < 5:
        # Very early: minimal self-play
        weights_dict['past_model'] = 0.02
        weights_dict['model'] = 0.02
        # Redistribute to calling_station (forces showdowns)
        weights_dict['calling_station'] = 0.35
    elif num_checkpoints < 20:
        # Early-mid: moderate self-play
        weights_dict['past_model'] = 0.05
        weights_dict['model'] = 0.03
        weights_dict['calling_station'] = 0.32
    # else: use default weights
    
    types = [k for k in weights_dict if k != 'past_model' or has_pool]
    weights = [weights_dict[k] for k in types]
    chooser = rng or random
    return chooser.choices(types, weights=weights)[0]


class AgentCache:
    """LRU cache for heuristic agents and model checkpoints."""
    
    def __init__(self, device: torch.device = None, max_models: int = 10):
        self._agents: Dict[str, Any] = {}
        self._models = ModelCache(device=device, max_size=max_models)
    
    def get_agent(self, agent_type: str, player_id: str) -> Optional[Any]:
        """Get or create a heuristic agent (cached)."""
        if agent_type in ('model', 'past_model'):
            return None
        key = f"{agent_type}_{player_id}"
        if key not in self._agents:
            self._agents[key] = create_agent(agent_type, player_id)
        return self._agents.get(key)
    
    def get_model(self, path: str, config: Dict[str, Any]) -> Optional[torch.nn.Module]:
        """Load and cache an opponent model checkpoint."""
        return self._models.get(path, config)
    
    def clear(self):
        """Clear all caches."""
        self._agents.clear()
        self._models.clear()
