#!/usr/bin/env python3
"""Opponent Selection and Agent Caching for Training."""

import random
from typing import Any, Dict, List, Optional, Tuple

import torch

from common import create_agent, ModelCache

# Training opponent distribution (must sum to ~1.0)
# IMPORTANT: Avoid opponents that enable degenerate strategies:
# - No always_fold (teaches "raise = instant win" → fold collapse)
# - Limited always_raise (can lead to raise-only spirals)
# - Higher weight on calling_station/hero_caller (forces model to play hands out)
#
# KEY INSIGHT: Heavy self-play causes catastrophic forgetting. Later checkpoints
# learn to beat recent checkpoints but forget how to beat earlier ones. This
# manifests as ELO regression: iter_500 loses to iter_100 despite "improving".
#
# SOLUTION v3: Use PFSP (Prioritized Fictitious Self-Play) for self-play.
# PFSP automatically focuses training on opponents we're losing to, preventing
# forgetting while maintaining diverse opponent coverage.
OPPONENT_WEIGHTS = {
    'calling_station': 0.25,  # Forces showdowns - critical for learning hand value
    'hero_caller': 0.12,      # Similar - calls with decent hands  
    'tight': 0.12,            # Folds bad hands, plays strong ones normally
    'heuristic': 0.10,        # Balanced play
    'aggressive': 0.10,       # Teaches defense against aggression
    'pfsp_model': 0.15,       # PFSP-weighted checkpoint selection (NEW - primary self-play)
    'anchor_model': 0.08,     # Anchor checkpoints (early milestones)
    'past_model': 0.02,       # Random past checkpoint (reduced, PFSP handles this better)
    'model': 0.01,            # Current model (exploration) - MINIMAL
    'loose_passive': 0.02,
    'random': 0.03,           # Robustness
    # Removed: always_fold, always_call, always_raise - degenerate opponents
}

# Anchor checkpoint intervals - these are milestone checkpoints that represent
# fundamental skills learned at different training phases. We ALWAYS include
# training against these to prevent forgetting.
ANCHOR_INTERVALS = [1, 50, 100, 200, 300, 400, 500]


def select_opponent_type(
    has_pool: bool = False,
    rng: Optional[random.Random] = None,
    num_checkpoints: int = 0,
    has_anchors: bool = False,
    has_pfsp: bool = False,
) -> str:
    """Select opponent type using weighted random selection.

    If `rng` is provided, it is used for deterministic selection (useful for
    reproducible training runs).
    
    SOLUTION v3: Use PFSP (Prioritized Fictitious Self-Play) for self-play.
    PFSP automatically focuses training on opponents we're losing to, preventing
    forgetting while maintaining diverse opponent coverage.
    
    Key insight: PFSP gives higher selection probability to checkpoints the
    current model loses to, ensuring we always train against our weaknesses.
    """
    # Make a copy of weights to modify
    weights_dict = dict(OPPONENT_WEIGHTS)
    
    # Thresholds based on num_checkpoints (with save_interval=50):
    # <3 checkpoints = ~iterations 1-100, <6 = ~iterations 100-250, etc.
    
    if num_checkpoints < 3:
        # Phase 1 (iter 1-100): Pure heuristic learning, NO self-play
        # Critical for learning fundamental poker skills
        weights_dict['past_model'] = 0.0
        weights_dict['anchor_model'] = 0.0
        weights_dict['pfsp_model'] = 0.0
        weights_dict['model'] = 0.0
        weights_dict['calling_station'] = 0.35
        weights_dict['hero_caller'] = 0.20
        weights_dict['tight'] = 0.18
        weights_dict['heuristic'] = 0.12
        weights_dict['aggressive'] = 0.10
        weights_dict['loose_passive'] = 0.03
        weights_dict['random'] = 0.02
    elif num_checkpoints < 6:
        # Phase 2 (iter 100-250): Minimal self-play (~8%), mostly heuristics
        # Still building fundamentals, avoid catastrophic forgetting
        weights_dict['past_model'] = 0.01
        weights_dict['pfsp_model'] = 0.04 if has_pfsp else 0.0
        weights_dict['anchor_model'] = 0.03 if has_anchors else 0.0
        weights_dict['model'] = 0.0  # No current-model self-play yet
        weights_dict['calling_station'] = 0.32
        weights_dict['hero_caller'] = 0.18
        weights_dict['tight'] = 0.16
        weights_dict['heuristic'] = 0.12
        weights_dict['aggressive'] = 0.10
        weights_dict['loose_passive'] = 0.02
        weights_dict['random'] = 0.02
    elif num_checkpoints < 15:
        # Phase 3 (iter 250-700): Moderate self-play (~15%), PFSP starting
        weights_dict['past_model'] = 0.02
        weights_dict['pfsp_model'] = 0.08 if has_pfsp else 0.0
        weights_dict['anchor_model'] = 0.05 if has_anchors else 0.0
        weights_dict['model'] = 0.0
        weights_dict['calling_station'] = 0.28
        weights_dict['hero_caller'] = 0.16
        weights_dict['tight'] = 0.14
        weights_dict['heuristic'] = 0.12
        weights_dict['aggressive'] = 0.10
        weights_dict['loose_passive'] = 0.03
        weights_dict['random'] = 0.02
    elif num_checkpoints < 30:
        # Phase 4 (iter 700-1400): PFSP as primary self-play (~20%)
        weights_dict['past_model'] = 0.02
        weights_dict['pfsp_model'] = 0.12 if has_pfsp else 0.0
        weights_dict['anchor_model'] = 0.06 if has_anchors else 0.0
        weights_dict['model'] = 0.0
        weights_dict['calling_station'] = 0.26
        weights_dict['hero_caller'] = 0.14
        weights_dict['tight'] = 0.14
        weights_dict['heuristic'] = 0.10
        weights_dict['aggressive'] = 0.10
        weights_dict['loose_passive'] = 0.03
        weights_dict['random'] = 0.03
    else:
        # Phase 5 (iter 1400+): Higher self-play (~25%) with PFSP + anchors
        weights_dict['past_model'] = 0.02
        weights_dict['pfsp_model'] = 0.15 if has_pfsp else 0.0
        weights_dict['anchor_model'] = 0.08 if has_anchors else 0.0
        weights_dict['model'] = 0.0  # Still no current-model self-play
        weights_dict['calling_station'] = 0.22
        weights_dict['hero_caller'] = 0.14
        weights_dict['tight'] = 0.14
        weights_dict['heuristic'] = 0.10
        weights_dict['aggressive'] = 0.10
        weights_dict['loose_passive'] = 0.02
        weights_dict['random'] = 0.03
    
    # Build candidate types: exclude types that aren't available
    types = []
    for k in weights_dict:
        if k == 'past_model' and not has_pool:
            continue
        if k == 'anchor_model' and not has_anchors:
            continue
        if k == 'pfsp_model' and not has_pfsp:
            continue
        types.append(k)
    
    weights = [weights_dict[k] for k in types]
    chooser = rng or random
    return chooser.choices(types, weights=weights)[0]


def get_anchor_checkpoints(
    checkpoints: List[Tuple[int, str]],
) -> List[Tuple[int, str]]:
    """Extract anchor checkpoints from checkpoint list.
    
    Anchors are checkpoints at milestone iterations (iter_1, iter_50, iter_100, etc.)
    that represent fundamental skills. We force regular training against these.
    
    Args:
        checkpoints: List of (iteration, path) tuples
        
    Returns:
        List of anchor (iteration, path) tuples
    """
    if not checkpoints:
        return []
    
    anchors = []
    checkpoint_dict = {it: path for it, path in checkpoints}
    
    for anchor_iter in ANCHOR_INTERVALS:
        if anchor_iter in checkpoint_dict:
            anchors.append((anchor_iter, checkpoint_dict[anchor_iter]))
    
    return anchors


def select_anchor_checkpoint(
    anchors: List[Tuple[int, str]],
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """Select an anchor checkpoint with emphasis on earliest ones.
    
    Early anchors (iter_1, iter_50) get higher weight because they represent
    the most fundamental skills that are easiest to forget.
    
    Args:
        anchors: List of anchor (iteration, path) tuples
        rng: Optional random generator
        
    Returns:
        Selected anchor checkpoint path
    """
    if not anchors:
        return None
    
    chooser = rng or random
    
    if len(anchors) == 1:
        return anchors[0][1]
    
    # Weight heavily toward earliest anchors
    # iter_1 gets 5x weight, iter_50 gets 3x, iter_100 gets 2x, etc.
    weights = []
    for i, (iteration, _) in enumerate(anchors):
        if iteration <= 1:
            weight = 5.0
        elif iteration <= 50:
            weight = 3.0
        elif iteration <= 100:
            weight = 2.0
        elif iteration <= 200:
            weight = 1.5
        else:
            weight = 1.0
        weights.append(weight)
    
    selected = chooser.choices(anchors, weights=weights)[0]
    return selected[1]


def select_checkpoint_with_priority(
    checkpoints: List[Tuple[int, str]],
    current_iteration: int,
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """Select a NON-ANCHOR checkpoint for recent self-play.
    
    This is for the 'past_model' opponent type - playing against recent 
    checkpoints (not anchors). Since we're heavily limiting this, we focus
    on moderately-old checkpoints (50-200 iterations behind current).
    
    Args:
        checkpoints: List of (iteration, path) tuples, sorted by iteration
        current_iteration: Current training iteration
        rng: Optional random generator for reproducibility
        
    Returns:
        Selected checkpoint path, or None if no checkpoints available
    """
    if not checkpoints:
        return None
    
    chooser = rng or random
    
    # Filter out anchor checkpoints - those are handled separately
    anchor_iters = set(ANCHOR_INTERVALS)
    non_anchors = [(it, p) for it, p in checkpoints if it not in anchor_iters]
    
    if not non_anchors:
        # Fall back to anchors if no non-anchors available
        return checkpoints[0][1] if checkpoints else None
    
    if len(non_anchors) == 1:
        return non_anchors[0][1]
    
    # For recent self-play, favor checkpoints 50-200 iterations behind current
    # This provides a challenging opponent without overfitting to latest
    weights = []
    for iteration, _ in non_anchors:
        distance = current_iteration - iteration
        
        if 50 <= distance <= 200:
            # Sweet spot: challenging but not too old
            weight = 2.0
        elif 20 <= distance < 50:
            # Pretty recent - moderate weight
            weight = 1.5
        elif 200 < distance <= 400:
            # Moderately old - still useful
            weight = 1.0
        elif distance < 20:
            # Too recent - low weight
            weight = 0.5
        else:
            # Very old non-anchor - minimal weight
            weight = 0.3
        
        weights.append(weight)
    
    selected = chooser.choices(non_anchors, weights=weights)[0]
    return selected[1]


class PFSPOpponentSelector:
    """Prioritized Fictitious Self-Play opponent selector.
    
    Implements the PFSP algorithm from AlphaStar: opponents that the current
    model loses to more often get higher selection priority. This focuses
    training on weaknesses and prevents catastrophic forgetting.
    
    Priority formula: priority[opp] = (1 - win_rate[opp])^p
    where p=2 emphasizes low win-rate opponents.
    """
    
    def __init__(self, priority_exponent: float = 2.0, min_priority: float = 0.1):
        """Initialize PFSP selector.
        
        Args:
            priority_exponent: Power to raise (1 - win_rate) to. Higher = more emphasis on losses.
            min_priority: Minimum priority to ensure all opponents get some play.
        """
        self.priority_exponent = priority_exponent
        self.min_priority = min_priority
        
        # Track win/loss stats per opponent
        # Key: opponent identifier (checkpoint iteration or heuristic name)
        # Value: {'wins': int, 'losses': int, 'draws': int}
        self._stats: Dict[str, Dict[str, int]] = {}
        
        # Cached priorities (recomputed when stats change)
        self._priorities: Dict[str, float] = {}
        self._dirty = True
    
    def update_stats(self, opponent_id: str, win_rate: float, rounds_played: int = 1) -> None:
        """Update win/loss statistics for an opponent.
        
        Args:
            opponent_id: Identifier for the opponent (e.g., "iter_50", "calling_station")
            win_rate: Win rate from recent evaluation (0.0 to 1.0)
            rounds_played: Number of rounds in the evaluation
        """
        if opponent_id not in self._stats:
            self._stats[opponent_id] = {'wins': 0, 'losses': 0, 'draws': 0}
        
        # Convert win_rate to approximate wins/losses
        wins = int(win_rate * rounds_played)
        losses = int((1 - win_rate) * rounds_played)
        draws = rounds_played - wins - losses
        
        self._stats[opponent_id]['wins'] += wins
        self._stats[opponent_id]['losses'] += losses
        self._stats[opponent_id]['draws'] += draws
        self._dirty = True
    
    def update_from_match_result(
        self, 
        opponent_id: str, 
        round_wins: int, 
        round_losses: int, 
        draws: int = 0
    ) -> None:
        """Update stats from a direct match result.
        
        Args:
            opponent_id: Identifier for the opponent
            round_wins: Number of rounds won by current model
            round_losses: Number of rounds lost by current model
            draws: Number of drawn rounds
        """
        if opponent_id not in self._stats:
            self._stats[opponent_id] = {'wins': 0, 'losses': 0, 'draws': 0}
        
        self._stats[opponent_id]['wins'] += round_wins
        self._stats[opponent_id]['losses'] += round_losses
        self._stats[opponent_id]['draws'] += draws
        self._dirty = True
    
    def _recompute_priorities(self) -> None:
        """Recompute priorities from current stats."""
        self._priorities.clear()
        
        for opp_id, stats in self._stats.items():
            total = stats['wins'] + stats['losses'] + stats['draws']
            if total == 0:
                # No data yet - use maximum priority
                self._priorities[opp_id] = 1.0
                continue
            
            # Win rate including draws as 0.5
            win_rate = (stats['wins'] + 0.5 * stats['draws']) / total
            
            # Priority: higher when we're losing more
            # (1 - win_rate)^p gives high priority to opponents we lose to
            priority = max(self.min_priority, (1 - win_rate) ** self.priority_exponent)
            self._priorities[opp_id] = priority
        
        self._dirty = False
    
    def get_priority(self, opponent_id: str) -> float:
        """Get selection priority for an opponent.
        
        Args:
            opponent_id: Identifier for the opponent
            
        Returns:
            Priority weight (higher = more likely to be selected)
        """
        if self._dirty:
            self._recompute_priorities()
        
        return self._priorities.get(opponent_id, 1.0)
    
    def select_checkpoint(
        self,
        checkpoints: List[Tuple[int, str]],
        rng: Optional[random.Random] = None,
    ) -> Optional[str]:
        """Select a checkpoint using PFSP priority weighting.
        
        Args:
            checkpoints: List of (iteration, path) tuples
            rng: Optional random generator for reproducibility
            
        Returns:
            Selected checkpoint path, or None if no checkpoints
        """
        if not checkpoints:
            return None
        
        if self._dirty:
            self._recompute_priorities()
        
        chooser = rng or random
        
        if len(checkpoints) == 1:
            return checkpoints[0][1]
        
        # Compute weights based on priorities
        weights = []
        for iteration, path in checkpoints:
            opp_id = f"iter_{iteration}"
            priority = self._priorities.get(opp_id, 1.0)
            weights.append(priority)
        
        # Normalize (not strictly necessary for choices, but good practice)
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(checkpoints)] * len(checkpoints)
        
        selected = chooser.choices(checkpoints, weights=weights)[0]
        return selected[1]
    
    def get_stats_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all opponent stats and priorities.
        
        Returns:
            Dict mapping opponent_id to stats and priority
        """
        if self._dirty:
            self._recompute_priorities()
        
        summary = {}
        for opp_id, stats in self._stats.items():
            total = stats['wins'] + stats['losses'] + stats['draws']
            win_rate = (stats['wins'] + 0.5 * stats['draws']) / total if total > 0 else 0.5
            summary[opp_id] = {
                **stats,
                'total': total,
                'win_rate': win_rate,
                'priority': self._priorities.get(opp_id, 1.0),
            }
        return summary
    
    def reset(self) -> None:
        """Reset all statistics."""
        self._stats.clear()
        self._priorities.clear()
        self._dirty = True


class AgentCache:
    """LRU cache for heuristic agents and model checkpoints."""
    
    def __init__(self, device: torch.device = None, max_models: int = 10):
        self._agents: Dict[str, Any] = {}
        self._models = ModelCache(device=device, max_size=max_models)
    
    def get_agent(self, agent_type: str, player_id: str) -> Optional[Any]:
        """Get or create a heuristic agent (cached)."""
        if agent_type in ('model', 'past_model', 'anchor_model', 'pfsp_model'):
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
