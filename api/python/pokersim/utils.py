"""
Core utility functions for poker gameplay.

This module provides functions shared between training, evaluation, and gameplay:
- Action conversion (action labels to game actions)
- State extraction from game state dictionaries
- Legal action mask creation for neural network inference
"""

from typing import Dict, Tuple, Any, List

import torch

from .config import (
    ACTION_MAP,
    ACTION_NAMES,
    NUM_ACTIONS,
    RAISE_SIZE_MAP,
    BET_SIZE_MAP,
    FEATURE_DIM,
)

# Re-export constants for backwards compatibility
__all__ = [
    'ACTION_MAP',
    'ACTION_NAMES',
    'NUM_ACTIONS',
    'RAISE_SIZE_MAP',
    'BET_SIZE_MAP',
    'FEATURE_DIM',
    'convert_action_label',
    'extract_state',
    'create_legal_actions_mask',
]


def convert_action_label(
    action_label: str,
    state: Dict[str, Any]
) -> Tuple[str, int]:
    """
    Convert action label to (action_type, amount).
    
    Unified raise_X% actions are converted to either 'bet' or 'raise' based on
    game context (whether there's already a bet to face).
    
    Args:
        action_label: Action label (e.g., 'raise_50%', 'call')
        state: Game state
    
    Returns:
        Tuple of (action_type, amount)
    """
    player_chips = state.get('player_chips', 0)
    
    # Simple actions (including bare 'bet' and 'raise' which get their amounts from caller)
    if action_label in ['fold', 'check', 'call', 'all_in', 'bet', 'raise']:
        return action_label, 0
    
    # Unified raise actions - convert to bet or raise based on game context
    if action_label.startswith('raise_'):
        pot = state.get('pot', 0)
        player_bet = state.get('player_bet', 0)
        current_bet = state.get('current_bet', 0)
        to_call = max(0, current_bet - player_bet)
        min_bet = state.get('min_bet', state.get('big_blind', 20))
        min_raise = state.get('min_raise_total', state.get('big_blind', 20))
        
        # Get size fraction
        size_fraction = RAISE_SIZE_MAP.get(action_label, 0.5)
        sizing_amount = int(pot * size_fraction) if pot > 0 else min_bet
        
        # Determine if this should be a bet or raise based on whether there's a bet to face
        if to_call == 0:
            # No bet to face - this is a BET
            amount = max(min_bet, sizing_amount)
            
            # If bet would use all chips, go all-in
            if amount >= player_chips:
                return 'all_in', 0
            
            return 'bet', amount
        else:
            # There's a bet to face - this is a RAISE
            # Amount is call + raise increment
            amount = to_call + sizing_amount
            
            # Ensure we meet minimum raise requirement
            if amount < min_raise:
                amount = min_raise
            
            # If raise would use all chips, go all-in
            if amount >= player_chips:
                return 'all_in', 0
            
            # If we can't afford the minimum raise, go all-in instead
            if player_chips < min_raise:
                return 'all_in', 0
            
            return 'raise', amount
    
    # Legacy support for bet_X% actions (convert to raise_X% logic)
    if action_label.startswith('bet_'):
        # Convert bet_X% to equivalent raise_X%
        raise_label = action_label.replace('bet_', 'raise_')
        return convert_action_label(raise_label, state)
    
    # Fallback
    return 'check', 0


def extract_state(game_state: Dict, player_id: str) -> Dict[str, Any]:
    """
    Extract state for a specific player from the raw API game state.
    
    Args:
        game_state: Raw game state from C++ API
        player_id: Player ID to extract state for
        
    Returns:
        Dictionary containing extracted state features
    """
    player = None
    for p in game_state['players']:
        if p['id'] == player_id:
            player = p
            break
    
    if player is None:
        return {}
    
    config = game_state.get('config', {})
    action_constraints = game_state.get('actionConstraints', {})
    
    # Compute max stack across all players for relative normalization
    max_stack = max(p.get('chips', 0) for p in game_state['players'])
    # Use at least starting_chips to avoid division by zero and handle edge cases
    starting_chips = config.get('startingChips', 1000)
    max_stack = max(max_stack, starting_chips)
    
    return {
        'player_id': player_id,
        'hole_cards': player.get('holeCards', []),
        'community_cards': game_state.get('communityCards', []),
        'pot': game_state.get('pot', 0),
        'current_bet': game_state.get('currentBet', 0),
        'player_chips': player.get('chips', 0),
        'player_bet': player.get('bet', 0),
        'player_total_bet': player.get('totalBet', 0),
        'stage': game_state.get('stage', 'Preflop'),
        'num_players': len(game_state['players']),
        'num_active': sum(1 for p in game_state['players'] if p.get('isInHand', False)),
        'position': player.get('position', 0),
        'is_dealer': player.get('isDealer', False),
        'is_small_blind': player.get('isSmallBlind', False),
        'is_big_blind': player.get('isBigBlind', False),
        'big_blind': config.get('bigBlind', 20),
        'small_blind': config.get('smallBlind', 10),
        'starting_chips': starting_chips,
        'max_stack': max_stack,  # Maximum stack across all players for relative normalization
        'to_call': action_constraints.get('toCall', 0),
        'min_bet': action_constraints.get('minBet', 20),
        'min_raise_total': action_constraints.get('minRaiseTotal', 20),
    }


def create_legal_actions_mask(legal_actions: List[str], device: torch.device) -> torch.Tensor:
    """
    Create a boolean mask for legal actions.
    
    With the unified action space, 'bet' and 'raise' both enable the same raise_X% actions.
    The convert_action_label function handles converting to the correct game action.
    
    Args:
        legal_actions: List of legal action strings (e.g., ['fold', 'call', 'raise'])
        device: Torch device
    
    Returns:
        Boolean tensor of shape (1, NUM_ACTIONS)
    """
    mask = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool, device=device)
    
    for action in legal_actions:
        if action in ACTION_MAP:
            # Simple action (fold, check, call, all_in)
            mask[0, ACTION_MAP[action]] = True
        elif action in ('bet', 'raise'):
            # Enable all raise_X% sizing actions
            # (they become bet or raise based on game context in convert_action_label)
            for action_name in ACTION_NAMES:
                if action_name.startswith('raise_'):
                    mask[0, ACTION_MAP[action_name]] = True
    
    return mask
