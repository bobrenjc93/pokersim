#!/usr/bin/env python3
"""
Enhanced State Encoding for RL Poker Agent

This module provides comprehensive state encoding including:
- Card information (hole cards, community cards)
- Pot and betting information
- Position and game stage
- Opponent modeling (action history, stack sizes)
- Game-theoretic features (pot odds, stack depth)
- Hand strength estimation
"""

import torch
from typing import Any, List, Dict, Tuple
import numpy as np


# Card and action mappings (same as train.py)
RANK_MAP = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 
            'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
SUIT_MAP = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
STAGE_MAP = {'Preflop': 0, 'Flop': 1, 'Turn': 2, 'River': 3, 'Complete': 4}

# Action encoding for history
ACTION_TYPE_MAP = {
    'fold': 0, 'check': 1, 'call': 2, 'bet': 3, 'raise': 4, 'all_in': 5
}


def encode_card(card: str) -> Tuple[int, int]:
    """Encode a card string into rank and suit indices"""
    if len(card) != 2:
        return (0, 0)
    rank = RANK_MAP.get(card[0], 0)
    suit = SUIT_MAP.get(card[1], 0)
    return (rank, suit)


def estimate_hand_strength(hole_cards: List[str], community_cards: List[str]) -> float:
    """
    Estimate hand strength using an improved heuristic.
    
    Returns value between 0 and 1 representing relative hand strength.
    """
    if not hole_cards or len(hole_cards) < 2:
        return 0.5
    
    # Extract ranks and suits
    ranks = [RANK_MAP.get(card[0], 0) for card in hole_cards]
    suits = [SUIT_MAP.get(card[1], 0) for card in hole_cards]
    
    comm_ranks = [RANK_MAP.get(card[0], 0) for card in community_cards]
    comm_suits = [SUIT_MAP.get(card[1], 0) for card in community_cards]
    
    all_ranks = ranks + comm_ranks
    all_suits = suits + comm_suits
    
    score = 0.0
    
    # 1. High Card Strength (normalized)
    # Based on hole cards only relative to board
    max_hole = max(ranks)
    score += max_hole / 25.0  # Base contribution (0.0 - 0.48)
    
    # 2. Pair / Sets / Quads Detection
    from collections import Counter
    rank_counts = Counter(all_ranks)
    max_count = max(rank_counts.values()) if rank_counts else 1
    
    # Check if we improved the board
    board_counts = Counter(comm_ranks)
    max_board_count = max(board_counts.values()) if board_counts else 1
    
    if max_count == 2:
        # One pair
        if max_board_count < 2:
             # We have a pair that board doesn't
             score += 0.3
        else:
             # Board paired, check if we have better kicker/pair
             score += 0.1
    elif max_count == 3:
        # Trips/Set
        score += 0.5
    elif max_count == 4:
        # Quads
        score += 0.9
    
    # Full House Check (3 of one, 2 of another)
    if max_count >= 3:
        counts = list(rank_counts.values())
        if counts.count(2) >= 1 or counts.count(3) >= 2:
             score = max(score, 0.7)
             
    # 3. Flush Detection
    suit_counts = Counter(all_suits)
    max_suit = max(suit_counts.values()) if suit_counts else 0
    
    if max_suit >= 5:
        score = max(score, 0.8)  # Made flush
    elif max_suit == 4:
        score += 0.15  # Flush draw
    
    # 4. Straight Detection
    unique_ranks = sorted(set(all_ranks))
    max_consecutive = 1
    current_consecutive = 1
    # Handle Ace low (0, 1, 2, 3, 12 -> 12 should count as -1)
    if 12 in unique_ranks: # Ace
        unique_ranks_ace_low = [-1] + unique_ranks
    else:
        unique_ranks_ace_low = unique_ranks
        
    for i in range(1, len(unique_ranks)):
        if unique_ranks[i] == unique_ranks[i-1] + 1:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1
            
    if max_consecutive >= 5:
        score = max(score, 0.75) # Made straight
    elif max_consecutive == 4:
        score += 0.1 # Straight draw
        
    # Normalize
    return min(1.0, max(0.0, score))


class ActionHistory:
    """Track action history for opponent modeling"""
    
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.actions: List[Dict[str, Any]] = []
    
    def add_action(self, player_id: str, action_type: str, amount: int, 
                   pot: int, stage: str):
        """Add an action to the history"""
        self.actions.append({
            'player_id': player_id,
            'action_type': action_type,
            'amount': amount,
            'pot': pot,
            'stage': stage,
            'pot_fraction': amount / max(1, pot) if amount > 0 else 0
        })
        
        # Keep only recent history
        if len(self.actions) > self.max_history:
            self.actions = self.actions[-self.max_history:]
    
    def get_opponent_features(self, current_player_id: str, 
                            num_opponents: int = 9) -> torch.Tensor:
        """
        Extract opponent modeling features from action history.
        
        Returns a fixed-size tensor with opponent statistics.
        """
        # Features per opponent:
        # - Action counts (fold, check, call, bet, raise, all_in) -> 6
        # - Average bet size (as fraction of pot) -> 1
        # - Aggression frequency (bet/raise rate) -> 1
        # - Last 3 actions (one-hot) -> 3 * 6 = 18
        # Total = 26 dims
        
        features = []
        
        # Aggregate statistics per opponent (we don't know exact opponent mapping,
        # so we'll use aggregate features)
        opponent_actions = [a for a in self.actions if a['player_id'] != current_player_id]
        
        if opponent_actions:
            # Action type distribution
            action_counts = {at: 0 for at in ACTION_TYPE_MAP.keys()}
            total_actions = len(opponent_actions)
            bet_sizes = []
            
            for action in opponent_actions:
                action_type = action['action_type']
                if action_type in action_counts:
                    action_counts[action_type] += 1
                if action['amount'] > 0:
                    bet_sizes.append(action['pot_fraction'])
            
            # Normalize action counts
            for action_type in ['fold', 'check', 'call', 'bet', 'raise', 'all_in']:
                features.append(action_counts.get(action_type, 0) / max(1, total_actions))
            
            # Average bet size
            avg_bet_size = np.mean(bet_sizes) if bet_sizes else 0.0
            features.append(avg_bet_size)
            
            # Aggression frequency (bet + raise rate)
            aggression = (action_counts.get('bet', 0) + action_counts.get('raise', 0)) / max(1, total_actions)
            features.append(aggression)
            
            # Last 3 opponent actions (one-hot over action types)
            # Padding with zeros if fewer than 3 actions
            last_actions = opponent_actions[-3:]
            # Pad to ensure we have 3
            while len(last_actions) < 3:
                last_actions.insert(0, {'action_type': 'none'})
                
            for action in last_actions:
                act_type = action.get('action_type', 'none')
                for target_type in ['fold', 'check', 'call', 'bet', 'raise', 'all_in']:
                    features.append(1.0 if act_type == target_type else 0.0)
        else:
            # No opponent history - use neutral values
            features.extend([0.0] * (6 + 1 + 1 + 18))  # action counts + avg bet + aggression + last 3 actions
        
        return torch.tensor(features, dtype=torch.float32)


class RLStateEncoder:
    """
    Enhanced state encoder for RL-based poker agent.
    
    Includes all features from basic encoder plus:
    - Opponent modeling (action history)
    - Game-theoretic features (pot odds, stack depth)
    - Hand strength estimation
    """
    
    def __init__(self):
        self.action_history = ActionHistory(max_history=20)
    
    def reset_history(self):
        """Reset action history (call at start of new hand)"""
        self.action_history = ActionHistory(max_history=20)
    
    def add_action(self, player_id: str, action_type: str, amount: int,
                   pot: int, stage: str):
        """Add action to history"""
        self.action_history.add_action(player_id, action_type, amount, pot, stage)
    
    def encode_state(self, state: Dict[str, Any]) -> torch.Tensor:
        """
        Encode poker state into comprehensive feature vector.
        
        Features (total ~200+ dimensions):
        1. Hole cards (2 × 17 = 34)
        2. Community cards (5 × 17 = 85)
        3. Pot info (5 features)
        4. Stage (5 one-hot)
        5. Position (6 features)
        6. Game-theoretic features (5 features)
        7. Opponent modeling (14 features)
        8. Hand strength (1 feature)
        
        Returns:
            Tensor of shape (feature_dim,)
        """
        features = []
        
        # 1. Hole cards encoding (same as before)
        hole_cards = state.get('hole_cards', [])
        for i in range(2):
            if i < len(hole_cards):
                rank, suit = encode_card(hole_cards[i])
                rank_onehot = [0] * 13
                rank_onehot[rank] = 1
                suit_onehot = [0] * 4
                suit_onehot[suit] = 1
                features.extend(rank_onehot + suit_onehot)
            else:
                features.extend([0] * 17)
        
        # 2. Community cards encoding
        community_cards = state.get('community_cards', [])
        for i in range(5):
            if i < len(community_cards):
                rank, suit = encode_card(community_cards[i])
                rank_onehot = [0] * 13
                rank_onehot[rank] = 1
                suit_onehot = [0] * 4
                suit_onehot[suit] = 1
                features.extend(rank_onehot + suit_onehot)
            else:
                features.extend([0] * 17)
        
        # 3. Pot and betting information (normalized)
        starting_chips = state.get('starting_chips', 1000.0)
        pot = state.get('pot', 0) / starting_chips
        current_bet = state.get('current_bet', 0) / starting_chips
        player_chips = state.get('player_chips', 0) / starting_chips
        player_bet = state.get('player_bet', 0) / starting_chips
        player_total_bet = state.get('player_total_bet', 0) / starting_chips
        
        features.extend([pot, current_bet, player_chips, player_bet, player_total_bet])
        
        # 4. Stage encoding
        stage = state.get('stage', 'Preflop')
        stage_idx = STAGE_MAP.get(stage, 0)
        stage_onehot = [0] * 5
        stage_onehot[stage_idx] = 1
        features.extend(stage_onehot)
        
        # 5. Position features
        num_players = state.get('num_players', 2)
        num_active = state.get('num_active', 2)
        position = state.get('position', 0)
        is_dealer = float(state.get('is_dealer', False))
        is_small_blind = float(state.get('is_small_blind', False))
        is_big_blind = float(state.get('is_big_blind', False))
        
        features.extend([
            num_players / 10.0,
            num_active / 10.0,
            position / 10.0,
            is_dealer,
            is_small_blind,
            is_big_blind
        ])
        
        # 6. Game-theoretic features (NEW)
        to_call = state.get('to_call', 0)
        pot_size = state.get('pot', 0)
        
        # Pot odds (probability needed to call profitably)
        pot_odds = to_call / max(1, pot_size + to_call)
        
        # Stack to pot ratio (important for decision making)
        stack_to_pot = state.get('player_chips', 0) / max(1, pot_size)
        
        # Effective stack depth (normalized by big blind)
        big_blind = state.get('big_blind', 20)
        effective_stack = state.get('player_chips', 0) / max(1, big_blind)
        effective_stack_normalized = min(1.0, effective_stack / 100.0)  # Cap at 100 BBs
        
        # Call amount as fraction of remaining chips
        call_fraction = to_call / max(1, state.get('player_chips', 1))
        
        # Pot commitment (how much of our stack is already in pot)
        pot_commitment = player_total_bet / max(1, player_total_bet + player_chips)
        
        features.extend([
            pot_odds,
            stack_to_pot,
            effective_stack_normalized,
            call_fraction,
            pot_commitment
        ])
        
        # 7. Opponent modeling features (NEW)
        player_id = state.get('player_id', 'unknown')
        opponent_features = self.action_history.get_opponent_features(player_id)
        features.extend(opponent_features.tolist())
        
        # 8. Hand strength estimation (NEW)
        hand_strength = estimate_hand_strength(hole_cards, community_cards)
        features.append(hand_strength)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def get_feature_dim(self) -> int:
        """Return the total feature dimension"""
        # Hole cards: 34
        # Community cards: 85
        # Pot info: 5
        # Stage: 5
        # Position: 6
        # Game-theoretic: 5
        # Opponent modeling: 26 (6 actions + 1 avg + 1 agg + 18 history)
        # Hand strength: 1
        return 34 + 85 + 5 + 5 + 6 + 5 + 26 + 1  # = 167

