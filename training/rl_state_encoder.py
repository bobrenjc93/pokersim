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

# Try to import C++ hand strength estimator for performance
try:
    import poker_api_binding
    _CPP_HAND_STRENGTH_AVAILABLE = hasattr(poker_api_binding, 'estimate_hand_strength')
except ImportError:
    _CPP_HAND_STRENGTH_AVAILABLE = False


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


def estimate_preflop_strength(hole_cards: List[str]) -> float:
    """
    Estimate preflop hand strength using poker fundamentals.
    
    Based on Sklansky-Chubukov rankings and equity simulations.
    Returns value between 0.0 (worst) and 1.0 (best).
    """
    if not hole_cards or len(hole_cards) < 2:
        return 0.3
    
    # Extract ranks
    ranks = sorted([RANK_MAP.get(card[0], 0) for card in hole_cards], reverse=True)
    suits = [SUIT_MAP.get(card[1], 0) for card in hole_cards]
    
    high_rank = ranks[0]  # Higher card (0-12 where 12=Ace)
    low_rank = ranks[1]   # Lower card
    is_suited = suits[0] == suits[1]
    is_pair = high_rank == low_rank
    gap = high_rank - low_rank
    
    # Base strength from high cards (normalized 0-0.35)
    base_strength = (high_rank + low_rank) / 48.0  # Max 24/48 = 0.5
    
    # Pair bonus (pairs are strong)
    if is_pair:
        # AA=1.0, KK=0.95, ..., 22=0.55
        pair_strength = 0.55 + (high_rank / 12.0) * 0.45
        return pair_strength
    
    # Premium unpaired hands
    # Ace-high
    if high_rank == 12:  # Ace
        if low_rank >= 10:  # AT+
            return 0.70 + (low_rank - 10) * 0.05 + (0.03 if is_suited else 0)  # AK=0.83, AQ=0.78, AJ=0.73, AT=0.70
        elif low_rank >= 7:  # A7-A9
            return 0.50 + (low_rank - 7) * 0.03 + (0.05 if is_suited else 0)
        else:  # A2-A6
            return 0.35 + (low_rank * 0.02) + (0.07 if is_suited else 0)
    
    # King-high
    if high_rank == 11:  # King
        if low_rank >= 10:  # KQ, KJ, KT
            return 0.55 + (low_rank - 10) * 0.05 + (0.04 if is_suited else 0)
        elif low_rank >= 7:
            return 0.40 + (low_rank - 7) * 0.03 + (0.05 if is_suited else 0)
        else:
            return 0.25 + (0.06 if is_suited else 0)
    
    # Queen-high
    if high_rank == 10:  # Queen
        if low_rank >= 9:  # QJ, QT
            return 0.48 + (low_rank - 9) * 0.04 + (0.04 if is_suited else 0)
        else:
            return 0.25 + (0.05 if is_suited else 0)
    
    # Connected cards (potential straights)
    connectivity_bonus = 0.0
    if gap == 1:  # Connectors
        connectivity_bonus = 0.08
    elif gap == 2:  # One-gappers
        connectivity_bonus = 0.04
    elif gap == 3:  # Two-gappers
        connectivity_bonus = 0.02
    
    # Suited bonus
    suited_bonus = 0.07 if is_suited else 0.0
    
    # Final calculation for other hands
    strength = base_strength + connectivity_bonus + suited_bonus
    
    # Clamp to reasonable range for non-premium hands
    return max(0.15, min(0.55, strength))


def estimate_hand_strength(hole_cards: List[str], community_cards: List[str]) -> float:
    """
    Estimate hand strength using improved heuristic.
    
    Combines preflop strength with made hand strength postflop.
    Returns value between 0 and 1 representing relative hand strength.
    
    Performance: Uses C++ implementation when available (~10x faster).
    """
    if not hole_cards or len(hole_cards) < 2:
        return 0.3
    
    # Use C++ implementation if available (much faster for episode collection)
    if _CPP_HAND_STRENGTH_AVAILABLE:
        try:
            return poker_api_binding.estimate_hand_strength(hole_cards, community_cards or [])
        except Exception:
            pass  # Fall back to Python implementation
    
    # Preflop: use preflop-specific evaluation
    if not community_cards:
        return estimate_preflop_strength(hole_cards)
    
    # Postflop: evaluate made hand + draws
    from collections import Counter
    
    # Extract ranks and suits
    hole_ranks = [RANK_MAP.get(card[0], 0) for card in hole_cards]
    hole_suits = [SUIT_MAP.get(card[1], 0) for card in hole_cards]
    
    comm_ranks = [RANK_MAP.get(card[0], 0) for card in community_cards]
    comm_suits = [SUIT_MAP.get(card[1], 0) for card in community_cards]
    
    all_ranks = hole_ranks + comm_ranks
    all_suits = hole_suits + comm_suits
    
    # Count ranks across all cards
    rank_counts = Counter(all_ranks)
    board_rank_counts = Counter(comm_ranks)
    
    # Count suits
    suit_counts = Counter(all_suits)
    board_suit_counts = Counter(comm_suits)
    
    # Determine made hand strength
    made_hand_rank = 0  # 0=high card, 1=pair, 2=two pair, 3=trips, 4=straight, 5=flush, 6=full house, 7=quads, 8=straight flush
    
    # Check for flush
    max_suit = max(suit_counts.values()) if suit_counts else 0
    has_flush = max_suit >= 5
    
    # Check if our hole cards contribute to the flush
    flush_suit = None
    if has_flush:
        for suit, count in suit_counts.items():
            if count >= 5:
                flush_suit = suit
                break
        hole_contributes_to_flush = flush_suit is not None and any(s == flush_suit for s in hole_suits)
    else:
        hole_contributes_to_flush = False
    
    # Check for straight
    unique_ranks = sorted(set(all_ranks))
    # Add ace-low straight possibility
    if 12 in unique_ranks:  # Ace
        unique_ranks_with_ace_low = [-1] + unique_ranks
    else:
        unique_ranks_with_ace_low = unique_ranks
    
    has_straight = False
    straight_high = 0
    for i in range(len(unique_ranks) - 4):
        if unique_ranks[i+4] - unique_ranks[i] == 4:
            has_straight = True
            straight_high = unique_ranks[i+4]
            break
    # Check wheel (A-2-3-4-5)
    if not has_straight and set([0,1,2,3,12]).issubset(set(all_ranks)):
        has_straight = True
        straight_high = 3  # 5-high
    
    # Check if hole cards contribute to straight
    hole_contributes_to_straight = False
    if has_straight:
        # Simplified check: at least one hole card is in the straight range
        hole_contributes_to_straight = any(
            r in range(max(0, straight_high-4), straight_high+1) or (r == 12 and straight_high <= 4)
            for r in hole_ranks
        )
    
    # Count pairs, trips, quads involving hole cards
    max_count = max(rank_counts.values()) if rank_counts else 1
    
    # Check if hole cards make the pair/trips/quads (not just the board)
    hole_contributes_pairs = False
    pair_rank = 0
    for rank in hole_ranks:
        if rank_counts[rank] >= 2:
            hole_contributes_pairs = True
            if rank_counts[rank] >= pair_rank:
                pair_rank = rank_counts[rank]
    
    # Two pair check
    pairs = [r for r, c in rank_counts.items() if c >= 2]
    has_two_pair = len(pairs) >= 2
    hole_contributes_two_pair = has_two_pair and any(r in pairs for r in hole_ranks)
    
    # Full house check
    trips = [r for r, c in rank_counts.items() if c >= 3]
    has_full_house = len(trips) >= 1 and (len(pairs) >= 2 or len(trips) >= 2)
    
    # Straight flush check
    has_straight_flush = has_flush and has_straight
    if has_straight_flush:
        # Verify same suit for straight
        for suit in range(4):
            suited_ranks = sorted([all_ranks[i] for i in range(len(all_ranks)) if all_suits[i] == suit])
            if len(suited_ranks) >= 5:
                for i in range(len(suited_ranks) - 4):
                    if suited_ranks[i+4] - suited_ranks[i] == 4:
                        made_hand_rank = 8
                        break
    
    # Assign made hand rank
    if made_hand_rank < 8:  # Not straight flush
        if max_count == 4 and hole_contributes_pairs:
            made_hand_rank = 7  # Quads
        elif has_full_house and (any(r in trips for r in hole_ranks) or hole_contributes_two_pair):
            made_hand_rank = 6  # Full house
        elif has_flush and hole_contributes_to_flush:
            made_hand_rank = 5  # Flush
        elif has_straight and hole_contributes_to_straight:
            made_hand_rank = 4  # Straight
        elif max_count == 3 and hole_contributes_pairs:
            made_hand_rank = 3  # Trips
        elif hole_contributes_two_pair:
            made_hand_rank = 2  # Two pair
        elif hole_contributes_pairs:
            made_hand_rank = 1  # Pair
        else:
            made_hand_rank = 0  # High card
    
    # Convert made hand rank to strength score
    # Base strength from made hand (0.0 to 0.85)
    made_hand_strengths = {
        0: 0.15,  # High card
        1: 0.35,  # Pair
        2: 0.50,  # Two pair
        3: 0.60,  # Trips
        4: 0.70,  # Straight
        5: 0.75,  # Flush
        6: 0.85,  # Full house
        7: 0.95,  # Quads
        8: 1.00,  # Straight flush
    }
    base_strength = made_hand_strengths[made_hand_rank]
    
    # Adjust based on kicker/pair strength
    kicker_bonus = 0.0
    if made_hand_rank <= 2:  # High card, pair, two pair
        high_hole = max(hole_ranks)
        kicker_bonus = (high_hole / 12.0) * 0.10
    
    # Draw potential (when not already made strong hand)
    draw_bonus = 0.0
    if made_hand_rank < 4:  # Less than straight
        # Flush draw
        for suit in range(4):
            my_suited = sum(1 for s in all_suits if s == suit)
            hole_in_suit = sum(1 for s in hole_suits if s == suit)
            if my_suited == 4 and hole_in_suit >= 1:
                draw_bonus += 0.08
        
        # Straight draw (open-ended or gutshot)
        for i in range(len(unique_ranks) - 3):
            if unique_ranks[i+3] - unique_ranks[i] <= 4:
                if any(r in range(unique_ranks[i], unique_ranks[i+3]+1) for r in hole_ranks):
                    draw_bonus += 0.05
                    break
    
    final_strength = min(1.0, base_strength + kicker_bonus + draw_bonus)
    
    return final_strength


def get_hand_category(hole_cards: List[str], community_cards: List[str]) -> str:
    """
    Get a categorical description of hand strength for logging/debugging.
    
    Returns one of: 'premium', 'strong', 'medium', 'weak', 'trash'
    """
    strength = estimate_hand_strength(hole_cards, community_cards)
    
    if strength >= 0.70:
        return 'premium'
    elif strength >= 0.50:
        return 'strong'
    elif strength >= 0.35:
        return 'medium'
    elif strength >= 0.25:
        return 'weak'
    else:
        return 'trash'


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
    
    Performance optimization: Pre-allocates feature buffer to avoid
    repeated tensor allocation during episode collection.
    """
    
    # Feature dimensions (must match get_feature_dim())
    _FEATURE_DIM = 167
    
    def __init__(self, use_buffer: bool = True):
        """
        Args:
            use_buffer: If True, pre-allocate feature buffer for faster encoding.
                       Set False if encoder will be used across threads.
        """
        self.action_history = ActionHistory(max_history=20)
        self.use_buffer = use_buffer
        # Pre-allocate buffer for features (avoids repeated tensor allocation)
        self._buffer = torch.zeros(self._FEATURE_DIM, dtype=torch.float32) if use_buffer else None
    
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
        
        Features (total 167 dimensions):
        1. Hole cards (2 × 17 = 34)
        2. Community cards (5 × 17 = 85)
        3. Pot info (5 features)
        4. Stage (5 one-hot)
        5. Position (6 features)
        6. Game-theoretic features (5 features)
        7. Opponent modeling (26 features)
        8. Hand strength (1 feature)
        
        Returns:
            Tensor of shape (feature_dim,)
        """
        # Use pre-allocated buffer if available for speed
        if self.use_buffer and self._buffer is not None:
            return self._encode_state_fast(state)
        
        # Fallback to list-based encoding
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
        
        # 3. Pot and betting information (normalized relative to max stack)
        # Use max_stack for relative normalization (1 = max stack, everything else is a fraction)
        max_stack = state.get('max_stack', state.get('starting_chips', 1000.0))
        if max_stack <= 0:
            max_stack = state.get('starting_chips', 1000.0)
        
        pot = state.get('pot', 0) / max_stack
        current_bet = state.get('current_bet', 0) / max_stack
        player_chips = state.get('player_chips', 0) / max_stack
        player_bet = state.get('player_bet', 0) / max_stack
        player_total_bet = state.get('player_total_bet', 0) / max_stack
        
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
    
    def _encode_state_fast(self, state: Dict[str, Any]) -> torch.Tensor:
        """
        Fast state encoding using pre-allocated buffer.
        
        This method fills the buffer in-place to avoid repeated tensor allocation,
        providing ~20% speedup during episode collection.
        """
        # Zero out buffer
        self._buffer.zero_()
        idx = 0
        
        # 1. Hole cards encoding (34 features)
        hole_cards = state.get('hole_cards', [])
        for i in range(2):
            if i < len(hole_cards):
                rank, suit = encode_card(hole_cards[i])
                self._buffer[idx + rank] = 1.0  # rank one-hot (13)
                self._buffer[idx + 13 + suit] = 1.0  # suit one-hot (4)
            idx += 17
        
        # 2. Community cards encoding (85 features)
        community_cards = state.get('community_cards', [])
        for i in range(5):
            if i < len(community_cards):
                rank, suit = encode_card(community_cards[i])
                self._buffer[idx + rank] = 1.0
                self._buffer[idx + 13 + suit] = 1.0
            idx += 17
        
        # 3. Pot and betting information (5 features)
        max_stack = state.get('max_stack', state.get('starting_chips', 1000.0))
        if max_stack <= 0:
            max_stack = state.get('starting_chips', 1000.0)
        
        self._buffer[idx] = state.get('pot', 0) / max_stack
        self._buffer[idx + 1] = state.get('current_bet', 0) / max_stack
        self._buffer[idx + 2] = state.get('player_chips', 0) / max_stack
        self._buffer[idx + 3] = state.get('player_bet', 0) / max_stack
        self._buffer[idx + 4] = state.get('player_total_bet', 0) / max_stack
        idx += 5
        
        # 4. Stage encoding (5 features)
        stage = state.get('stage', 'Preflop')
        stage_idx = STAGE_MAP.get(stage, 0)
        self._buffer[idx + stage_idx] = 1.0
        idx += 5
        
        # 5. Position features (6 features)
        self._buffer[idx] = state.get('num_players', 2) / 10.0
        self._buffer[idx + 1] = state.get('num_active', 2) / 10.0
        self._buffer[idx + 2] = state.get('position', 0) / 10.0
        self._buffer[idx + 3] = float(state.get('is_dealer', False))
        self._buffer[idx + 4] = float(state.get('is_small_blind', False))
        self._buffer[idx + 5] = float(state.get('is_big_blind', False))
        idx += 6
        
        # 6. Game-theoretic features (5 features)
        to_call = state.get('to_call', 0)
        pot_size = state.get('pot', 0)
        pot_odds = to_call / max(1, pot_size + to_call)
        stack_to_pot = state.get('player_chips', 0) / max(1, pot_size)
        big_blind = state.get('big_blind', 20)
        effective_stack = state.get('player_chips', 0) / max(1, big_blind)
        effective_stack_normalized = min(1.0, effective_stack / 100.0)
        call_fraction = to_call / max(1, state.get('player_chips', 1))
        player_total_bet = state.get('player_total_bet', 0) / max_stack
        player_chips_norm = state.get('player_chips', 0) / max_stack
        pot_commitment = player_total_bet / max(0.001, player_total_bet + player_chips_norm)
        
        self._buffer[idx] = pot_odds
        self._buffer[idx + 1] = stack_to_pot
        self._buffer[idx + 2] = effective_stack_normalized
        self._buffer[idx + 3] = call_fraction
        self._buffer[idx + 4] = pot_commitment
        idx += 5
        
        # 7. Opponent modeling features (26 features)
        player_id = state.get('player_id', 'unknown')
        opponent_features = self.action_history.get_opponent_features(player_id)
        self._buffer[idx:idx + 26] = opponent_features
        idx += 26
        
        # 8. Hand strength estimation (1 feature)
        hand_strength = estimate_hand_strength(hole_cards, community_cards)
        self._buffer[idx] = hand_strength
        
        # Return a clone so caller can store it without affecting buffer
        return self._buffer.clone()
    
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

