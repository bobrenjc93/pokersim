"""
Poker agents for gameplay and training.

This module provides various agent implementations:
- ModelAgent: Uses trained neural network for action selection
- RandomAgent: Selects random legal actions
- HeuristicAgent: Rule-based agent for baseline comparison
- TightAgent: Tight-aggressive agent
- LoosePassiveAgent: Loose-passive calling station
- AggressiveAgent: Loose-aggressive bluffer
- CallingStationAgent: Calls most bets including all-ins
- HeroCallerAgent: Calls down suspected bluffs
- SimpleAgent: Base class for simple agents
- AlwaysRaiseAgent, AlwaysCallAgent, AlwaysFoldAgent: Deterministic agents for testing
"""

import random
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import (
    ACTION_MAP,
    ACTION_NAMES,
    NUM_ACTIONS,
    FEATURE_DIM,
    RANK_MAP,
    SUIT_MAP,
)
from .utils import convert_action_label, create_legal_actions_mask
from .state_encoder import RLStateEncoder


__all__ = [
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
]


class ModelAgent:
    """
    Agent that uses a trained neural network model for action selection.
    
    This agent:
    1. Encodes the game state using RLStateEncoder
    2. Feeds state to actor-critic model
    3. Samples action from policy distribution
    4. Converts action to game-compatible format
    """
    
    def __init__(
        self,
        player_id: str,
        name: str,
        model_path: str = None,
        model = None,
        device: torch.device = None,
        temperature: float = 1.0,
        deterministic: bool = False
    ):
        """
        Args:
            player_id: Player ID
            name: Player name
            model_path: Path to trained model checkpoint (optional if model is provided)
            model: Pre-loaded model instance (optional)
            device: Device to run model on
            temperature: Sampling temperature (higher = more random)
            deterministic: If True, always pick argmax action
        """
        self.player_id = player_id
        self.name = name
        self.temperature = temperature
        self.deterministic = deterministic
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # Load model or use provided one
        if model is not None:
            self.model = model
            self.model.to(self.device)
            self.model.eval()
            # Create new encoder
            self.encoder = RLStateEncoder()
        elif model_path is not None:
            self.model, self.encoder = self._load_model(model_path)
            self.model.eval()
        else:
            raise ValueError("Must provide either model_path or model")
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        from .model import PokerActorCritic
        
        # weights_only=False is required to load checkpoints with numpy scalars
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get model configuration from checkpoint
        input_dim = checkpoint.get('input_dim', FEATURE_DIM)
        dropout = checkpoint.get('dropout', 0.1)

        # Try to get model architecture from checkpoint (preferred)
        # These are now saved explicitly during training
        state_dict = checkpoint.get('model_state_dict', {})
        
        # PRIORITY 1: Use explicitly saved architecture parameters
        if 'hidden_dim' in checkpoint and 'num_heads' in checkpoint:
            hidden_dim = checkpoint['hidden_dim']
            num_heads = checkpoint['num_heads']
            num_layers = checkpoint.get('num_layers', 4)
        # PRIORITY 2: Infer from state dict shape
        elif 'pos_encoding' in state_dict:
            # pos_encoding shape: [1, 14, hidden_dim]
            hidden_dim = state_dict['pos_encoding'].shape[2]
            
            # Infer num_layers from transformer layer count
            max_layer = 0
            for key in state_dict.keys():
                if key.startswith('transformer.layers.'):
                    try:
                        layer_idx = int(key.split('.')[2])
                        max_layer = max(max_layer, layer_idx)
                    except (IndexError, ValueError):
                        pass
            num_layers = max_layer + 1
            
            # IMPORTANT: Try to infer num_heads from attention weights
            # The attention weights have shape that reveals num_heads
            num_heads = checkpoint.get('num_heads', None)
            if num_heads is None:
                # Try to infer from self_attn.in_proj_weight shape
                # For MultiheadAttention, in_proj_weight shape is [3*hidden_dim, hidden_dim]
                # We can't directly infer num_heads, so try common values
                for candidate_heads in [16, 8, 4]:
                    if hidden_dim % candidate_heads == 0:
                        num_heads = candidate_heads
                        break
                else:
                    num_heads = 8  # Fallback default
        else:
            # PRIORITY 3: Fallback to defaults
            hidden_dim = checkpoint.get('hidden_dim', 256)
            num_heads = checkpoint.get('num_heads', 8)
            num_layers = checkpoint.get('num_layers', 4)
        
        # Create model
        model = PokerActorCritic(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        # Create encoder
        encoder = RLStateEncoder()
        
        return model, encoder
    
    def reset_hand(self):
        """Reset state for new hand (clear action history)"""
        self.encoder.reset_history()
    
    def observe_action(self, player_id: str, action_type: str, amount: int, 
                      pot: int, stage: str):
        """Observe an action (for opponent modeling)"""
        self.encoder.add_action(player_id, action_type, amount, pot, stage)
    
    def select_action(
        self,
        state: Dict[str, Any],
        legal_actions: List[str],
        return_probs: bool = False
    ) -> Tuple[str, int, str, ...]:
        """
        Select an action using the trained model.
        
        Args:
            state: Game state dictionary
            legal_actions: List of legal action strings
            return_probs: If True, also return the full probability distribution
        
        Returns:
            Tuple of (action_type, amount, action_label) or
            Tuple of (action_type, amount, action_label, action_probs) if return_probs=True
        """
        # Encode state
        state_tensor = self.encoder.encode_state(state).unsqueeze(0).to(self.device)
        
        # Create legal actions mask
        legal_actions_mask = self._create_legal_actions_mask(legal_actions)
        
        # Get action probabilities from model
        with torch.no_grad():
            action_probs, value = self.model.get_action_probs(
                state_tensor,
                legal_actions_mask,
                temperature=self.temperature
            )
        
        # Select action
        if self.deterministic:
            action_idx = action_probs.argmax().item()
        else:
            action_idx = torch.multinomial(action_probs.squeeze(0), 1).item()
        
        # Convert action index to action name
        action_label = ACTION_NAMES[action_idx]
        
        # Convert to game-compatible format
        action_type, amount = self._convert_action(action_label, state)
        
        if return_probs:
            # Return probabilities as a dict mapping action names to probabilities
            probs_dict = {
                ACTION_NAMES[i]: action_probs[0, i].item() 
                for i in range(len(ACTION_NAMES))
            }
            return action_type, amount, action_label, probs_dict
        
        return action_type, amount, action_label
    
    def get_action_distribution(
        self,
        state: Dict[str, Any],
        legal_actions: List[str]
    ) -> Dict[str, float]:
        """
        Get the full probability distribution over actions without selecting one.
        
        Args:
            state: Game state dictionary
            legal_actions: List of legal action strings
        
        Returns:
            Dict mapping action names to probabilities
        """
        # Encode state
        state_tensor = self.encoder.encode_state(state).unsqueeze(0).to(self.device)
        
        # Create legal actions mask
        legal_actions_mask = self._create_legal_actions_mask(legal_actions)
        
        # Get action probabilities from model
        with torch.no_grad():
            action_probs, value = self.model.get_action_probs(
                state_tensor,
                legal_actions_mask,
                temperature=self.temperature
            )
        
        # Return probabilities as a dict mapping action names to probabilities
        probs_dict = {
            ACTION_NAMES[i]: action_probs[0, i].item() 
            for i in range(len(ACTION_NAMES))
        }
        return probs_dict
    
    def _create_legal_actions_mask(self, legal_actions: List[str]) -> torch.Tensor:
        """Wrapper for standalone function"""
        return create_legal_actions_mask(legal_actions, self.device)
    
    def _convert_action(self, action_label: str, state: Dict[str, Any]) -> Tuple[str, int]:
        """Wrapper for standalone function"""
        return convert_action_label(action_label, state)


def load_model_agent(
    player_id: str,
    name: str,
    model_path: str = None,
    model = None,
    temperature: float = 1.0,
    deterministic: bool = False,
    device: torch.device = None
) -> ModelAgent:
    """
    Factory function to create a ModelAgent.
    
    Args:
        player_id: Player ID
        name: Player name  
        model_path: Path to trained model
        model: Pre-loaded model
        temperature: Sampling temperature
        deterministic: Use deterministic action selection
        device: Device
    
    Returns:
        ModelAgent instance
    """
    return ModelAgent(
        player_id=player_id,
        name=name,
        model_path=model_path,
        model=model,
        temperature=temperature,
        deterministic=deterministic,
        device=device
    )


class RandomAgent:
    """
    Agent that selects random valid actions.
    Useful for baselines and initial training.
    """
    def __init__(self, player_id: str, name: str):
        self.player_id = player_id
        self.name = name
    
    def reset_hand(self):
        pass
        
    def observe_action(self, player_id: str, action_type: str, amount: int, pot: int, stage: str):
        pass
        
    def select_action(self, state: Dict[str, Any], legal_actions: List[str]) -> Tuple[str, int, str]:
        action = random.choice(legal_actions)
        
        if action == 'bet':
            size_fractions = [0.5, 0.75, 1.0]
            size_fraction = random.choice(size_fractions)
            action_label = f'bet_{int(size_fraction*100)}%'
        elif action == 'raise':
            size_fractions = [0.5, 0.75, 1.0]
            size_fraction = random.choice(size_fractions)
            action_label = f'raise_{int(size_fraction*100)}%'
        else:
            action_label = action
            
        action_type, amount = convert_action_label(action_label, state)
        return action_type, amount, action_label


class HeuristicAgent:
    """
    Rule-based agent for baseline comparison.
    Implements a simple aggressive strategy.
    """
    def __init__(self, player_id: str, name: str):
        self.player_id = player_id
        self.name = name
        
    def reset_hand(self):
        pass
        
    def observe_action(self, player_id: str, action_type: str, amount: int, pot: int, stage: str):
        pass
        
    def _parse_card_rank(self, card) -> str:
        """Parse card rank from either string format ('9H', 'TH') or dict format."""
        if isinstance(card, str):
            # String format: first char is rank (T for 10)
            return card[0] if card else '2'
        elif isinstance(card, dict):
            return card.get('rank', '2')
        return '2'
    
    def _get_hand_strength(self, hole_cards: List, community_cards: List) -> float:
        """
        Estimate hand strength (0.0 to 1.0).
        Simple proxy: pair check, high card, etc.
        For a real implementation, we'd use a hand evaluator library.
        """
        # Placeholder for simple heuristic
        # Return random strength for now, biased by card ranks
        if not hole_cards or len(hole_cards) < 2:
            return 0.0
            
        # Parse ranks - handles both string format ("9H") and dict format ({"rank": "9", "suit": "H"})
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 
                   'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        ranks = []
        for c in hole_cards:
            r = self._parse_card_rank(c)
            ranks.append(rank_map.get(r, 2))
            
        h1, h2 = sorted(ranks, reverse=True)
        
        # Preflop heuristic
        if not community_cards:
            # Pocket pair
            if h1 == h2:
                return 0.5 + (h1 / 14.0) * 0.5
            
            # High cards
            return (h1 / 14.0) * 0.6 + (h2 / 14.0) * 0.2
            
        # Postflop - use deterministic heuristic based on board texture
        # FIXED: Removed random.random() component that was causing high variance
        # and making the agent behavior unpredictable (harder to exploit consistently)
        # In a real bot we'd use a proper hand evaluator
        
        # Check for pairs with board
        comm_ranks = []
        for c in community_cards:
            r = self._parse_card_rank(c)
            comm_ranks.append(rank_map.get(r, 2))
        
        # Check if we paired the board
        paired_high = h1 in comm_ranks
        paired_low = h2 in comm_ranks
        
        base_strength = (h1 / 14.0) * 0.3 + (h2 / 14.0) * 0.15
        
        if paired_high and paired_low:
            # Two pair
            return min(0.85, base_strength + 0.45)
        elif paired_high:
            # Top pair
            return min(0.75, base_strength + 0.30)
        elif paired_low:
            # Bottom pair
            return min(0.60, base_strength + 0.20)
        else:
            # No pair - mostly high card strength with small board-dependent bonus
            num_high_cards_on_board = sum(1 for r in comm_ranks if r >= 10)
            board_scare = num_high_cards_on_board * 0.03
            return max(0.15, min(0.50, base_strength + 0.10 - board_scare))

    def select_action(self, state: Dict[str, Any], legal_actions: List[str]) -> Tuple[str, int, str]:
        # Get hand strength
        strength = self._get_hand_strength(state.get('hole_cards', []), state.get('community_cards', []))
        
        # Determine desired action based on strength
        can_check = 'check' in legal_actions
        can_call = 'call' in legal_actions
        can_bet = 'bet' in legal_actions
        can_raise = 'raise' in legal_actions
        
        # Very strong hand -> Bet/Raise
        if strength > 0.8:
            if can_raise:
                action_label = 'raise_100%'
            elif can_bet:
                action_label = 'raise_75%'
            elif can_call:
                action_label = 'call'
            else:
                action_label = 'check' if can_check else 'fold'
                
        # Strong hand -> Bet small / Call
        elif strength > 0.6:
            if can_bet:
                action_label = 'raise_50%'
            elif can_call:
                action_label = 'call'
            else:
                action_label = 'check' if can_check else 'fold'
        
        # Medium hand -> Check/Call if cheap
        elif strength > 0.4:
            if can_check:
                action_label = 'check'
            elif can_call:
                # Call if not too expensive relative to stack (simplified)
                action_label = 'call'
            else:
                action_label = 'fold'
        
        # Weak hand -> Check/Fold (bluff occasionally)
        else:
            if random.random() < 0.1 and can_bet: # Bluff 10%
                action_label = 'raise_50%'
            elif can_check:
                action_label = 'check'
            else:
                action_label = 'fold'
        
        # Fallback validation
        action_type, amount = convert_action_label(action_label, state)
        
        # If conversion failed or action not legal (e.g. raise not legal), fallback
        base_type = action_type
        if base_type == 'bet' and not can_bet:
             action_label = 'check' if can_check else 'fold'
        elif base_type == 'raise' and not can_raise:
             action_label = 'call' if can_call else 'fold'
             
        # Re-convert
        action_type, amount = convert_action_label(action_label, state)
        
        return action_type, amount, action_label


class TightAgent:
    """
    Tight-aggressive agent that only plays premium hands.
    
    This agent is crucial for training because it teaches the model that:
    - Not all aggression works (tight player folds weak hands)
    - Bluffing has diminishing returns against tight players
    - Need to have actual hand strength to win
    """
    def __init__(self, player_id: str, name: str):
        self.player_id = player_id
        self.name = name
        
    def reset_hand(self):
        pass
        
    def observe_action(self, player_id: str, action_type: str, amount: int, pot: int, stage: str):
        pass
        
    def _parse_card(self, card) -> Tuple[int, int]:
        """Parse card into (rank, suit) values."""
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 
                   'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        suit_map = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
        
        if isinstance(card, str):
            rank = rank_map.get(card[0], 2)
            suit = suit_map.get(card[1], 0) if len(card) > 1 else 0
        elif isinstance(card, dict):
            rank = rank_map.get(card.get('rank', '2'), 2)
            suit = suit_map.get(card.get('suit', 'C'), 0)
        else:
            rank, suit = 2, 0
        return rank, suit
    
    def _get_preflop_strength(self, hole_cards: List) -> float:
        """Get preflop hand strength (0-1)."""
        if not hole_cards or len(hole_cards) < 2:
            return 0.0
            
        ranks_suits = [self._parse_card(c) for c in hole_cards]
        ranks = sorted([r for r, s in ranks_suits], reverse=True)
        suits = [s for r, s in ranks_suits]
        
        h1, h2 = ranks[0], ranks[1]
        is_suited = suits[0] == suits[1]
        is_pair = h1 == h2
        
        # Premium pairs: AA, KK, QQ, JJ
        if is_pair and h1 >= 11:
            return 0.9 + (h1 - 11) * 0.025
        
        # Medium pairs: TT-77
        if is_pair and h1 >= 7:
            return 0.6 + (h1 - 7) * 0.05
        
        # Small pairs: 66-22
        if is_pair:
            return 0.35 + (h1 - 2) * 0.03
        
        # Big Ace (AK, AQ, AJ, AT)
        if h1 == 14:  # Ace
            if h2 >= 10:
                base = 0.65 + (h2 - 10) * 0.05
                return base + (0.05 if is_suited else 0)
            elif h2 >= 7:
                return 0.45 + (0.05 if is_suited else 0)
            else:
                return 0.30 + (0.06 if is_suited else 0)
        
        # Broadway cards
        if h1 >= 10 and h2 >= 10:
            return 0.50 + (0.05 if is_suited else 0)
        
        # Suited connectors
        if is_suited and abs(h1 - h2) == 1:
            return 0.35 + (h1 / 14) * 0.1
        
        # Other suited cards
        if is_suited:
            return 0.25 + (h1 / 14) * 0.1
        
        # Offsuit connected
        if abs(h1 - h2) <= 2:
            return 0.20 + (h1 / 14) * 0.1
        
        # Trash
        return 0.15

    def select_action(self, state: Dict[str, Any], legal_actions: List[str]) -> Tuple[str, int, str]:
        hole_cards = state.get('hole_cards', [])
        community_cards = state.get('community_cards', [])
        
        can_check = 'check' in legal_actions
        can_call = 'call' in legal_actions
        can_bet = 'bet' in legal_actions
        can_raise = 'raise' in legal_actions
        
        # Preflop - tight hand selection
        if not community_cards:
            strength = self._get_preflop_strength(hole_cards)
            
            # Only play premium hands (top ~15%)
            if strength >= 0.65:
                # Premium - raise/3-bet
                if can_raise:
                    action_label = 'raise_75%'
                elif can_bet:
                    action_label = 'raise_75%'
                elif can_call:
                    action_label = 'call'
                else:
                    action_label = 'check' if can_check else 'fold'
            elif strength >= 0.50:
                # Strong but not premium - call or small raise
                if can_call:
                    action_label = 'call'
                elif can_check:
                    action_label = 'check'
                else:
                    action_label = 'fold'
            elif strength >= 0.35:
                # Medium - only call in position, otherwise fold
                if can_check:
                    action_label = 'check'
                else:
                    action_label = 'fold'
            else:
                # Weak/trash - always fold if facing bet
                action_label = 'check' if can_check else 'fold'
        else:
            # Postflop - simplified: bet/raise strong, check/fold weak
            # Use simple heuristic for postflop strength
            strength = self._get_preflop_strength(hole_cards)  # Simplified
            
            to_call = state.get('to_call', 0)
            pot = state.get('pot', 0)
            pot_odds = to_call / max(1, pot + to_call) if to_call > 0 else 0
            
            # Strong made hand - bet for value
            if strength >= 0.65:
                if can_bet:
                    action_label = 'raise_50%'
                elif can_raise:
                    action_label = 'raise_50%'
                elif can_call:
                    action_label = 'call'
                else:
                    action_label = 'check' if can_check else 'fold'
            # Medium - play passively
            elif strength >= 0.45:
                if can_check:
                    action_label = 'check'
                elif can_call and pot_odds < 0.3:
                    action_label = 'call'
                else:
                    action_label = 'fold'
            # Weak - check/fold
            else:
                action_label = 'check' if can_check else 'fold'
        
        action_type, amount = convert_action_label(action_label, state)
        return action_type, amount, action_label


class LoosePassiveAgent:
    """
    Loose-passive agent that calls too much but rarely raises.
    
    This agent is useful for training because:
    - Teaches value of thin value bets (they call with weak hands)
    - Shows that passive play is exploitable
    - Provides contrast to tight aggressive play
    """
    def __init__(self, player_id: str, name: str):
        self.player_id = player_id
        self.name = name
        
    def reset_hand(self):
        pass
        
    def observe_action(self, player_id: str, action_type: str, amount: int, pot: int, stage: str):
        pass

    def select_action(self, state: Dict[str, Any], legal_actions: List[str]) -> Tuple[str, int, str]:
        can_check = 'check' in legal_actions
        can_call = 'call' in legal_actions
        can_bet = 'bet' in legal_actions
        can_raise = 'raise' in legal_actions
        
        # Loose-passive: calls most of the time, rarely bets/raises
        roll = random.random()
        
        # Can check - usually check
        if can_check:
            if roll < 0.85:
                action_label = 'check'
            elif can_bet:
                action_label = 'raise_33%'  # Small bet sometimes
            else:
                action_label = 'check'
        # Facing bet - call most of the time
        elif can_call:
            if roll < 0.75:
                action_label = 'call'
            elif roll < 0.80 and can_raise:
                action_label = 'raise_50%'  # Rare raise
            else:
                action_label = 'fold'  # Sometimes fold
        else:
            action_label = 'fold'
        
        action_type, amount = convert_action_label(action_label, state)
        return action_type, amount, action_label


class AggressiveAgent:
    """
    Loose-aggressive agent that bets and raises frequently.
    
    This agent is useful for training because:
    - Tests model's ability to call down with marginal hands
    - Shows that not all aggression has real hands behind it
    - Provides pressure scenarios
    """
    def __init__(self, player_id: str, name: str):
        self.player_id = player_id
        self.name = name
        
    def reset_hand(self):
        pass
        
    def observe_action(self, player_id: str, action_type: str, amount: int, pot: int, stage: str):
        pass

    def select_action(self, state: Dict[str, Any], legal_actions: List[str]) -> Tuple[str, int, str]:
        can_check = 'check' in legal_actions
        can_call = 'call' in legal_actions
        can_bet = 'bet' in legal_actions
        can_raise = 'raise' in legal_actions
        
        roll = random.random()
        
        # Aggressive: bets and raises frequently
        if can_bet:
            if roll < 0.65:
                # Bet most of the time when we can
                sizes = ['raise_50%', 'raise_75%', 'raise_100%']
                action_label = random.choice(sizes)
            elif roll < 0.85:
                action_label = 'check' if can_check else random.choice(['raise_50%', 'raise_75%'])
            else:
                action_label = 'check' if can_check else 'raise_50%'
        elif can_raise:
            if roll < 0.55:
                # Raise frequently
                sizes = ['raise_50%', 'raise_75%', 'raise_100%']
                action_label = random.choice(sizes)
            elif roll < 0.85:
                action_label = 'call'
            else:
                action_label = 'fold'
        elif can_call:
            if roll < 0.70:
                action_label = 'call'
            else:
                action_label = 'fold'
        elif can_check:
            action_label = 'check'
        else:
            action_label = 'fold'
        
        action_type, amount = convert_action_label(action_label, state)
        return action_type, amount, action_label


class CallingStationAgent:
    """
    Calling Station agent that calls almost everything, especially all-ins.
    
    CRITICAL FOR TRAINING: This agent is essential for teaching the model that
    all-in with weak hands LOSES MONEY. If the model only plays against folding
    opponents, it will learn that all-in "works" because opponents fold.
    
    This agent CALLS most bets and raises, including all-ins, to show the model
    that trash hands lose at showdown.
    """
    def __init__(self, player_id: str, name: str):
        self.player_id = player_id
        self.name = name
        
    def reset_hand(self):
        pass
        
    def observe_action(self, player_id: str, action_type: str, amount: int, pot: int, stage: str):
        pass
    
    def _parse_card(self, card) -> Tuple[int, int]:
        """Parse card into (rank, suit) values."""
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 
                   'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        suit_map = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
        
        if isinstance(card, str):
            rank = rank_map.get(card[0], 2)
            suit = suit_map.get(card[1], 0) if len(card) > 1 else 0
        elif isinstance(card, dict):
            rank = rank_map.get(card.get('rank', '2'), 2)
            suit = suit_map.get(card.get('suit', 'C'), 0)
        else:
            rank, suit = 2, 0
        return rank, suit
    
    def _get_hand_strength(self, hole_cards: List) -> float:
        """Simple hand strength estimate (higher = better)."""
        if not hole_cards or len(hole_cards) < 2:
            return 0.3
        
        ranks_suits = [self._parse_card(c) for c in hole_cards]
        ranks = sorted([r for r, s in ranks_suits], reverse=True)
        suits = [s for r, s in ranks_suits]
        
        h1, h2 = ranks[0], ranks[1]
        is_suited = suits[0] == suits[1]
        is_pair = h1 == h2
        
        # Pairs
        if is_pair:
            return 0.5 + (h1 / 14) * 0.4
        
        # High cards
        base = (h1 / 14) * 0.4 + (h2 / 14) * 0.2
        if is_suited:
            base += 0.08
        if abs(h1 - h2) <= 2:
            base += 0.05
        
        return max(0.15, min(0.85, base))

    def select_action(self, state: Dict[str, Any], legal_actions: List[str]) -> Tuple[str, int, str]:
        """
        Calling Station strategy: Call almost everything with any reasonable hand.
        Key: This agent CALLS ALL-INS frequently to punish weak all-in strategies.
        """
        can_check = 'check' in legal_actions
        can_call = 'call' in legal_actions
        can_bet = 'bet' in legal_actions
        can_raise = 'raise' in legal_actions
        
        hole_cards = state.get('hole_cards', [])
        hand_strength = self._get_hand_strength(hole_cards)
        to_call = state.get('to_call', 0)
        pot = state.get('pot', 0)
        player_chips = state.get('player_chips', 0)
        
        # Calculate if this is an all-in situation (to_call >= player_chips)
        is_facing_all_in = to_call >= player_chips * 0.5  # Facing large bet
        
        roll = random.random()
        
        # Check if we can
        if can_check:
            if roll < 0.10 and can_bet:
                # Occasionally bet (10%)
                action_label = 'raise_33%'
            else:
                action_label = 'check'
        # Facing bet or raise - CALL FREQUENTLY
        elif can_call:
            # CRITICAL: Call even all-ins with any decent hand
            # This is what punishes trashy all-in strategies
            if is_facing_all_in:
                # Facing all-in: call if hand is remotely decent
                if hand_strength >= 0.25:  # Call with almost anything
                    action_label = 'call'
                elif roll < 0.40:  # Even call some trash (40% of time)
                    action_label = 'call'
                else:
                    action_label = 'fold'
            else:
                # Regular bet: almost always call
                if hand_strength >= 0.20:
                    action_label = 'call'
                elif roll < 0.60:  # Call 60% of trash too
                    action_label = 'call'
                else:
                    action_label = 'fold'
        else:
            action_label = 'fold'
        
        action_type, amount = convert_action_label(action_label, state)
        return action_type, amount, action_label


class HeroCallerAgent:
    """
    Hero Caller agent that specifically calls down suspected bluffs.
    
    This agent evaluates hand strength more carefully and makes "hero calls"
    with medium-strength hands when facing aggression. This teaches the model
    that bluffing and aggressive all-ins don't always work.
    """
    def __init__(self, player_id: str, name: str):
        self.player_id = player_id
        self.name = name
        self.opponent_aggression_count = 0  # Track opponent aggression this hand
        
    def reset_hand(self):
        self.opponent_aggression_count = 0
        
    def observe_action(self, player_id: str, action_type: str, amount: int, pot: int, stage: str):
        # Track opponent aggression
        if player_id != self.player_id and action_type in ['bet', 'raise', 'all_in']:
            self.opponent_aggression_count += 1
    
    def _parse_card(self, card) -> Tuple[int, int]:
        """Parse card into (rank, suit) values."""
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 
                   'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        suit_map = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
        
        if isinstance(card, str):
            rank = rank_map.get(card[0], 2)
            suit = suit_map.get(card[1], 0) if len(card) > 1 else 0
        elif isinstance(card, dict):
            rank = rank_map.get(card.get('rank', '2'), 2)
            suit = suit_map.get(card.get('suit', 'C'), 0)
        else:
            rank, suit = 2, 0
        return rank, suit
    
    def _get_hand_strength(self, hole_cards: List, community_cards: List) -> float:
        """Estimate hand strength (0-1)."""
        if not hole_cards or len(hole_cards) < 2:
            return 0.3
        
        ranks_suits = [self._parse_card(c) for c in hole_cards]
        ranks = sorted([r for r, s in ranks_suits], reverse=True)
        suits = [s for r, s in ranks_suits]
        
        h1, h2 = ranks[0], ranks[1]
        is_suited = suits[0] == suits[1]
        is_pair = h1 == h2
        
        # Preflop strength
        if not community_cards:
            if is_pair:
                return 0.5 + (h1 / 14) * 0.45
            base = (h1 / 14) * 0.4 + (h2 / 14) * 0.2
            if is_suited:
                base += 0.08
            return max(0.15, min(0.75, base))
        
        # Postflop: simplified - add pair bonus if we hit
        comm_ranks = [self._parse_card(c)[0] for c in community_cards]
        paired = h1 in comm_ranks or h2 in comm_ranks
        two_pair = h1 in comm_ranks and h2 in comm_ranks
        
        base = (h1 / 14) * 0.25 + (h2 / 14) * 0.1
        if is_pair:
            base += 0.20
        if paired:
            base += 0.25
        if two_pair:
            base += 0.35
        
        return max(0.15, min(0.90, base))

    def select_action(self, state: Dict[str, Any], legal_actions: List[str]) -> Tuple[str, int, str]:
        """
        Hero Caller strategy: Call down with medium+ hands when opponent is aggressive.
        The more aggressive the opponent has been, the lighter we call.
        """
        can_check = 'check' in legal_actions
        can_call = 'call' in legal_actions
        can_bet = 'bet' in legal_actions
        can_raise = 'raise' in legal_actions
        
        hole_cards = state.get('hole_cards', [])
        community_cards = state.get('community_cards', [])
        hand_strength = self._get_hand_strength(hole_cards, community_cards)
        to_call = state.get('to_call', 0)
        pot = state.get('pot', 0)
        
        # Adjust call threshold based on opponent aggression
        # More aggressive opponent = lighter calls
        call_threshold = max(0.20, 0.45 - (self.opponent_aggression_count * 0.10))
        
        roll = random.random()
        
        if can_check:
            if hand_strength >= 0.55 and can_bet and roll < 0.50:
                action_label = 'raise_50%'
            else:
                action_label = 'check'
        elif can_call:
            # Hero call logic: call more loosely when opponent is aggressive
            if hand_strength >= call_threshold:
                action_label = 'call'
            elif roll < 0.20:  # Occasional light call even with weak hands
                action_label = 'call'
            else:
                action_label = 'fold'
        else:
            action_label = 'fold'
        
        action_type, amount = convert_action_label(action_label, state)
        return action_type, amount, action_label


class SimpleAgent:
    """
    Base class for simple stateless heuristic agents.
    
    Provides common __init__, reset_hand, and observe_action methods.
    Subclasses only need to implement _choose_action_label().
    """
    def __init__(self, player_id: str, name: str):
        self.player_id = player_id
        self.name = name
        
    def reset_hand(self):
        pass
        
    def observe_action(self, player_id: str, action_type: str, amount: int, pot: int, stage: str):
        pass

    def _choose_action_label(self, state: Dict[str, Any], legal_actions: List[str]) -> str:
        """Override in subclasses to return the chosen action label."""
        raise NotImplementedError
    
    def select_action(self, state: Dict[str, Any], legal_actions: List[str]) -> Tuple[str, int, str]:
        """Select action using subclass's _choose_action_label."""
        action_label = self._choose_action_label(state, legal_actions)
        action_type, amount = convert_action_label(action_label, state)
        return action_type, amount, action_label


class AlwaysRaiseAgent(SimpleAgent):
    """
    Always raises/bets when possible, otherwise call/check.
    Useful for testing model robustness against hyper-aggressive opponents.
    """
    def _choose_action_label(self, state: Dict[str, Any], legal_actions: List[str]) -> str:
        if 'raise' in legal_actions:
            return 'raise_100%'
        if 'bet' in legal_actions:
            return 'raise_100%'
        if 'all_in' in legal_actions:
            return 'all_in'
        if 'call' in legal_actions:
            return 'call'
        if 'check' in legal_actions:
            return 'check'
        return 'fold'


class AlwaysCallAgent(SimpleAgent):
    """
    Always checks or calls, never bets/raises.
    Useful for testing value betting - this agent will call you down.
    """
    def _choose_action_label(self, state: Dict[str, Any], legal_actions: List[str]) -> str:
        if 'check' in legal_actions:
            return 'check'
        if 'call' in legal_actions:
            return 'call'
        return 'fold'


class AlwaysFoldAgent(SimpleAgent):
    """
    Always folds when facing a bet, checks if free.
    Useful for testing if the model learns to steal blinds and apply pressure.
    """
    def _choose_action_label(self, state: Dict[str, Any], legal_actions: List[str]) -> str:
        if 'check' in legal_actions:
            return 'check'
        return 'fold'

