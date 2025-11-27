#!/usr/bin/env python3
"""
Model-based Agent for Poker

This agent uses a trained neural network model to select actions.
It integrates with the existing agent framework and can be used for:
- Self-play training
- Evaluation against other agents
- Actual gameplay
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from pathlib import Path
import random

from rl_state_encoder import RLStateEncoder
from rl_model import PokerActorCritic


# Action mapping (same as train.py)
ACTION_MAP = {
    'fold': 0, 'check': 1, 'call': 2,
    'bet_10%': 3, 'bet_25%': 4, 'bet_33%': 5, 'bet_50%': 6, 'bet_75%': 7,
    'bet_100%': 8, 'bet_150%': 9, 'bet_200%': 10, 'bet_300%': 11,
    'raise_10%': 12, 'raise_25%': 13, 'raise_33%': 14, 'raise_50%': 15, 'raise_75%': 16,
    'raise_100%': 17, 'raise_150%': 18, 'raise_200%': 19, 'raise_300%': 20,
    'all_in': 21
}

ACTION_NAMES = [
    'fold', 'check', 'call',
    'bet_10%', 'bet_25%', 'bet_33%', 'bet_50%', 'bet_75%', 'bet_100%', 'bet_150%', 'bet_200%', 'bet_300%',
    'raise_10%', 'raise_25%', 'raise_33%', 'raise_50%', 'raise_75%', 'raise_100%', 'raise_150%', 'raise_200%', 'raise_300%',
    'all_in'
]

# Bet size percentages
BET_SIZE_MAP = {
    'bet_10%': 0.10, 'bet_25%': 0.25, 'bet_33%': 0.33, 'bet_50%': 0.50, 'bet_75%': 0.75,
    'bet_100%': 1.0, 'bet_150%': 1.5, 'bet_200%': 2.0, 'bet_300%': 3.0,
    'raise_10%': 0.10, 'raise_25%': 0.25, 'raise_33%': 0.33, 'raise_50%': 0.50, 'raise_75%': 0.75,
    'raise_100%': 1.0, 'raise_150%': 1.5, 'raise_200%': 2.0, 'raise_300%': 3.0,
}


def convert_action_label(
    action_label: str,
    state: Dict[str, Any]
) -> Tuple[str, int]:
    """
    Convert action label to (action_type, amount).
    
    Args:
        action_label: Action label (e.g., 'bet_50%', 'call')
        state: Game state
    
    Returns:
        Tuple of (action_type, amount)
    """
    player_chips = state.get('player_chips', 0)
    
    # Simple actions
    if action_label in ['fold', 'check', 'call', 'all_in']:
        return action_label, 0
    
    # Bet actions
    if action_label.startswith('bet_'):
        pot = state.get('pot', 0)
        min_bet = state.get('min_bet', state.get('big_blind', 20))
        
        # Get bet size fraction
        size_fraction = BET_SIZE_MAP.get(action_label, 0.5)
        
        # Calculate amount based on pot size
        target_amount = int(pot * size_fraction) if pot > 0 else min_bet
        amount = max(min_bet, target_amount)
        
        # If bet would use all or most of our chips, go all-in instead
        if amount >= player_chips:
            return 'all_in', 0
        
        return 'bet', amount
    
    # Raise actions
    if action_label.startswith('raise_'):
        pot = state.get('pot', 0)
        player_bet = state.get('player_bet', 0)
        current_bet = state.get('current_bet', 0)
        to_call = max(0, current_bet - player_bet)
        min_raise = state.get('min_raise_total', state.get('big_blind', 20))
        
        # Get raise size fraction
        size_fraction = BET_SIZE_MAP.get(action_label, 0.5)
        raise_size = int(pot * size_fraction) if pot > 0 else min_raise
        
        # The amount is the additional chips to add (call + raise increment)
        amount = to_call + raise_size
        
        # Ensure we meet minimum raise requirement
        if amount < min_raise:
            amount = min_raise
        
        # If raise would use all or most of our chips, go all-in
        if amount >= player_chips:
            return 'all_in', 0
        
        # If we can't afford the minimum raise, go all-in instead
        if player_chips < min_raise:
            return 'all_in', 0
        
        return 'raise', amount
    
    # Fallback
    return 'check', 0


def create_legal_actions_mask(legal_actions: List[str], device: torch.device) -> torch.Tensor:
    """
    Create a boolean mask for legal actions.
    
    Args:
        legal_actions: List of legal action strings (e.g., ['fold', 'call', 'raise'])
        device: Torch device
    
    Returns:
        Boolean tensor of shape (1, num_actions)
    """
    mask = torch.zeros(1, len(ACTION_NAMES), dtype=torch.bool, device=device)
    
    for action in legal_actions:
        if action in ACTION_MAP:
            # Simple action (fold, check, call, all_in)
            mask[0, ACTION_MAP[action]] = True
        elif action == 'bet':
            # Enable all bet sizes
            for action_name in ACTION_NAMES:
                if action_name.startswith('bet_'):
                    mask[0, ACTION_MAP[action_name]] = True
        elif action == 'raise':
            # Enable all raise sizes
            for action_name in ACTION_NAMES:
                if action_name.startswith('raise_'):
                    mask[0, ACTION_MAP[action_name]] = True
    
    return mask


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
        'starting_chips': config.get('startingChips', 1000),
        'to_call': action_constraints.get('toCall', 0),
        'min_bet': action_constraints.get('minBet', 20),
        'min_raise_total': action_constraints.get('minRaiseTotal', 20),
    }


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
        model: PokerActorCritic = None,
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
    
    def _load_model(self, model_path: str) -> Tuple[PokerActorCritic, RLStateEncoder]:
        """Load trained model from checkpoint"""
        # weights_only=False is required to load checkpoints with numpy scalars
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get model configuration from checkpoint
        input_dim = checkpoint.get('input_dim', 155)  # Default to RL encoder dim
        dropout = checkpoint.get('dropout', 0.1)

        # Try to infer model architecture from state_dict if not in checkpoint
        state_dict = checkpoint.get('model_state_dict', {})
        
        if 'hidden_dim' in checkpoint:
            hidden_dim = checkpoint['hidden_dim']
            num_heads = checkpoint.get('num_heads', 8)
            num_layers = checkpoint.get('num_layers', 4)
        elif 'pos_encoding' in state_dict:
            # Infer from state dict
            # pos_encoding shape: [1, 14, hidden_dim]
            hidden_dim = state_dict['pos_encoding'].shape[2]
            
            # Infer num_layers
            max_layer = 0
            for key in state_dict.keys():
                if key.startswith('transformer.layers.'):
                    try:
                        layer_idx = int(key.split('.')[2])
                        max_layer = max(max_layer, layer_idx)
                    except (IndexError, ValueError):
                        pass
            num_layers = max_layer + 1
            
            # Default num_heads to 8 (standard for this project)
            num_heads = 8
            
            # Ensure hidden_dim is divisible by num_heads
            if hidden_dim % num_heads != 0:
                # Try 4 heads if 8 doesn't work
                if hidden_dim % 4 == 0:
                    num_heads = 4
        else:
            # Fallback to defaults
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
        legal_actions: List[str]
    ) -> Tuple[str, int, str]:
        """
        Select an action using the trained model.
        
        Args:
            state: Game state dictionary
            legal_actions: List of legal action strings
        
        Returns:
            Tuple of (action_type, amount, action_label)
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
        
        return action_type, amount, action_label
    
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
    model: PokerActorCritic = None,
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
            
        # Postflop - just random noise mixed with preflop strength for this simple baseline
        # In a real bot we'd evaluate the full hand
        return (h1 / 14.0) * 0.4 + (h2 / 14.0) * 0.1 + random.random() * 0.5

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
                action_label = 'bet_75%'
            elif can_call:
                action_label = 'call'
            else:
                action_label = 'check' if can_check else 'fold'
                
        # Strong hand -> Bet small / Call
        elif strength > 0.6:
            if can_bet:
                action_label = 'bet_50%'
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
                action_label = 'bet_50%'
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
