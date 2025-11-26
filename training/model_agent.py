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


def calculate_bet_amount(
    pot: int,
    size_fraction: float,
    min_amount: int,
    max_amount: int
) -> int:
    """Calculate bet amount based on pot size and constraints"""
    target = int(pot * size_fraction)
    amount = max(min_amount, target)
    amount = min(max_amount, amount)
    return amount


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
    # Simple actions
    if action_label in ['fold', 'check', 'call', 'all_in']:
        return action_label, 0
    
    # Bet actions
    if action_label.startswith('bet_'):
        pot = state.get('pot', 0)
        min_bet = state.get('min_bet', state.get('big_blind', 20))
        max_bet = state.get('player_chips', 0)
        
        # Get bet size fraction
        size_fraction = BET_SIZE_MAP.get(action_label, 0.5)
        
        # Calculate amount
        amount = calculate_bet_amount(pot, size_fraction, min_bet, max_bet)
        
        return 'bet', amount
    
    # Raise actions
    if action_label.startswith('raise_'):
        pot = state.get('pot', 0)
        player_bet = state.get('player_bet', 0)
        player_chips = state.get('player_chips', 0)
        current_bet = state.get('current_bet', 0)
        to_call = current_bet - player_bet
        
        # Get raise size fraction
        size_fraction = BET_SIZE_MAP.get(action_label, 0.5)
        raise_size = int(pot * size_fraction)
        
        # The amount is: call + raise (this is ADDITIONAL chips to add)
        # Total bet will be: player_bet + amount
        amount = to_call + raise_size
        
        # Ensure we meet minimum raise requirement
        # min_raise_total is the total chips needed (including call)
        min_raise_total = state.get('min_raise_total', state.get('big_blind', 20))
        if amount < min_raise_total:
            amount = min_raise_total
        
        # Cap at player's available chips
        amount = min(amount, player_chips)
        
        # CRITICAL FIX: If after capping we can't meet minimum raise, go all-in instead
        # This prevents invalid raises that the API will reject
        if amount < min_raise_total:
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
        model_path: str,
        device: torch.device = None,
        temperature: float = 1.0,
        deterministic: bool = False
    ):
        """
        Args:
            player_id: Player ID
            name: Player name
            model_path: Path to trained model checkpoint
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
        
        # Load model
        self.model, self.encoder = self._load_model(model_path)
        self.model.eval()  # Set to evaluation mode
    
    def _load_model(self, model_path: str) -> Tuple[PokerActorCritic, RLStateEncoder]:
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model configuration from checkpoint
        input_dim = checkpoint.get('input_dim', 155)  # Default to RL encoder dim
        hidden_dim = checkpoint.get('hidden_dim', 256)
        num_heads = checkpoint.get('num_heads', 8)
        num_layers = checkpoint.get('num_layers', 4)
        dropout = checkpoint.get('dropout', 0.1)
        
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
    
    def _calculate_bet_amount(self, pot: int, size_fraction: float, min_amount: int, max_amount: int) -> int:
        """Wrapper for standalone function"""
        return calculate_bet_amount(pot, size_fraction, min_amount, max_amount)


def load_model_agent(
    player_id: str,
    name: str,
    model_path: str,
    temperature: float = 1.0,
    deterministic: bool = False
) -> ModelAgent:
    """
    Factory function to create a ModelAgent.
    
    Args:
        player_id: Player ID
        name: Player name  
        model_path: Path to trained model
        temperature: Sampling temperature
        deterministic: Use deterministic action selection
    
    Returns:
        ModelAgent instance
    """
    return ModelAgent(
        player_id=player_id,
        name=name,
        model_path=model_path,
        temperature=temperature,
        deterministic=deterministic
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
