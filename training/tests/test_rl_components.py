
import pytest
import torch

from rl_state_encoder import RLStateEncoder
from rl_model import PokerActorCritic, create_actor_critic

def test_rl_state_encoder():
    """Test that state encoder produces correct shape"""
    encoder = RLStateEncoder()
    expected_dim = 167  # Updated dimension
    assert encoder.get_feature_dim() == expected_dim
    
    # dummy state
    state = {
        'hole_cards': ['Ah', 'Kd'],
        'community_cards': ['Th', 'Jh', 'Qh'],
        'pot': 100,
        'current_bet': 20,
        'player_chips': 900,
        'player_bet': 20,
        'stage': 'Flop',
        'num_players': 2,
        'position': 0,
        'big_blind': 20,
        'to_call': 0
    }
    
    features = encoder.encode_state(state)
    assert isinstance(features, torch.Tensor)
    assert features.shape == (expected_dim,)

def test_poker_actor_critic():
    """Test that model forward pass works"""
    input_dim = 167
    hidden_dim = 64
    model = create_actor_critic(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_layers=2
    )
    
    # Batch of 2
    batch_size = 2
    x = torch.randn(batch_size, input_dim)
    
    # Forward
    action_logits, value = model(x)
    
    assert action_logits.shape == (batch_size, 22)  # 22 actions
    assert value.shape == (batch_size, 1)
    
    # Probabilities
    probs, val = model.get_action_probs(x)
    assert probs.shape == (batch_size, 22)
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size))

