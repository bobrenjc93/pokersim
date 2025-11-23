"""
Configuration presets for RL training data generation.

Import and use presets:
    from config import get_preset
    config = get_preset('heads_up')
    
Or customize game configs:
    from config import DEFAULT_GAME_CONFIG
    my_config = DEFAULT_GAME_CONFIG.copy()
    my_config['startingChips'] = 2000
"""

# =============================================================================
# Model Version
# =============================================================================

# Model architecture versions (imported from train.py for consistency)
# This determines which directories to use for data and models
MODEL_VERSION = 4  # Current model version (1=legacy feed-forward, 2=transformer)

# =============================================================================
# Game Configuration
# =============================================================================

DEFAULT_GAME_CONFIG = {
    'smallBlind': 10,
    'bigBlind': 20,
    'startingChips': 1000,
    'minPlayers': 2,
    'maxPlayers': 10,
}

# High stakes game
HIGH_STAKES_CONFIG = {
    'smallBlind': 50,
    'bigBlind': 100,
    'startingChips': 5000,
    'minPlayers': 2,
    'maxPlayers': 10,
}

# Short stack game
SHORT_STACK_CONFIG = {
    'smallBlind': 10,
    'bigBlind': 20,
    'startingChips': 200,  # Only 10 big blinds
    'minPlayers': 2,
    'maxPlayers': 10,
}

# Deep stack game
DEEP_STACK_CONFIG = {
    'smallBlind': 5,
    'bigBlind': 10,
    'startingChips': 5000,  # 500 big blinds
    'minPlayers': 2,
    'maxPlayers': 10,
}

# Tournament settings (increasing blinds could be added)
TOURNAMENT_CONFIG = {
    'smallBlind': 25,
    'bigBlind': 50,
    'startingChips': 1500,
    'minPlayers': 6,
    'maxPlayers': 9,
}


# =============================================================================
# Agent Configuration
# =============================================================================

# Available agent types for training data generation
AGENT_TYPES = [
    'random',      # Uniformly random actions
    'call',        # Always call/check
    'tight',       # Play only premium hands
    'aggressive',  # Bet/raise frequently
    'mixed',       # Random mix of agent types
]


# =============================================================================
# Presets for Common Scenarios
# =============================================================================

PRESETS = {
    'quick_test': {
        'game': DEFAULT_GAME_CONFIG,
        'num_players': 2,
        'agent_type': 'random',
    },
    
    'heads_up': {
        'game': DEFAULT_GAME_CONFIG,
        'num_players': 2,
        'agent_type': 'mixed',
    },
    
    'full_ring': {
        'game': DEFAULT_GAME_CONFIG,
        'num_players': 9,
        'agent_type': 'mixed',
    },
    
    'tournament': {
        'game': TOURNAMENT_CONFIG,
        'num_players': 6,
        'agent_type': 'tight',
    },
    
    'high_stakes': {
        'game': HIGH_STAKES_CONFIG,
        'num_players': 6,
        'agent_type': 'aggressive',
    },
}


def get_preset(name):
    """
    Get a preset configuration.
    
    Args:
        name: Name of the preset ('quick_test', 'heads_up', etc.)
    
    Returns:
        Dictionary with preset configuration
    
    Example:
        config = get_preset('heads_up')
        print(config['game'])
        print(config['rollout'])
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name].copy()

