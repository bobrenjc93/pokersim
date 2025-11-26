"""
Configuration for RL training.

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

# Model version for RL training
MODEL_VERSION = 17  # Current RL model version

# Default directory for saving/loading models
DEFAULT_MODELS_DIR = f"/tmp/pokersim/rl_models_v{MODEL_VERSION}"

# =============================================================================
# Logging Configuration
# =============================================================================

# Log levels: 0=minimal, 1=normal, 2=verbose
LOG_LEVEL = 1  # Default to normal logging (shows errors and progress)

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
# (Kept for reference or future use)
# AGENT_TYPES = [
#     'random',      # Uniformly random actions
#     'call',        # Always call/check
#     'tight',       # Play only premium hands
#     'aggressive',  # Bet/raise frequently
#     'mixed',       # Random mix of agent types
# ]


