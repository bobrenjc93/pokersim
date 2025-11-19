"""
Configuration for RL training.

Only constants that are actively imported elsewhere are defined here.
"""

# =============================================================================
# Model Version
# =============================================================================

# Model version for RL training
# v11: Critical bug fixes for ELO drift and HeuristicAgent determinism
#      - Fixed position swap bug in play_freezeout_round
#      - Aligned training distribution with ELO evaluation
#      - Removed random.random() from HeuristicAgent postflop hand strength
MODEL_VERSION = 13

# Default directory for saving/loading models
DEFAULT_MODELS_DIR = f"/tmp/pokersim/rl_models_v{MODEL_VERSION}"

# =============================================================================
# Logging Configuration
# =============================================================================

# Log levels: 0=minimal, 1=normal, 2=verbose
LOG_LEVEL = 1  # Default to normal logging (shows errors and progress)

