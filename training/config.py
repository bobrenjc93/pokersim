"""
Configuration for RL training.

Only constants that are actively imported elsewhere are defined here.
"""

# =============================================================================
# Model Version
# =============================================================================

# Model version for RL training
MODEL_VERSION = 32  # Current RL model version (v31: diverse opponent pool - AlwaysRaise, AlwaysCall, AlwaysFold agents)

# Default directory for saving/loading models
DEFAULT_MODELS_DIR = f"/tmp/pokersim/rl_models_v{MODEL_VERSION}"

# =============================================================================
# Logging Configuration
# =============================================================================

# Log levels: 0=minimal, 1=normal, 2=verbose
LOG_LEVEL = 1  # Default to normal logging (shows errors and progress)
