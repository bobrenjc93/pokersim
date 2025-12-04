"""
Configuration for RL training.

Only constants that are actively imported elsewhere are defined here.
"""

# =============================================================================
# Model Version
# =============================================================================

# Model version for RL training
# v43: Unified action space - consolidated bet/raise into single raise_X% actions (13 actions instead of 22)
MODEL_VERSION = 3

# Default directory for saving/loading models
DEFAULT_MODELS_DIR = f"/tmp/pokersim/rl_models_v{MODEL_VERSION}"

# =============================================================================
# Logging Configuration
# =============================================================================

# Log levels: 0=minimal, 1=normal, 2=verbose
LOG_LEVEL = 1  # Default to normal logging (shows errors and progress)
