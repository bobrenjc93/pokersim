"""
Configuration for RL training and ELO evaluation.
"""

import os
import torch

# --------------------------------------------------------------------------------------
# Model directory/version selection
# --------------------------------------------------------------------------------------
#
# Historically this repo used a single hard-coded MODEL_VERSION that drove both:
# - training output directory (e.g. /tmp/pokersim/rl_models_v17)
# - ELO server default models directory
#
# That makes it very easy to accidentally *train one version and evaluate another*,
# leading to confusing symptoms like "iter_1 is best" simply because you're looking
# at a different run directory.
#
# Make the choice explicit and overrideable:
# - Set POKERSIM_MODEL_VERSION=17 to use /tmp/pokersim/rl_models_v17
# - Or set POKERSIM_MODELS_DIR=/path/to/models to force a specific directory
#
MODEL_VERSION = int(os.environ.get("POKERSIM_MODEL_VERSION", "26"))

# Default directory for saving/loading models
DEFAULT_MODELS_DIR = os.environ.get("POKERSIM_MODELS_DIR", f"/tmp/pokersim/rl_models_v{MODEL_VERSION}")


def detect_device() -> torch.device:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ELO Arena Configuration (shared between training and evaluation)
# Stack and blinds (25 BB effective stack)
ELO_STARTING_STACK = 1000
ELO_BIG_BLIND = 40
ELO_SMALL_BLIND = 20

# Match format
ELO_ROUNDS_PER_MATCH = 50
ELO_MAX_HANDS_PER_ROUND = 200
ELO_WIN_THRESHOLD = 0.5

# Reward structure for training
DEFAULT_HAND_REWARD_SCALE = 0.01
DEFAULT_MATCH_WIN_BONUS = 2.0

# Logging: 0=minimal, 1=normal, 2=verbose
LOG_LEVEL = 1

