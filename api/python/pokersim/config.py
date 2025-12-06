"""
Configuration constants for the pokersim package.

Central location for shared constants to avoid magic numbers throughout the codebase.
This module is shared between training, elo evaluation, and gameplay code.
"""

from typing import Dict

# =============================================================================
# Model Version
# =============================================================================

# Model version for RL training
# v5: Unified action space - consolidated bet/raise into single raise_X% actions (13 actions instead of 22)
# v6: Previous version (inherited issues)
# v7: Critical bug fixes for ELO drift:
#     - Fixed position swap bug in play_freezeout_round (was inverting profits half the time!)
#     - Fixed opponent distribution mismatch: Heuristic now 25% (was only 15%, ELO uses heuristic as baseline)
#     - Aligned training distribution with ELO evaluation for consistent skill transfer
# v8: HeuristicAgent determinism fix:
#     - CRITICAL: Removed random.random() from HeuristicAgent postflop hand strength
#     - Old code: hand_strength = preflop * 0.5 + random.random() * 0.5 (50% noise!)
#     - New code: deterministic board-texture based evaluation
#     - This was causing high variance in ELO and preventing consistent exploitation
MODEL_VERSION = 10

# Default directory for saving/loading models
DEFAULT_MODELS_DIR = f"/tmp/pokersim/rl_models_v{MODEL_VERSION}"

# =============================================================================
# Action Space
# =============================================================================

# Unified action space - bet and raise consolidated into "raise" (contextually becomes bet or raise)
# 13 total actions: fold, check, call, 9 raise sizes, all_in
ACTION_MAP: Dict[str, int] = {
    'fold': 0, 'check': 1, 'call': 2,
    'raise_10%': 3, 'raise_25%': 4, 'raise_33%': 5, 'raise_50%': 6, 'raise_75%': 7,
    'raise_100%': 8, 'raise_150%': 9, 'raise_200%': 10, 'raise_300%': 11,
    'all_in': 12
}

ACTION_NAMES = [
    'fold', 'check', 'call',
    'raise_10%', 'raise_25%', 'raise_33%', 'raise_50%', 'raise_75%', 'raise_100%', 'raise_150%', 'raise_200%', 'raise_300%',
    'all_in'
]

NUM_ACTIONS = len(ACTION_NAMES)  # 13

# Raise size percentages (of pot)
RAISE_SIZE_MAP: Dict[str, float] = {
    'raise_10%': 0.10, 'raise_25%': 0.25, 'raise_33%': 0.33, 'raise_50%': 0.50, 'raise_75%': 0.75,
    'raise_100%': 1.0, 'raise_150%': 1.5, 'raise_200%': 2.0, 'raise_300%': 3.0,
}

# Backwards compatibility alias
BET_SIZE_MAP = RAISE_SIZE_MAP

# =============================================================================
# Neural Network Dimensions
# =============================================================================

# Feature dimension output by RLStateEncoder.encode_state()
# Breakdown:
#   - Hole cards: 2 × 17 = 34
#   - Community cards: 5 × 17 = 85
#   - Pot info: 5
#   - Stage: 5 (one-hot)
#   - Position: 6
#   - Game-theoretic features: 5
#   - Opponent modeling: 26
#   - Hand strength: 1
# Total: 34 + 85 + 5 + 5 + 6 + 5 + 26 + 1 = 167
FEATURE_DIM = 167

# =============================================================================
# Card Encoding
# =============================================================================

RANK_MAP: Dict[str, int] = {
    '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
    'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12
}

SUIT_MAP: Dict[str, int] = {'C': 0, 'D': 1, 'H': 2, 'S': 3}

STAGE_MAP: Dict[str, int] = {'Preflop': 0, 'Flop': 1, 'Turn': 2, 'River': 3, 'Complete': 4}

# Action encoding for history
ACTION_TYPE_MAP: Dict[str, int] = {
    'fold': 0, 'check': 1, 'call': 2, 'bet': 3, 'raise': 4, 'all_in': 5
}

# =============================================================================
# Logging Configuration
# =============================================================================

# Log levels: 0=minimal, 1=normal, 2=verbose
LOG_LEVEL = 1  # Default to normal logging (shows errors and progress)

# =============================================================================
# Game Configuration Defaults
# =============================================================================

# Default game parameters for training and evaluation
# These values are used consistently across training, ELO evaluation, and gameplay
DEFAULT_SMALL_BLIND = 10
DEFAULT_BIG_BLIND = 20
DEFAULT_STARTING_CHIPS = 1000  # 50 BB (1000 chips / 20 BB = 50 big blinds)

# Match configuration
DEFAULT_MAX_HANDS_PER_ROUND = 200  # Safety limit per freezeout round
DEFAULT_ROUNDS_PER_MATCH = 51  # Best-of-51 for reduced variance in ELO
DEFAULT_MAX_RAISES_PER_ROUND = 4  # Limit raises per betting round
