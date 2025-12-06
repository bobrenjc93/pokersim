"""
Reward shaping utilities for poker RL training.

This module provides shared reward shaping functions used across training,
batched rollouts, and parallel rollouts.
"""

from typing import Any, Dict


def compute_action_shaping_reward(
    action_type: str,
    hand_strength: float,
    state: Dict[str, Any],
    facing_aggression: bool,
    step_in_hand: int = 0,
    total_steps_estimate: int = 6,
) -> float:
    """
    Compute per-step reward shaping based on action appropriateness.
    
    This provides dense learning signal to help the model learn proper
    hand-action correlations with temporal consistency.
    
    Key principles:
    - Folding weak hands = context-dependent (facing aggression matters)
    - All-in with weak/trash hands = PENALIZED (regardless of outcome)
    - Checking/calling with appropriate hands = mildly rewarded
    - Betting/raising strong hands = rewarded
    - Temporal consistency: earlier decisions affect later rewards
    
    Args:
        action_type: The action taken (fold, check, call, bet, raise, all_in)
        hand_strength: Estimated hand strength (0.0 to 1.0)
        state: Current game state
        facing_aggression: Whether opponent just bet/raised
        step_in_hand: Current step number in hand (0-indexed)
        total_steps_estimate: Estimated total steps in hand
    
    Returns:
        Shaping reward (positive/negative value)
    """
    reward = 0.0
    
    # Define hand strength thresholds (calibrated for realistic poker)
    TRASH_THRESHOLD = 0.30      # Hands we should almost always fold (bottom ~30%)
    WEAK_THRESHOLD = 0.45       # Marginal hands - careful play required
    MEDIUM_THRESHOLD = 0.55     # Playable hands
    STRONG_THRESHOLD = 0.70     # Value betting hands
    PREMIUM_THRESHOLD = 0.85    # Premium hands - can go all-in comfortably
    
    # Get game context
    pot = state.get('pot', 0)
    to_call = state.get('to_call', 0)
    player_chips = state.get('player_chips', 0)
    stage = state.get('stage', 'Preflop')
    
    # Pot odds: what fraction of pot we need to win to break even
    pot_odds = to_call / max(1, pot + to_call) if to_call > 0 else 0
    
    # Commitment level: how much of our stack is already in
    player_bet = state.get('player_bet', 0)
    commitment = player_bet / max(1, player_bet + player_chips)
    
    # Stack-to-pot ratio (SPR) - low SPR justifies more aggression
    spr = player_chips / max(1, pot) if pot > 0 else 10.0
    
    # Temporal discount: earlier actions have more impact on hand outcome
    # This encourages good decisions from the start
    temporal_weight = 1.0 + (0.2 * (total_steps_estimate - step_in_hand) / max(1, total_steps_estimate))
    
    # === REWARD SHAPING RULES ===
    
    # 1. FOLDING
    if action_type == 'fold':
        if hand_strength < TRASH_THRESHOLD:
            # Folding trash facing aggression is neutral to slightly positive
            reward += 0.02 if facing_aggression else -0.03
        elif hand_strength < WEAK_THRESHOLD:
            # Folding weak hands - context dependent
            reward += -0.02 if facing_aggression else -0.08
        elif hand_strength < MEDIUM_THRESHOLD:
            # Folding medium hands is usually bad
            reward += -0.08 if facing_aggression else -0.12
        elif hand_strength < STRONG_THRESHOLD:
            # Folding decent hands is VERY BAD
            reward -= 0.15
        else:
            # Folding strong hands is TERRIBLE
            reward -= 0.22
    
    # 2. ALL-IN - Strong penalties for weak all-ins
    elif action_type == 'all_in':
        if hand_strength >= PREMIUM_THRESHOLD:
            # All-in with premium hands is excellent
            reward += 0.18
        elif hand_strength >= STRONG_THRESHOLD:
            # All-in with strong hands is good
            reward += 0.10
        elif hand_strength >= MEDIUM_THRESHOLD:
            # All-in with medium hands is risky
            if stage == 'Preflop' and hand_strength >= 0.60:
                reward += 0.02
            elif commitment > 0.6:
                reward += 0.0  # Pot committed
            elif spr < 2.0:
                reward += 0.0  # Low SPR justifies
            else:
                reward -= 0.12
        elif hand_strength >= WEAK_THRESHOLD:
            # ALL-IN WITH WEAK HANDS IS BAD
            if commitment > 0.7:
                reward -= 0.06
            else:
                reward -= 0.30
        else:
            # ALL-IN WITH TRASH IS TERRIBLE
            if commitment > 0.8:
                reward -= 0.15
            else:
                reward -= 0.45
    
    # 3. BET / RAISE (aggressive actions)
    elif action_type in ['bet', 'raise']:
        if hand_strength >= STRONG_THRESHOLD:
            reward += 0.10
        elif hand_strength >= MEDIUM_THRESHOLD:
            reward += 0.04 if stage in ['Preflop', 'Flop'] else 0.02
        elif hand_strength >= WEAK_THRESHOLD:
            reward -= 0.05
        else:
            reward -= 0.10
    
    # 4. CALL
    elif action_type == 'call':
        if hand_strength >= STRONG_THRESHOLD:
            reward += 0.08
        elif hand_strength >= MEDIUM_THRESHOLD:
            reward += 0.06
        elif hand_strength >= WEAK_THRESHOLD:
            if pot_odds > 0 and hand_strength > pot_odds * 1.2:
                reward += 0.04
            else:
                reward += 0.02
        else:
            reward -= 0.04
    
    # 5. CHECK
    elif action_type == 'check':
        if hand_strength >= STRONG_THRESHOLD:
            reward += 0.02  # Slowplay consideration
        elif hand_strength >= MEDIUM_THRESHOLD:
            reward += 0.05  # Pot control
        elif hand_strength < WEAK_THRESHOLD:
            reward += 0.06  # Pot control with weak hand
        else:
            reward += 0.04
    
    # Stage multiplier (later streets have higher stakes)
    stage_multiplier = {
        'Preflop': 1.0,
        'Flop': 1.15,
        'Turn': 1.30,
        'River': 1.50
    }.get(stage, 1.0)
    
    return reward * stage_multiplier * temporal_weight

