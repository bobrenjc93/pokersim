#!/usr/bin/env python3
"""
Monte Carlo Simulation for Poker Training

This module provides multi-runout simulation capabilities for:
1. Training: Play each hand N times to calculate regret-based rewards
2. Arena: "Run it N times" equity calculation for pot splitting

Key concept: Instead of relying on single outcomes (high variance), 
we simulate multiple outcomes by sampling actions from policies 
and dealing out remaining cards to get expected values.
"""

import random
import copy
from typing import Any, Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F

try:
    import poker_api_binding
except ImportError:
    poker_api_binding = None

try:
    import orjson as json
except ImportError:
    import json


def _call_api(game_config: Dict, history: List[Dict]) -> Dict:
    """Call poker API to get game state."""
    payload = {
        'config': {
            **game_config,
            'seed': random.randint(0, 1000000)
        },
        'history': history
    }
    
    try:
        payload_bytes = json.dumps(payload)
        payload_str = payload_bytes.decode('utf-8') if isinstance(payload_bytes, bytes) else payload_bytes
        response_str = poker_api_binding.process_request(payload_str)
        return json.loads(response_str)
    except Exception as e:
        return {'success': False, 'error': str(e)}


def simulate_runouts(
    game_config: Dict,
    history: List[Dict],
    num_runouts: int = 50,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Simulate multiple runouts from the current game state.
    
    This is the "run it N times" functionality - we simulate the remaining
    cards and actions multiple times to get expected values for each player.
    
    Args:
        game_config: Game configuration dict
        history: Current game history (list of actions)
        num_runouts: Number of times to simulate the remaining hand
        verbose: Print debug info
        
    Returns:
        Dict mapping player_id to average equity (0.0 to 1.0)
    """
    if not history:
        return {}
    
    # Get current state
    response = _call_api(game_config, history)
    if not response.get('success'):
        return {}
    
    game_state = response['gameState']
    stage = game_state.get('stage', '').lower()
    
    # If game is already complete, just return current results
    if stage in ['complete', 'showdown']:
        return _calculate_single_equity(game_state, game_config)
    
    # Track results across runouts
    player_wins: Dict[str, float] = {}
    player_ids = [p['id'] for p in game_state['players']]
    for pid in player_ids:
        player_wins[pid] = 0.0
    
    # Run multiple simulations
    for i in range(num_runouts):
        # Copy history for this simulation
        sim_history = copy.deepcopy(history)
        
        # Simulate to completion with random actions
        result = _simulate_to_completion(game_config, sim_history)
        
        if result:
            # Normalize results to equity (0-1)
            total_pot = sum(result.values())
            if total_pot > 0:
                for pid, winnings in result.items():
                    # winnings is chips - starting_chips, convert to equity
                    equity = (winnings / total_pot + 0.5) if total_pot > 0 else 0.5
                    equity = max(0.0, min(1.0, equity))
                    player_wins[pid] += equity
    
    # Average the results
    if num_runouts > 0:
        for pid in player_wins:
            player_wins[pid] /= num_runouts
    
    return player_wins


def _simulate_to_completion(
    game_config: Dict,
    history: List[Dict]
) -> Optional[Dict[str, float]]:
    """
    Simulate a hand to completion using random actions.
    
    Returns dict mapping player_id to profit (chips won/lost).
    """
    response = _call_api(game_config, history)
    if not response.get('success'):
        return None
    
    game_state = response['gameState']
    max_steps = 100
    step = 0
    
    while step < max_steps:
        step += 1
        
        stage = game_state.get('stage', '').lower()
        if stage in ['complete', 'showdown']:
            break
        
        current_player_id = game_state.get('currentPlayerId')
        if not current_player_id or current_player_id == 'none':
            break
        
        # Get legal actions
        legal_actions = game_state.get('actionConstraints', {}).get('legalActions', [])
        if not legal_actions:
            break
        
        # Select random action
        action_type, amount = _random_action(legal_actions, game_state)
        
        # Apply action
        history.append({
            'type': 'playerAction',
            'playerId': current_player_id,
            'action': action_type,
            'amount': amount
        })
        
        response = _call_api(game_config, history)
        if not response.get('success'):
            return None
        
        game_state = response['gameState']
    
    # Calculate profits
    starting_chips = game_config.get('startingChips', 1000)
    profits = {}
    for p in game_state.get('players', []):
        profits[p['id']] = p['chips'] - starting_chips
    
    return profits


def _random_action(legal_actions: List[str], game_state: Dict) -> Tuple[str, int]:
    """Select a random legal action."""
    action = random.choice(legal_actions)
    
    # Get state info for bet sizing
    pot = game_state.get('pot', 0)
    constraints = game_state.get('actionConstraints', {})
    min_bet = constraints.get('minBet', 20)
    min_raise = constraints.get('minRaiseTotal', 20)
    
    if action in ['fold', 'check', 'call', 'all_in']:
        return action, 0
    elif action == 'bet':
        # Random bet size (50-100% pot)
        size = random.uniform(0.5, 1.0)
        amount = max(min_bet, int(pot * size))
        return 'bet', amount
    elif action == 'raise':
        # Random raise size (50-100% pot)
        size = random.uniform(0.5, 1.0)
        amount = max(min_raise, int(pot * size))
        return 'raise', amount
    
    return 'check', 0


def _calculate_single_equity(game_state: Dict, game_config: Dict) -> Dict[str, float]:
    """Calculate equity from a completed game state."""
    starting_chips = game_config.get('startingChips', 1000)
    
    equities = {}
    for p in game_state.get('players', []):
        profit = p['chips'] - starting_chips
        # Convert profit to equity (0.5 is break-even)
        # Scale so full stack win = 1.0, full stack loss = 0.0
        equity = 0.5 + (profit / (2 * starting_chips))
        equity = max(0.0, min(1.0, equity))
        equities[p['id']] = equity
    
    return equities


def calculate_action_regrets(
    game_config: Dict,
    history: List[Dict],
    current_player_id: str,
    model,
    encoder,
    device: torch.device,
    num_runouts: int = 50
) -> Dict[str, float]:
    """
    Calculate regret for each action by simulating multiple outcomes.
    
    This implements Monte Carlo Counterfactual Regret (MCCFR) style calculation:
    1. Get action probabilities from the model
    2. For each legal action, simulate N runouts
    3. Calculate expected value of each action
    4. Regret = EV(best action) - EV(action taken)
    
    Args:
        game_config: Game configuration
        history: Current game history
        current_player_id: ID of player to calculate regrets for
        model: The policy model
        encoder: State encoder
        device: Torch device
        num_runouts: Number of simulations per action
        
    Returns:
        Dict mapping action labels to their regrets
    """
    from model_agent import (
        ACTION_NAMES, extract_state, create_legal_actions_mask, convert_action_label
    )
    
    # Get current state
    response = _call_api(game_config, history)
    if not response.get('success'):
        return {}
    
    game_state = response['gameState']
    
    # Extract state for the current player
    state_dict = extract_state(game_state, current_player_id)
    legal_actions = game_state.get('actionConstraints', {}).get('legalActions', [])
    
    if not legal_actions:
        return {}
    
    # Get action probabilities from model
    state_tensor = encoder.encode_state(state_dict).unsqueeze(0).to(device)
    legal_mask = create_legal_actions_mask(legal_actions, device)
    
    with torch.no_grad():
        action_logits, _ = model(state_tensor, legal_mask)
        action_probs = F.softmax(action_logits, dim=-1).squeeze(0)
    
    # Calculate EV for each legal action
    action_evs = {}
    
    for action_idx, action_name in enumerate(ACTION_NAMES):
        # Skip illegal actions
        if not legal_mask[0, action_idx]:
            continue
        
        prob = action_probs[action_idx].item()
        if prob < 0.001:  # Skip very unlikely actions for efficiency
            continue
        
        # Convert action name to (type, amount)
        action_type, amount = convert_action_label(action_name, state_dict)
        
        # Simulate this action N times
        ev_sum = 0.0
        successful_sims = 0
        
        for _ in range(num_runouts):
            # Copy history and add this action
            sim_history = copy.deepcopy(history)
            sim_history.append({
                'type': 'playerAction',
                'playerId': current_player_id,
                'action': action_type,
                'amount': amount
            })
            
            # Simulate to completion
            result = _simulate_to_completion(game_config, sim_history)
            
            if result and current_player_id in result:
                # Normalize profit to [-1, 1] range
                starting_chips = game_config.get('startingChips', 1000)
                profit = result[current_player_id]
                normalized_ev = profit / starting_chips
                ev_sum += normalized_ev
                successful_sims += 1
        
        if successful_sims > 0:
            action_evs[action_name] = ev_sum / successful_sims
    
    if not action_evs:
        return {}
    
    # Calculate regrets
    # Regret = EV(best action) - EV(this action)
    best_ev = max(action_evs.values())
    regrets = {}
    for action_name, ev in action_evs.items():
        regrets[action_name] = best_ev - ev
    
    return regrets


def compute_regret_weighted_reward(
    episode_reward: float,
    regrets: List[Dict[str, float]],
    actions_taken: List[str],
    regret_weight: float = 0.5
) -> float:
    """
    Compute a reward that incorporates regret information.
    
    The idea is to adjust the raw episode reward based on the regret
    of the actions taken. High regret actions should be penalized
    even if the episode was lucky, and low regret actions should
    be less penalized even if unlucky.
    
    Args:
        episode_reward: Raw reward from the episode (-1 to 1)
        regrets: List of regret dicts for each decision point
        actions_taken: List of action labels taken at each decision
        regret_weight: How much to weight regret vs raw reward (0-1)
        
    Returns:
        Adjusted reward
    """
    if not regrets or not actions_taken:
        return episode_reward
    
    # Calculate average regret for actions taken
    total_regret = 0.0
    count = 0
    
    for regret_dict, action in zip(regrets, actions_taken):
        if action in regret_dict:
            total_regret += regret_dict[action]
            count += 1
    
    if count == 0:
        return episode_reward
    
    avg_regret = total_regret / count
    
    # Adjust reward: lower regret = less penalty for losses, more reward for wins
    # High regret = more penalty, less reward
    # avg_regret is typically 0-1, where 0 is optimal play
    regret_adjustment = -avg_regret * regret_weight
    
    # Blend raw reward with regret-adjusted reward
    adjusted_reward = episode_reward * (1 - regret_weight) + (episode_reward + regret_adjustment) * regret_weight
    
    return adjusted_reward


class MultiRunoutEvaluator:
    """
    Evaluator that runs multiple simulations to calculate equity.
    
    This is used in arena matches to "run it N times" for more
    accurate pot splitting.
    """
    
    def __init__(self, game_config: Dict, num_runouts: int = 50):
        self.game_config = game_config
        self.num_runouts = num_runouts
    
    def calculate_equity(
        self,
        history: List[Dict],
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Calculate equity for each player by running multiple simulations.
        
        Returns dict mapping player_id to equity (0.0 to 1.0).
        """
        return simulate_runouts(
            self.game_config,
            history,
            self.num_runouts,
            verbose
        )
    
    def split_pot(
        self,
        history: List[Dict],
        pot_size: int,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Split the pot based on equity calculated from multiple runouts.
        
        Args:
            history: Game history
            pot_size: Total pot size to split
            verbose: Print debug info
            
        Returns:
            Dict mapping player_id to pot share in chips
        """
        equities = self.calculate_equity(history, verbose)
        
        if not equities:
            return {}
        
        # Normalize equities to sum to 1
        total_equity = sum(equities.values())
        if total_equity <= 0:
            # Equal split
            num_players = len(equities)
            return {pid: pot_size / num_players for pid in equities}
        
        # Split pot by equity
        pot_shares = {}
        for pid, eq in equities.items():
            pot_shares[pid] = (eq / total_equity) * pot_size
        
        return pot_shares

