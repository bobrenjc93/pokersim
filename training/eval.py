#!/usr/bin/env python3
"""
Evaluation Script - Play Against Random Agent

This script evaluates a trained model by:
1. Loading the trained model
2. Playing hands against a random agent
3. Computing performance metrics (win rate, profit)

Prerequisites:
- Trained model (from train.py)
- Running API server

Usage:
    python eval.py --model /tmp/pokersim/models/poker_model.pt
    python eval.py --model /tmp/pokersim/models/poker_model.pt --num-hands 100 --num-players 2
"""

import argparse
import sys
import random
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import orjson as json
except ImportError:
    import json

import torch

# Import agent classes
from model_agent import ModelAgent, RandomAgent

# Import poker_api_binding
try:
    import poker_api_binding
except ImportError:
    print("Error: 'poker_api_binding' not found. Please compile the binding (cd api && make module).")
    sys.exit(1)


class GameEvaluator:
    """Evaluates agents by playing games via the API"""
    
    def __init__(self):
        pass
    
    def check_server(self) -> bool:
        """Check if API binding is working"""
        try:
            test_payload = {
                'config': {'seed': 123},
                'history': []
            }
            
            payload_bytes = json.dumps(test_payload)
            payload_str = payload_bytes.decode('utf-8') if isinstance(payload_bytes, bytes) else payload_bytes
            
            response_str = poker_api_binding.process_request(payload_str)
            response = json.loads(response_str)
            return response.get('success', False)
        except:
            return False

    def play_hand(
        self,
        agents: List[Any],
        config: Dict[str, Any],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Play a single hand.
        
        Args:
            agents: List of agent objects (must have player_id and select_action method)
            config: Game configuration
            verbose: Print detailed progress
            
        Returns:
            Dictionary with hand results (rewards, etc.)
        """
        # Initialize game history
        history = []
        for agent in agents:
            history.append({
                'type': 'addPlayer',
                'playerId': agent.player_id,
                'playerName': agent.name
            })
            # Reset agent state
            agent.reset_hand()
        
        # Get initial state
        response = self._call_api(config, history)
        if not response['success']:
            return {'success': False, 'error': response.get('error')}
        
        game_state = response['gameState']
        
        # Main game loop
        max_steps = 1000
        step = 0
        terminal_stages = {'complete', 'showdown'}
        
        while step < max_steps:
            step += 1
            
            # Check terminal condition
            current_stage = game_state.get('stage', '').lower()
            if current_stage in terminal_stages:
                break
            
            # Get current player
            current_player_id = game_state.get('currentPlayerId')
            if not current_player_id or current_player_id == 'none':
                break
            
            # Find agent for current player
            current_agent = None
            for agent in agents:
                if agent.player_id == current_player_id:
                    current_agent = agent
                    break
            
            if current_agent is None:
                break
            
            # Extract state
            state_dict = self._extract_state(game_state, current_player_id)
            legal_actions = self._get_legal_actions(game_state)
            
            if not legal_actions:
                break
            
            # Agent selects action
            action_type, amount, action_label = current_agent.select_action(state_dict, legal_actions)
            
            if verbose:
                print(f"  {current_agent.name}: {action_label} ({amount})")
            
            # Observe action for all agents (for opponent modeling)
            stage = state_dict.get('stage', 'Preflop')
            pot = state_dict.get('pot', 0)
            for agent in agents:
                agent.observe_action(current_player_id, action_type, amount, pot, stage)
            
            # Apply action
            history.append({
                'type': 'playerAction',
                'playerId': current_player_id,
                'action': action_type,
                'amount': amount
            })
            
            # Get new state
            response = self._call_api(config, history)
            if not response['success']:
                return {'success': False, 'error': response.get('error')}
            
            game_state = response['gameState']
        
        # Calculate rewards (chips won/lost)
        rewards = {}
        initial_chips = config['startingChips']
        
        for player_data in game_state['players']:
            player_id = player_data['id']
            rewards[player_id] = player_data['chips'] - initial_chips
            
        return {
            'success': True,
            'rewards': rewards,
            'steps': step
        }

    def _call_api(self, config: Dict, history: List[Dict]) -> Dict:
        """Call poker API server"""
        payload = {
            'config': config,
            'history': history
        }
        
        try:
            payload_bytes = json.dumps(payload)
            payload_str = payload_bytes.decode('utf-8') if isinstance(payload_bytes, bytes) else payload_bytes
            
            response_str = poker_api_binding.process_request(payload_str)
            return json.loads(response_str)
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _extract_state(self, game_state: Dict, player_id: str) -> Dict[str, Any]:
        """Extract state for a specific player"""
        player = None
        for p in game_state['players']:
            if p['id'] == player_id:
                player = p
                break
        
        if player is None:
            return {}
        
        config = game_state.get('config', {})
        action_constraints = game_state.get('actionConstraints', {})
        
        return {
            'player_id': player_id,
            'hole_cards': player.get('holeCards', []),
            'community_cards': game_state.get('communityCards', []),
            'pot': game_state.get('pot', 0),
            'current_bet': game_state.get('currentBet', 0),
            'player_chips': player.get('chips', 0),
            'player_bet': player.get('bet', 0),
            'player_total_bet': player.get('totalBet', 0),
            'stage': game_state.get('stage', 'Preflop'),
            'num_players': len(game_state['players']),
            'num_active': sum(1 for p in game_state['players'] if p.get('isInHand', False)),
            'position': player.get('position', 0),
            'is_dealer': player.get('isDealer', False),
            'is_small_blind': player.get('isSmallBlind', False),
            'is_big_blind': player.get('isBigBlind', False),
            'big_blind': config.get('bigBlind', 20),
            'small_blind': config.get('smallBlind', 10),
            'starting_chips': config.get('startingChips', 1000),
            'to_call': action_constraints.get('toCall', 0),
            'min_bet': action_constraints.get('minBet', 20),
            'min_raise_total': action_constraints.get('minRaiseTotal', 20),
        }
    
    def _get_legal_actions(self, game_state: Dict) -> List[str]:
        """Get legal actions from game state"""
        action_constraints = game_state.get('actionConstraints', {})
        return action_constraints.get('legalActions', [])


def play_vs_random(
    model_path: str,
    num_hands: int = 100,
    num_players: int = 2,
    small_blind: int = 10,
    big_blind: int = 20,
    starting_chips: int = 1000,
    verbose: bool = False,
    device_name: str = "cpu"
) -> Dict[str, Any]:
    """
    Play hands against random agent(s).
    """
    # Setup device
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    evaluator = GameEvaluator()
    
    if not evaluator.check_server():
        print(f"✗ Error: Binding check failed.")
        return {}
    
    print(f"✓ Binding check passed")
    print(f"  Playing {num_hands} hands with {num_players} players")
    
    # Track statistics
    model_id = "p0"
    stats = {
        'hands_played': 0,
        'hands_won': 0,
        'hands_lost': 0,
        'hands_tied': 0,
        'total_profit': 0,
        'profits': []
    }
    
    start_time = time.time()
    
    for hand_num in range(num_hands):
        # Create agents
        agents = []
        
        # Model agent is player 0
        try:
            model_agent = ModelAgent(
                player_id=model_id,
                name="ModelAgent",
                model_path=model_path,
                device=device,
                deterministic=True  # Use deterministic actions for evaluation
            )
            agents.append(model_agent)
        except Exception as e:
            print(f"Error loading model: {e}")
            return {}
        
        # Random agents
        for i in range(1, num_players):
            random_agent = RandomAgent(f"p{i}", f"RandomAgent{i}")
            agents.append(random_agent)
        
        # Game config
        config = {
            'smallBlind': small_blind,
            'bigBlind': big_blind,
            'startingChips': starting_chips,
            'minPlayers': num_players,
            'maxPlayers': num_players,
            'seed': random.randint(0, 1000000)
        }
        
        # Play hand
        result = evaluator.play_hand(agents, config, verbose)
        
        if not result['success']:
            print(f"Error in hand {hand_num+1}: {result.get('error')}")
            continue
            
        # Update stats
        profit = result['rewards'].get(model_id, 0)
        stats['total_profit'] += profit
        stats['profits'].append(profit)
        stats['hands_played'] += 1
        
        if profit > 0:
            stats['hands_won'] += 1
        elif profit < 0:
            stats['hands_lost'] += 1
        else:
            stats['hands_tied'] += 1
            
        # Progress
        if (hand_num + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (hand_num + 1) / elapsed
            eta = (num_hands - hand_num - 1) / rate if rate > 0 else 0
            win_rate = stats['hands_won'] / stats['hands_played'] * 100
            avg_profit = stats['total_profit'] / stats['hands_played']
            print(f"  Hand {hand_num+1}/{num_hands} | Win Rate: {win_rate:.1f}% | Avg Profit: {avg_profit:.1f} | ETA: {eta:.0f}s")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Poker AI Model")
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num-hands', type=int, default=100, help='Number of hands to play')
    parser.add_argument('--num-players', type=int, default=2, help='Number of players')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Device to use')
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1
        
    stats = play_vs_random(
        model_path=str(model_path),
        num_hands=args.num_hands,
        num_players=args.num_players,
        verbose=args.verbose,
        device_name=args.device
    )
    
    if stats:
        print("\nEvaluation Complete!")
        print(f"Hands Played: {stats['hands_played']}")
        print(f"Win Rate: {stats['hands_won'] / stats['hands_played'] * 100:.2f}%")
        print(f"Total Profit: {stats['total_profit']}")
        print(f"Avg Profit/Hand: {stats['total_profit'] / stats['hands_played']:.2f}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
