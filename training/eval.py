#!/usr/bin/env python3
"""
Evaluation Script - Play Against Random Agent

This script evaluates a trained model by:
1. Loading the trained model
2. Playing hands against a random agent
3. Computing performance metrics (win rate, profit)

Prerequisites:
- Trained model (from train.py)

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

# Import from common package - shared simulation and agent logic
from common import (
    ModelAgent,
    RandomAgent,
    extract_state,
    GameConfig,
    PokerSimulator,
    check_binding_available,
)


class GameEvaluator:
    """
    Evaluates agents by playing games via the API.
    
    Uses common.simulation.PokerSimulator for the core game logic.
    """
    
    def __init__(self):
        self.simulator = None
    
    def check_server(self) -> bool:
        """Check if API binding is working"""
        return check_binding_available()

    def play_hand(
        self,
        agents: List[Any],
        config: Dict[str, Any],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Play a single hand using the common simulator.
        
        Args:
            agents: List of agent objects (must have player_id and select_action method)
            config: Game configuration
            verbose: Print detailed progress
            
        Returns:
            Dictionary with hand results (rewards, etc.)
        """
        # Create simulator with the given config
        sim_config = GameConfig(
            num_players=len(agents),
            small_blind=config.get('smallBlind', 10),
            big_blind=config.get('bigBlind', 20),
            starting_chips=config.get('startingChips', 1000),
            min_players=config.get('minPlayers', 2),
            max_players=config.get('maxPlayers', 2),
            seed=config.get('seed')
        )
        simulator = PokerSimulator(sim_config)
        
        # Convert agent list to dict
        agents_dict = {agent.player_id: agent for agent in agents}
        
        # Verbose callback
        def on_action(player_id, action_type, amount, action_label, game_state):
            if verbose:
                agent = agents_dict.get(player_id)
                name = agent.name if agent else player_id
                print(f"  {name}: {action_label} ({amount})")
        
        # Play the hand using common simulator
        result = simulator.play_hand(
            agents=agents_dict,
            on_action=on_action if verbose else None
        )
        
        if not result.get('success'):
            return {'success': False, 'error': result.get('error')}
        
        return {
            'success': True,
            'rewards': result.get('profits', {}),
            'steps': result.get('hands_played', 1)
        }


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
    
    # Create agents (load model once)
    agents_pool = []
    
    # Model agent is player 0
    try:
        model_agent = ModelAgent(
            player_id=model_id,
            name="ModelAgent",
            model_path=model_path,
            device=device,
            deterministic=True  # Use deterministic actions for evaluation
        )
        agents_pool.append(model_agent)
    except Exception as e:
        print(f"Error loading model: {e}")
        return {}
    
    # Random agents
    for i in range(1, num_players):
        random_agent = RandomAgent(f"p{i}", f"RandomAgent{i}")
        agents_pool.append(random_agent)
    
    start_time = time.time()
    
    for hand_num in range(num_hands):
        # Use the agents pool
        agents = agents_pool
        
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
