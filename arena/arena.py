#!/usr/bin/env python3
"""
Poker AI Arena
Play models against each other and evaluate performance.
"""

import sys
import os
import argparse
import re
import time
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import concurrent.futures

import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Add training directory to path to import modules
TRAINING_DIR = Path(__file__).parent.parent / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.append(str(TRAINING_DIR))

try:
    from rl_state_encoder import RLStateEncoder
    from model_agent import (
        ACTION_NAMES, 
        convert_action_label,
        create_legal_actions_mask,
        extract_state,
        ModelAgent,
        RandomAgent,
        HeuristicAgent,
        load_model_agent
    )
    # Helper for type hinting
    from rl_model import PokerActorCritic
    import poker_api_binding
    from config import DEFAULT_MODELS_DIR
except ImportError as e:
    print(f"Error importing modules from training directory: {e}")
    print(f"Make sure {TRAINING_DIR} exists and contains the required files.")
    sys.exit(1)


class EloRating:
    """
    Simple ELO rating system implementation.
    """
    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1000.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = defaultdict(lambda: initial_rating)
        
    def get_rating(self, player_id: str) -> float:
        return self.ratings[player_id]
        
    def get_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B"""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
        
    def update_ratings(self, player_a: str, player_b: str, score_a: float) -> Tuple[float, float]:
        """
        Update ratings based on match result.
        score_a: 1.0 for win, 0.5 for draw, 0.0 for loss
        """
        ra = self.ratings[player_a]
        rb = self.ratings[player_b]
        
        expected_a = self.get_expected_score(ra, rb)
        
        new_ra = ra + self.k_factor * (score_a - expected_a)
        new_rb = rb + self.k_factor * ((1.0 - score_a) - (1.0 - expected_a))
        
        self.ratings[player_a] = new_ra
        self.ratings[player_b] = new_rb
        
        return new_ra, new_rb


class Arena:
    def __init__(self, device: str = "cpu", output_dir: str = "arena_results"):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Game config (standard heads-up)
        self.game_config = {
            'num_players': 2,
            'smallBlind': 10,
            'bigBlind': 20,
            'startingChips': 1000,
            'minPlayers': 2,
            'maxPlayers': 2
        }
        
        # Cache for loaded models to avoid reloading
        self.model_cache = {}

    def load_model(self, checkpoint_path: Path) -> PokerActorCritic:
        """Load a model from a checkpoint"""
        path_str = str(checkpoint_path)
        if path_str in self.model_cache:
            return self.model_cache[path_str]
        
        try:
            # Use ModelAgent to handle robust model loading
            agent = load_model_agent("dummy", "dummy", path_str, device=self.device)
            model = agent.model
            self.model_cache[path_str] = model
            return model
            
        except Exception as e:
            print(f"Error loading model {checkpoint_path}: {e}")
            raise

    def _call_api(self, history: List[Dict]) -> Dict:
        """Call poker API binding"""
        payload = {
            'config': {
                **self.game_config,
                'seed': random.randint(0, 1000000)
            },
            'history': history
        }
        
        try:
            payload_str = json.dumps(payload)
            response_str = poker_api_binding.process_request(payload_str)
            return json.loads(response_str)
        except Exception as e:
            print(f"API Error: {e}")
            return {'success': False, 'error': str(e)}

    def play_hand(self, agent_a_config: Dict, agent_b_config: Dict) -> Dict[str, float]:
        """
        Play a single hand between two agents.
        
        Args:
            agent_a_config: Dict with 'type' ('model', 'random', 'heuristic') and 'model'/'path' if needed
            agent_b_config: Dict with 'type' ('model', 'random', 'heuristic') and 'model'/'path' if needed
        """
        
        def create_agent(player_id: str, config: Dict) -> Any:
            name = config.get('name', f"{config['type']}_{player_id}")
            if config['type'] == 'model':
                model = config.get('model')
                if model is None and 'path' in config:
                    model = self.load_model(Path(config['path']))
                
                return load_model_agent(player_id, name, model=model, 
                                      device=self.device, deterministic=config.get('deterministic', True))
            elif config['type'] == 'heuristic':
                return HeuristicAgent(player_id, name)
            else:
                return RandomAgent(player_id, name)

        agent_a = create_agent('p0', agent_a_config)
        agent_b = create_agent('p1', agent_b_config)
        
        agents = {
            'p0': agent_a,
            'p1': agent_b
        }
        
        history = [
            {'type': 'addPlayer', 'playerId': 'p0', 'playerName': agent_a.name},
            {'type': 'addPlayer', 'playerId': 'p1', 'playerName': agent_b.name}
        ]
        
        # Initial state
        response = self._call_api(history)
        if not response['success']:
            return {'p0': 0, 'p1': 0, 'error': True}
            
        game_state = response['gameState']
        max_steps = 1000
        step = 0
        
        while step < max_steps:
            step += 1
            
            stage = game_state.get('stage', '').lower()
            if stage in ['complete', 'showdown']:
                break
                
            current_player_id = game_state.get('currentPlayerId')
            if not current_player_id or current_player_id == 'none':
                break
                
            agent = agents[current_player_id]
            
            # Get legal actions
            legal_actions = game_state.get('actionConstraints', {}).get('legalActions', [])
            if not legal_actions:
                break
            
            # Extract state
            state_dict = extract_state(game_state, current_player_id)
            
            # Get action
            if hasattr(agent, 'select_action'):
                action_type, amount, action_label = agent.select_action(state_dict, legal_actions)
            else:
                # Fallback for weird agent types?
                action_type = 'check'
                amount = 0
            
            # Notify all agents (update history/encoders)
            pot = game_state.get('pot', 0)
            stage_name = game_state.get('stage', 'Preflop')
            
            for pid, ag in agents.items():
                if hasattr(ag, 'observe_action'):
                    ag.observe_action(current_player_id, action_type, amount, pot, stage_name)
            
            # Apply action
            history.append({
                'type': 'playerAction',
                'playerId': current_player_id,
                'action': action_type,
                'amount': amount
            })
            
            response = self._call_api(history)
            if not response['success']:
                break
            game_state = response['gameState']
            
        # Calculate rewards
        rewards = {}
        initial_chips = self.game_config['startingChips']
        big_blind = self.game_config['bigBlind']
        
        for p in game_state['players']:
            # Return result in Big Blinds
            profit = p['chips'] - initial_chips
            rewards[p['id']] = profit / big_blind
            
        return rewards

    def play_match(self, agent_a_config: Dict, agent_b_config: Dict, 
                   num_hands: int = 100) -> Dict:
        """Play a match between two agents"""
        name_a = agent_a_config.get('name', 'Agent A')
        name_b = agent_b_config.get('name', 'Agent B')
        print(f"Playing {name_a} vs {name_b} ({num_hands} hands)...")
        
        # Pre-load models to avoid loading them for every hand
        # We create copies of the config to avoid modifying the caller's dict
        config_a = agent_a_config.copy()
        config_b = agent_b_config.copy()
        
        if config_a.get('type') == 'model' and 'model' not in config_a and 'path' in config_a:
            print(f"  Loading model A from {config_a['path']}...")
            config_a['model'] = self.load_model(Path(config_a['path']))
            
        if config_b.get('type') == 'model' and 'model' not in config_b and 'path' in config_b:
            print(f"  Loading model B from {config_b['path']}...")
            config_b['model'] = self.load_model(Path(config_b['path']))
        
        results_a = []
        results_b = []
        wins_a = 0
        wins_b = 0
        
        total_rewards_a = 0
        total_rewards_b = 0
        
        # Use a single executor for all hands
        max_workers = os.cpu_count() or 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_futures = []
            batch_swaps = []
            
            # Submit all hands
            for j in range(num_hands):
                swap = (j % 2 == 1)
                batch_swaps.append(swap)
                if swap:
                    # Swap positions (B is p0, A is p1)
                    batch_futures.append(executor.submit(self.play_hand, config_b, config_a))
                else:
                    # A is p0, B is p1
                    batch_futures.append(executor.submit(self.play_hand, config_a, config_b))
            
            # Collect results with progress bar
            for k, f in tqdm(enumerate(batch_futures), total=num_hands, desc="Simulating"):
                try:
                    res = f.result()
                    if res.get('error'): continue
                    
                    swap = batch_swaps[k]
                    
                    if swap:
                        # p0 is B, p1 is A
                        reward_a = res['p1']
                        reward_b = res['p0']
                    else:
                        # p0 is A, p1 is B
                        reward_a = res['p0']
                        reward_b = res['p1']
                        
                    results_a.append(reward_a)
                    results_b.append(reward_b)
                    total_rewards_a += reward_a
                    total_rewards_b += reward_b
                    
                    if reward_a > 0: wins_a += 1
                    if reward_b > 0: wins_b += 1
                    
                except Exception as e:
                    print(f"Error: {e}")
        
        n = len(results_a)
        if n == 0: return {}
        
        return {
            'agent_a': name_a,
            'agent_b': name_b,
            'hands': n,
            'win_rate_a': wins_a / n,
            'win_rate_b': wins_b / n,
            'avg_bb_a': total_rewards_a / n,
            'avg_bb_b': total_rewards_b / n,
            'bb_100_a': (total_rewards_a / n) * 100
        }

def parse_checkpoints(directory: Path) -> List[Tuple[int, Path]]:
    """Find and sort checkpoints by iteration number"""
    checkpoints = []
    pattern = re.compile(r"poker_rl_iter_(\d+)\.pt")
    
    if not directory.exists():
        return []
        
    for f in directory.glob("*.pt"):
        if f.name == "poker_rl_baseline.pt":
            checkpoints.append((0, f))
            continue
            
        match = pattern.match(f.name)
        if match:
            iteration = int(match.group(1))
            checkpoints.append((iteration, f))
            
    return sorted(checkpoints, key=lambda x: x[0])

def ascii_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    """Print a simple ASCII plot to stdout"""
    print(f"\n=== {title} ===")
    if df.empty:
        print("No data to plot.")
        return

    xs = df[x_col].tolist()
    ys = df[y_col].tolist()
    
    if not xs: return

    min_y, max_y = min(ys), max(ys)
    # Fix range if flat
    if abs(max_y - min_y) < 1e-6:
        max_y += 1.0
        min_y -= 1.0
    
    range_y = max_y - min_y
    width = 40
    
    print(f"{'Iter':>5} | {'BB/100':>8} | Chart")
    print("-" * (17 + width))
    
    for x, y in zip(xs, ys):
        # Normalize to 0..width
        pos = int(((y - min_y) / range_y) * width)
        pos = max(0, min(width, pos))
        
        # Create bar
        # If y is 0, we want to show where 0 is? 
        # For simplicity, just show magnitude relative to min/max for now
        # Or better: just a bar proportional to value? 
        # Let's just do a simple position marker
        line = [" "] * (width + 1)
        line[pos] = "o"
        
        # If zero line crosses, mark it
        if min_y < 0 < max_y:
            zero_pos = int(((0 - min_y) / range_y) * width)
            if line[zero_pos] == " ":
                line[zero_pos] = "|"
        
        print(f"{x:5d} | {y:8.2f} | {''.join(line)}")
    print("-" * (17 + width))


def main():
    parser = argparse.ArgumentParser(description="Poker AI Arena")
    parser.add_argument('--models-dir', type=str, default=DEFAULT_MODELS_DIR, help=f'Directory containing model checkpoints (default: {DEFAULT_MODELS_DIR})')
    parser.add_argument('--output-dir', type=str, default='arena_results', help='Output directory for results')
    parser.add_argument('--hands', type=int, default=1000, help='Number of hands per match')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda/mps)')
    parser.add_argument('--mode', type=str, choices=['ladder', 'all-vs-all', 'vs-random', 'elo'], default='ladder',
                       help='Evaluation mode: ladder (vs previous), all-vs-all, vs-random, or elo')
    
    args = parser.parse_args()
    
    arena = Arena(device=args.device, output_dir=args.output_dir)
    models_dir = Path(args.models_dir)
    
    checkpoints = parse_checkpoints(models_dir)
    if not checkpoints:
        print(f"No checkpoints found in {models_dir}")
        return
        
    print(f"Found {len(checkpoints)} checkpoints.")
    
    all_results = []
    
    if args.mode == 'vs-random':
        print("Running evaluation vs Random...")
        for iter_num, path in checkpoints:
            config_a = {'type': 'model', 'path': str(path), 'name': f"Iter_{iter_num}"}
            config_b = {'type': 'random', 'name': "Random"}
            
            res = arena.play_match(config_a, config_b, num_hands=args.hands)
            res['iteration'] = iter_num
            all_results.append(res)
            print(f"Iter {iter_num}: Win Rate {res['win_rate_a']:.2%}, BB/100 {res['bb_100_a']:.2f}")
            
    elif args.mode == 'ladder':
        print("Running Ladder evaluation (vs previous version)...")
        # Sort by iteration
        sorted_cps = sorted(checkpoints, key=lambda x: x[0])
        
        for i in range(1, len(sorted_cps)):
            iter_a, path_a = sorted_cps[i]
            iter_b, path_b = sorted_cps[i-1]
            
            config_a = {'type': 'model', 'path': str(path_a), 'name': f"Iter_{iter_a}"}
            config_b = {'type': 'model', 'path': str(path_b), 'name': f"Iter_{iter_b}"}
            
            res = arena.play_match(config_a, config_b, num_hands=args.hands)
            res['iteration'] = iter_a
            res['opponent_iteration'] = iter_b
            all_results.append(res)
            print(f"Iter {iter_a} vs {iter_b}: Win Rate {res['win_rate_a']:.2%}, BB/100 {res['bb_100_a']:.2f}")

    elif args.mode == 'elo':
        print("Running ELO evaluation...")
        elo = EloRating()
        
        # Include baselines in ELO tracking
        elo.ratings['Random'] = 800.0 # Bad baseline
        elo.ratings['Heuristic'] = 1200.0 # Better baseline
        
        # Sort by iteration to simulate progression
        sorted_cps = sorted(checkpoints, key=lambda x: x[0])
        
        # Initialize ratings for all models
        for iter_num, _ in sorted_cps:
             elo.ratings[f"Iter_{iter_num}"] = 1000.0
        
        # Play matches
        # Strategy: Play each model against:
        # 1. Random
        # 2. Heuristic
        # 3. Previous model
        # 4. A few recent models
        
        for i in range(len(sorted_cps)):
            iter_a, path_a = sorted_cps[i]
            name_a = f"Iter_{iter_a}"
            config_a = {'type': 'model', 'path': str(path_a), 'name': name_a}
            
            opponents = []
            # Always play Random and Heuristic
            opponents.append({'type': 'random', 'name': 'Random'})
            opponents.append({'type': 'heuristic', 'name': 'Heuristic'})
            
            # Play against previous 3 models
            start_idx = max(0, i - 3)
            for j in range(start_idx, i):
                iter_b, path_b = sorted_cps[j]
                opponents.append({'type': 'model', 'path': str(path_b), 'name': f"Iter_{iter_b}"})
            
            for opp_config in opponents:
                name_b = opp_config['name']
                res = arena.play_match(config_a, opp_config, num_hands=args.hands // len(opponents)) # Distribute hands
                
                # Update ELO
                score_a = res['win_rate_a'] # Win rate as score approximation
                # Or use binary win/loss? Win rate is smoother.
                
                new_ra, new_rb = elo.update_ratings(name_a, name_b, score_a)
                
                res['elo_a'] = new_ra
                res['elo_b'] = new_rb
                res['iteration'] = iter_a
                all_results.append(res)
                
                print(f"  Result: {name_a} ({new_ra:.1f}) vs {name_b} ({new_rb:.1f}) -> WR {score_a:.2%}")

    # Save results
    df = pd.DataFrame(all_results)
    csv_path = arena.output_dir / f"results_{args.mode}_{int(time.time())}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Plotting
    if not df.empty:
        plt.figure(figsize=(12, 6))
        
        if args.mode == 'vs-random':
            sns.lineplot(data=df, x='iteration', y='bb_100_a', marker='o')
            plt.title('Performance vs Random Agent')
            plt.ylabel('BB/100')
            plt.xlabel('Training Iteration')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
        elif args.mode == 'ladder':
            sns.lineplot(data=df, x='iteration', y='bb_100_a', marker='o')
            plt.title('Performance vs Previous Version')
            plt.ylabel('BB/100 (vs Previous)')
            plt.xlabel('Training Iteration')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
        elif args.mode == 'elo':
             # Filter for just model iterations
             elo_data = []
             for k, v in elo.ratings.items():
                 if k.startswith('Iter_'):
                     iter_num = int(k.split('_')[1])
                     elo_data.append({'iteration': iter_num, 'elo': v})
             
             elo_df = pd.DataFrame(elo_data).sort_values('iteration')
             sns.lineplot(data=elo_df, x='iteration', y='elo', marker='o')
             plt.title('ELO Rating Progression')
             plt.ylabel('ELO')
             plt.xlabel('Training Iteration')
             plt.grid(True, alpha=0.3)
        
        plot_path = arena.output_dir / f"plot_{args.mode}_{int(time.time())}.png"
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        
        # ASCII dump
        if args.mode == 'vs-random':
            ascii_plot(df, 'iteration', 'bb_100_a', 'Performance vs Random')
        elif args.mode == 'ladder':
            ascii_plot(df, 'iteration', 'bb_100_a', 'Performance vs Previous Version')
        elif args.mode == 'elo':
             ascii_plot(elo_df, 'iteration', 'elo', 'ELO Rating')

if __name__ == "__main__":
    main()
