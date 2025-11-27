"""
Core Poker Arena Logic
"""
import sys
import os
import re
import json
import random
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Tuple, TYPE_CHECKING

import torch
import pandas as pd

# Add training directory to path to import modules
TRAINING_DIR = Path(__file__).parent.parent / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.append(str(TRAINING_DIR))

from model_agent import (
    extract_state,
    RandomAgent,
    HeuristicAgent,
    load_model_agent
)
from monte_carlo import MultiRunoutEvaluator, simulate_runouts
import poker_api_binding
from config import DEFAULT_MODELS_DIR

if TYPE_CHECKING:
    from rl_model import PokerActorCritic

class Arena:
    def __init__(self, device: str = "cpu", output_dir: str = "arena_results", num_runouts: int = 0):
        """
        Initialize the Arena.
        
        Args:
            device: Torch device for model inference
            output_dir: Directory for saving results
            num_runouts: Number of Monte Carlo runouts for equity calculation (0=disabled, 50=recommended)
                        When enabled, pots are split based on average equity across multiple simulations
                        ("run it N times" feature)
        """
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Game config (standard heads-up)
        self.game_config = {
            'num_players': 2,
            'smallBlind': 20,
            'bigBlind': 40,
            'startingChips': 4000,
            'minPlayers': 2,
            'maxPlayers': 2
        }
        
        # Monte Carlo runout settings
        self.num_runouts = num_runouts  # 0 = disabled, 50 = default when enabled
        if num_runouts > 0:
            self.runout_evaluator = MultiRunoutEvaluator(self.game_config, num_runouts)
        else:
            self.runout_evaluator = None
        
        # Cache for loaded models to avoid reloading
        self.model_cache = {}

    def load_model(self, checkpoint_path: Path) -> "PokerActorCritic":
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

    def play_hand(self, agent_a_config: Dict, agent_b_config: Dict, capture_details: bool = False) -> Dict[str, Any]:
        """
        Play a single hand between two agents.
        
        Args:
            agent_a_config: Dict with 'type' ('model', 'random', 'heuristic') and 'model'/'path' if needed
            agent_b_config: Dict with 'type' ('model', 'random', 'heuristic') and 'model'/'path' if needed
            capture_details: If True, include detailed hand history for replay
        """
        # Betting rules: cap raises per betting round to prevent infinite loops
        MAX_RAISES_PER_ROUND = 4  # Standard poker cap
        
        def create_agent(player_id: str, config: Dict) -> Any:
            name = config.get('name', f"{config['type']}_{player_id}")
            if config['type'] == 'model':
                model = config.get('model')
                if model is None and 'path' in config:
                    model = self.load_model(Path(config['path']))
                
                # Default to stochastic sampling (deterministic=False) to sample from probability distribution
                return load_model_agent(player_id, name, model=model, 
                                      device=self.device, deterministic=config.get('deterministic', False))
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
        
        # Track raises per stage for betting caps
        raises_this_round = 0
        current_betting_stage = None
        
        # Capture hand details for replay
        hand_details = None
        if capture_details:
            hand_details = {
                'hole_cards': {},
                'community_cards': [],
                'actions': [],
                'stages': {},  # stage -> {pot, actions}
                'final_state': None
            }
            # Capture initial hole cards after deal
            for p in game_state.get('players', []):
                hand_details['hole_cards'][p['id']] = p.get('holeCards', [])
        
        current_stage = None
        
        while step < max_steps:
            step += 1
            
            stage = game_state.get('stage', '').lower()
            if stage in ['complete', 'showdown']:
                break
            
            # Reset raise counter when stage changes
            if stage != current_betting_stage:
                current_betting_stage = stage
                raises_this_round = 0
            
            # Track stage transitions for hand details
            if capture_details and stage != current_stage:
                current_stage = stage
                hand_details['community_cards'] = game_state.get('communityCards', [])
                if stage not in hand_details['stages']:
                    hand_details['stages'][stage] = {
                        'pot': game_state.get('pot', 0),
                        'community_cards': game_state.get('communityCards', [])
                    }
                
            current_player_id = game_state.get('currentPlayerId')
            if not current_player_id or current_player_id == 'none':
                break
                
            agent = agents[current_player_id]
            
            # Get legal actions
            legal_actions = game_state.get('actionConstraints', {}).get('legalActions', [])
            if not legal_actions:
                break
            
            # Apply raise cap: if we've hit the raise cap, remove raise/bet from legal actions
            if raises_this_round >= MAX_RAISES_PER_ROUND:
                legal_actions = [a for a in legal_actions if a not in ['raise', 'bet']]
                # If no actions left after filtering, something is wrong
                if not legal_actions:
                    legal_actions = ['fold']
            
            # Extract state
            state_dict = extract_state(game_state, current_player_id)
            
            # Get action
            if hasattr(agent, 'select_action'):
                action_type, amount, action_label = agent.select_action(state_dict, legal_actions)
            else:
                # Fallback for weird agent types?
                action_type = 'check'
                amount = 0
            
            # Enforce all-in when player doesn't have enough chips for the selected action
            player_chips = state_dict.get('player_chips', 0)
            to_call = state_dict.get('to_call', 0)
            
            if action_type in ['bet', 'raise']:
                # If the bet/raise amount exceeds available chips, go all-in instead
                if amount >= player_chips:
                    action_type = 'all_in'
                    amount = 0
            elif action_type == 'call':
                # If can't afford the call, go all-in
                if to_call >= player_chips and player_chips > 0:
                    action_type = 'all_in'
                    amount = 0
            
            # Track raises for the cap
            if action_type in ['raise', 'bet']:
                raises_this_round += 1
            
            # Capture action details
            if capture_details:
                # Get player name
                player_name = None
                for p in game_state.get('players', []):
                    if p['id'] == current_player_id:
                        player_name = p.get('name', current_player_id)
                        break
                
                hand_details['actions'].append({
                    'player_id': current_player_id,
                    'player_name': player_name,
                    'action': action_type,
                    'amount': amount,
                    'stage': game_state.get('stage', 'Unknown'),
                    'pot_before': game_state.get('pot', 0)
                })
            
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
        
        # Use multi-runout equity calculation if enabled
        if self.num_runouts > 0 and self.runout_evaluator:
            # Calculate equity for each player by running multiple simulations
            # This implements "run it N times" - splitting the pot based on 
            # average equity across multiple board runouts
            equities = self.runout_evaluator.calculate_equity(history)
            
            if equities:
                # Calculate total pot from player contributions
                total_pot = sum(initial_chips - p['chips'] + p['chips'] 
                               for p in game_state['players'])
                # Actually: pot is money in the middle. Start with what we put in.
                total_pot = game_state.get('pot', 0)
                
                # If pot is 0 (e.g., someone folded preflop), calculate from chip changes
                if total_pot == 0:
                    total_in_pot = sum(initial_chips - p['chips'] for p in game_state['players'] if p['chips'] < initial_chips)
                    total_pot = abs(total_in_pot)
                
                # Split pot based on equity
                pot_shares = self.runout_evaluator.split_pot(history, total_pot)
                
                for p in game_state['players']:
                    pid = p['id']
                    # Calculate profit: equity share of pot - amount put in
                    player_contribution = initial_chips - p['chips'] + (p['chips'] - initial_chips if p['chips'] > initial_chips else 0)
                    # Simpler: just use equity to determine winnings
                    if pid in pot_shares:
                        # Equity share minus what we would have had if we just kept our chips
                        equity_winnings = pot_shares[pid] - (total_pot * 0.5)  # vs expected break-even share
                        rewards[pid] = equity_winnings / big_blind
                    else:
                        # Fallback to actual result
                        profit = p['chips'] - initial_chips
                        rewards[pid] = profit / big_blind
            else:
                # Fallback to actual results if equity calculation failed
                for p in game_state['players']:
                    profit = p['chips'] - initial_chips
                    rewards[p['id']] = profit / big_blind
        else:
            # Standard single-outcome calculation
            for p in game_state['players']:
                # Return result in Big Blinds
                profit = p['chips'] - initial_chips
                rewards[p['id']] = profit / big_blind
        
        # Capture final state
        if capture_details:
            hand_details['community_cards'] = game_state.get('communityCards', [])
            hand_details['final_state'] = {
                'stage': game_state.get('stage', 'Unknown'),
                'pot': game_state.get('pot', 0),
                'players': [{
                    'id': p['id'],
                    'name': p.get('name', p['id']),
                    'chips': p['chips'],
                    'hole_cards': p.get('holeCards', []),
                    'profit_bb': rewards.get(p['id'], 0)
                } for p in game_state.get('players', [])]
            }
            # Add equity information if calculated
            if self.num_runouts > 0 and self.runout_evaluator:
                equities = self.runout_evaluator.calculate_equity(history)
                hand_details['final_state']['equities'] = equities
                hand_details['final_state']['num_runouts'] = self.num_runouts
            rewards['details'] = hand_details
            
        return rewards

    def play_match(self, agent_a_config: Dict, agent_b_config: Dict, 
                   num_hands: int = 100) -> Dict:
        """Play a match between two agents"""
        name_a = agent_a_config.get('name', 'Agent A')
        name_b = agent_b_config.get('name', 'Agent B')
        # print(f"Playing {name_a} vs {name_b} ({num_hands} hands)...")
        
        # Pre-load models to avoid loading them for every hand
        # We create copies of the config to avoid modifying the caller's dict
        config_a = agent_a_config.copy()
        config_b = agent_b_config.copy()
        
        if config_a.get('type') == 'model' and 'model' not in config_a and 'path' in config_a:
            # print(f"  Loading model A from {config_a['path']}...")
            config_a['model'] = self.load_model(Path(config_a['path']))
            
        if config_b.get('type') == 'model' and 'model' not in config_b and 'path' in config_b:
            # print(f"  Loading model B from {config_b['path']}...")
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
            
            # Collect results
            # Removed tqdm for server usage, but in CLI we might want it.
            # For shared usage, we just iterate
            for k, f in enumerate(batch_futures):
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

class RoundRobinTournament:
    """
    Round-robin tournament that plays random matches between all checkpoint pairs
    and tracks performance over time.
    """
    
    def __init__(self, arena: Arena):
        self.arena = arena
        self.results: Dict[int, Dict[str, Any]] = {}  # iteration -> performance stats
        
    def run_tournament(
        self, 
        checkpoints: List[Tuple[int, Path]], 
        hands_per_match: int = 100,
        max_matches_per_checkpoint: int = 10,
        include_random: bool = True,
        include_heuristic: bool = True,
        callback=None
    ) -> pd.DataFrame:
        """
        Run a round-robin tournament between checkpoints.
        
        Args:
            checkpoints: List of (iteration, path) tuples
            hands_per_match: Number of hands per match
            max_matches_per_checkpoint: Max opponents per checkpoint (for large checkpoint sets)
            include_random: Include Random agent as opponent
            include_heuristic: Include Heuristic agent as opponent
            callback: Optional callback function for progress updates
            
        Returns:
            DataFrame with performance metrics for each checkpoint
        """
        if not checkpoints:
            return pd.DataFrame()
            
        # Initialize results tracking for each checkpoint
        for iter_num, path in checkpoints:
            self.results[iter_num] = {
                'iteration': iter_num,
                'path': str(path),
                'total_hands': 0,
                'total_wins': 0,
                'total_bb': 0.0,
                'matches_played': 0,
                'opponents_beaten': 0,
                'vs_random_bb100': None,
                'vs_heuristic_bb100': None,
                'vs_models_bb100': None,
            }
        
        sorted_checkpoints = sorted(checkpoints, key=lambda x: x[0])
        total_checkpoints = len(sorted_checkpoints)
        
        # Build match list
        matches = []
        
        for i, (iter_a, path_a) in enumerate(sorted_checkpoints):
            opponents = []
            
            # Add baseline agents
            if include_random:
                opponents.append(('random', {'type': 'random', 'name': 'Random'}))
            if include_heuristic:
                opponents.append(('heuristic', {'type': 'heuristic', 'name': 'Heuristic'}))
            
            # Add other checkpoints as opponents (sample if too many)
            other_checkpoints = [(iter_b, path_b) for iter_b, path_b in sorted_checkpoints if iter_b != iter_a]
            
            if len(other_checkpoints) > max_matches_per_checkpoint:
                # Sample: always include first, last, and random selection in between
                sampled = []
                if other_checkpoints:
                    sampled.append(other_checkpoints[0])  # earliest
                    sampled.append(other_checkpoints[-1])  # latest
                    
                    # Random sample from middle
                    middle = other_checkpoints[1:-1]
                    sample_size = min(max_matches_per_checkpoint - 2, len(middle))
                    if sample_size > 0:
                        sampled.extend(random.sample(middle, sample_size))
                other_checkpoints = sampled
            
            for iter_b, path_b in other_checkpoints:
                opponents.append(('model', {
                    'type': 'model',
                    'path': str(path_b),
                    'name': f'Iter_{iter_b}'
                }))
            
            for opp_type, opp_config in opponents:
                matches.append((iter_a, path_a, opp_type, opp_config))
        
        total_matches = len(matches)
        print(f"Running round-robin tournament: {total_checkpoints} checkpoints, {total_matches} matches")
        
        # Shuffle matches for fair sampling
        random.shuffle(matches)
        
        # Run matches
        for match_idx, (iter_a, path_a, opp_type, opp_config) in enumerate(matches):
            config_a = {'type': 'model', 'path': str(path_a), 'name': f'Iter_{iter_a}'}
            
            if callback:
                callback({
                    'type': 'match_started',
                    'match_idx': match_idx,
                    'total_matches': total_matches,
                    'agent_a': config_a['name'],
                    'agent_b': opp_config['name'],
                    'iteration': iter_a
                })
            
            try:
                result = self.arena.play_match(config_a, opp_config, num_hands=hands_per_match)
                
                if result:
                    # Update stats for this checkpoint
                    stats = self.results[iter_a]
                    stats['total_hands'] += result.get('hands', 0)
                    stats['total_wins'] += int(result.get('win_rate_a', 0) * result.get('hands', 0))
                    stats['total_bb'] += result.get('avg_bb_a', 0) * result.get('hands', 0)
                    stats['matches_played'] += 1
                    
                    if result.get('win_rate_a', 0) > 0.5:
                        stats['opponents_beaten'] += 1
                    
                    # Track specific opponent types
                    bb100 = result.get('bb_100_a', 0)
                    if opp_type == 'random':
                        stats['vs_random_bb100'] = bb100
                    elif opp_type == 'heuristic':
                        stats['vs_heuristic_bb100'] = bb100
                    else:
                        # Average vs models
                        if stats['vs_models_bb100'] is None:
                            stats['vs_models_bb100'] = bb100
                        else:
                            n = stats['matches_played'] - (1 if include_random else 0) - (1 if include_heuristic else 0)
                            if n > 0:
                                stats['vs_models_bb100'] = (stats['vs_models_bb100'] * (n - 1) + bb100) / n
                    
                    if callback:
                        callback({
                            'type': 'match_complete',
                            'iteration': iter_a,
                            'opponent': opp_config['name'],
                            'win_rate_a': result.get('win_rate_a', 0),
                            'bb_100_a': bb100,
                            'agent_a': config_a['name'],
                            'agent_b': opp_config['name'],
                            'hands': result.get('hands', 0)
                        })
                        
            except Exception as e:
                print(f"Error in match Iter_{iter_a} vs {opp_config['name']}: {e}")
                if callback:
                    callback({'type': 'error', 'message': str(e)})
        
        # Calculate final metrics
        rows = []
        for iter_num in sorted(self.results.keys()):
            stats = self.results[iter_num]
            total_hands = stats['total_hands']
            
            row = {
                'iteration': iter_num,
                'matches_played': stats['matches_played'],
                'total_hands': total_hands,
                'win_rate': stats['total_wins'] / total_hands if total_hands > 0 else 0,
                'bb_100': (stats['total_bb'] / total_hands * 100) if total_hands > 0 else 0,
                'opponents_beaten': stats['opponents_beaten'],
                'vs_random_bb100': stats['vs_random_bb100'],
                'vs_heuristic_bb100': stats['vs_heuristic_bb100'],
                'vs_models_bb100': stats['vs_models_bb100'],
            }
            rows.append(row)
            
            if callback:
                callback({
                    'type': 'checkpoint_summary',
                    'iteration': iter_num,
                    **row
                })
        
        df = pd.DataFrame(rows)
        return df
    
    def get_performance_over_time(self) -> pd.DataFrame:
        """Get DataFrame showing how each checkpoint performs over training iterations."""
        rows = []
        for iter_num in sorted(self.results.keys()):
            stats = self.results[iter_num]
            total_hands = stats['total_hands']
            rows.append({
                'iteration': iter_num,
                'win_rate': stats['total_wins'] / total_hands if total_hands > 0 else 0,
                'bb_100': (stats['total_bb'] / total_hands * 100) if total_hands > 0 else 0,
                'vs_random_bb100': stats['vs_random_bb100'] or 0,
                'vs_heuristic_bb100': stats['vs_heuristic_bb100'] or 0,
                'vs_models_bb100': stats['vs_models_bb100'] or 0,
            })
        return pd.DataFrame(rows)

