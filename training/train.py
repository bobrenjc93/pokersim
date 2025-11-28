#!/usr/bin/env python3
"""
Reinforcement Learning Training Script for Poker AI

This script uses PPO (Proximal Policy Optimization) with self-play to train
a poker agent from scratch. The agent learns optimal strategies through
experience and exploration.

Key features:
- PPO algorithm for stable policy learning
- Self-play against past model versions
- Transformer-based actor-critic architecture
- Comprehensive logging and checkpointing

Usage:
    python train.py --iterations 5000 --episodes-per-iter 200 --ppo-epochs 10
"""

import concurrent.futures
import argparse
import random
import sys
import os
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

try:
    import orjson as json
except ImportError:
    import json

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

# Import RL components
from rl_state_encoder import RLStateEncoder, estimate_hand_strength, get_hand_category
from rl_model import PokerActorCritic, create_actor_critic
from ppo import PPOTrainer
from model_agent import (
    ModelAgent, 
    RandomAgent,
    HeuristicAgent,
    TightAgent,
    LoosePassiveAgent,
    AggressiveAgent,
    CallingStationAgent,
    HeroCallerAgent,
    ACTION_MAP, 
    ACTION_NAMES, 
    BET_SIZE_MAP,
    convert_action_label,
    create_legal_actions_mask,
    extract_state
)
from rl_state_encoder import estimate_hand_strength

# Import model version and log level from config
from config import MODEL_VERSION, LOG_LEVEL, DEFAULT_MODELS_DIR

# Import Monte Carlo simulation for multi-runout regret calculation
from monte_carlo import (
    simulate_runouts,
    calculate_action_regrets,
    compute_regret_weighted_reward,
    MultiRunoutEvaluator
)

# Import poker_api_binding for direct C++ calls
try:
    import poker_api_binding
except ImportError:
    print("Error: 'poker_api_binding' not found. Please compile the binding (cd api && make module).")
    sys.exit(1)


# =============================================================================
# RL Training Session
# =============================================================================

class RLTrainingSession:
    """
    Manages RL training session for poker.
    
    Handles:
    - Episode collection (playing hands)
    - Trajectory storage
    - PPO updates
    - Model checkpointing
    - Self-play opponent management
    """
    
    def __init__(
        self,
        model: PokerActorCritic,
        ppo_trainer: PPOTrainer,
        game_config: Dict[str, Any],
        device: torch.device,
        output_dir: Path,
        tensorboard_dir: Optional[Path] = None,
        log_level: int = 0,
        num_runouts: int = 0,
        regret_weight: float = 0.5
    ):
        """
        Args:
            model: Actor-critic model
            ppo_trainer: PPO trainer
            game_config: Game configuration
            device: Device for training
            output_dir: Directory for model checkpoints
            tensorboard_dir: Directory for TensorBoard logs
            log_level: Logging verbosity (0=minimal, 1=normal, 2=verbose)
            num_runouts: Number of runouts for Monte Carlo regret calculation (0 = disabled)
            regret_weight: Weight for regret-based reward adjustment (0-1)
        """
        self.model = model
        self.ppo_trainer = ppo_trainer
        self.game_config = game_config
        self.device = device
        self.output_dir = output_dir
        self.tensorboard_dir = tensorboard_dir
        self.log_level = log_level
        
        # Multi-runout regret calculation settings
        self.num_runouts = num_runouts  # 0 = disabled, 50 = default when enabled
        self.regret_weight = regret_weight
        
        # Opponent pool for self-play (stores past model versions)
        self.opponent_pool: List[Path] = []
        self.max_opponent_pool_size = 30  # Increased for better diversity and plateau prevention
        
        # Cache for loaded opponent models to avoid reloading
        self._opponent_model_cache: Dict[str, PokerActorCritic] = {}
        self._max_opponent_cache_size = 10  # Keep more models in memory for faster sampling
        
        # Track win rates against each opponent for prioritized sampling
        self.opponent_win_rates: Dict[str, float] = {}  # checkpoint_path -> win_rate
        self._opponent_game_counts: Dict[str, int] = {}  # checkpoint_path -> num_games
        
        # TensorBoard writer
        self.writer = None
        if tensorboard_dir:
            log_dir = tensorboard_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            if self.log_level >= 1:
                print(f"📊 TensorBoard logging to: {log_dir}")
        
        # Statistics
        self.stats = {
            'iteration': 0,
            'total_episodes': 0,
            'total_timesteps': 0,
            'avg_reward': deque(maxlen=100),
            'avg_episode_length': deque(maxlen=100),
            'win_rate': deque(maxlen=100),
            'avg_value_estimate': deque(maxlen=100),
            'explained_variance_history': deque(maxlen=100)
        }
        
        # Plateau detection tracking
        self._plateau_counter = 0  # How many iterations we've been in plateau
        self._last_plateau_response_iter = 0  # Last iteration we responded to plateau
        self._initial_entropy_coef = None  # Will be set from ppo_trainer
    
    def collect_episode(
        self,
        use_opponent_pool: bool = True,
        verbose: bool = False,
        deterministic: bool = False
    ) -> Dict[str, Any]:
        """
        Collect one episode (complete poker hand) using current policy.
        
        Args:
            use_opponent_pool: Whether to use past models as opponents
            verbose: Print detailed progress
            deterministic: Whether to use deterministic action selection (argmax)
        
        Returns:
            Episode data dict with states, actions, rewards, etc.
        """
        # Create encoder for this episode
        encoder = RLStateEncoder()
        
        # Create agents
        num_players = self.game_config['num_players']
        agents = []
        
        # Main agent (uses current model)
        main_player_id = 'p0'
        agents.append({
            'id': main_player_id,
            'type': 'model',
            'encoder': encoder
        })
        
        # Opponent agents with diverse opponent selection for better generalization
        # KEY INSIGHT: Need opponents that PUNISH all-in with weak hands by CALLING
        # The main problem was that folding opponents make all-in "work"
        for i in range(1, num_players):
            player_id = f'p{i}'
            
            # Opponent selection probabilities (REBALANCED TO PUNISH BAD ALL-INS):
            # - CallingStation: 25% - CRITICAL: Calls all-ins, shows model trash loses at showdown
            # - HeroCaller: 15% - Calls down with medium hands when opponent is aggressive
            # - Past checkpoint: 15% (when pool available) - diverse self-play
            # - Current model: 10% - immediate self-play for Nash equilibrium
            # - TightAgent: 15% - Punishes by only playing premium hands
            # - Heuristic: 10% - Strategic baseline
            # - AggressiveAgent: 5% - Teaches calling down bluffs
            # - LoosePassive: 3% - Rewards value betting
            # - Random: 2% - Minimal exploration
            roll = random.random()
            
            if roll < 0.25:
                # CallingStationAgent (25% of time) - CRITICAL FOR FIXING ALL-IN PROBLEM
                # This agent CALLS all-ins with any decent hand, showing the model
                # that trash hands lose at showdown
                agents.append({
                    'id': player_id,
                    'type': 'calling_station'
                })
            elif roll < 0.40:
                # HeroCallerAgent (15% of time) - Calls down suspected bluffs
                agents.append({
                    'id': player_id,
                    'type': 'hero_caller'
                })
            elif use_opponent_pool and self.opponent_pool and roll < 0.55:
                # Use past checkpoint (15% of time when pool available)
                checkpoint_path = self._sample_opponent_checkpoint()
                opponent_model = self._load_opponent_model(checkpoint_path)
                if opponent_model is not None:
                    agents.append({
                        'id': player_id,
                        'type': 'past_model',
                        'model': opponent_model,
                        'encoder': RLStateEncoder(),
                        'checkpoint_path': str(checkpoint_path)
                    })
                else:
                    agents.append({'id': player_id, 'type': 'calling_station'})
            elif use_opponent_pool and roll < 0.65:
                # Use current model as opponent (10% of time)
                agents.append({
                    'id': player_id,
                    'type': 'model',
                    'encoder': RLStateEncoder()
                })
            elif roll < 0.80:
                # Use TightAgent (15% of time) - Only plays premium hands
                agents.append({
                    'id': player_id,
                    'type': 'tight'
                })
            elif roll < 0.90:
                # Use heuristic agent (10% of time)
                agents.append({
                    'id': player_id,
                    'type': 'heuristic'
                })
            elif roll < 0.95:
                # Use AggressiveAgent (5% of time) - teaches model to call down
                agents.append({
                    'id': player_id,
                    'type': 'aggressive'
                })
            elif roll < 0.98:
                # Use LoosePassive agent (3% of time) - rewards value betting
                agents.append({
                    'id': player_id,
                    'type': 'loose_passive'
                })
            else:
                # Use random agent (2% of time)
                agents.append({
                    'id': player_id,
                    'type': 'random'
                })
        
        # Initialize game history
        history = []
        for agent in agents:
            history.append({
                'type': 'addPlayer',
                'playerId': agent['id'],
                'playerName': f"Player_{agent['id']}"
            })
        
        # Get initial state
        response = self._call_api(history)
        if not response['success']:
            return {'states': [], 'actions': [], 'rewards': {}, 'success': False}
        
        game_state = response['gameState']
        
        # Determine position (out of position = acting first post-flop)
        # In heads-up, the dealer (button) acts last post-flop
        is_out_of_position = not game_state.get('players', [{}])[0].get('isDealer', False)
        
        # Track which opponent type and checkpoint we're facing for win rate tracking
        opponent_type_faced = None
        opponent_checkpoint_faced = None
        for agent in agents:
            if agent['id'] != main_player_id:
                opponent_type_faced = agent.get('type')
                if opponent_type_faced == 'past_model':
                    # Store checkpoint path for prioritized sampling updates
                    opponent_checkpoint_faced = agent.get('checkpoint_path', '')
                break
        
        # Track episode data
        episode = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': {},
            'legal_actions_masks': [],
            'dones': [],
            'success': True,
            # Reward shaping tracking
            'action_types': [],  # Track action types taken by main player
            'action_labels': [],  # Track action labels for regret calculation
            'hand_strengths': [],  # Track hand strength at each decision
            'step_rewards': [],  # Per-step reward shaping
            'regrets': [],  # Per-step regrets from Monte Carlo simulation
            'folded_to_aggression': False,  # Did main player fold to bet/raise?
            'won_uncontested': False,  # Did main player win without showdown?
            'is_out_of_position': is_out_of_position,
            'opponent_type': opponent_type_faced,
            'opponent_checkpoint': opponent_checkpoint_faced,
        }
        
        # Track opponent aggression (bets/raises by opponents)
        last_action_was_opponent_aggression = False
        
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
                if agent['id'] == current_player_id:
                    current_agent = agent
                    break
            
            if current_agent is None:
                break
            
            # Extract state
            state_dict = extract_state(game_state, current_player_id)
            legal_actions = self._get_legal_actions(game_state)
            
            if not legal_actions:
                break
            
            # Select action based on agent type
            if current_player_id == main_player_id:
                # Main agent always uses current model with trajectory tracking
                state_tensor = encoder.encode_state(state_dict).unsqueeze(0).to(self.device)
                legal_mask = self._create_legal_actions_mask(legal_actions)
                
                # Compute hand strength for reward shaping
                hole_cards = state_dict.get('hole_cards', [])
                community_cards = state_dict.get('community_cards', [])
                hand_strength = estimate_hand_strength(hole_cards, community_cards)
                
                with torch.no_grad():
                    action_logits, value = self.model(state_tensor, legal_mask)
                    action_probs = F.softmax(action_logits, dim=-1)
                    
                    if deterministic:
                        action_idx = torch.argmax(action_probs.squeeze(0)).item()
                        log_prob = F.log_softmax(action_logits, dim=-1)[0, action_idx]
                    else:
                        action_idx = torch.multinomial(action_probs.squeeze(0), 1).item()
                        log_prob = F.log_softmax(action_logits, dim=-1)[0, action_idx]
                
                # Store trajectory data for main agent
                episode['states'].append(state_tensor.squeeze(0).cpu())
                episode['actions'].append(action_idx)
                episode['log_probs'].append(log_prob.cpu())
                episode['values'].append(value.squeeze().cpu())
                episode['legal_actions_masks'].append(legal_mask.squeeze(0).cpu())
                episode['dones'].append(0)  # Will set last one to 1
                
                action_label = self._idx_to_action_label(action_idx)
                action_type, amount = self._convert_action(action_label, state_dict)
                
                # Track action types and hand strength for reward shaping
                episode['action_types'].append(action_type)
                episode['action_labels'].append(action_label)
                episode['hand_strengths'].append(hand_strength)
                
                # Calculate regrets using Monte Carlo simulation if enabled
                if self.num_runouts > 0:
                    regrets = calculate_action_regrets(
                        self.game_config,
                        history,
                        main_player_id,
                        self.model,
                        encoder,
                        self.device,
                        num_runouts=self.num_runouts
                    )
                    episode['regrets'].append(regrets)
                else:
                    episode['regrets'].append({})
                
                # Compute per-step reward shaping based on action appropriateness
                # Include step number for temporal consistency weighting
                step_reward = self._compute_action_shaping_reward(
                    action_type, hand_strength, state_dict, last_action_was_opponent_aggression,
                    step_in_hand=len(episode['states']),  # Current step in hand
                    total_steps_estimate=6  # Average poker hand has ~4-8 actions
                )
                episode['step_rewards'].append(step_reward)
                
                # Check if main player folded to opponent aggression
                if action_type == 'fold' and last_action_was_opponent_aggression:
                    episode['folded_to_aggression'] = True
                
            elif current_agent['type'] == 'heuristic':
                # Use HeuristicAgent for action selection
                heuristic = HeuristicAgent(current_player_id, "Heuristic")
                action_type, amount, action_label = heuristic.select_action(state_dict, legal_actions)
            
            elif current_agent['type'] == 'tight':
                # Use TightAgent - only plays premium hands, punishes random aggression
                tight_agent = TightAgent(current_player_id, "Tight")
                action_type, amount, action_label = tight_agent.select_action(state_dict, legal_actions)
            
            elif current_agent['type'] == 'aggressive':
                # Use AggressiveAgent - bets/raises frequently
                aggressive_agent = AggressiveAgent(current_player_id, "Aggressive")
                action_type, amount, action_label = aggressive_agent.select_action(state_dict, legal_actions)
            
            elif current_agent['type'] == 'loose_passive':
                # Use LoosePassiveAgent - calls too much, rewards value betting
                loose_passive_agent = LoosePassiveAgent(current_player_id, "LoosePassive")
                action_type, amount, action_label = loose_passive_agent.select_action(state_dict, legal_actions)
            
            elif current_agent['type'] == 'calling_station':
                # Use CallingStationAgent - CRITICAL for punishing trash all-ins
                # This agent calls all-ins to show model that weak hands lose at showdown
                calling_station = CallingStationAgent(current_player_id, "CallingStation")
                action_type, amount, action_label = calling_station.select_action(state_dict, legal_actions)
            
            elif current_agent['type'] == 'hero_caller':
                # Use HeroCallerAgent - calls down suspected bluffs with medium hands
                hero_caller = HeroCallerAgent(current_player_id, "HeroCaller")
                action_type, amount, action_label = hero_caller.select_action(state_dict, legal_actions)
                
            elif current_agent['type'] == 'past_model':
                # Use past checkpoint model for action selection
                opponent_model = current_agent['model']
                opponent_encoder = current_agent['encoder']
                state_tensor = opponent_encoder.encode_state(state_dict).unsqueeze(0).to(self.device)
                legal_mask = self._create_legal_actions_mask(legal_actions)
                
                with torch.no_grad():
                    action_logits, _ = opponent_model(state_tensor, legal_mask)
                    action_probs = F.softmax(action_logits, dim=-1)
                    action_idx = torch.multinomial(action_probs.squeeze(0), 1).item()
                
                action_label = self._idx_to_action_label(action_idx)
                action_type, amount = self._convert_action(action_label, state_dict)
                
            elif current_agent['type'] == 'model':
                # Opponent using current model (self-play)
                opponent_encoder = current_agent['encoder']
                state_tensor = opponent_encoder.encode_state(state_dict).unsqueeze(0).to(self.device)
                legal_mask = self._create_legal_actions_mask(legal_actions)
                
                with torch.no_grad():
                    action_logits, _ = self.model(state_tensor, legal_mask)
                    action_probs = F.softmax(action_logits, dim=-1)
                    action_idx = torch.multinomial(action_probs.squeeze(0), 1).item()
                
                action_label = self._idx_to_action_label(action_idx)
                action_type, amount = self._convert_action(action_label, state_dict)
                
            else:
                # Random agent (default)
                action_type, amount, action_label = self._random_action(legal_actions, state_dict)
            
            # Track opponent aggression for fold-to-aggression detection
            if current_player_id != main_player_id:
                last_action_was_opponent_aggression = action_type in ('bet', 'raise', 'all_in')
            else:
                last_action_was_opponent_aggression = False
            
            # Observe action for all agents' encoders
            stage = state_dict.get('stage', 'Preflop')
            pot = state_dict.get('pot', 0)
            for agent in agents:
                if agent['type'] in ('model', 'past_model') and 'encoder' in agent:
                    agent['encoder'].add_action(current_player_id, action_type, amount, pot, stage)
            
            # Apply action
            history.append({
                'type': 'playerAction',
                'playerId': current_player_id,
                'action': action_type,
                'amount': amount
            })
            
            # Get new state
            response = self._call_api(history)
            if not response['success']:
                episode['success'] = False
                break
            
            game_state = response['gameState']
        
        # Calculate rewards (chips won/lost)
        final_chips = {}
        initial_chips = self.game_config['startingChips']
        
        for player_data in game_state['players']:
            player_id = player_data['id']
            final_chips[player_id] = player_data['chips']
            episode['rewards'][player_id] = final_chips[player_id] - initial_chips
        
        # Detect if pot was won uncontested (no showdown, opponent folded)
        final_stage = game_state.get('stage', '').lower()
        main_profit = episode['rewards'].get(main_player_id, 0)
        if main_profit > 0 and final_stage == 'complete':
            # Won without showdown - opponent folded
            episode['won_uncontested'] = True
        
        # Mark last state as done
        if episode['dones']:
            episode['dones'][-1] = 1
        
        # Normalize rewards (divide by starting chips for scale)
        main_reward = episode['rewards'].get(main_player_id, 0) / initial_chips
        
        # Apply regret-weighted reward adjustment if regrets were calculated
        if self.num_runouts > 0 and episode['regrets'] and episode['action_labels']:
            main_reward = compute_regret_weighted_reward(
                main_reward,
                episode['regrets'],
                episode['action_labels'],
                regret_weight=self.regret_weight
            )
        
        episode['main_reward'] = main_reward
        
        # Convert episode data to tensors
        if episode['states']:
            episode['states'] = torch.stack(episode['states'])
            episode['actions'] = torch.tensor(episode['actions'], dtype=torch.long)
            episode['log_probs'] = torch.stack(episode['log_probs'])
            episode['values'] = torch.stack(episode['values'])
            episode['legal_actions_masks'] = torch.stack(episode['legal_actions_masks'])
            episode['dones'] = torch.tensor(episode['dones'], dtype=torch.float32)
        
        return episode
    
    def evaluate_vs_random(self, num_episodes: int = 50) -> Dict[str, float]:
        """
        Evaluate current model against random agent.
        
        Args:
            num_episodes: Number of episodes to play
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.log_level >= 1:
            print(f"\n📊 Evaluating vs Random ({num_episodes} episodes)...")
            
        wins = 0
        total_reward_bb = 0
        total_episodes = 0
        
        # Track action counts for evaluation
        eval_action_counts = {name: 0 for name in ACTION_NAMES}
        
        # Run evaluation episodes
        # We use concurrent execution for speed, similar to train_iteration
        max_workers = min(num_episodes, 10)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Force use_opponent_pool=False to play against Random
            # Force deterministic=True for evaluation
            futures = [
                executor.submit(self.collect_episode, use_opponent_pool=False, verbose=False, deterministic=True)
                for _ in range(num_episodes)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    episode = future.result()
                    
                    if not episode['success']:
                        continue
                        
                    total_episodes += 1
                    
                    # Update eval action counts
                    for act_idx in episode['actions']:
                        eval_action_counts[ACTION_NAMES[act_idx.item()]] += 1
                    
                    # Check if won (profit > 0)
                    main_reward = episode['main_reward'] # This is normalized by starting chips
                    
                    # Calculate raw profit in chips
                    starting_chips = self.game_config['startingChips']
                    raw_profit = main_reward * starting_chips
                    
                    # Convert to Big Blinds
                    big_blind = self.game_config['bigBlind']
                    bb_profit = raw_profit / big_blind
                    total_reward_bb += bb_profit
                    
                    if raw_profit > 0:
                        wins += 1
                        
                except Exception as e:
                    print(f"⚠️  Error in evaluation episode: {e}")
        
        if total_episodes == 0:
            return {}
            
        win_rate = wins / total_episodes
        avg_bb_per_hand = total_reward_bb / total_episodes
        bb_per_100 = avg_bb_per_hand * 100
        
        if self.log_level >= 1:
            print(f"  Win Rate: {win_rate:.2%}")
            print(f"  BB/100: {bb_per_100:.2f}")
            
            # Print action distribution for evaluation
            total_eval_actions = sum(eval_action_counts.values())
            if total_eval_actions > 0:
                print("  Eval Action Distribution:")
                grouped_counts = {'fold': 0, 'check': 0, 'call': 0, 'bet': 0, 'raise': 0, 'all_in': 0}
                for name, count in eval_action_counts.items():
                    if name in grouped_counts:
                        grouped_counts[name] += count
                    elif name.startswith('bet_'):
                        grouped_counts['bet'] += count
                    elif name.startswith('raise_'):
                        grouped_counts['raise'] += count
                
                for act in ['fold', 'check', 'call', 'bet', 'raise', 'all_in']:
                    count = grouped_counts[act]
                    pct = count / total_eval_actions
                    print(f"    {act.ljust(8)}: {pct:.1%} ({count})")
            
        # Log to TensorBoard
        if self.writer:
            iteration = self.stats['iteration']
            self.writer.add_scalar('Evaluation/WinRate_vs_Random', win_rate, iteration)
            self.writer.add_scalar('Evaluation/BB100_vs_Random', bb_per_100, iteration)
            
        return {
            'win_rate': win_rate,
            'bb_per_100': bb_per_100
        }

    def evaluate_vs_heuristic(self, num_episodes: int = 50) -> Dict[str, float]:
        """
        Evaluate current model against HeuristicAgent.
        
        Args:
            num_episodes: Number of episodes to play
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.log_level >= 1:
            print(f"\n📊 Evaluating vs Heuristic ({num_episodes} episodes)...")
            
        wins = 0
        total_reward_bb = 0
        total_episodes = 0
        
        # Track action counts for evaluation
        eval_action_counts = {name: 0 for name in ACTION_NAMES}
        
        # Run evaluation episodes
        max_workers = min(num_episodes, 10)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use a special method to collect episodes vs heuristic
            futures = [
                executor.submit(self._collect_episode_vs_heuristic, deterministic=True)
                for _ in range(num_episodes)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    episode = future.result()
                    
                    if not episode['success']:
                        continue
                        
                    total_episodes += 1
                    
                    # Update eval action counts
                    for act_idx in episode['actions']:
                        eval_action_counts[ACTION_NAMES[act_idx.item()]] += 1
                    
                    # Check if won (profit > 0)
                    main_reward = episode['main_reward']
                    
                    # Calculate raw profit in chips
                    starting_chips = self.game_config['startingChips']
                    raw_profit = main_reward * starting_chips
                    
                    # Convert to Big Blinds
                    big_blind = self.game_config['bigBlind']
                    bb_profit = raw_profit / big_blind
                    total_reward_bb += bb_profit
                    
                    if raw_profit > 0:
                        wins += 1
                        
                except Exception as e:
                    print(f"⚠️  Error in heuristic evaluation episode: {e}")
        
        if total_episodes == 0:
            return {}
            
        win_rate = wins / total_episodes
        avg_bb_per_hand = total_reward_bb / total_episodes
        bb_per_100 = avg_bb_per_hand * 100
        
        if self.log_level >= 1:
            print(f"  Win Rate: {win_rate:.2%}")
            print(f"  BB/100: {bb_per_100:.2f}")
            
            # Print action distribution for evaluation
            total_eval_actions = sum(eval_action_counts.values())
            if total_eval_actions > 0:
                print("  Eval Action Distribution:")
                grouped_counts = {'fold': 0, 'check': 0, 'call': 0, 'bet': 0, 'raise': 0, 'all_in': 0}
                for name, count in eval_action_counts.items():
                    if name in grouped_counts:
                        grouped_counts[name] += count
                    elif name.startswith('bet_'):
                        grouped_counts['bet'] += count
                    elif name.startswith('raise_'):
                        grouped_counts['raise'] += count
                
                for act in ['fold', 'check', 'call', 'bet', 'raise', 'all_in']:
                    count = grouped_counts[act]
                    pct = count / total_eval_actions
                    print(f"    {act.ljust(8)}: {pct:.1%} ({count})")
            
        # Log to TensorBoard
        if self.writer:
            iteration = self.stats['iteration']
            self.writer.add_scalar('Evaluation/WinRate_vs_Heuristic', win_rate, iteration)
            self.writer.add_scalar('Evaluation/BB100_vs_Heuristic', bb_per_100, iteration)
            
        return {
            'win_rate': win_rate,
            'bb_per_100': bb_per_100
        }
    
    def _collect_episode_vs_heuristic(self, deterministic: bool = True) -> Dict[str, Any]:
        """
        Collect one episode specifically against HeuristicAgent.
        Similar to collect_episode but forces heuristic opponent.
        """
        # Create encoder for this episode
        encoder = RLStateEncoder()
        
        # Create agents - main agent vs heuristic
        main_player_id = 'p0'
        agents = [
            {'id': main_player_id, 'type': 'model', 'encoder': encoder},
            {'id': 'p1', 'type': 'heuristic'}
        ]
        
        # Initialize game history
        history = []
        for agent in agents:
            history.append({
                'type': 'addPlayer',
                'playerId': agent['id'],
                'playerName': f"Player_{agent['id']}"
            })
        
        # Get initial state
        response = self._call_api(history)
        if not response['success']:
            return {'states': [], 'actions': [], 'rewards': {}, 'success': False}
        
        game_state = response['gameState']
        
        # Track episode data
        episode = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': {},
            'legal_actions_masks': [],
            'dones': [],
            'success': True
        }
        
        # Main game loop
        max_steps = 1000
        step = 0
        terminal_stages = {'complete', 'showdown'}
        
        while step < max_steps:
            step += 1
            
            current_stage = game_state.get('stage', '').lower()
            if current_stage in terminal_stages:
                break
            
            current_player_id = game_state.get('currentPlayerId')
            if not current_player_id or current_player_id == 'none':
                break
            
            # Find agent for current player
            current_agent = None
            for agent in agents:
                if agent['id'] == current_player_id:
                    current_agent = agent
                    break
            
            if current_agent is None:
                break
            
            # Extract state
            state_dict = extract_state(game_state, current_player_id)
            legal_actions = self._get_legal_actions(game_state)
            
            if not legal_actions:
                break
            
            # Select action based on agent type
            if current_player_id == main_player_id:
                # Main agent uses current model
                state_tensor = encoder.encode_state(state_dict).unsqueeze(0).to(self.device)
                legal_mask = self._create_legal_actions_mask(legal_actions)
                
                with torch.no_grad():
                    action_logits, value = self.model(state_tensor, legal_mask)
                    action_probs = F.softmax(action_logits, dim=-1)
                    
                    if deterministic:
                        action_idx = torch.argmax(action_probs.squeeze(0)).item()
                        log_prob = F.log_softmax(action_logits, dim=-1)[0, action_idx]
                    else:
                        action_idx = torch.multinomial(action_probs.squeeze(0), 1).item()
                        log_prob = F.log_softmax(action_logits, dim=-1)[0, action_idx]
                
                # Store trajectory data
                episode['states'].append(state_tensor.squeeze(0).cpu())
                episode['actions'].append(action_idx)
                episode['log_probs'].append(log_prob.cpu())
                episode['values'].append(value.squeeze().cpu())
                episode['legal_actions_masks'].append(legal_mask.squeeze(0).cpu())
                episode['dones'].append(0)
                
                action_label = self._idx_to_action_label(action_idx)
                action_type, amount = self._convert_action(action_label, state_dict)
            else:
                # Heuristic opponent
                heuristic = HeuristicAgent(current_player_id, "Heuristic")
                action_type, amount, action_label = heuristic.select_action(state_dict, legal_actions)
            
            # Observe action for encoder
            stage = state_dict.get('stage', 'Preflop')
            pot = state_dict.get('pot', 0)
            encoder.add_action(current_player_id, action_type, amount, pot, stage)
            
            # Apply action
            history.append({
                'type': 'playerAction',
                'playerId': current_player_id,
                'action': action_type,
                'amount': amount
            })
            
            # Get new state
            response = self._call_api(history)
            if not response['success']:
                episode['success'] = False
                break
            
            game_state = response['gameState']
        
        # Calculate rewards
        initial_chips = self.game_config['startingChips']
        
        for player_data in game_state['players']:
            player_id = player_data['id']
            episode['rewards'][player_id] = player_data['chips'] - initial_chips
        
        # Mark last state as done
        if episode['dones']:
            episode['dones'][-1] = 1
        
        # Normalize rewards
        main_reward = episode['rewards'].get(main_player_id, 0) / initial_chips
        episode['main_reward'] = main_reward
        
        # Convert episode data to tensors
        if episode['states']:
            episode['states'] = torch.stack(episode['states'])
            episode['actions'] = torch.tensor(episode['actions'], dtype=torch.long)
            episode['log_probs'] = torch.stack(episode['log_probs'])
            episode['values'] = torch.stack(episode['values'])
            episode['legal_actions_masks'] = torch.stack(episode['legal_actions_masks'])
            episode['dones'] = torch.tensor(episode['dones'], dtype=torch.float32)
        
        return episode

    def train_iteration(
        self,
        num_episodes: int,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Run one training iteration (collect episodes + PPO update).
        
        Args:
            num_episodes: Number of episodes to collect
            verbose: Print detailed progress
        
        Returns:
            Dictionary of training statistics
        """
        if self.log_level >= 1:
            print(f"\n{'='*70}")
            print(f"Training Iteration {self.stats['iteration'] + 1}")
            print(f"{'='*70}")
            print(f"Collecting {num_episodes} episodes...")
        
        all_states = []
        all_actions = []
        all_log_probs = []
        all_values = []
        all_advantages = []
        all_returns = []
        all_legal_masks = []
        all_hand_strengths = []  # For auxiliary hand strength prediction loss
        
        episode_rewards = []
        episode_lengths = []
        
        # Track action counts
        action_counts = {name: 0 for name in ACTION_NAMES}
        
        # Collect episodes in parallel
        # Use all available CPUs (leaving a few for system)
        # Cap at 20 to avoid excessive overhead if CPU count is very high
        num_workers = min(num_episodes, max(1, (os.cpu_count() or 4) - 1))
        max_workers = min(num_workers, 20)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.collect_episode, use_opponent_pool=True, verbose=verbose)
                for _ in range(num_episodes)
            ]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    episode = future.result()
                    
                    if not episode['success'] or len(episode['states']) == 0:
                        continue
                    
                    # Update statistics (moved from collect_episode)
                    self.stats['total_episodes'] += 1
                    self.stats['total_timesteps'] += len(episode['states'])
                    self.stats['avg_reward'].append(episode['main_reward'])
                    self.stats['avg_episode_length'].append(len(episode['states']))
                    self.stats['win_rate'].append(1.0 if episode['main_reward'] > 0 else 0.0)
                    
                    # Update opponent win rate tracking for prioritized sampling
                    opponent_checkpoint = episode.get('opponent_checkpoint')
                    if opponent_checkpoint and episode.get('opponent_type') == 'past_model':
                        won = 1.0 if episode['main_reward'] > 0 else 0.0
                        if opponent_checkpoint not in self._opponent_game_counts:
                            self._opponent_game_counts[opponent_checkpoint] = 0
                            self.opponent_win_rates[opponent_checkpoint] = 0.5  # Initial estimate
                        
                        # Update running average
                        count = self._opponent_game_counts[opponent_checkpoint]
                        old_win_rate = self.opponent_win_rates[opponent_checkpoint]
                        # Exponential moving average with decay
                        alpha = 0.1  # Learning rate for win rate estimate
                        self.opponent_win_rates[opponent_checkpoint] = (1 - alpha) * old_win_rate + alpha * won
                        self._opponent_game_counts[opponent_checkpoint] = count + 1
                    
                    # Update action counts
                    for act_idx in episode['actions']:
                        action_counts[ACTION_NAMES[act_idx.item()]] += 1
                    
                    # Track average value estimate
                    if len(episode['values']) > 0:
                        avg_value = sum([v.item() for v in episode['values']]) / len(episode['values'])
                        self.stats['avg_value_estimate'].append(avg_value)
                    
                    # Compute advantages and returns using GAE
                    rewards = torch.tensor([episode['main_reward']], dtype=torch.float32)
                    
                    # For poker, we get reward at end of hand, so we need to propagate it back
                    num_steps = len(episode['states'])
                    step_rewards = torch.zeros(num_steps)
                    
                    # Normalize reward to be in range [-1, 1]
                    # episode['main_reward'] is already normalized by starting_chips in collect_episode
                    normalized_reward = episode['main_reward']
                    # Relax clipping to allow for > 1.0 wins in deep stack scenarios, but cap extreme values
                    normalized_reward = max(-2.0, min(5.0, normalized_reward))
                    
                    # === REWARD SHAPING for better learning signal ===
                    # Key principle: Shaping must be STRONG enough to overwhelm poker variance
                    
                    # 1. Per-step action appropriateness rewards (based on hand strength)
                    # These provide dense signal about whether actions matched hand quality
                    per_step_shaping = episode.get('step_rewards', [])
                    if per_step_shaping:
                        for idx, shaping_reward in enumerate(per_step_shaping):
                            if idx < num_steps:
                                step_rewards[idx] += shaping_reward
                    
                    # 2. Episode-level shaping (STRONGLY ENHANCED)
                    episode_shaping = 0.0
                    
                    hand_strengths = episode.get('hand_strengths', [])
                    action_types = episode.get('action_types', [])
                    
                    # Detect problematic all-in behavior with detailed tracking
                    all_in_with_weak = False
                    all_in_with_trash = False
                    all_in_count = 0
                    fold_count = 0
                    
                    if hand_strengths and action_types:
                        for hs, act in zip(hand_strengths, action_types):
                            if act == 'all_in':
                                all_in_count += 1
                                if hs < 0.45:  # Weak threshold raised
                                    all_in_with_weak = True
                                if hs < 0.30:  # Trash threshold raised
                                    all_in_with_trash = True
                            if act == 'fold':
                                fold_count += 1
                    
                    # === KEY ANTI-ALL-IN PENALTIES ===
                    # These MUST be strong enough that the model learns:
                    # "All-in with weak hands is ALWAYS bad, even if I got lucky this time"
                    
                    if all_in_with_trash:
                        if normalized_reward > 0:
                            # WON by luck with trash all-in
                            # CRITICAL: Must HEAVILY penalize even winning to prevent exploitation learning
                            episode_shaping -= 0.40  # "Lucky win" - still terrible play
                        else:
                            # LOST with trash all-in
                            # Expected result, but reinforce that this was bad
                            episode_shaping -= 0.30  # Strong additional penalty
                    elif all_in_with_weak:
                        if normalized_reward > 0:
                            # Won with weak all-in - reduce reward significantly
                            episode_shaping -= 0.25  # Penalize lucky wins
                        else:
                            # Lost with weak all-in - expected, but reinforce
                            episode_shaping -= 0.15
                    
                    # === FOLD PENALTY TO DISCOURAGE OVER-FOLDING ===
                    # Penalize folding to encourage playing and learning from outcomes
                    if fold_count > 0 and hand_strengths:
                        first_hand_strength = hand_strengths[0]
                        if first_hand_strength < 0.25:
                            # Folded with true trash - acceptable (no penalty)
                            pass
                        elif first_hand_strength < 0.35:
                            # Folded with weak hand - small penalty to encourage playing
                            episode_shaping -= 0.03
                        elif first_hand_strength < 0.50:
                            # Folded with playable hand - bigger penalty
                            episode_shaping -= 0.08
                        else:
                            # Folded with decent+ hand - severe penalty
                            episode_shaping -= 0.15
                    
                    # === HAND STRENGTH CORRELATION ===
                    if hand_strengths:
                        avg_hand_strength = sum(hand_strengths) / len(hand_strengths)
                        
                        if normalized_reward > 0:  # Won the hand
                            if avg_hand_strength > 0.70:
                                # Won with strong hand - expected and good value extraction
                                episode_shaping += 0.10
                            elif avg_hand_strength > 0.55:
                                # Won with medium hand - good play
                                episode_shaping += 0.05
                            elif avg_hand_strength < 0.35:
                                # Won with weak hand (bluff worked)
                                # Small bonus only if we didn't do trash all-in
                                if not all_in_with_trash and not all_in_with_weak:
                                    episode_shaping += 0.03
                        else:  # Lost the hand
                            if avg_hand_strength > 0.70:
                                # Lost with strong hand (cooler/bad luck)
                                # Reduce penalty - this wasn't misplay
                                episode_shaping += 0.05
                            elif avg_hand_strength < 0.35:
                                # Lost with weak hand
                                # If we folded, that's good. If we played, that's bad.
                                if fold_count == 0:
                                    # Played weak hand without folding - bad
                                    episode_shaping -= 0.08
                    
                    # Bonus for winning pots uncontested (good aggression that induces folds)
                    # But NOT if we used trash all-in to do it
                    if episode.get('won_uncontested', False) and normalized_reward > 0:
                        if not all_in_with_trash and not all_in_with_weak:
                            episode_shaping += 0.06  # Reward controlled aggression
                        # No bonus for all-in forcing fold - that's not skillful
                    
                    # Position-aware adjustment (playing OOP is harder)
                    if episode.get('is_out_of_position', False) and normalized_reward < 0:
                        normalized_reward *= 0.95
                    
                    # Apply episode-level shaping to final reward
                    normalized_reward += episode_shaping
                    
                    # Main reward at the end
                    step_rewards[-1] += normalized_reward
                    
                    # Compute GAE
                    advantages, returns = self.ppo_trainer.compute_gae(
                        step_rewards,
                        episode['values'],
                        episode['dones'],
                        torch.tensor(0.0)  # No next value (terminal)
                    )
                    
                    # Store episode data
                    all_states.append(episode['states'])
                    all_actions.append(episode['actions'])
                    all_log_probs.append(episode['log_probs'])
                    all_values.append(episode['values'])
                    all_advantages.append(advantages)
                    all_returns.append(returns)
                    all_legal_masks.append(episode['legal_actions_masks'])
                    
                    # Store hand strengths for auxiliary loss
                    if episode.get('hand_strengths'):
                        hs_tensor = torch.tensor(episode['hand_strengths'], dtype=torch.float32)
                        all_hand_strengths.append(hs_tensor)
                    
                    episode_rewards.append(episode['main_reward'])
                    episode_lengths.append(num_steps)
                    
                    if self.log_level >= 2 and (i + 1) % 10 == 0:
                        avg_reward = sum(episode_rewards[-10:]) / len(episode_rewards[-10:])
                        print(f"  Episode {i+1}/{num_episodes} - Avg Reward (last 10): {avg_reward:.3f}")
                        
                except Exception as e:
                    print(f"⚠️  Error in episode collection: {e}")
                    traceback.print_exc()
        
        if not all_states:
            if self.log_level >= 1:
                print("⚠️  No successful episodes collected!")
            return {}
        
        # Concatenate all episode data
        # NOTE: Keep on CPU for offloading
        states = torch.cat(all_states, dim=0)
        actions = torch.cat(all_actions, dim=0)
        old_log_probs = torch.cat(all_log_probs, dim=0)
        old_values = torch.cat(all_values, dim=0)
        advantages = torch.cat(all_advantages, dim=0)
        returns = torch.cat(all_returns, dim=0)
        legal_masks = torch.cat(all_legal_masks, dim=0)
        
        # Concatenate hand strengths for auxiliary loss
        hand_strengths = None
        if all_hand_strengths:
            hand_strengths = torch.cat(all_hand_strengths, dim=0)
        
        # Per-batch reward normalization to handle poker variance
        # This helps stabilize learning when reward distributions vary significantly
        if len(episode_rewards) > 1:
            reward_mean = sum(episode_rewards) / len(episode_rewards)
            reward_std = (sum((r - reward_mean) ** 2 for r in episode_rewards) / len(episode_rewards)) ** 0.5
            reward_std = max(reward_std, 0.1)  # Prevent division by very small values
            
            # Log reward statistics for monitoring
            if self.log_level >= 2:
                print(f"  Reward stats: mean={reward_mean:.4f}, std={reward_std:.4f}")
        
        # === ACTION RATIO GUARDRAILS ===
        # Apply global adjustment to advantages based on action ratios
        # This nudges the model towards more balanced play
        total_actions_count = sum(action_counts.values())
        if total_actions_count > 0:
            # Calculate action rates
            grouped = {'fold': 0, 'check': 0, 'call': 0, 'bet': 0, 'raise': 0, 'all_in': 0}
            for name, count in action_counts.items():
                if name in grouped:
                    grouped[name] += count
                elif name.startswith('bet_'):
                    grouped['bet'] += count
                elif name.startswith('raise_'):
                    grouped['raise'] += count
            
            fold_rate = grouped['fold'] / total_actions_count
            check_rate = grouped['check'] / total_actions_count
            call_rate = grouped['call'] / total_actions_count
            
            # Penalize high fold rate by reducing advantages when fold rate > 40%
            # This creates global pressure to explore non-fold actions
            if fold_rate > 0.40:
                # Scale penalty by how much over threshold
                excess_fold_rate = fold_rate - 0.40
                fold_penalty = excess_fold_rate * 0.5  # Max ~0.3 penalty at 100% fold
                advantages = advantages - fold_penalty
                if self.log_level >= 2:
                    print(f"  ⚠️ Applied fold rate penalty: -{fold_penalty:.4f} (fold rate: {fold_rate:.1%})")
            
            # Bonus for reasonable check+call rate (staying in hands)
            passive_rate = check_rate + call_rate
            if passive_rate > 0.25:
                # Small bonus for healthy passive play rate
                passive_bonus = min(0.1, (passive_rate - 0.25) * 0.3)
                advantages = advantages + passive_bonus
                if self.log_level >= 2:
                    print(f"  ✓ Applied passive play bonus: +{passive_bonus:.4f} (check+call rate: {passive_rate:.1%})")
        
        if self.log_level >= 1:
            print(f"\nCollected {len(states)} timesteps from {len(episode_rewards)} episodes")
            print(f"  Avg Episode Reward: {sum(episode_rewards)/len(episode_rewards):.3f}")
            print(f"  Avg Episode Length: {sum(episode_lengths)/len(episode_lengths):.1f}")
            print(f"  Win Rate: {sum(1 for r in episode_rewards if r > 0) / len(episode_rewards):.2%}")
            
            # Print action distribution
            total_actions = sum(action_counts.values())
            if total_actions > 0:
                print("\n  Action Distribution:")
                # Group bets and raises
                grouped_counts = {'fold': 0, 'check': 0, 'call': 0, 'bet': 0, 'raise': 0, 'all_in': 0}
                for name, count in action_counts.items():
                    if name in grouped_counts:
                        grouped_counts[name] += count
                    elif name.startswith('bet_'):
                        grouped_counts['bet'] += count
                    elif name.startswith('raise_'):
                        grouped_counts['raise'] += count
                
                for act in ['fold', 'check', 'call', 'bet', 'raise', 'all_in']:
                    count = grouped_counts[act]
                    pct = count / total_actions
                    print(f"    {act.ljust(8)}: {pct:.1%} ({count})")
                
                # Alert if all-in rate is too high (should be < 10% for healthy play)
                all_in_rate = grouped_counts['all_in'] / total_actions
                fold_rate = grouped_counts['fold'] / total_actions
                if all_in_rate > 0.15:
                    print(f"    ⚠️  ALL-IN RATE TOO HIGH: {all_in_rate:.1%} (target: <15%)")
                elif all_in_rate > 0.10:
                    print(f"    ⚡ All-in rate elevated: {all_in_rate:.1%} (target: <10%)")
                else:
                    print(f"    ✓ All-in rate healthy: {all_in_rate:.1%}")
                
                if fold_rate > 0.50:
                    print(f"    ⚠️  FOLD RATE TOO HIGH: {fold_rate:.1%} (target: <40%)")
                elif fold_rate > 0.40:
                    print(f"    ⚡ Fold rate elevated: {fold_rate:.1%} (target: <40%)")
                else:
                    print(f"    ✓ Fold rate healthy: {fold_rate:.1%}")
                
                # Check/call rate should be reasonable  
                passive_rate = (grouped_counts['check'] + grouped_counts['call']) / total_actions
                if passive_rate < 0.20:
                    print(f"    ⚠️  CHECK+CALL RATE TOO LOW: {passive_rate:.1%} (target: >25%)")
                elif passive_rate < 0.25:
                    print(f"    ⚡ Check+call rate low: {passive_rate:.1%} (target: >25%)")
                else:
                    print(f"    ✓ Check+call rate healthy: {passive_rate:.1%}")

            print(f"\nRunning PPO update...")
        
        # PPO update (with auxiliary hand strength prediction loss)
        ppo_stats = self.ppo_trainer.update(
            states, actions, old_log_probs, old_values, advantages, returns, legal_masks,
            hand_strengths=hand_strengths,
            verbose=(self.log_level >= 2)
        )
        
        # Entropy coefficient is now handled by adaptive entropy scheduler if enabled
        # Only manually anneal if adaptive entropy is disabled
        if not hasattr(self.ppo_trainer, 'use_adaptive_entropy') or not self.ppo_trainer.use_adaptive_entropy:
            self.ppo_trainer.entropy_coef = max(0.01, self.ppo_trainer.entropy_coef * 0.9999)
        
        # Log entropy coefficient
        if self.writer:
            entropy_coef = (
                self.ppo_trainer.entropy_scheduler.coef 
                if hasattr(self.ppo_trainer, 'entropy_scheduler') and self.ppo_trainer.entropy_scheduler
                else self.ppo_trainer.entropy_coef
            )
            self.writer.add_scalar('PPO/EntropyCoef', entropy_coef, self.stats['iteration'])
        
        # Plateau detection and response
        if self.detect_plateau():
            self._plateau_counter += 1
            if self._plateau_counter >= 50:  # Respond after 50 consecutive plateau iterations
                self.respond_to_plateau()
                self._plateau_counter = 0
        else:
            self._plateau_counter = 0
        
        # Track explained variance for convergence monitoring
        if 'explained_variance' in ppo_stats:
            self.stats['explained_variance_history'].append(ppo_stats['explained_variance'])
        
        if self.log_level >= 1:
            print(f"  Policy Loss: {ppo_stats['policy_loss']:.4f}")
            print(f"  Value Loss: {ppo_stats['value_loss']:.4f}")
            print(f"  Hand Strength Loss: {ppo_stats.get('hand_strength_loss', 0.0):.4f}")
            print(f"  Entropy: {ppo_stats['entropy']:.4f}")
            print(f"  KL Divergence: {ppo_stats['kl_divergence']:.4f}")
            print(f"  Grad Norm: {ppo_stats.get('grad_norm', 0.0):.4f}")
            
            if 'value_mean' in ppo_stats and 'return_mean' in ppo_stats:
                print(f"  Value Mean: {ppo_stats['value_mean']:.4f} vs Return Mean: {ppo_stats['return_mean']:.4f}")
                
            print(f"  Explained Variance: {ppo_stats['explained_variance']:.4f}")
            
            # Convergence indicator
            if len(self.stats['explained_variance_history']) >= 10:
                recent_ev = list(self.stats['explained_variance_history'])[-10:]
                avg_ev = sum(recent_ev) / len(recent_ev)
                if avg_ev > 0.8:
                    print(f"  ✓ Convergence indicator: GOOD (EV={avg_ev:.3f})")
                elif avg_ev > 0.5:
                    print(f"  ⚡ Convergence indicator: FAIR (EV={avg_ev:.3f})")
                else:
                    print(f"  ⚠️  Convergence indicator: POOR (EV={avg_ev:.3f})")
        
        # Log to TensorBoard
        if self.writer:
            iteration = self.stats['iteration']
            self.writer.add_scalar('Episode/AvgReward', sum(episode_rewards)/len(episode_rewards), iteration)
            self.writer.add_scalar('Episode/AvgLength', sum(episode_lengths)/len(episode_lengths), iteration)
            self.writer.add_scalar('Episode/WinRate', sum(1 for r in episode_rewards if r > 0) / len(episode_rewards), iteration)
            
            for key, value in ppo_stats.items():
                self.writer.add_scalar(f'PPO/{key}', value, iteration)
            
            # Log convergence metrics
            if self.stats['avg_value_estimate']:
                self.writer.add_scalar('Convergence/AvgValueEstimate', 
                                      sum(self.stats['avg_value_estimate']) / len(self.stats['avg_value_estimate']), 
                                      iteration)
            
            # Log action distribution metrics (critical for monitoring hand selection learning)
            total_actions = sum(action_counts.values())
            if total_actions > 0:
                grouped_counts = {'fold': 0, 'check': 0, 'call': 0, 'bet': 0, 'raise': 0, 'all_in': 0}
                for name, count in action_counts.items():
                    if name in grouped_counts:
                        grouped_counts[name] += count
                    elif name.startswith('bet_'):
                        grouped_counts['bet'] += count
                    elif name.startswith('raise_'):
                        grouped_counts['raise'] += count
                
                for act, count in grouped_counts.items():
                    self.writer.add_scalar(f'Actions/{act}_rate', count / total_actions, iteration)
                
                # Key metrics for hand selection health
                self.writer.add_scalar('HandSelection/AllInRate', grouped_counts['all_in'] / total_actions, iteration)
                self.writer.add_scalar('HandSelection/FoldRate', grouped_counts['fold'] / total_actions, iteration)
                self.writer.add_scalar('HandSelection/AggressionRate', 
                                      (grouped_counts['bet'] + grouped_counts['raise'] + grouped_counts['all_in']) / total_actions, 
                                      iteration)
        
        self.stats['iteration'] += 1
        
        return ppo_stats
    
    def save_checkpoint(self, name: str = "latest"):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f"poker_rl_{name}.pt"
        
        self.ppo_trainer.save_checkpoint(
            str(checkpoint_path),
            epoch=self.stats['iteration'],
            input_dim=self.model.input_dim,
            total_episodes=self.stats['total_episodes'],
            total_timesteps=self.stats['total_timesteps']
        )
        
        # Only print every 10th checkpoint in minimal mode
        if self.log_level >= 1 or (self.stats['iteration'] % 100 == 0):
            print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Add to opponent pool
        if len(self.opponent_pool) >= self.max_opponent_pool_size:
            # Remove oldest
            self.opponent_pool.pop(0)
        
        self.opponent_pool.append(checkpoint_path)
    
    def _load_opponent_model(self, checkpoint_path: Path) -> PokerActorCritic:
        """
        Load a past checkpoint for use as opponent in self-play.
        
        Caches loaded models to avoid repeated disk reads.
        """
        path_str = str(checkpoint_path)
        
        # Check cache first
        if path_str in self._opponent_model_cache:
            return self._opponent_model_cache[path_str]
        
        # Load from disk
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Create model with same architecture as current model
            model = create_actor_critic(
                input_dim=self.model.input_dim,
                hidden_dim=self.model.hidden_dim,
                num_heads=self.model.num_heads,
                num_layers=len(self.model.transformer.layers),
                dropout=0.1,  # Default dropout
                gradient_checkpointing=False  # Disable for inference
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            # Manage cache size
            if len(self._opponent_model_cache) >= self._max_opponent_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._opponent_model_cache))
                del self._opponent_model_cache[oldest_key]
            
            self._opponent_model_cache[path_str] = model
            return model
            
        except Exception as e:
            if self.log_level >= 1:
                print(f"⚠️  Failed to load opponent model {checkpoint_path}: {e}")
            return None
    
    def _sample_opponent_checkpoint(self) -> Path:
        """
        Sample an opponent checkpoint from the pool.
        
        Uses prioritized sampling based on win rates when available,
        otherwise falls back to uniform random sampling.
        """
        if not self.opponent_pool:
            return None
        
        # If we have win rate data, use prioritized sampling
        if hasattr(self, 'opponent_win_rates') and self.opponent_win_rates:
            # Prioritize opponents we struggle against (lower win rate = higher priority)
            # But also occasionally sample easy opponents to prevent forgetting
            weights = []
            for checkpoint in self.opponent_pool:
                path_str = str(checkpoint)
                if path_str in self.opponent_win_rates:
                    win_rate = self.opponent_win_rates[path_str]
                    # Inverse win rate: harder opponents get higher weight
                    # Add small constant to avoid zero weights
                    weight = 1.0 - win_rate + 0.1
                else:
                    # Unknown opponents get medium weight
                    weight = 0.6
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                # Sample based on weights
                idx = random.choices(range(len(self.opponent_pool)), weights=weights, k=1)[0]
                return self.opponent_pool[idx]
        
        # Fallback to uniform random sampling
        return random.choice(self.opponent_pool)
    
    def detect_plateau(self) -> bool:
        """
        Detect if training has plateaued based on win rate variance.
        
        Returns True if win rate has low variance over recent iterations,
        indicating the model is stuck at a local optimum.
        """
        if len(self.stats['win_rate']) < 100:
            return False
        
        import numpy as np
        recent = list(self.stats['win_rate'])[-100:]
        variance = np.var(recent)
        mean_win_rate = np.mean(recent)
        
        # Plateau detected if:
        # 1. Win rate variance is very low (not improving or declining)
        # 2. Win rate is in a "stuck" range (not dominating, not losing badly)
        is_low_variance = variance < 0.001
        is_stuck_range = 0.3 < mean_win_rate < 0.7
        
        return is_low_variance and is_stuck_range
    
    def respond_to_plateau(self):
        """
        Respond to detected plateau by resetting exploration parameters.
        
        Actions taken:
        1. Reset entropy coefficient to encourage more exploration
        2. Clear opponent win rates to re-evaluate all opponents
        3. Log the plateau response
        """
        # Don't respond too frequently
        min_iterations_between_responses = 100
        if self.stats['iteration'] - self._last_plateau_response_iter < min_iterations_between_responses:
            return
        
        self._last_plateau_response_iter = self.stats['iteration']
        
        # 1. Reset entropy coefficient to initial value for more exploration
        if self._initial_entropy_coef is not None:
            self.ppo_trainer.entropy_coef = self._initial_entropy_coef
            if self.log_level >= 1:
                print(f"🔄 Plateau detected! Reset entropy to {self._initial_entropy_coef}")
        
        # 2. Clear opponent win rates to re-evaluate fresh
        self.opponent_win_rates.clear()
        self._opponent_game_counts.clear()
        
        # 3. Log to TensorBoard
        if self.writer:
            self.writer.add_scalar('Plateau/ResponseTriggered', 1.0, self.stats['iteration'])
        
        if self.log_level >= 1:
            print(f"🔄 Plateau response at iteration {self.stats['iteration']}")
            print(f"   - Entropy coefficient reset")
            print(f"   - Opponent win rates cleared")
    
    def _call_api(self, history: List[Dict]) -> Dict:
        """Call poker API server"""
        payload = {
            'config': {
                **self.game_config,
                'seed': random.randint(0, 1000000)
            },
            'history': history
        }
        
        try:
            # Use direct C++ binding
            # orjson.dumps returns bytes, decode to str if binding requires str
            payload_bytes = json.dumps(payload)
            payload_str = payload_bytes.decode('utf-8') if isinstance(payload_bytes, bytes) else payload_bytes
            
            response_str = poker_api_binding.process_request(payload_str)
            return json.loads(response_str)
        except Exception as e:
            error_msg = f"Binding error: {e}"
            print(f"⚠️  {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def _get_legal_actions(self, game_state: Dict) -> List[str]:
        """Get legal actions from game state"""
        action_constraints = game_state.get('actionConstraints', {})
        return action_constraints.get('legalActions', [])
    
    def _create_legal_actions_mask(self, legal_actions: List[str]) -> torch.Tensor:
        """Create legal actions mask tensor"""
        return create_legal_actions_mask(legal_actions, self.device)
    
    def _idx_to_action_label(self, idx: int) -> str:
        """Convert action index to label"""
        return ACTION_NAMES[idx]
    
    def _convert_action(self, action_label: str, state: Dict) -> Tuple[str, int]:
        """Convert action label to (type, amount)"""
        return convert_action_label(action_label, state)
    
    def _random_action(self, legal_actions: List[str], state: Dict) -> Tuple[str, int, str]:
        """Generate random action"""
        action = random.choice(legal_actions)
        
        if action == 'bet':
            size_fractions = [0.5, 0.75, 1.0]
            size_fraction = random.choice(size_fractions)
            action_label = f'bet_{int(size_fraction*100)}%'
        elif action == 'raise':
            size_fractions = [0.5, 0.75, 1.0]
            size_fraction = random.choice(size_fractions)
            action_label = f'raise_{int(size_fraction*100)}%'
        else:
            action_label = action
            
        action_type, amount = self._convert_action(action_label, state)
        return action_type, amount, action_label
    
    def _compute_action_shaping_reward(
        self,
        action_type: str,
        hand_strength: float,
        state: Dict,
        facing_aggression: bool,
        step_in_hand: int = 0,
        total_steps_estimate: int = 4
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
        - Temporal consistency: early decisions affect later rewards
        
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
        starting_chips = state.get('starting_chips', 1000)
        
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


# =============================================================================
# Main Training Function
# =============================================================================

def main() -> int:
    """Main RL training function"""
    parser = argparse.ArgumentParser(
        description="Train a poker AI using reinforcement learning (PPO)"
    )
    
    # Training parameters - optimized for convergence
    parser.add_argument('--iterations', type=int, default=5000,
                       help='Number of training iterations (default: 5000)')
    parser.add_argument('--episodes-per-iter', type=int, default=100,
                       help='Episodes per iteration (default: 100)')
    parser.add_argument('--ppo-epochs', type=int, default=15,
                       help='PPO epochs per update (default: 15)')
    parser.add_argument('--mini-batch-size', type=int, default=128,
                       help='Mini-batch size for PPO (default: 128)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                       help='Gradient accumulation steps for larger effective batch (default: 4)')
    
    # Game configuration
    parser.add_argument('--num-players', type=int, default=2,
                       help='Number of players (default: 2 for heads-up)')
    parser.add_argument('--small-blind', type=int, default=10,
                       help='Small blind (default: 10)')
    parser.add_argument('--big-blind', type=int, default=20,
                       help='Big blind (default: 20)')
    parser.add_argument('--starting-chips', type=int, default=1000,
                       help='Starting chips (default: 1000)')
    
    # PPO hyperparameters - optimized for stability and convergence
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda (default: 0.95)')
    parser.add_argument('--clip-epsilon', type=float, default=0.2,
                       help='PPO clip epsilon (default: 0.2)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                       help='Entropy coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                       help='Value loss coefficient (default: 0.5)')
    parser.add_argument('--hand-strength-loss-coef', type=float, default=0.10,
                       help='Hand strength prediction loss coefficient for auxiliary task (default: 0.10)')
    
    # Advanced PPO features
    parser.add_argument('--lr-warmup-steps', type=int, default=100,
                       help='Learning rate warmup steps (default: 100)')
    parser.add_argument('--advantage-clip', type=float, default=10.0,
                       help='Clip extreme advantages (default: 10.0)')
    parser.add_argument('--no-popart', action='store_true',
                       help='Disable PopArt value normalization')
    parser.add_argument('--no-adaptive-entropy', action='store_true',
                       help='Disable adaptive entropy scheduling')
    
    # Model architecture
    parser.add_argument('--hidden-dim', type=int, default=512,
                       help='Hidden dimension (default: 512)')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads (default: 8)')
    parser.add_argument('--num-layers', type=int, default=4,
                       help='Number of transformer layers (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    parser.add_argument('--no-gradient-checkpointing', action='store_true',
                       help='Disable gradient checkpointing (enabled by default for large models)')
    
    # I/O
    parser.add_argument('--output-dir', type=str, 
                       default=DEFAULT_MODELS_DIR,
                       help='Output directory for models')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save checkpoint every N iterations (default: 100)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint to resume from')
    
    # TensorBoard
    parser.add_argument('--tensorboard-dir', type=str, default=f'/tmp/pokersim/tensorboard_v{MODEL_VERSION}',
                       help='Directory for TensorBoard logs')
    parser.add_argument('--no-tensorboard', action='store_true',
                       help='Disable TensorBoard logging')
    
    parser.add_argument('--eval-interval', type=int, default=20,
                       help='Evaluate against random every N iterations (default: 20)')
    parser.add_argument('--eval-episodes', type=int, default=50,
                       help='Number of episodes for evaluation (default: 50)')
    
    # Monte Carlo multi-runout settings
    parser.add_argument('--num-runouts', type=int, default=0,
                       help='Number of Monte Carlo runouts per decision for regret calculation (0=disabled, 50=recommended)')
    parser.add_argument('--regret-weight', type=float, default=0.5,
                       help='Weight for regret-based reward adjustment (default: 0.5)')
    
    # Other
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if LOG_LEVEL >= 1:
        print("✓ Using fast C++ bindings for PokerEngine")
    
    # Setup Accelerator
    # gradient_accumulation_steps is handled manually in PPO by mini_batch_size vs full batch
    accelerator = Accelerator()
    device = accelerator.device
    
    if LOG_LEVEL >= 1:
        print(f"✓ Using device: {device}")
        if accelerator.mixed_precision == 'fp16':
            print("✓ Using FP16 mixed precision")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = Path(args.tensorboard_dir) if not args.no_tensorboard else None
    
    # Create model
    if LOG_LEVEL >= 1:
        print("\n📦 Creating Actor-Critic model...")
    
    # Get input dimension from encoder
    temp_encoder = RLStateEncoder()
    input_dim = temp_encoder.get_feature_dim()
    
    model = create_actor_critic(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gradient_checkpointing=not args.no_gradient_checkpointing
    )
    # accelerator.prepare will handle device placement
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if LOG_LEVEL >= 1:
        print(f"✓ Model created: {num_params:,} parameters")
    
    # Create PPO trainer with enhanced features
    if LOG_LEVEL >= 1:
        print("\n🎯 Creating PPO trainer...")
    ppo_trainer = PPOTrainer(
        model=model,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        hand_strength_loss_coef=args.hand_strength_loss_coef,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_schedule_steps=args.iterations,  # Align LR schedule with total training iterations
        lr_warmup_steps=args.lr_warmup_steps,
        use_popart=not args.no_popart,
        use_adaptive_entropy=not args.no_adaptive_entropy,
        advantage_clip=args.advantage_clip,
        device=device,
        accelerator=accelerator
    )
    if LOG_LEVEL >= 1:
        print(f"✓ PPO trainer created")
        print(f"  Learning rate: {args.learning_rate} (with {args.lr_warmup_steps} warmup steps)")
        print(f"  PPO epochs: {args.ppo_epochs}")
        print(f"  Episodes per iteration: {args.episodes_per_iter}")
        print(f"  PopArt value normalization: {'enabled' if not args.no_popart else 'disabled'}")
        print(f"  Adaptive entropy scheduling: {'enabled' if not args.no_adaptive_entropy else 'disabled'}")
    
    # Load checkpoint if provided
    start_iteration = 0
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"\n📂 Loading checkpoint from: {checkpoint_path}")
            checkpoint = ppo_trainer.load_checkpoint(str(checkpoint_path))
            start_iteration = checkpoint.get('epoch', 0)
            print(f"✓ Resumed from iteration {start_iteration}")
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}, starting from scratch")
    
    # Game configuration
    game_config = {
        'num_players': args.num_players,
        'smallBlind': args.small_blind,
        'bigBlind': args.big_blind,
        'startingChips': args.starting_chips,
        'minPlayers': args.num_players,
        'maxPlayers': args.num_players
    }
    
    # Create training session
    if LOG_LEVEL >= 1:
        print("\n🚀 Starting RL training session...")
        print(f"  Total iterations: {args.iterations}")
        print(f"  Episodes per iteration: {args.episodes_per_iter}")
        print(f"  Estimated total episodes: {args.iterations * args.episodes_per_iter:,}")
        if args.num_runouts > 0:
            print(f"  Monte Carlo runouts per decision: {args.num_runouts}")
            print(f"  Regret weight: {args.regret_weight}")
    
    session = RLTrainingSession(
        model=model,
        ppo_trainer=ppo_trainer,
        game_config=game_config,
        device=device,
        output_dir=output_dir,
        tensorboard_dir=tensorboard_dir,
        log_level=LOG_LEVEL,
        num_runouts=args.num_runouts,
        regret_weight=args.regret_weight
    )
    
    # Store initial entropy coefficient for plateau response
    session._initial_entropy_coef = args.entropy_coef
    
    # Test API connection before starting
    if LOG_LEVEL >= 1:
        print(f"\n🔗 API Configuration:")
        print(f"   Using internal C++ binding (no server required)")
        
        # Test binding
        try:
            test_payload = {
                'config': {'seed': 123},
                'history': []
            }
            
            payload_bytes = json.dumps(test_payload)
            payload_str = payload_bytes.decode('utf-8') if isinstance(payload_bytes, bytes) else payload_bytes
            
            test_response = json.loads(poker_api_binding.process_request(payload_str))
            if test_response.get('success'):
                    print(f"✓ Binding test successful\n")
            else:
                    print(f"⚠️  Binding test failed: {test_response.get('error')}\n")
        except Exception as e:
            print(f"❌ Binding test raised exception: {e}\n")
    
    # Set starting iteration if resuming
    if start_iteration > 0:
        session.stats['iteration'] = start_iteration
    
    # Training loop
    for iteration in range(start_iteration, args.iterations):
        # Train one iteration
        stats = session.train_iteration(
            num_episodes=args.episodes_per_iter,
            verbose=args.verbose
        )
        
        # Show minimal progress every 10 iterations
        if LOG_LEVEL == 0 and (iteration + 1) % 10 == 0:
            avg_reward = sum(list(session.stats['avg_reward'])[-10:]) / min(10, len(list(session.stats['avg_reward'])))
            win_rate = sum(list(session.stats['win_rate'])[-10:]) / min(10, len(list(session.stats['win_rate'])))
            print(f"Iter {iteration+1}/{args.iterations} - Avg Reward: {avg_reward:.3f}, Win Rate: {win_rate:.2%}")
        
        # Save checkpoint periodically or on first iteration
        if (iteration + 1) % args.save_interval == 0 or iteration == 0:
            session.save_checkpoint(name=f"iter_{iteration+1}")
            session.save_checkpoint(name="latest")
            
            # Save baseline after first iteration
            if iteration == 0:
                session.save_checkpoint(name="baseline")
            
        # Evaluate periodically
        if (iteration + 1) % args.eval_interval == 0:
            session.evaluate_vs_random(num_episodes=args.eval_episodes)
            session.evaluate_vs_heuristic(num_episodes=args.eval_episodes)
    
    # Save final model
    session.save_checkpoint(name="final")
    
    print("\n✓ Training complete!")
    print(f"  Total episodes: {session.stats['total_episodes']}")
    print(f"  Total timesteps: {session.stats['total_timesteps']}")
    if session.stats['avg_reward']:
        print(f"  Final avg reward: {sum(list(session.stats['avg_reward']))/len(list(session.stats['avg_reward'])):.3f}")
    if session.stats['win_rate']:
        print(f"  Final win rate: {sum(list(session.stats['win_rate']))/len(list(session.stats['win_rate'])):.2%}")
    
    if session.writer:
        session.writer.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
