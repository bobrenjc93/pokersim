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
from rl_state_encoder import RLStateEncoder
from rl_model import PokerActorCritic, create_actor_critic
from ppo import PPOTrainer
from model_agent import (
    ModelAgent, 
    RandomAgent,
    HeuristicAgent,
    ACTION_MAP, 
    ACTION_NAMES, 
    BET_SIZE_MAP,
    convert_action_label,
    create_legal_actions_mask,
    extract_state
)

# Import model version and log level from config
from config import MODEL_VERSION, LOG_LEVEL, DEFAULT_MODELS_DIR

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
        log_level: int = 0
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
        """
        self.model = model
        self.ppo_trainer = ppo_trainer
        self.game_config = game_config
        self.device = device
        self.output_dir = output_dir
        self.tensorboard_dir = tensorboard_dir
        self.log_level = log_level
        
        # Opponent pool for self-play (stores past model versions)
        self.opponent_pool: List[Path] = []
        self.max_opponent_pool_size = 10  # Increased from 5 for more diversity
        
        # Cache for loaded opponent models to avoid reloading
        self._opponent_model_cache: Dict[str, PokerActorCritic] = {}
        self._max_opponent_cache_size = 5  # Keep at most 5 models in memory
        
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
        for i in range(1, num_players):
            player_id = f'p{i}'
            
            # Opponent selection probabilities:
            # - Past checkpoint: 30% (when pool available) - diverse self-play
            # - Heuristic: 20% - strategic baseline  
            # - Current model: 15% - immediate self-play
            # - Random: 35% - exploration baseline
            roll = random.random()
            
            if use_opponent_pool and self.opponent_pool and roll < 0.30:
                # Use past checkpoint (30% of time when pool available)
                checkpoint_path = random.choice(self.opponent_pool)
                opponent_model = self._load_opponent_model(checkpoint_path)
                if opponent_model is not None:
                    agents.append({
                        'id': player_id,
                        'type': 'past_model',
                        'model': opponent_model,
                        'encoder': RLStateEncoder()
                    })
                else:
                    # Fallback to random if loading failed
                    agents.append({'id': player_id, 'type': 'random'})
            elif use_opponent_pool and roll < 0.50:
                # Use heuristic agent (20% of time)
                agents.append({
                    'id': player_id,
                    'type': 'heuristic'
                })
            elif use_opponent_pool and roll < 0.65:
                # Use current model as opponent (15% of time)
                agents.append({
                    'id': player_id,
                    'type': 'model',
                    'encoder': RLStateEncoder()
                })
            else:
                # Use random agent (35% of time, or 100% if not using pool)
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
                
            elif current_agent['type'] == 'heuristic':
                # Use HeuristicAgent for action selection
                heuristic = HeuristicAgent(current_player_id, "Heuristic")
                action_type, amount, action_label = heuristic.select_action(state_dict, legal_actions)
                
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
        
        # Mark last state as done
        if episode['dones']:
            episode['dones'][-1] = 1
        
        # Normalize rewards (divide by starting chips for scale)
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
                    
                    # Add intermediate shaping rewards for better learning signal
                    # Small reward for staying in the hand (engagement)
                    # REMOVED: step_rewards[step_idx] = 0.01 - we want to encourage folding bad hands
                    if num_steps > 1:
                        for step_idx in range(num_steps - 1):
                             step_rewards[step_idx] = 0.0
                    
                    # Main reward at the end
                    step_rewards[-1] = normalized_reward
                    
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

            print(f"\nRunning PPO update...")
        
        # PPO update
        ppo_stats = self.ppo_trainer.update(
            states, actions, old_log_probs, old_values, advantages, returns, legal_masks,
            verbose=(self.log_level >= 2)
        )
        
        # Anneal entropy coefficient (slower decay to 0.005 minimum for better exploration)
        # Changed from 0.999 decay to 0.9995 and minimum from 0.001 to 0.005
        self.ppo_trainer.entropy_coef = max(0.005, self.ppo_trainer.entropy_coef * 0.9995)
        if self.writer:
             self.writer.add_scalar('PPO/EntropyCoef', self.ppo_trainer.entropy_coef, self.stats['iteration'])
        
        # Track explained variance for convergence monitoring
        if 'explained_variance' in ppo_stats:
            self.stats['explained_variance_history'].append(ppo_stats['explained_variance'])
        
        if self.log_level >= 1:
            print(f"  Policy Loss: {ppo_stats['policy_loss']:.4f}")
            print(f"  Value Loss: {ppo_stats['value_loss']:.4f}")
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
    
    # Create PPO trainer
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
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        lr_schedule_steps=args.iterations,  # Align LR schedule with total training iterations
        device=device,
        accelerator=accelerator
    )
    if LOG_LEVEL >= 1:
        print(f"✓ PPO trainer created")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  PPO epochs: {args.ppo_epochs}")
        print(f"  Episodes per iteration: {args.episodes_per_iter}")
    
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
    
    session = RLTrainingSession(
        model=model,
        ppo_trainer=ppo_trainer,
        game_config=game_config,
        device=device,
        output_dir=output_dir,
        tensorboard_dir=tensorboard_dir,
        log_level=LOG_LEVEL
    )
    
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
