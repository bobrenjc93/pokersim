import sys
import os
import json
import random
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union

try:
    import poker_api_binding
except ImportError:
    print("Error: 'poker_api_binding' not found.")
    sys.exit(1)

from model_agent import convert_action_label

class VecPokerEnv:
    """
    Vectorized Poker Environment.
    Manages multiple C++ Game instances for batched training.
    Handles opponent moves internally to present a single-agent view.
    """
    
    def __init__(
        self,
        num_envs: int,
        config: Dict[str, Any],
        opponent_pool: List[Any],
        device: str = "cpu"
    ):
        self.num_envs = num_envs
        self.config = config
        self.opponent_pool = opponent_pool
        self.device = device
        
        # Create GameConfig
        self.game_config = poker_api_binding.GameConfig()
        self.game_config.smallBlind = config.get('smallBlind', 10)
        self.game_config.bigBlind = config.get('bigBlind', 20)
        self.game_config.startingChips = config.get('startingChips', 1000)
        self.game_config.minPlayers = config.get('minPlayers', 2)
        self.game_config.maxPlayers = config.get('maxPlayers', 2)
        
        self.games = [None] * num_envs
        self.hero_ids = [None] * num_envs # 'p0' or 'p1'
        self.opponents = [None] * num_envs # Agent object
        
        # Initialize games
        for i in range(num_envs):
            self._reset_env(i)
            
    def _reset_env(self, env_idx: int):
        """Reset a single environment"""
        # Set new random seed
        self.game_config.seed = random.randint(0, 1000000)
        game = poker_api_binding.Game(self.game_config)
        
        # Setup players
        # Heads-up: p0 and p1
        game.add_player("p0", "Player 0", self.game_config.startingChips)
        game.add_player("p1", "Player 1", self.game_config.startingChips)
        
        # Assign Hero and Opponent
        if random.random() < 0.5:
            self.hero_ids[env_idx] = "p0"
            opp_id = "p1"
        else:
            self.hero_ids[env_idx] = "p1"
            opp_id = "p0"
            
        # Pick opponent
        if self.opponent_pool:
            self.opponents[env_idx] = random.choice(self.opponent_pool)
            # Reset opponent internal state if needed
            if hasattr(self.opponents[env_idx], 'reset_hand'):
                self.opponents[env_idx].reset_hand()
        else:
            self.opponents[env_idx] = None
        
        game.start_hand()
        self.games[env_idx] = game
        
    def reset(self) -> List[Dict]:
        """
        Reset all environments and return initial states for Hero.
        """
        for i in range(self.num_envs):
            self._reset_env(i)
            
        # Fast-forward to Hero's turn
        return self._fast_forward_all()
        
    def step(self, actions: List[str]) -> Tuple[List[Dict], List[float], List[bool], List[Dict]]:
        """
        Apply Hero actions and advance.
        
        Args:
            actions: List of action labels (e.g., 'fold', 'raise_50%') for each env.
            
        Returns:
            states: New states for Hero
            rewards: Rewards for Hero
            dones: Done flags
            infos: Info dicts
        """
        rewards = [0.0] * self.num_envs
        dones = [False] * self.num_envs
        infos = [{} for _ in range(self.num_envs)]
        
        # Apply Hero actions
        for i in range(self.num_envs):
            game = self.games[i]
            hero_id = self.hero_ids[i]
            opponent = self.opponents[i]
            
            # Skip if already done (should be handled by auto-reset, but check safety)
            if game.get_stage_name() == "Complete":
                # Ensure we mark as done if we haven't already
                # But usually we reset immediately when done is detected
                pass
                
            # Verify it's Hero's turn
            current = game.get_current_player_id()
            if current == hero_id:
                # Get state for action conversion
                state_dict = game.get_state_dict()
                
                action_type, amount = convert_action_label(actions[i], state_dict)
                
                # Notify opponent (if stateful)
                if opponent and hasattr(opponent, 'observe_action'):
                    opponent.observe_action(hero_id, action_type, amount, state_dict.get('pot', 0), state_dict.get('stage', 'Preflop'))
                
                # Process action
                success = game.process_action(hero_id, action_type, amount)
                if not success:
                    # Fallback to fold
                    game.process_action(hero_id, "fold", 0)
        
        # Fast forward through opponents
        states = self._fast_forward_all()
        
        # Check for completion and compute rewards
        for i in range(self.num_envs):
            game = self.games[i]
            
            if game.get_stage_name() == "Complete":
                dones[i] = True
                
                # Get final state
                final_state = game.get_state_dict()
                
                hero_chips = 0
                for p in final_state['players']:
                    if p['id'] == self.hero_ids[i]:
                        hero_chips = p['chips']
                        break
                
                profit = hero_chips - self.game_config.startingChips
                rewards[i] = profit / self.game_config.startingChips
                
                # Info
                infos[i]['terminal_observation'] = final_state
                infos[i]['outcome'] = 'win' if profit > 0 else 'loss' if profit < 0 else 'tie'
                
                # Auto-reset
                self._reset_env(i)
                # Fast forward new game
                new_states = self._fast_forward_env(i)
                states[i] = new_states
            else:
                rewards[i] = 0.0
                
        return states, rewards, dones, infos

    def _fast_forward_env(self, env_idx: int) -> Dict:
        """
        Advance a single environment until it's Hero's turn or game ends.
        Returns the state dict for Hero.
        """
        game = self.games[env_idx]
        hero_id = self.hero_ids[env_idx]
        opponent = self.opponents[env_idx]
        
        max_steps = 100
        step = 0
        
        while step < max_steps:
            stage = game.get_stage_name()
            if stage == "Complete":
                break
                
            current_id = game.get_current_player_id()
            if not current_id:
                advanced = game.advance_game()
                if not advanced:
                    break
                continue
                
            if current_id == hero_id:
                break
            
            # Opponent's turn
            state_dict = game.get_state_dict()
            
            legal_actions = state_dict.get('actionConstraints', {}).get('legalActions', [])
            
            if opponent:
                # Assuming select_action returns (type, amount, label) or similar
                # We use duck typing
                if hasattr(opponent, 'select_action'):
                    res = opponent.select_action(state_dict, legal_actions)
                    if len(res) == 3:
                        act_type, act_amount, _ = res
                    else:
                        act_type, act_amount = res
                elif hasattr(opponent, 'act'):
                    # Simpler interface
                    res = opponent.act(state_dict, legal_actions)
                    if isinstance(res, tuple):
                        act_type, act_amount = res
                    else:
                        act_type, act_amount = convert_action_label(res, state_dict)
                else:
                    # Fallback
                    act_type, act_amount = "check", 0
                    
                # Notify opponent of their own action (if they are stateful/self-updating?)
                # Usually observe_action is for *others*, but some agents track history
                if hasattr(opponent, 'observe_action'):
                    opponent.observe_action(current_id, act_type, act_amount, state_dict.get('pot', 0), state_dict.get('stage', 'Preflop'))
            else:
                # Random fallback
                act_type = "check"
                if "check" not in legal_actions:
                    act_type = "fold"
                act_amount = 0
            
            game.process_action(current_id, act_type, act_amount)
            step += 1
            
        state_dict = game.get_state_dict()
        return state_dict

    def _fast_forward_all(self) -> List[Dict]:
        """Fast forward all envs"""
        states = []
        for i in range(self.num_envs):
            states.append(self._fast_forward_env(i))
        return states
