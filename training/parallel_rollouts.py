#!/usr/bin/env python3
"""
Multi-Process Rollout Collection for Poker RL Training

This module implements parallel episode collection using Python multiprocessing.
Each worker process has its own copy of the model and poker engine binding,
allowing true parallelism without GIL limitations.

Key features:
- Worker processes collect episodes independently
- Model weights are synced from main process periodically
- Uses torch.multiprocessing for proper CUDA/MPS tensor handling
- Efficient communication via shared memory and queues
"""

import os
import random
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

# Set multiprocessing start method early
# 'spawn' is required for CUDA compatibility on all platforms
try:
    mp.set_start_method('spawn', force=False)
except RuntimeError:
    pass  # Already set

try:
    import orjson as json
except ImportError:
    import json


@dataclass
class EpisodeData:
    """Container for episode data returned by workers."""
    states: torch.Tensor  # (num_steps, state_dim)
    actions: torch.Tensor  # (num_steps,)
    log_probs: torch.Tensor  # (num_steps,)
    values: torch.Tensor  # (num_steps,)
    legal_masks: torch.Tensor  # (num_steps, num_actions)
    dones: torch.Tensor  # (num_steps,)
    main_reward: float
    step_rewards: List[float]
    hand_strengths: List[float]
    action_types: List[str]
    action_labels: List[str]
    regrets: List[Dict]
    opponent_type: str
    opponent_checkpoint: Optional[str]
    folded_to_aggression: bool
    won_uncontested: bool
    is_out_of_position: bool
    success: bool


def episode_data_to_dict(ep: EpisodeData) -> Dict[str, Any]:
    """Convert EpisodeData to dictionary for pickling."""
    return {
        'states': ep.states,
        'actions': ep.actions,
        'log_probs': ep.log_probs,
        'values': ep.values,
        'legal_masks': ep.legal_masks,
        'dones': ep.dones,
        'main_reward': ep.main_reward,
        'step_rewards': ep.step_rewards,
        'hand_strengths': ep.hand_strengths,
        'action_types': ep.action_types,
        'action_labels': ep.action_labels,
        'regrets': ep.regrets,
        'opponent_type': ep.opponent_type,
        'opponent_checkpoint': ep.opponent_checkpoint,
        'folded_to_aggression': ep.folded_to_aggression,
        'won_uncontested': ep.won_uncontested,
        'is_out_of_position': ep.is_out_of_position,
        'success': ep.success,
    }


def dict_to_episode_data(d: Dict[str, Any]) -> EpisodeData:
    """Convert dictionary back to EpisodeData."""
    return EpisodeData(**d)


class RolloutWorker:
    """
    Worker that collects episodes in a separate process.
    
    Each worker:
    1. Has its own copy of the model (loaded from shared state dict)
    2. Has its own poker_api_binding instance
    3. Collects episodes and sends them back via a queue
    """
    
    def __init__(
        self,
        worker_id: int,
        game_config: Dict[str, Any],
        model_config: Dict[str, Any],
        device: str = 'cpu',  # Workers typically use CPU for inference
    ):
        self.worker_id = worker_id
        self.game_config = game_config
        self.model_config = model_config
        self.device = torch.device(device)
        
        # These will be initialized in the worker process
        self.model = None
        self.encoder = None
        self.poker_api = None
        
    def initialize(self):
        """Initialize model and encoder in the worker process."""
        # Import here to ensure imports happen in the worker process
        from rl_model import create_actor_critic
        from rl_state_encoder import RLStateEncoder
        
        # Create model
        self.model = create_actor_critic(
            input_dim=self.model_config['input_dim'],
            hidden_dim=self.model_config['hidden_dim'],
            num_heads=self.model_config['num_heads'],
            num_layers=self.model_config['num_layers'],
            dropout=self.model_config.get('dropout', 0.1),
            gradient_checkpointing=False  # Disable for inference
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Create encoder
        self.encoder = RLStateEncoder()
        
        # Import poker_api_binding
        try:
            import poker_api_binding
            self.poker_api = poker_api_binding
        except ImportError:
            raise RuntimeError("poker_api_binding not found in worker process")
    
    def update_model_weights(self, state_dict: Dict):
        """Update model weights from main process."""
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def collect_episode(
        self,
        opponent_pool: List[str] = None,
        use_opponent_pool: bool = True,
        deterministic: bool = False,
        num_runouts: int = 0,
        regret_weight: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Collect one episode (complete poker hand).
        
        This is similar to RLTrainingSession.collect_episode but runs in a worker process.
        """
        from model_agent import (
            RandomAgent, HeuristicAgent, TightAgent, LoosePassiveAgent,
            AggressiveAgent, CallingStationAgent, HeroCallerAgent,
            AlwaysRaiseAgent, AlwaysCallAgent, AlwaysFoldAgent,
            ACTION_MAP, ACTION_NAMES, convert_action_label, 
            create_legal_actions_mask, extract_state
        )
        from rl_state_encoder import RLStateEncoder, estimate_hand_strength
        
        # Create fresh encoder for this episode
        encoder = RLStateEncoder()
        
        # Setup agents
        num_players = self.game_config.get('num_players', 2)
        agents = []
        
        # Main agent (uses current model)
        main_player_id = 'p0'
        agents.append({
            'id': main_player_id,
            'type': 'model',
            'encoder': encoder
        })
        
        # Opponent selection (same distribution as train.py)
        for i in range(1, num_players):
            player_id = f'p{i}'
            roll = random.random()
            
            if roll < 0.23:
                agents.append({'id': player_id, 'type': 'calling_station'})
            elif roll < 0.36:
                agents.append({'id': player_id, 'type': 'hero_caller'})
            elif use_opponent_pool and opponent_pool and roll < 0.51:
                checkpoint_path = random.choice(opponent_pool)
                agents.append({
                    'id': player_id,
                    'type': 'past_model',
                    'checkpoint_path': checkpoint_path
                })
            elif use_opponent_pool and roll < 0.61:
                agents.append({'id': player_id, 'type': 'model', 'encoder': RLStateEncoder()})
            elif roll < 0.74:
                agents.append({'id': player_id, 'type': 'tight'})
            elif roll < 0.82:
                agents.append({'id': player_id, 'type': 'heuristic'})
            elif roll < 0.86:
                agents.append({'id': player_id, 'type': 'aggressive'})
            elif roll < 0.89:
                agents.append({'id': player_id, 'type': 'loose_passive'})
            elif roll < 0.92:
                agents.append({'id': player_id, 'type': 'always_call'})
            elif roll < 0.94:
                agents.append({'id': player_id, 'type': 'always_raise'})
            elif roll < 0.96:
                agents.append({'id': player_id, 'type': 'always_fold'})
            else:
                agents.append({'id': player_id, 'type': 'random'})
        
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
            return self._empty_episode(success=False)
        
        game_state = response['gameState']
        
        # Determine position
        is_out_of_position = not game_state.get('players', [{}])[0].get('isDealer', False)
        
        # Track opponent info
        opponent_type_faced = None
        opponent_checkpoint_faced = None
        for agent in agents:
            if agent['id'] != main_player_id:
                opponent_type_faced = agent.get('type')
                if opponent_type_faced == 'past_model':
                    opponent_checkpoint_faced = agent.get('checkpoint_path', '')
                break
        
        # Episode tracking
        episode = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'legal_actions_masks': [],
            'dones': [],
            'action_types': [],
            'action_labels': [],
            'hand_strengths': [],
            'step_rewards': [],
            'regrets': [],
            'folded_to_aggression': False,
            'won_uncontested': False,
            'is_out_of_position': is_out_of_position,
            'opponent_type': opponent_type_faced,
            'opponent_checkpoint': opponent_checkpoint_faced,
            'success': True,
        }
        
        last_action_was_opponent_aggression = False
        
        # Opponent agent instances cache
        opponent_agents = {}
        opponent_models = {}
        
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
                legal_mask = create_legal_actions_mask(legal_actions, self.device)
                
                # Compute hand strength
                hole_cards = state_dict.get('hole_cards', [])
                community_cards = state_dict.get('community_cards', [])
                hand_strength = estimate_hand_strength(hole_cards, community_cards)
                
                with torch.no_grad():
                    action_logits, value = self.model(state_tensor, legal_mask)
                    action_probs = F.softmax(action_logits, dim=-1)
                    
                    if deterministic:
                        action_idx = torch.argmax(action_probs.squeeze(0)).item()
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
                
                action_label = ACTION_NAMES[action_idx]
                action_type, amount = convert_action_label(action_label, state_dict)
                
                episode['action_types'].append(action_type)
                episode['action_labels'].append(action_label)
                episode['hand_strengths'].append(hand_strength)
                episode['regrets'].append({})  # Regrets computed if num_runouts > 0
                
                # Compute step reward shaping
                step_reward = self._compute_action_shaping_reward(
                    action_type, hand_strength, state_dict, last_action_was_opponent_aggression
                )
                episode['step_rewards'].append(step_reward)
                
                if action_type == 'fold' and last_action_was_opponent_aggression:
                    episode['folded_to_aggression'] = True
                    
            else:
                # Opponent action selection
                action_type, amount, action_label = self._select_opponent_action(
                    current_agent, state_dict, legal_actions,
                    opponent_agents, opponent_models
                )
            
            # Track opponent aggression
            if current_player_id != main_player_id:
                last_action_was_opponent_aggression = action_type in ('bet', 'raise', 'all_in')
            else:
                last_action_was_opponent_aggression = False
            
            # Observe action for all agents
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
            
            response = self._call_api(history)
            if not response['success']:
                episode['success'] = False
                break
            
            game_state = response['gameState']
        
        # Calculate rewards
        initial_chips = self.game_config.get('startingChips', 1000)
        
        for player_data in game_state.get('players', []):
            if player_data['id'] == main_player_id:
                final_chips = player_data['chips']
                episode['main_reward'] = (final_chips - initial_chips) / initial_chips
                break
        else:
            episode['main_reward'] = 0.0
        
        # Detect uncontested win
        final_stage = game_state.get('stage', '').lower()
        if episode['main_reward'] > 0 and final_stage == 'complete':
            episode['won_uncontested'] = True
        
        # Mark last state as done
        if episode['dones']:
            episode['dones'][-1] = 1
        
        # Convert to tensors
        if episode['states']:
            episode['states'] = torch.stack(episode['states'])
            episode['actions'] = torch.tensor(episode['actions'], dtype=torch.long)
            episode['log_probs'] = torch.stack(episode['log_probs'])
            episode['values'] = torch.stack(episode['values'])
            episode['legal_actions_masks'] = torch.stack(episode['legal_actions_masks'])
            episode['dones'] = torch.tensor(episode['dones'], dtype=torch.float32)
        
        return episode
    
    def _call_api(self, history: List[Dict]) -> Dict:
        """Call poker API."""
        payload = {
            'config': {
                **self.game_config,
                'seed': random.randint(0, 1000000)
            },
            'history': history
        }
        
        try:
            payload_bytes = json.dumps(payload)
            payload_str = payload_bytes.decode('utf-8') if isinstance(payload_bytes, bytes) else payload_bytes
            response_str = self.poker_api.process_request(payload_str)
            return json.loads(response_str)
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _get_legal_actions(self, game_state: Dict) -> List[str]:
        """Get legal actions from game state."""
        action_constraints = game_state.get('actionConstraints', {})
        return action_constraints.get('legalActions', [])
    
    def _select_opponent_action(
        self,
        agent: Dict,
        state_dict: Dict,
        legal_actions: List[str],
        opponent_agents: Dict,
        opponent_models: Dict
    ) -> Tuple[str, int, str]:
        """Select action for an opponent agent."""
        from model_agent import (
            RandomAgent, HeuristicAgent, TightAgent, LoosePassiveAgent,
            AggressiveAgent, CallingStationAgent, HeroCallerAgent,
            AlwaysRaiseAgent, AlwaysCallAgent, AlwaysFoldAgent,
            ACTION_NAMES, convert_action_label, create_legal_actions_mask
        )
        from rl_state_encoder import RLStateEncoder
        
        agent_type = agent['type']
        player_id = agent['id']
        
        if agent_type == 'random':
            action = random.choice(legal_actions)
            if action == 'bet':
                action_label = f'bet_{random.choice([50, 75, 100])}%'
            elif action == 'raise':
                action_label = f'raise_{random.choice([50, 75, 100])}%'
            else:
                action_label = action
            action_type, amount = convert_action_label(action_label, state_dict)
            return action_type, amount, action_label
        
        # Get or create agent instance
        if player_id not in opponent_agents:
            agent_classes = {
                'heuristic': HeuristicAgent,
                'tight': TightAgent,
                'aggressive': AggressiveAgent,
                'loose_passive': LoosePassiveAgent,
                'calling_station': CallingStationAgent,
                'hero_caller': HeroCallerAgent,
                'always_raise': AlwaysRaiseAgent,
                'always_call': AlwaysCallAgent,
                'always_fold': AlwaysFoldAgent,
                'random': RandomAgent,
            }
            
            if agent_type in agent_classes:
                opponent_agents[player_id] = agent_classes[agent_type](player_id, agent_type)
        
        if agent_type in ['heuristic', 'tight', 'aggressive', 'loose_passive', 
                          'calling_station', 'hero_caller', 'always_raise',
                          'always_call', 'always_fold', 'random']:
            agent_instance = opponent_agents.get(player_id)
            if agent_instance:
                return agent_instance.select_action(state_dict, legal_actions)
        
        elif agent_type == 'model':
            # Self-play with current model
            encoder = agent.get('encoder', RLStateEncoder())
            state_tensor = encoder.encode_state(state_dict).unsqueeze(0).to(self.device)
            legal_mask = create_legal_actions_mask(legal_actions, self.device)
            
            with torch.no_grad():
                action_logits, _ = self.model(state_tensor, legal_mask)
                action_probs = F.softmax(action_logits, dim=-1)
                action_idx = torch.multinomial(action_probs.squeeze(0), 1).item()
            
            action_label = ACTION_NAMES[action_idx]
            action_type, amount = convert_action_label(action_label, state_dict)
            return action_type, amount, action_label
        
        elif agent_type == 'past_model':
            # Load past model if not cached
            checkpoint_path = agent.get('checkpoint_path')
            if checkpoint_path and checkpoint_path not in opponent_models:
                try:
                    from rl_model import create_actor_critic
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                    past_model = create_actor_critic(
                        input_dim=self.model_config['input_dim'],
                        hidden_dim=self.model_config['hidden_dim'],
                        num_heads=self.model_config['num_heads'],
                        num_layers=self.model_config['num_layers'],
                        dropout=0.1,
                        gradient_checkpointing=False
                    )
                    past_model.load_state_dict(checkpoint['model_state_dict'])
                    past_model.to(self.device)
                    past_model.eval()
                    opponent_models[checkpoint_path] = past_model
                except Exception:
                    opponent_models[checkpoint_path] = None
            
            past_model = opponent_models.get(checkpoint_path)
            if past_model:
                encoder = agent.get('encoder', RLStateEncoder())
                state_tensor = encoder.encode_state(state_dict).unsqueeze(0).to(self.device)
                legal_mask = create_legal_actions_mask(legal_actions, self.device)
                
                with torch.no_grad():
                    action_logits, _ = past_model(state_tensor, legal_mask)
                    action_probs = F.softmax(action_logits, dim=-1)
                    action_idx = torch.multinomial(action_probs.squeeze(0), 1).item()
                
                action_label = ACTION_NAMES[action_idx]
                action_type, amount = convert_action_label(action_label, state_dict)
                return action_type, amount, action_label
        
        # Fallback to random
        action = random.choice(legal_actions) if legal_actions else 'fold'
        action_type, amount = convert_action_label(action, state_dict)
        return action_type, amount, action
    
    def _compute_action_shaping_reward(
        self,
        action_type: str,
        hand_strength: float,
        state: Dict,
        facing_aggression: bool
    ) -> float:
        """Compute per-step reward shaping (same logic as train.py)."""
        reward = 0.0
        
        TRASH_THRESHOLD = 0.30
        WEAK_THRESHOLD = 0.45
        MEDIUM_THRESHOLD = 0.55
        STRONG_THRESHOLD = 0.70
        PREMIUM_THRESHOLD = 0.85
        
        if action_type == 'fold':
            if hand_strength < TRASH_THRESHOLD:
                reward += 0.02 if facing_aggression else -0.03
            elif hand_strength < WEAK_THRESHOLD:
                reward += -0.02 if facing_aggression else -0.08
            elif hand_strength < MEDIUM_THRESHOLD:
                reward += -0.08 if facing_aggression else -0.12
            elif hand_strength < STRONG_THRESHOLD:
                reward -= 0.15
            else:
                reward -= 0.22
        
        elif action_type == 'all_in':
            player_chips = state.get('player_chips', 0)
            player_bet = state.get('player_bet', 0)
            commitment = player_bet / max(1, player_bet + player_chips)
            pot = state.get('pot', 0)
            spr = player_chips / max(1, pot) if pot > 0 else 10.0
            
            if hand_strength >= PREMIUM_THRESHOLD:
                reward += 0.18
            elif hand_strength >= STRONG_THRESHOLD:
                reward += 0.10
            elif hand_strength >= MEDIUM_THRESHOLD:
                if commitment > 0.6 or spr < 2.0:
                    reward += 0.0
                else:
                    reward -= 0.12
            elif hand_strength >= WEAK_THRESHOLD:
                reward -= 0.06 if commitment > 0.7 else -0.30
            else:
                reward -= 0.15 if commitment > 0.8 else -0.45
        
        elif action_type in ['bet', 'raise']:
            if hand_strength >= STRONG_THRESHOLD:
                reward += 0.10
            elif hand_strength >= MEDIUM_THRESHOLD:
                reward += 0.04
            elif hand_strength >= WEAK_THRESHOLD:
                reward -= 0.05
            else:
                reward -= 0.10
        
        elif action_type == 'call':
            if hand_strength >= STRONG_THRESHOLD:
                reward += 0.08
            elif hand_strength >= MEDIUM_THRESHOLD:
                reward += 0.06
            elif hand_strength >= WEAK_THRESHOLD:
                reward += 0.02
            else:
                reward -= 0.04
        
        elif action_type == 'check':
            if hand_strength >= STRONG_THRESHOLD:
                reward += 0.02
            elif hand_strength >= MEDIUM_THRESHOLD:
                reward += 0.05
            elif hand_strength < WEAK_THRESHOLD:
                reward += 0.06
            else:
                reward += 0.04
        
        return reward
    
    def _empty_episode(self, success: bool = True) -> Dict[str, Any]:
        """Return an empty episode dict."""
        return {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'legal_actions_masks': [],
            'dones': [],
            'main_reward': 0.0,
            'step_rewards': [],
            'hand_strengths': [],
            'action_types': [],
            'action_labels': [],
            'regrets': [],
            'opponent_type': None,
            'opponent_checkpoint': None,
            'folded_to_aggression': False,
            'won_uncontested': False,
            'is_out_of_position': False,
            'success': success,
        }


def worker_fn(
    worker_id: int,
    game_config: Dict[str, Any],
    model_config: Dict[str, Any],
    model_state_dict: Dict,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    shutdown_event: mp.Event,
    opponent_pool: List[str],
    device: str = 'cpu',
):
    """
    Worker process function.
    
    Args:
        worker_id: Worker ID
        game_config: Game configuration
        model_config: Model configuration  
        model_state_dict: Initial model state dict
        task_queue: Queue to receive tasks from main process
        result_queue: Queue to send results to main process
        shutdown_event: Event to signal shutdown
        opponent_pool: List of checkpoint paths for opponent pool
        device: Device for inference
    """
    try:
        # Initialize worker
        worker = RolloutWorker(worker_id, game_config, model_config, device)
        worker.initialize()
        worker.update_model_weights(model_state_dict)
        
        # Signal that we're ready
        result_queue.put({'type': 'ready', 'worker_id': worker_id})
        
        while not shutdown_event.is_set():
            try:
                # Get task with timeout
                task = task_queue.get(timeout=0.1)
            except Exception:
                continue
            
            if task is None:
                break
            
            task_type = task.get('type')
            
            if task_type == 'collect_episode':
                # Collect episode
                use_pool = task.get('use_opponent_pool', True)
                deterministic = task.get('deterministic', False)
                
                episode = worker.collect_episode(
                    opponent_pool=opponent_pool,
                    use_opponent_pool=use_pool,
                    deterministic=deterministic,
                )
                
                result_queue.put({
                    'type': 'episode',
                    'worker_id': worker_id,
                    'episode': episode
                })
            
            elif task_type == 'update_weights':
                # Update model weights
                new_state_dict = task.get('state_dict')
                if new_state_dict:
                    worker.update_model_weights(new_state_dict)
                result_queue.put({'type': 'weights_updated', 'worker_id': worker_id})
            
            elif task_type == 'update_opponent_pool':
                # Update opponent pool
                opponent_pool = task.get('opponent_pool', [])
                result_queue.put({'type': 'pool_updated', 'worker_id': worker_id})
            
            elif task_type == 'shutdown':
                break
        
    except Exception as e:
        result_queue.put({
            'type': 'error',
            'worker_id': worker_id,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


class ParallelRolloutManager:
    """
    Manages multiple worker processes for parallel episode collection.
    
    Usage:
        manager = ParallelRolloutManager(
            num_workers=4,
            game_config={...},
            model_config={...},
            model=my_model,
            device='cpu'
        )
        manager.start()
        
        # Collect episodes
        episodes = manager.collect_episodes(num_episodes=100)
        
        # Update model weights after training
        manager.update_model_weights(new_state_dict)
        
        # Cleanup
        manager.shutdown()
    """
    
    def __init__(
        self,
        num_workers: int,
        game_config: Dict[str, Any],
        model_config: Dict[str, Any],
        model: torch.nn.Module,
        device: str = 'cpu',
        opponent_pool: List[str] = None,
    ):
        self.num_workers = num_workers
        self.game_config = game_config
        self.model_config = model_config
        self.model = model
        self.device = device
        self.opponent_pool = opponent_pool or []
        
        # Process management
        self.processes: List[mp.Process] = []
        self.task_queues: List[mp.Queue] = []
        self.result_queue = mp.Queue()
        self.shutdown_event = mp.Event()
        
        self._started = False
    
    def start(self):
        """Start worker processes."""
        if self._started:
            return
        
        # Get current model state dict (move to CPU for sharing)
        model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        
        # Create shared opponent pool
        mp_manager = mp.Manager()
        shared_opponent_pool = list(self.opponent_pool)
        
        for i in range(self.num_workers):
            task_queue = mp.Queue()
            self.task_queues.append(task_queue)
            
            p = mp.Process(
                target=worker_fn,
                args=(
                    i,
                    self.game_config,
                    self.model_config,
                    model_state_dict,
                    task_queue,
                    self.result_queue,
                    self.shutdown_event,
                    shared_opponent_pool,
                    self.device,
                ),
                daemon=True
            )
            p.start()
            self.processes.append(p)
        
        # Wait for all workers to be ready
        ready_count = 0
        while ready_count < self.num_workers:
            result = self.result_queue.get()
            if result['type'] == 'ready':
                ready_count += 1
            elif result['type'] == 'error':
                raise RuntimeError(f"Worker {result['worker_id']} failed: {result['error']}")
        
        self._started = True
    
    def collect_episodes(
        self,
        num_episodes: int,
        use_opponent_pool: bool = True,
        deterministic: bool = False,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Collect episodes in parallel.
        
        Args:
            num_episodes: Number of episodes to collect
            use_opponent_pool: Whether to use opponent pool
            deterministic: Use deterministic action selection
            verbose: Print progress
        
        Returns:
            List of episode dictionaries
        """
        if not self._started:
            raise RuntimeError("Manager not started. Call start() first.")
        
        # Distribute tasks round-robin
        for i in range(num_episodes):
            worker_idx = i % self.num_workers
            self.task_queues[worker_idx].put({
                'type': 'collect_episode',
                'use_opponent_pool': use_opponent_pool,
                'deterministic': deterministic,
            })
        
        # Collect results
        episodes = []
        collected = 0
        errors = 0
        
        while collected < num_episodes:
            result = self.result_queue.get()
            
            if result['type'] == 'episode':
                episode = result['episode']
                if episode.get('success', False) and episode.get('states') is not None:
                    if isinstance(episode['states'], torch.Tensor) and len(episode['states']) > 0:
                        episodes.append(episode)
                collected += 1
                
                if verbose and collected % 50 == 0:
                    print(f"  Collected {collected}/{num_episodes} episodes...")
            
            elif result['type'] == 'error':
                errors += 1
                collected += 1
                if verbose:
                    print(f"  Worker {result['worker_id']} error: {result['error']}")
        
        return episodes
    
    def update_model_weights(self, state_dict: Dict = None):
        """Update model weights in all workers."""
        if not self._started:
            return
        
        if state_dict is None:
            state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        
        # Send update command to all workers
        for tq in self.task_queues:
            tq.put({
                'type': 'update_weights',
                'state_dict': state_dict
            })
        
        # Wait for acknowledgements
        acks = 0
        while acks < self.num_workers:
            result = self.result_queue.get()
            if result['type'] == 'weights_updated':
                acks += 1
    
    def update_opponent_pool(self, opponent_pool: List[str]):
        """Update opponent pool in all workers."""
        self.opponent_pool = opponent_pool
        
        if not self._started:
            return
        
        for tq in self.task_queues:
            tq.put({
                'type': 'update_opponent_pool',
                'opponent_pool': opponent_pool
            })
        
        # Wait for acknowledgements
        acks = 0
        while acks < self.num_workers:
            result = self.result_queue.get()
            if result['type'] == 'pool_updated':
                acks += 1
    
    def shutdown(self):
        """Shutdown all worker processes."""
        if not self._started:
            return
        
        self.shutdown_event.set()
        
        # Send shutdown command
        for tq in self.task_queues:
            try:
                tq.put({'type': 'shutdown'})
            except Exception:
                pass
        
        # Wait for processes to finish
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        # Clean up queues
        for tq in self.task_queues:
            tq.close()
        self.result_queue.close()
        
        self._started = False


def collect_episodes_parallel(
    num_episodes: int,
    num_workers: int,
    game_config: Dict[str, Any],
    model_config: Dict[str, Any],
    model: torch.nn.Module,
    opponent_pool: List[str] = None,
    use_opponent_pool: bool = True,
    deterministic: bool = False,
    device: str = 'cpu',
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    One-shot parallel episode collection.
    
    Creates workers, collects episodes, and shuts down.
    Use ParallelRolloutManager for persistent workers across iterations.
    """
    manager = ParallelRolloutManager(
        num_workers=num_workers,
        game_config=game_config,
        model_config=model_config,
        model=model,
        device=device,
        opponent_pool=opponent_pool or [],
    )
    
    try:
        manager.start()
        episodes = manager.collect_episodes(
            num_episodes=num_episodes,
            use_opponent_pool=use_opponent_pool,
            deterministic=deterministic,
            verbose=verbose,
        )
        return episodes
    finally:
        manager.shutdown()

