#!/usr/bin/env python3
"""
Batched Rollout Collection for Efficient Episode Collection

This module implements batched inference for episode collection, which provides
significant speedups by:
1. Running multiple games simultaneously
2. Batching states that need model inference together
3. Running a single batched forward pass instead of many individual passes

Key insight: Model inference is the bottleneck during episode collection.
Batching N games together amortizes the PyTorch overhead and enables GPU parallelism.

Expected speedup: 2-5x compared to sequential episode collection.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

try:
    import poker_api_binding
except ImportError:
    poker_api_binding = None

from rl_state_encoder import RLStateEncoder, estimate_hand_strength
from model_agent import (
    ACTION_NAMES, convert_action_label, create_legal_actions_mask, extract_state,
    HeuristicAgent, TightAgent, AggressiveAgent, LoosePassiveAgent,
    CallingStationAgent, HeroCallerAgent, AlwaysRaiseAgent, AlwaysCallAgent,
    AlwaysFoldAgent, RandomAgent
)


@dataclass
class GameInstance:
    """Tracks state for a single game during batched collection."""
    game: Any  # poker_api_binding.Game
    agents: List[Dict[str, Any]]
    main_player_id: str
    encoder: RLStateEncoder
    
    # Episode data
    states: List[torch.Tensor] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    legal_masks: List[torch.Tensor] = field(default_factory=list)
    dones: List[int] = field(default_factory=list)
    action_types: List[str] = field(default_factory=list)
    action_labels: List[str] = field(default_factory=list)
    hand_strengths: List[float] = field(default_factory=list)
    step_rewards: List[float] = field(default_factory=list)
    
    # Tracking
    is_done: bool = False
    last_opponent_aggressive: bool = False
    opponent_type: Optional[str] = None
    opponent_checkpoint: Optional[str] = None
    is_out_of_position: bool = False
    folded_to_aggression: bool = False
    won_uncontested: bool = False
    
    # Pending inference (when main player needs to act)
    pending_state_tensor: Optional[torch.Tensor] = None
    pending_legal_mask: Optional[torch.Tensor] = None
    pending_hand_strength: float = 0.0


class BatchedRolloutCollector:
    """
    Collects episodes using batched model inference for efficiency.
    
    This collector manages multiple simultaneous games and batches together
    all states that need model inference for a single forward pass.
    
    Usage:
        collector = BatchedRolloutCollector(
            model=model,
            game_config=game_config,
            device=device,
            batch_size=32
        )
        episodes = collector.collect_episodes(num_episodes=100)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        game_config: Dict[str, Any],
        device: torch.device,
        batch_size: int = 32,
        opponent_pool: List[str] = None,
        deterministic: bool = False,
    ):
        """
        Args:
            model: Actor-critic model for policy inference
            game_config: Game configuration dict
            device: Device for model inference
            batch_size: Number of games to run simultaneously
            opponent_pool: List of checkpoint paths for past model opponents
            deterministic: Use argmax instead of sampling
        """
        self.model = model
        self.game_config = game_config
        self.device = device
        self.batch_size = batch_size
        self.opponent_pool = opponent_pool or []
        self.deterministic = deterministic
        
        # Cache for opponent models
        self._opponent_model_cache: Dict[str, torch.nn.Module] = {}
        
        # Heuristic agent instances (reuse to avoid allocation)
        self._agent_cache: Dict[str, Any] = {}
    
    def _create_game(self) -> Tuple[Any, List[Dict], str, RLStateEncoder]:
        """Create a new game instance with agents."""
        config = poker_api_binding.GameConfig()
        config.smallBlind = self.game_config.get('smallBlind', 10)
        config.bigBlind = self.game_config.get('bigBlind', 20)
        config.startingChips = self.game_config.get('startingChips', 1000)
        config.minPlayers = self.game_config.get('minPlayers', 2)
        config.maxPlayers = self.game_config.get('maxPlayers', 2)
        config.seed = random.randint(0, 1000000)
        
        game = poker_api_binding.Game(config)
        
        # Create agents
        num_players = self.game_config.get('num_players', 2)
        agents = []
        encoder = RLStateEncoder(use_buffer=True)
        
        # Main agent
        main_player_id = 'p0'
        agents.append({
            'id': main_player_id,
            'type': 'model',
            'encoder': encoder
        })
        
        # Opponent agents (same distribution as train.py)
        for i in range(1, num_players):
            player_id = f'p{i}'
            roll = random.random()
            
            if roll < 0.23:
                agents.append({'id': player_id, 'type': 'calling_station'})
            elif roll < 0.36:
                agents.append({'id': player_id, 'type': 'hero_caller'})
            elif self.opponent_pool and roll < 0.51:
                checkpoint_path = random.choice(self.opponent_pool)
                agents.append({
                    'id': player_id,
                    'type': 'past_model',
                    'checkpoint_path': checkpoint_path,
                    'encoder': RLStateEncoder(use_buffer=True)
                })
            elif roll < 0.61:
                agents.append({
                    'id': player_id,
                    'type': 'model',
                    'encoder': RLStateEncoder(use_buffer=True)
                })
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
        
        # Add players and start hand
        for agent in agents:
            game.add_player(agent['id'], f"Player_{agent['id']}", config.startingChips)
        game.start_hand()
        
        return game, agents, main_player_id, encoder
    
    def _get_heuristic_agent(self, agent_type: str, player_id: str):
        """Get or create a heuristic agent instance."""
        cache_key = f"{agent_type}_{player_id}"
        if cache_key not in self._agent_cache:
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
                self._agent_cache[cache_key] = agent_classes[agent_type](player_id, agent_type)
        return self._agent_cache.get(cache_key)
    
    def _load_opponent_model(self, checkpoint_path: str) -> Optional[torch.nn.Module]:
        """Load opponent model from checkpoint (cached)."""
        if checkpoint_path in self._opponent_model_cache:
            return self._opponent_model_cache[checkpoint_path]
        
        try:
            from rl_model import create_actor_critic
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Infer model config
            state_dict = checkpoint.get('model_state_dict', {})
            hidden_dim = checkpoint.get('hidden_dim', 256)
            num_heads = checkpoint.get('num_heads', 8)
            num_layers = checkpoint.get('num_layers', 4)
            input_dim = checkpoint.get('input_dim', 167)
            
            if 'pos_encoding' in state_dict:
                hidden_dim = state_dict['pos_encoding'].shape[2]
            
            model = create_actor_critic(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=0.1,
                gradient_checkpointing=False
            )
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self._opponent_model_cache[checkpoint_path] = model
            return model
        except Exception:
            return None
    
    def _step_game(self, instance: GameInstance) -> bool:
        """
        Advance a game by one step.
        
        Returns True if game should continue, False if done.
        Sets instance.pending_* if main player needs to act.
        """
        game = instance.game
        
        # Check if game is done
        stage = game.get_stage_name().lower()
        if stage in {'complete', 'showdown'}:
            instance.is_done = True
            return False
        
        # Get current player
        current_player_id = game.get_current_player_id()
        if not current_player_id:
            if not game.advance_game():
                instance.is_done = True
                return False
            return True
        
        # Find agent for current player
        current_agent = None
        for agent in instance.agents:
            if agent['id'] == current_player_id:
                current_agent = agent
                break
        
        if current_agent is None:
            instance.is_done = True
            return False
        
        # Get game state
        game_state = game.get_state_dict()
        state_dict = extract_state(game_state, current_player_id)
        legal_actions = game_state.get('actionConstraints', {}).get('legalActions', [])
        
        if not legal_actions:
            instance.is_done = True
            return False
        
        # Main player needs model inference - defer to batch
        if current_player_id == instance.main_player_id:
            state_tensor = instance.encoder.encode_state(state_dict)
            legal_mask = create_legal_actions_mask(legal_actions, 'cpu').squeeze(0)
            
            hole_cards = state_dict.get('hole_cards', [])
            community_cards = state_dict.get('community_cards', [])
            hand_strength = estimate_hand_strength(hole_cards, community_cards)
            
            instance.pending_state_tensor = state_tensor
            instance.pending_legal_mask = legal_mask
            instance.pending_hand_strength = hand_strength
            return True  # Wait for batched inference
        
        # Opponent action - handle immediately
        action_type, amount = self._select_opponent_action(
            current_agent, state_dict, legal_actions
        )
        
        # Track opponent aggression
        instance.last_opponent_aggressive = action_type in ('bet', 'raise', 'all_in')
        
        # Process action
        if not game.process_action(current_player_id, action_type, amount):
            game.process_action(current_player_id, "fold", 0)
        
        return True
    
    def _select_opponent_action(
        self,
        agent: Dict,
        state_dict: Dict,
        legal_actions: List[str]
    ) -> Tuple[str, int]:
        """Select action for an opponent agent."""
        agent_type = agent['type']
        player_id = agent['id']
        
        # Heuristic agents
        if agent_type in ['heuristic', 'tight', 'aggressive', 'loose_passive',
                          'calling_station', 'hero_caller', 'always_raise',
                          'always_call', 'always_fold', 'random']:
            agent_instance = self._get_heuristic_agent(agent_type, player_id)
            if agent_instance:
                action_type, amount, _ = agent_instance.select_action(state_dict, legal_actions)
                return action_type, amount
        
        # Model-based opponents (current or past model)
        if agent_type in ['model', 'past_model']:
            if agent_type == 'past_model':
                checkpoint_path = agent.get('checkpoint_path')
                model = self._load_opponent_model(checkpoint_path) if checkpoint_path else None
            else:
                model = self.model
            
            if model is not None:
                encoder = agent.get('encoder', RLStateEncoder(use_buffer=True))
                state_tensor = encoder.encode_state(state_dict).unsqueeze(0).to(self.device)
                legal_mask = create_legal_actions_mask(legal_actions, self.device)
                
                with torch.no_grad():
                    action_logits, _ = model(state_tensor, legal_mask)
                    action_probs = F.softmax(action_logits, dim=-1)
                    action_idx = torch.multinomial(action_probs.squeeze(0), 1).item()
                
                action_label = ACTION_NAMES[action_idx]
                action_type, amount = convert_action_label(action_label, state_dict)
                return action_type, amount
        
        # Fallback
        return 'check' if 'check' in legal_actions else 'fold', 0
    
    def _batched_model_inference(
        self,
        instances: List[GameInstance]
    ) -> None:
        """
        Run batched model inference for all instances that have pending states.
        
        This is the key optimization - instead of N individual forward passes,
        we do 1 batched forward pass.
        """
        # Collect instances that need inference
        pending = [(i, inst) for i, inst in enumerate(instances) 
                   if inst.pending_state_tensor is not None and not inst.is_done]
        
        if not pending:
            return
        
        # Batch states and masks
        states = torch.stack([inst.pending_state_tensor for _, inst in pending]).to(self.device)
        masks = torch.stack([inst.pending_legal_mask for _, inst in pending]).to(self.device)
        
        # Single batched forward pass!
        with torch.no_grad():
            action_logits, values = self.model(states, masks)
            action_probs = F.softmax(action_logits, dim=-1)
            
            if self.deterministic:
                action_indices = torch.argmax(action_probs, dim=-1)
            else:
                action_indices = torch.multinomial(action_probs, 1).squeeze(-1)
            
            log_probs = F.log_softmax(action_logits, dim=-1)
            selected_log_probs = log_probs.gather(1, action_indices.unsqueeze(-1)).squeeze(-1)
        
        # Distribute results back to instances
        for batch_idx, (_, inst) in enumerate(pending):
            action_idx = action_indices[batch_idx].item()
            log_prob = selected_log_probs[batch_idx].cpu()
            value = values[batch_idx].squeeze().cpu()
            
            # Get action details
            action_label = ACTION_NAMES[action_idx]
            game_state = inst.game.get_state_dict()
            state_dict = extract_state(game_state, inst.main_player_id)
            action_type, amount = convert_action_label(action_label, state_dict)
            
            # Store trajectory data
            inst.states.append(inst.pending_state_tensor.cpu())
            inst.actions.append(action_idx)
            inst.log_probs.append(log_prob)
            inst.values.append(value)
            inst.legal_masks.append(inst.pending_legal_mask.cpu())
            inst.dones.append(0)
            inst.action_types.append(action_type)
            inst.action_labels.append(action_label)
            inst.hand_strengths.append(inst.pending_hand_strength)
            inst.step_rewards.append(0.0)  # Will be computed later
            
            # Check fold to aggression
            if action_type == 'fold' and inst.last_opponent_aggressive:
                inst.folded_to_aggression = True
            
            # Apply action
            if not inst.game.process_action(inst.main_player_id, action_type, amount):
                inst.game.process_action(inst.main_player_id, "fold", 0)
            
            # Clear pending
            inst.pending_state_tensor = None
            inst.pending_legal_mask = None
    
    def _finalize_episode(self, inst: GameInstance) -> Dict[str, Any]:
        """Convert GameInstance to episode dict."""
        game_state = inst.game.get_state_dict()
        starting_chips = self.game_config.get('startingChips', 1000)
        
        # Calculate reward
        main_reward = 0.0
        for player in game_state.get('players', []):
            if player['id'] == inst.main_player_id:
                profit = player['chips'] - starting_chips
                main_reward = profit / starting_chips
                break
        
        # Detect uncontested win
        final_stage = game_state.get('stage', '').lower()
        if main_reward > 0 and final_stage == 'complete':
            inst.won_uncontested = True
        
        # Mark last state as done
        if inst.dones:
            inst.dones[-1] = 1
        
        # Build episode dict
        episode = {
            'states': torch.stack(inst.states) if inst.states else torch.tensor([]),
            'actions': torch.tensor(inst.actions, dtype=torch.long) if inst.actions else torch.tensor([], dtype=torch.long),
            'log_probs': torch.stack(inst.log_probs) if inst.log_probs else torch.tensor([]),
            'values': torch.stack(inst.values) if inst.values else torch.tensor([]),
            'legal_actions_masks': torch.stack(inst.legal_masks) if inst.legal_masks else torch.tensor([]),
            'dones': torch.tensor(inst.dones, dtype=torch.float32) if inst.dones else torch.tensor([], dtype=torch.float32),
            'main_reward': main_reward,
            'step_rewards': inst.step_rewards,
            'hand_strengths': inst.hand_strengths,
            'action_types': inst.action_types,
            'action_labels': inst.action_labels,
            'regrets': [{} for _ in inst.actions],
            'opponent_type': inst.opponent_type,
            'opponent_checkpoint': inst.opponent_checkpoint,
            'folded_to_aggression': inst.folded_to_aggression,
            'won_uncontested': inst.won_uncontested,
            'is_out_of_position': inst.is_out_of_position,
            'success': len(inst.states) > 0,
        }
        
        return episode
    
    def collect_episodes(self, num_episodes: int, verbose: bool = False) -> List[Dict[str, Any]]:
        """
        Collect episodes using batched inference.
        
        Args:
            num_episodes: Number of episodes to collect
            verbose: Print progress
            
        Returns:
            List of episode dictionaries
        """
        episodes = []
        active_instances: List[GameInstance] = []
        episodes_started = 0
        max_steps_per_game = 1000
        
        self.model.eval()
        
        while len(episodes) < num_episodes:
            # Start new games to fill batch
            while len(active_instances) < self.batch_size and episodes_started < num_episodes:
                game, agents, main_player_id, encoder = self._create_game()
                
                # Get initial game state
                game_state = game.get_state_dict()
                is_oop = not game_state.get('players', [{}])[0].get('isDealer', False)
                
                # Track opponent info
                opp_type = None
                opp_checkpoint = None
                for agent in agents:
                    if agent['id'] != main_player_id:
                        opp_type = agent.get('type')
                        if opp_type == 'past_model':
                            opp_checkpoint = agent.get('checkpoint_path')
                        break
                
                inst = GameInstance(
                    game=game,
                    agents=agents,
                    main_player_id=main_player_id,
                    encoder=encoder,
                    opponent_type=opp_type,
                    opponent_checkpoint=opp_checkpoint,
                    is_out_of_position=is_oop,
                )
                active_instances.append(inst)
                episodes_started += 1
            
            if not active_instances:
                break
            
            # Step all active games until they need inference or are done
            for inst in active_instances:
                steps = 0
                while not inst.is_done and inst.pending_state_tensor is None and steps < max_steps_per_game:
                    self._step_game(inst)
                    steps += 1
            
            # Run batched inference for all games that need it
            self._batched_model_inference(active_instances)
            
            # Collect finished episodes and remove from active
            still_active = []
            for inst in active_instances:
                if inst.is_done:
                    episode = self._finalize_episode(inst)
                    if episode['success']:
                        episodes.append(episode)
                        if verbose and len(episodes) % 50 == 0:
                            print(f"  Collected {len(episodes)}/{num_episodes} episodes...")
                else:
                    still_active.append(inst)
            active_instances = still_active
        
        return episodes[:num_episodes]


def collect_episodes_batched(
    model: torch.nn.Module,
    game_config: Dict[str, Any],
    device: torch.device,
    num_episodes: int,
    batch_size: int = 32,
    opponent_pool: List[str] = None,
    deterministic: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convenience function to collect episodes using batched inference.
    
    Args:
        model: Actor-critic model
        game_config: Game configuration
        device: Device for inference
        num_episodes: Number of episodes to collect
        batch_size: Number of simultaneous games
        opponent_pool: List of checkpoint paths for past model opponents
        deterministic: Use argmax instead of sampling
        verbose: Print progress
        
    Returns:
        List of episode dictionaries
    """
    collector = BatchedRolloutCollector(
        model=model,
        game_config=game_config,
        device=device,
        batch_size=batch_size,
        opponent_pool=opponent_pool,
        deterministic=deterministic,
    )
    return collector.collect_episodes(num_episodes, verbose=verbose)

