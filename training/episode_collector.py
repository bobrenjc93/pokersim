#!/usr/bin/env python3
"""Episode Collection for Poker RL Training."""

import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from common.binding_loader import load_poker_api_binding

poker_api_binding = load_poker_api_binding()

from common import (
    RLStateEncoder, ACTION_NAMES, convert_action_label,
    create_legal_actions_mask, extract_state, estimate_hand_strength,
    GameConfig, create_binding_config,
)
from agents import (
    AgentCache, select_opponent_type, select_checkpoint_with_priority,
    get_anchor_checkpoints, select_anchor_checkpoint, PFSPOpponentSelector,
)

TERMINAL_STAGES = frozenset({'complete', 'showdown'})
MAX_STEPS_PER_EPISODE = 500

# Action index for fold (must match ACTION_NAMES order in common)
FOLD_ACTION_IDX = 0  # 'fold' is typically index 0


class EpisodeCollector:
    """Collects training episodes from self-play poker games."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        game_config: Dict[str, Any],
        device: torch.device,
        model_config: Optional[Dict[str, Any]] = None,
        opponent_pool: Optional[List[str]] = None,
        seed: Optional[int] = None,
        current_iteration: int = 0,
        pfsp_selector: Optional[PFSPOpponentSelector] = None,
    ):
        if poker_api_binding is None:
            raise RuntimeError("poker_api_binding not found")
        
        self.model = model
        self.game_config = game_config
        self.device = device
        self.model_config = model_config or {}
        self.opponent_pool = opponent_pool or []
        self.seed = seed
        self.current_iteration = current_iteration
        self._rng = random.Random(seed) if seed is not None else random
        self._cache = AgentCache(device=device)
        self._encoder = RLStateEncoder()
        # Track checkpoint info for priority selection
        self._checkpoint_info: List[Tuple[int, str]] = []
        # Track anchor checkpoints separately
        self._anchor_checkpoints: List[Tuple[int, str]] = []
        # PFSP selector for prioritized opponent selection
        self.pfsp_selector = pfsp_selector
    
    def update_opponent_pool(self, pool: List[str], checkpoint_info: Optional[List[Tuple[int, str]]] = None):
        """Update opponent pool with optional checkpoint metadata for priority selection.
        
        Args:
            pool: List of checkpoint paths
            checkpoint_info: Optional list of (iteration, path) tuples for priority selection
        """
        self.opponent_pool = pool
        if checkpoint_info is not None:
            self._checkpoint_info = checkpoint_info
        elif pool:
            # Infer iteration from path if not provided
            import re
            pattern = re.compile(r"iter_(\d+)\.pt")
            self._checkpoint_info = []
            for p in pool:
                match = pattern.search(str(p))
                if match:
                    self._checkpoint_info.append((int(match.group(1)), p))
                else:
                    self._checkpoint_info.append((0, p))
            self._checkpoint_info.sort(key=lambda x: x[0])
        
        # Extract anchor checkpoints
        self._anchor_checkpoints = get_anchor_checkpoints(self._checkpoint_info)

    def _reset_encoders(self, opp_encoder: Optional[RLStateEncoder] = None) -> None:
        """Reset per-hand encoder state (action history, etc.)."""
        # RLStateEncoder includes opponent modeling features via action history;
        # even if we don't feed observations today, make the reset explicit to avoid
        # cross-hand leakage if we add it later.
        self._encoder.reset_history()
        if opp_encoder is not None:
            opp_encoder.reset_history()
    
    def _create_opponent(self) -> Tuple[Optional[torch.nn.Module], Optional[RLStateEncoder], Optional[Any], str]:
        """Create opponent based on weighted selection strategy.
        
        Uses separate handling for:
        - pfsp_model: PFSP-weighted checkpoint selection (primary self-play mechanism)
        - anchor_model: Milestone checkpoints (iter_1, iter_50, etc.) for fundamental skills
        - past_model: Recent non-anchor checkpoints for moderate self-play
        - heuristic types: Calling station, tight, etc. for diverse opponents
        """
        opp_type = select_opponent_type(
            has_pool=bool(self.opponent_pool),
            rng=self._rng if isinstance(self._rng, random.Random) else None,
            num_checkpoints=len(self.opponent_pool),
            has_anchors=bool(self._anchor_checkpoints),
            has_pfsp=self.pfsp_selector is not None and bool(self._checkpoint_info),
        )
        
        # Handle PFSP-weighted checkpoint selection (prioritizes opponents we lose to)
        if opp_type == 'pfsp_model' and self.pfsp_selector and self._checkpoint_info:
            selected_path = self.pfsp_selector.select_checkpoint(
                self._checkpoint_info,
                rng=self._rng if isinstance(self._rng, random.Random) else None,
            )
            if selected_path:
                model = self._cache.get_model(selected_path, self.model_config)
                if model:
                    return model, RLStateEncoder(), None, opp_type
            # Fallback to anchor selection if PFSP fails
            opp_type = 'anchor_model'
        
        # Handle anchor checkpoint selection
        if opp_type == 'anchor_model' and self._anchor_checkpoints:
            selected_path = select_anchor_checkpoint(
                self._anchor_checkpoints,
                rng=self._rng if isinstance(self._rng, random.Random) else None,
            )
            if selected_path:
                model = self._cache.get_model(selected_path, self.model_config)
                if model:
                    return model, RLStateEncoder(), None, opp_type
            # Fallback to calling_station if anchor load fails
            opp_type = 'calling_station'
        
        # Handle regular past checkpoint selection (non-anchors)
        if opp_type == 'past_model' and self.opponent_pool:
            if self._checkpoint_info:
                selected_path = select_checkpoint_with_priority(
                    self._checkpoint_info,
                    self.current_iteration,
                    rng=self._rng if isinstance(self._rng, random.Random) else None,
                )
            else:
                # Fallback to random selection if no checkpoint info
                selected_path = self._rng.choice(self.opponent_pool)
            
            if selected_path:
                model = self._cache.get_model(selected_path, self.model_config)
                if model:
                    return model, RLStateEncoder(), None, opp_type
            opp_type = 'calling_station'
        
        if opp_type == 'model':
            return self.model, RLStateEncoder(), None, opp_type
        
        return None, None, self._cache.get_agent(opp_type, 'p1'), opp_type
    
    def _sample_action(self, model: torch.nn.Module, encoder: RLStateEncoder,
                       state_dict: Dict, legal: List[str]) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from model policy."""
        state = encoder.encode_state(state_dict).unsqueeze(0).to(self.device)
        mask = create_legal_actions_mask(legal, self.device)
        
        with torch.no_grad():
            logits, val = model(state, mask)
            probs = F.softmax(logits, dim=-1)
            action_idx = torch.multinomial(probs, 1).squeeze(-1).item()
            log_prob = F.log_softmax(logits, dim=-1)[0, action_idx]
        
        return action_idx, state.squeeze(0).cpu(), mask.squeeze(0).cpu(), log_prob.cpu(), val.squeeze().cpu()
    
    def _opponent_action(self, opp_model, opp_encoder, opp_agent, state_dict: Dict, legal: List[str]) -> Tuple[str, int]:
        """Get opponent's action."""
        if opp_model and opp_encoder:
            action_idx, *_ = self._sample_action(opp_model, opp_encoder, state_dict, legal)
            return convert_action_label(ACTION_NAMES[action_idx], state_dict)
        
        if opp_agent:
            action, amount, _ = opp_agent.select_action(state_dict, legal)
            return action, amount
        
        return convert_action_label(random.choice(legal), state_dict)

    def _compute_main_reward(self, final_state: Dict[str, Any]) -> float:
        """Compute final normalized profit reward for p0."""
        starting = self.game_config.get('startingChips', 1000)
        for p in final_state.get('players', []):
            if p.get('id') == 'p0':
                return (p.get('chips', starting) - starting) / max(1, starting)
        return 0.0
    
    def collect_episode(self) -> Dict[str, Any]:
        """Collect a single training episode."""
        opp_model, opp_encoder, opp_agent, opp_type = self._create_opponent()
        self._reset_encoders(opp_encoder)
        
        cfg = GameConfig.from_dict(self.game_config)
        game = poker_api_binding.Game(create_binding_config(cfg, seed=self._rng.randint(0, 1_000_000)))
        # IMPORTANT (correctness / eval parity):
        # Elo matches alternate positions every other hand. If training always
        # starts from the same seat/position, the policy can overfit to that
        # specific seat and look "worse" in Elo even if it improved in-training.
        #
        # The C++ Game assigns positions based on player join order, so we
        # randomize join order each episode while keeping stable IDs (p0 is
        # always the learning agent whose trajectory we record).
        if self._rng.random() < 0.5:
            game.add_player('p1', 'Opponent', cfg.starting_chips)
            game.add_player('p0', 'Agent', cfg.starting_chips)
        else:
            game.add_player('p0', 'Agent', cfg.starting_chips)
            game.add_player('p1', 'Opponent', cfg.starting_chips)
        game.start_hand()
        
        # Trajectory buffers
        states, actions, log_probs, values, masks = [], [], [], [], []
        dones, hand_strengths, step_rewards = [], [], []
        
        for _ in range(MAX_STEPS_PER_EPISODE):
            stage = game.get_stage_name().lower()
            if stage in TERMINAL_STAGES:
                break
            
            player = game.get_current_player_id()
            if not player:
                if not game.advance_game():
                    break
                continue
            
            game_state = game.get_state_dict()
            legal = game_state.get('actionConstraints', {}).get('legalActions', [])
            if not legal:
                break
            
            state_dict = extract_state(game_state, player)
            
            if player == 'p0':
                action_idx, state, mask, log_prob, val = self._sample_action(
                    self.model, self._encoder, state_dict, legal
                )
                
                states.append(state)
                actions.append(action_idx)
                log_probs.append(log_prob)
                values.append(val)
                masks.append(mask)
                dones.append(0)
                hand_strengths.append(
                    estimate_hand_strength(state_dict.get('hole_cards', []), state_dict.get('community_cards', []))
                )
                step_rewards.append(0.0)
                
                action, amount = convert_action_label(ACTION_NAMES[action_idx], state_dict)
            else:
                action, amount = self._opponent_action(opp_model, opp_encoder, opp_agent, state_dict, legal)
            
            if not game.process_action(player, action, amount):
                game.process_action(player, 'fold', 0)
        
        if not states:
            return {'success': False}
        
        # Calculate reward (normalized profit)
        final_state = game.get_state_dict()
        main_reward = self._compute_main_reward(final_state)
        
        dones[-1] = 1
        
        # Track fold rate for collapse detection
        num_folds = sum(1 for a in actions if a == FOLD_ACTION_IDX)
        fold_rate = num_folds / len(actions) if actions else 0.0
        
        return {
            'states': torch.stack(states),
            'actions': torch.tensor(actions, dtype=torch.long),
            'log_probs': torch.stack(log_probs),
            'values': torch.stack(values),
            'legal_actions_masks': torch.stack(masks),
            'dones': torch.tensor(dones, dtype=torch.float32),
            'main_reward': main_reward,
            'step_rewards': step_rewards,
            'hand_strengths': hand_strengths,
            'fold_rate': fold_rate,
            'num_steps': len(actions),
            'opponent_type': opp_type,  # Track opponent for PFSP updates
            'success': True,
        }
    
    def collect_episodes(self, num_episodes: int, verbose: bool = False) -> List[Dict[str, Any]]:
        """Collect multiple training episodes."""
        # Ensure we don't permanently toggle the trainer model's mode.
        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        episodes = []
        
        for i in range(num_episodes):
            ep = self.collect_episode()
            if ep.get('success'):
                episodes.append(ep)
            if verbose and (i + 1) % 50 == 0:
                print(f"  {i + 1}/{num_episodes} episodes...")
        # Restore mode (important for dropout + gradient-checkpointing in updates).
        if was_training:
            self.model.train()
        return episodes
