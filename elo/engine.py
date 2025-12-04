"""
ELO Rating Engine for Poker AI Models

Implements the classic ELO rating system where players gain/lose rating points
based on match outcomes and their relative ratings.
"""
import sys
import os
import re
import json
import random
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

import torch

# Add training directory to path to import modules
TRAINING_DIR = Path(__file__).parent.parent / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.append(str(TRAINING_DIR))

from model_agent import (
    extract_state,
    ModelAgent,
    RandomAgent,
    HeuristicAgent,
    TightAgent,
    LoosePassiveAgent,
    AggressiveAgent,
    CallingStationAgent,
    HeroCallerAgent,
    AlwaysRaiseAgent,
    AlwaysCallAgent,
    AlwaysFoldAgent,
    load_model_agent
)
import poker_api_binding
from config import DEFAULT_MODELS_DIR

if TYPE_CHECKING:
    from rl_model import PokerActorCritic


# Hand logging configuration
HAND_LOGS_DIR = Path(os.environ.get('HAND_LOGS_DIR', '/tmp/pokersim/hand_logs'))
HAND_LOG_FREQUENCY = 100  # Log 1 out of every N hands


class HandLogger:
    """
    Logger for recording poker hands with full action history and model predictions.
    Saves hands to disk in JSON format for analysis with the hand-viewer.
    """
    
    def __init__(self, logs_dir: Path = HAND_LOGS_DIR):
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.hand_counter = 0
        self.current_hand = None
    
    def should_log(self) -> bool:
        """Check if the current hand should be logged (1 in HAND_LOG_FREQUENCY)."""
        return self.hand_counter % HAND_LOG_FREQUENCY == 0
    
    def start_hand(
        self,
        player_a_id: str,
        player_b_id: str,
        player_a_name: str,
        player_b_name: str,
        player_a_config: Dict,
        player_b_config: Dict,
        starting_stack_a: int,
        starting_stack_b: int
    ):
        """Start logging a new hand."""
        self.hand_counter += 1
        
        if not self.should_log():
            self.current_hand = None
            return
        
        self.current_hand = {
            'hand_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'players': [
                {
                    'player_id': player_a_id,
                    'name': player_a_name,
                    'model_path': player_a_config.get('path', ''),
                    'agent_type': player_a_config.get('type', 'unknown'),
                    'starting_stack': starting_stack_a,
                    'hole_cards': [],
                    'is_dealer': False,
                    'is_small_blind': False,
                    'is_big_blind': False
                },
                {
                    'player_id': player_b_id,
                    'name': player_b_name,
                    'model_path': player_b_config.get('path', ''),
                    'agent_type': player_b_config.get('type', 'unknown'),
                    'starting_stack': starting_stack_b,
                    'hole_cards': [],
                    'is_dealer': False,
                    'is_small_blind': False,
                    'is_big_blind': False
                }
            ],
            'community_cards': {
                'flop': [],
                'turn': None,
                'river': None
            },
            'actions': [],
            'result': {}
        }
    
    def set_hole_cards(self, player_id: str, hole_cards: List):
        """Record hole cards for a player."""
        if not self.current_hand:
            return
        
        for p in self.current_hand['players']:
            if p['player_id'] == player_id:
                p['hole_cards'] = hole_cards
                break
    
    def set_player_positions(self, game_state: Dict):
        """Record player positions (dealer, SB, BB) from game state."""
        if not self.current_hand:
            return
        
        for game_player in game_state.get('players', []):
            player_id = game_player.get('id')
            for p in self.current_hand['players']:
                if p['player_id'] == player_id:
                    p['is_dealer'] = game_player.get('isDealer', False)
                    p['is_small_blind'] = game_player.get('isSmallBlind', False)
                    p['is_big_blind'] = game_player.get('isBigBlind', False)
                    break
    
    def set_community_cards(self, stage: str, cards: List):
        """Record community cards for a stage."""
        if not self.current_hand:
            return
        
        stage_lower = stage.lower()
        if stage_lower == 'flop' and cards:
            self.current_hand['community_cards']['flop'] = cards[:3]
        elif stage_lower == 'turn' and cards and len(cards) > 3:
            self.current_hand['community_cards']['turn'] = cards[3]
        elif stage_lower == 'river' and cards and len(cards) > 4:
            self.current_hand['community_cards']['river'] = cards[4]
    
    def log_action(
        self,
        player_id: str,
        player_name: str,
        action_type: str,
        action_label: str,
        amount: int,
        stage: str,
        pot: int,
        predictions: Optional[Dict[str, float]] = None
    ):
        """Log an action with optional model predictions."""
        if not self.current_hand:
            return
        
        action_data = {
            'player_id': player_id,
            'player_name': player_name,
            'action_type': action_type,
            'action_label': action_label,
            'amount': amount,
            'stage': stage,
            'pot_after': pot,
            'predictions': predictions or {}
        }
        
        self.current_hand['actions'].append(action_data)
    
    def end_hand(
        self,
        winner_id: str,
        winner_name: str,
        final_pot: int,
        profits: Dict[str, int]
    ):
        """Finalize and save the hand log."""
        if not self.current_hand:
            return
        
        self.current_hand['result'] = {
            'winner_id': winner_id,
            'winner_name': winner_name,
            'final_pot': final_pot,
            'profits': profits
        }
        
        # Save to file
        filename = f"{self.current_hand['hand_id']}.json"
        filepath = self.logs_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.current_hand, f, indent=2)
        except Exception as e:
            print(f"Failed to save hand log: {e}")
        
        self.current_hand = None
    
    def cancel_hand(self):
        """Cancel current hand logging (e.g., on error)."""
        self.current_hand = None


@dataclass
class EloRating:
    """Represents an ELO rating for a player."""
    rating: float = 2500.0  # Start at 2500 for Go/Chess-like scale (top ~4-5k)
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    rating_history: List[Tuple[int, float]] = field(default_factory=list)  # (game_num, rating)
    
    def __post_init__(self):
        if not self.rating_history:
            self.rating_history = [(0, self.rating)]


class EloCalculator:
    """
    Classic ELO rating calculator (Go/Chess-like scale).
    
    Uses a higher initial rating (2500) to allow top players to reach 4000-5000,
    similar to professional Go/Chess rating systems.
    
    The ELO system calculates expected scores based on rating differences,
    then updates ratings based on actual vs expected performance.
    """
    
    def __init__(self, k_factor: float = 40.0, initial_rating: float = 2500.0):
        """
        Initialize ELO calculator.
        
        Args:
            k_factor: Maximum rating change per game (32 is standard, 16 for established players)
            initial_rating: Starting rating for new players (1500 is standard)
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A against player B.
        
        Uses the formula: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        
        Args:
            rating_a: Player A's current rating
            rating_b: Player B's current rating
            
        Returns:
            Expected score (probability of winning) for player A, between 0 and 1
        """
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))
    
    def calculate_new_ratings(
        self, 
        rating_a: float, 
        rating_b: float, 
        score_a: float,
        k_factor_a: Optional[float] = None,
        k_factor_b: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate new ratings after a match.
        
        Args:
            rating_a: Player A's current rating
            rating_b: Player B's current rating  
            score_a: Actual score for player A (1=win, 0.5=draw, 0=loss)
            k_factor_a: Optional K-factor for player A (uses default if None)
            k_factor_b: Optional K-factor for player B (uses default if None)
            
        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        k_a = k_factor_a if k_factor_a is not None else self.k_factor
        k_b = k_factor_b if k_factor_b is not None else self.k_factor
        
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a
        
        score_b = 1.0 - score_a
        
        new_rating_a = rating_a + k_a * (score_a - expected_a)
        new_rating_b = rating_b + k_b * (score_b - expected_b)
        
        return new_rating_a, new_rating_b
    
    def get_rating_change(self, rating_a: float, rating_b: float, score_a: float) -> float:
        """
        Get the rating change for player A.
        
        Args:
            rating_a: Player A's current rating
            rating_b: Player B's current rating
            score_a: Actual score for player A (1=win, 0.5=draw, 0=loss)
            
        Returns:
            Rating change for player A (positive or negative)
        """
        expected_a = self.expected_score(rating_a, rating_b)
        return self.k_factor * (score_a - expected_a)


class PokerEloArena:
    """
    Arena for running poker matches and tracking ELO ratings.
    
    Plays poker hands between AI models/bots and updates their ELO ratings
    based on match outcomes. Uses freezeout format where players play until
    one runs out of chips (1000 chips each, 40 BB).
    
    Each match consists of multiple rounds (default 9) to reduce variance.
    The winner is determined by who wins the most rounds (best-of-N).
    """
    
    # Fixed match parameters for consistent evaluation
    STARTING_STACK = 1000  # 25 BB
    BIG_BLIND = 40
    SMALL_BLIND = 20
    MAX_HANDS = 200  # Safety limit (25 BB shouldn't last more than ~50-100 hands typically)
    ROUNDS_PER_MATCH = 51  # Best-of-51 rounds per match to reduce variance
    
    def __init__(
        self, 
        device: str = "cpu",
        k_factor: float = 40.0,
        initial_rating: float = 2500.0
    ):
        """
        Initialize the ELO Arena (Go/Chess-like scale).
        
        Args:
            device: Torch device for model inference
            k_factor: K-factor for ELO calculations (default 40 for faster differentiation)
            initial_rating: Starting ELO for new players (2500 for Go/Chess-like scale)
        """
        self.device = torch.device(device)
        self.elo_calc = EloCalculator(k_factor=k_factor, initial_rating=initial_rating)
        
        # Game config (standard heads-up) - matches starting stack for simplicity
        self.game_config = {
            'num_players': 2,
            'smallBlind': self.SMALL_BLIND,
            'bigBlind': self.BIG_BLIND,
            'startingChips': self.STARTING_STACK,
            'minPlayers': 2,
            'maxPlayers': 2
        }
        
        # Player ratings
        self.ratings: Dict[str, EloRating] = {}
        
        # Match history
        self.matches: List[Dict[str, Any]] = []
        
        # Cache for loaded models
        self.model_cache = {}
        
        # Hand logger for saving hands to disk
        self.hand_logger = HandLogger()
    
    def load_model(self, checkpoint_path: Path) -> "PokerActorCritic":
        """Load a model from a checkpoint."""
        path_str = str(checkpoint_path)
        if path_str in self.model_cache:
            return self.model_cache[path_str]
        
        try:
            agent = load_model_agent("dummy", "dummy", path_str, device=self.device)
            model = agent.model
            self.model_cache[path_str] = model
            return model
        except Exception as e:
            print(f"Error loading model {checkpoint_path}: {e}")
            raise
    
    def _call_api(self, history: List[Dict], starting_chips: Optional[int] = None) -> Dict:
        """Call poker API binding."""
        config = {
            **self.game_config,
            'seed': random.randint(0, 1000000)
        }
        # Override starting chips if provided
        if starting_chips is not None:
            config['startingChips'] = starting_chips
            
        payload = {
            'config': config,
            'history': history
        }
        
        try:
            payload_str = json.dumps(payload)
            response_str = poker_api_binding.process_request(payload_str)
            return json.loads(response_str)
        except Exception as e:
            print(f"API Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_or_create_rating(self, player_id: str) -> EloRating:
        """Get or create ELO rating for a player."""
        if player_id not in self.ratings:
            self.ratings[player_id] = EloRating(rating=self.elo_calc.initial_rating)
        return self.ratings[player_id]
    
    def _create_agent(self, player_id: str, config: Dict) -> Any:
        """Create an agent from config."""
        name = config.get('name', f"{config['type']}_{player_id}")
        if config['type'] == 'model':
            model = config.get('model')
            if model is None and 'path' in config:
                model = self.load_model(Path(config['path']))
            return load_model_agent(player_id, name, model=model, 
                                   device=self.device, deterministic=config.get('deterministic', False))
        elif config['type'] == 'heuristic':
            return HeuristicAgent(player_id, name)
        elif config['type'] == 'tight':
            return TightAgent(player_id, name)
        elif config['type'] == 'loose_passive':
            return LoosePassiveAgent(player_id, name)
        elif config['type'] == 'aggressive':
            return AggressiveAgent(player_id, name)
        elif config['type'] == 'calling_station':
            return CallingStationAgent(player_id, name)
        elif config['type'] == 'hero_caller':
            return HeroCallerAgent(player_id, name)
        elif config['type'] == 'always_raise':
            return AlwaysRaiseAgent(player_id, name)
        elif config['type'] == 'always_call':
            return AlwaysCallAgent(player_id, name)
        elif config['type'] == 'always_fold':
            return AlwaysFoldAgent(player_id, name)
        else:
            return RandomAgent(player_id, name)

    def play_hand(
        self, 
        agents: Dict,
        config_a: Optional[Dict] = None,
        config_b: Optional[Dict] = None,
        stack_p0: Optional[int] = None,
        stack_p1: Optional[int] = None
    ) -> Tuple[int, int, bool]:
        """
        Play a single hand between two agents.
        
        Args:
            agents: Dict mapping player_id to agent instance
            config_a: Optional config for player A (for logging)
            config_b: Optional config for player B (for logging)
            stack_p0: Starting stack for p0 (defaults to STARTING_STACK)
            stack_p1: Starting stack for p1 (defaults to STARTING_STACK)
            
        Returns:
            Tuple of (profit_p0, profit_p1, error)
        """
        agent_a = agents['p0']
        agent_b = agents['p1']
        
        # Use provided stacks or default
        actual_stack_p0 = stack_p0 if stack_p0 is not None else self.STARTING_STACK
        actual_stack_p1 = stack_p1 if stack_p1 is not None else self.STARTING_STACK
        
        # For heads-up, both players must have the same stack in the API
        # Use the minimum of both stacks (effective stack)
        effective_stack = min(actual_stack_p0, actual_stack_p1)
        
        # Start hand logging with actual individual stacks
        if config_a and config_b:
            self.hand_logger.start_hand(
                player_a_id='p0',
                player_b_id='p1',
                player_a_name=agent_a.name,
                player_b_name=agent_b.name,
                player_a_config=config_a,
                player_b_config=config_b,
                starting_stack_a=actual_stack_p0,
                starting_stack_b=actual_stack_p1
            )
        
        history = [
            {'type': 'addPlayer', 'playerId': 'p0', 'playerName': agent_a.name},
            {'type': 'addPlayer', 'playerId': 'p1', 'playerName': agent_b.name}
        ]
        
        response = self._call_api(history, starting_chips=effective_stack)
        if not response['success']:
            self.hand_logger.cancel_hand()
            return 0, 0, True
            
        game_state = response['gameState']
        raises_this_round = 0
        current_betting_stage = None
        last_logged_stage = None
        
        # Log hole cards and positions if logging this hand
        if self.hand_logger.current_hand:
            for p in game_state.get('players', []):
                self.hand_logger.set_hole_cards(p['id'], p.get('holeCards', []))
            self.hand_logger.set_player_positions(game_state)
        
        for _ in range(200):  # Max steps per hand
            stage = game_state.get('stage', '').lower()
            if stage in ['complete', 'showdown']:
                break
            
            if stage != current_betting_stage:
                current_betting_stage = stage
                raises_this_round = 0
            
            # Log community cards when stage changes
            if self.hand_logger.current_hand and stage != last_logged_stage:
                community_cards = game_state.get('communityCards', [])
                self.hand_logger.set_community_cards(stage, community_cards)
                last_logged_stage = stage
            
            current_player_id = game_state.get('currentPlayerId')
            if not current_player_id or current_player_id == 'none':
                break
                
            agent = agents[current_player_id]
            legal_actions = game_state.get('actionConstraints', {}).get('legalActions', [])
            if not legal_actions:
                break
            
            # Limit raises per betting round
            if raises_this_round >= 4:
                legal_actions = [a for a in legal_actions if a not in ['raise', 'bet']]
                if not legal_actions:
                    legal_actions = ['fold']
            
            state_dict = extract_state(game_state, current_player_id)
            
            # Get action with predictions if it's a ModelAgent and we're logging
            predictions = None
            if self.hand_logger.current_hand and isinstance(agent, ModelAgent):
                # ModelAgent supports return_probs to get probability distribution
                action_type, amount, action_label, predictions = agent.select_action(
                    state_dict, legal_actions, return_probs=True
                )
            else:
                action_type, amount, action_label = agent.select_action(state_dict, legal_actions)
            
            player_chips = state_dict.get('player_chips', 0)
            to_call = state_dict.get('to_call', 0)
            
            # Convert to all_in if needed
            original_action_type = action_type
            if action_type in ['bet', 'raise'] and amount >= player_chips:
                action_type, amount = 'all_in', 0
            elif action_type == 'call' and to_call >= player_chips > 0:
                action_type, amount = 'all_in', 0
            
            if original_action_type in ['raise', 'bet']:
                raises_this_round += 1
            
            # Calculate the pot from chip differences (API's pot field is unreliable)
            # Pot = total chips bet = effective_stack * num_players - sum(current_chips)
            def calc_pot_from_chips(gs, eff_stack):
                total_chips_remaining = sum(p.get('chips', eff_stack) for p in gs.get('players', []))
                return (eff_stack * 2) - total_chips_remaining
            
            pot_before_action = calc_pot_from_chips(game_state, effective_stack)
            
            # Store action info for logging after API call
            pending_action_log = None
            if self.hand_logger.current_hand:
                pending_action_log = {
                    'player_id': current_player_id,
                    'player_name': agent.name,
                    'action_type': action_type,
                    'action_label': action_label or action_type,
                    'amount': amount,
                    'stage': stage,
                    'predictions': predictions,
                    'pot_before': pot_before_action
                }
            
            history.append({
                'type': 'playerAction',
                'playerId': current_player_id,
                'action': action_type,
                'amount': amount
            })
            
            response = self._call_api(history, starting_chips=effective_stack)
            if not response['success']:
                self.hand_logger.cancel_hand()
                return 0, 0, True
            game_state = response['gameState']
            
            # Log action with pot value
            # Calculate pot after action from chip differences
            if pending_action_log:
                new_stage = game_state.get('stage', '').lower()
                
                # If the hand ended (chips distributed), use pot_before as it represents the pot won
                if new_stage in ['complete', 'showdown']:
                    pot_after = pending_action_log['pot_before']
                else:
                    # Calculate pot from chip differences after the action
                    pot_after = calc_pot_from_chips(game_state, effective_stack)
                
                self.hand_logger.log_action(
                    player_id=pending_action_log['player_id'],
                    player_name=pending_action_log['player_name'],
                    action_type=pending_action_log['action_type'],
                    action_label=pending_action_log['action_label'],
                    amount=pending_action_log['amount'],
                    stage=pending_action_log['stage'],
                    pot=pot_after,
                    predictions=pending_action_log['predictions']
                )
        
        # Calculate profit/loss in chips (based on effective stack used in this hand)
        profit_a = profit_b = 0
        for p in game_state['players']:
            profit = p['chips'] - effective_stack
            if p['id'] == 'p0':
                profit_a = profit
            else:
                profit_b = profit
        
        # End hand logging
        if self.hand_logger.current_hand:
            winner_id = 'p0' if profit_a > 0 else ('p1' if profit_b > 0 else None)
            winner_name = agent_a.name if profit_a > 0 else (agent_b.name if profit_b > 0 else 'Draw')
            # Calculate final pot from the max pot seen during actions
            # (game_state.pot is 0 after hand ends because chips are distributed)
            actions = self.hand_logger.current_hand.get('actions', [])
            final_pot = max((a.get('pot_after', 0) for a in actions), default=0) if actions else 0
            
            self.hand_logger.end_hand(
                winner_id=winner_id or '',
                winner_name=winner_name,
                final_pot=final_pot,
                profits={'p0': profit_a, 'p1': profit_b}
            )
            
        return profit_a, profit_b, False
    
    def play_freezeout_round(
        self, 
        agents: Dict,
        agents_swapped: Dict,
        config_a: Optional[Dict] = None,
        config_b: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Play a single freezeout round between two agents.
        
        Players start with 1000 chips (25 BB) and play until one runs out.
        
        Args:
            agents: Dict mapping player_id to agent instance (normal order)
            agents_swapped: Dict mapping player_id to agent instance (swapped order)
            config_a: Optional config for player A (for logging)
            config_b: Optional config for player B (for logging)
            
        Returns:
            Round result with winner info
        """
        # Initialize chip stacks (1000 chips = 25 BB)
        stack_a = self.STARTING_STACK
        stack_b = self.STARTING_STACK
        
        hands_played = 0
        hand_wins_a = 0
        hand_wins_b = 0
        
        # Play until one player is bust or max hands reached
        while stack_a > 0 and stack_b > 0 and hands_played < self.MAX_HANDS:
            # Alternate positions each hand
            swap = (hands_played % 2 == 1)
            
            try:
                if swap:
                    # B is p0 (button/SB), A is p1 (BB)
                    profit_p0, profit_p1, error = self.play_hand(
                        agents_swapped, 
                        config_a=config_b,  # B is p0
                        config_b=config_a,  # A is p1
                        stack_p0=stack_b,   # B's stack
                        stack_p1=stack_a    # A's stack
                    )
                    profit_a, profit_b = profit_p1, profit_p0
                else:
                    # A is p0 (button/SB), B is p1 (BB)
                    profit_a, profit_b, error = self.play_hand(
                        agents, 
                        config_a=config_a, 
                        config_b=config_b,
                        stack_p0=stack_a,   # A's stack
                        stack_p1=stack_b    # B's stack
                    )
                
                if error:
                    continue
                
                hands_played += 1
                stack_a += profit_a
                stack_b += profit_b
                
                if profit_a > 0:
                    hand_wins_a += 1
                if profit_b > 0:
                    hand_wins_b += 1
                    
            except Exception as e:
                print(f"Error in hand: {e}")
                continue
        
        if hands_played == 0:
            return {'error': True}
        
        # Determine round outcome
        # Winner is the player with chips remaining (or more chips if max hands reached)
        if stack_a > stack_b:
            winner = 'a'
        elif stack_b > stack_a:
            winner = 'b'
        else:
            winner = 'draw'
        
        return {
            'error': False,
            'hands_played': hands_played,
            'final_stack_a': stack_a,
            'final_stack_b': stack_b,
            'hand_wins_a': hand_wins_a,
            'hand_wins_b': hand_wins_b,
            'winner': winner
        }

    def play_match(
        self, 
        player_a_id: str, 
        player_b_id: str,
        config_a: Dict,
        config_b: Dict
    ) -> Dict[str, Any]:
        """
        Play a best-of-N match between two players and update ELO ratings.
        
        Each match consists of ROUNDS_PER_MATCH (default 51) freezeout rounds.
        The winner is whoever wins the most rounds (26+ out of 51).
        ELO is only updated after all rounds are complete.
        
        Args:
            player_a_id: Unique identifier for player A
            player_b_id: Unique identifier for player B
            config_a: Agent configuration for player A
            config_b: Agent configuration for player B
            
        Returns:
            Match result with ELO changes
        """
        # Pre-load models (uses cache)
        if config_a.get('type') == 'model' and 'model' not in config_a and 'path' in config_a:
            config_a = {**config_a, 'model': self.load_model(Path(config_a['path']))}
            
        if config_b.get('type') == 'model' and 'model' not in config_b and 'path' in config_b:
            config_b = {**config_b, 'model': self.load_model(Path(config_b['path']))}
        
        # Create agents once (reuse across all rounds)
        agent_a = self._create_agent('p0', config_a)
        agent_b = self._create_agent('p1', config_b)
        agents = {'p0': agent_a, 'p1': agent_b}
        agents_swapped = {'p0': agent_b, 'p1': agent_a}
        
        # Track results across all rounds
        round_wins_a = 0
        round_wins_b = 0
        total_hands_played = 0
        total_hand_wins_a = 0
        total_hand_wins_b = 0
        round_results = []
        
        # Play ROUNDS_PER_MATCH rounds (best-of-9 by default)
        for round_num in range(self.ROUNDS_PER_MATCH):
            result = self.play_freezeout_round(
                agents, 
                agents_swapped,
                config_a=config_a,
                config_b=config_b
            )
            
            if result.get('error'):
                continue
            
            round_results.append(result)
            total_hands_played += result['hands_played']
            total_hand_wins_a += result['hand_wins_a']
            total_hand_wins_b += result['hand_wins_b']
            
            if result['winner'] == 'a':
                round_wins_a += 1
            elif result['winner'] == 'b':
                round_wins_b += 1
            # draws don't count toward either player's round wins
        
        if len(round_results) == 0:
            return {'error': True}
        
        # Determine match outcome for ELO based on round wins
        if round_wins_a > round_wins_b:
            score_a = 1.0  # A wins the match
        elif round_wins_b > round_wins_a:
            score_a = 0.0  # B wins the match
        else:
            score_a = 0.5  # Draw (equal round wins)
        
        # Get current ratings
        rating_a = self.get_or_create_rating(player_a_id)
        rating_b = self.get_or_create_rating(player_b_id)
        
        old_rating_a = rating_a.rating
        old_rating_b = rating_b.rating
        
        # Calculate new ratings
        new_rating_a, new_rating_b = self.elo_calc.calculate_new_ratings(
            old_rating_a, old_rating_b, score_a
        )
        
        # Update ratings
        rating_a.rating = new_rating_a
        rating_b.rating = new_rating_b
        rating_a.games_played += 1
        rating_b.games_played += 1
        
        match_num = len(self.matches) + 1
        rating_a.rating_history.append((match_num, new_rating_a))
        rating_b.rating_history.append((match_num, new_rating_b))
        
        if score_a == 1.0:
            rating_a.wins += 1
            rating_b.losses += 1
        elif score_a == 0.0:
            rating_a.losses += 1
            rating_b.wins += 1
        else:
            rating_a.draws += 1
            rating_b.draws += 1
        
        # Record match
        match_result = {
            'match_num': match_num,
            'player_a': player_a_id,
            'player_b': player_b_id,
            'rounds_played': len(round_results),
            'round_wins_a': round_wins_a,
            'round_wins_b': round_wins_b,
            'hands_played': total_hands_played,
            'hand_wins_a': total_hand_wins_a,
            'hand_wins_b': total_hand_wins_b,
            'score_a': score_a,
            'old_rating_a': old_rating_a,
            'old_rating_b': old_rating_b,
            'new_rating_a': new_rating_a,
            'new_rating_b': new_rating_b,
            'rating_change_a': new_rating_a - old_rating_a,
            'rating_change_b': new_rating_b - old_rating_b,
        }
        
        self.matches.append(match_result)
        
        return match_result
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get current leaderboard sorted by ELO rating."""
        leaderboard = []
        for player_id, rating in self.ratings.items():
            leaderboard.append({
                'player_id': player_id,
                'rating': rating.rating,
                'games_played': rating.games_played,
                'wins': rating.wins,
                'losses': rating.losses,
                'draws': rating.draws,
                'win_rate': rating.wins / rating.games_played if rating.games_played > 0 else 0
            })
        
        return sorted(leaderboard, key=lambda x: x['rating'], reverse=True)


def parse_checkpoints(directory: Path) -> List[Tuple[int, Path]]:
    """Find and sort checkpoints by iteration number.
    
    Returns:
        List of (iteration_number, path) tuples sorted by iteration.
        Baseline model uses iteration -1 to distinguish from actual iterations.
    """
    checkpoints = []
    pattern = re.compile(r"poker_rl_iter_(\d+)\.pt")
    
    if not directory.exists():
        return []
        
    for f in directory.glob("*.pt"):
        if f.name == "poker_rl_baseline.pt":
            # Use -1 for baseline to distinguish from actual iteration 0
            checkpoints.append((-1, f))
            continue
            
        match = pattern.match(f.name)
        if match:
            iteration = int(match.group(1))
            checkpoints.append((iteration, f))
            
    return sorted(checkpoints, key=lambda x: x[0])


def select_spread_checkpoints(
    checkpoints: List[Tuple[int, Path]], 
    max_checkpoints: int
) -> List[Tuple[int, Path]]:
    """
    Select checkpoints with maximum coverage across all iterations.
    
    Uses farthest-first (maximin distance) greedy selection with logarithmic
    scaling of iteration numbers. This gives more weight to early iterations
    where learning changes are typically most dramatic, while still maintaining
    good coverage of later training.
    
    Strategy:
    - Always keep first checkpoint (baseline/-1 or earliest iteration)
    - Always keep last checkpoint (most recent)
    - Use log-scaled iteration numbers for distance calculation
    - Greedily select remaining checkpoints to maximize minimum distance
      to already-selected checkpoints (farthest-first traversal)
    
    Args:
        checkpoints: List of (iteration_number, path) tuples, sorted by iteration
        max_checkpoints: Maximum number of checkpoints to select
        
    Returns:
        List of (iteration_number, path) tuples with maximized spread
    """
    n = len(checkpoints)
    if n <= max_checkpoints:
        return checkpoints
    
    # Extract iteration numbers and apply log scaling for distance calculation
    # Add offset to handle baseline (-1) and iter_0 cases
    iterations = [cp[0] for cp in checkpoints]
    min_iter = min(iterations)
    offset = 2 - min_iter  # Ensure all values are >= 2 for log scaling
    
    def log_scale(iter_num: int) -> float:
        """Apply log scaling to iteration number."""
        return math.log(iter_num + offset)
    
    # Pre-compute log-scaled values for all checkpoints
    log_iterations = [log_scale(it) for it in iterations]
    
    # Start with first and last checkpoints (always included)
    selected_indices = [0, n - 1]
    selected_log_values = {log_iterations[0], log_iterations[n - 1]}
    
    # Greedily select remaining checkpoints using farthest-first on log scale
    remaining_slots = max_checkpoints - 2
    available_indices = set(range(1, n - 1))
    
    for _ in range(remaining_slots):
        if not available_indices:
            break
        
        best_idx = None
        best_min_dist = -1.0
        
        # Find the checkpoint that maximizes minimum log-distance to selected set
        for idx in available_indices:
            log_val = log_iterations[idx]
            # Calculate minimum distance to any already-selected checkpoint (in log space)
            min_dist = min(abs(log_val - sel_log) for sel_log in selected_log_values)
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            selected_log_values.add(log_iterations[best_idx])
            available_indices.remove(best_idx)
    
    # Sort and return selected checkpoints
    sorted_indices = sorted(selected_indices)
    return [checkpoints[i] for i in sorted_indices]

