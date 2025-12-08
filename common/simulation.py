#!/usr/bin/env python3
"""
Poker Game Simulation Engine

This module provides the core simulation logic for playing poker hands,
shared between the training system and ELO evaluation system.

Classes:
- PokerSimulator: Core simulation engine using C++ poker_api_binding
- DirectGameSimulator: Direct C++ Game binding for high-performance training
- GameConfig: Configuration for poker games
- HandLogger: Logger for recording poker hands with full action history
"""

import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

# Import poker_api_binding (native extension).
# Use the robust loader so console-script entrypoints can still find local builds.
from .binding_loader import load_poker_api_binding

poker_api_binding = load_poker_api_binding()
_BINDING_AVAILABLE = poker_api_binding is not None

from .model_agent import extract_state, convert_action_label

if TYPE_CHECKING:
    from .model_agent import ModelAgent


@dataclass
class GameConfig:
    """Configuration for a poker game."""
    num_players: int = 2
    small_blind: int = 10
    big_blind: int = 20
    starting_chips: int = 1000
    min_players: int = 2
    max_players: int = 2
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            'numPlayers': self.num_players,
            'smallBlind': self.small_blind,
            'bigBlind': self.big_blind,
            'startingChips': self.starting_chips,
            'minPlayers': self.min_players,
            'maxPlayers': self.max_players,
            'seed': self.seed if self.seed is not None else random.randint(0, 1000000)
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'GameConfig':
        """
        Create a GameConfig from a dictionary.
        
        Supports both snake_case (small_blind) and camelCase (smallBlind) keys.
        """
        return cls(
            num_players=config.get('num_players', config.get('numPlayers', 2)),
            small_blind=config.get('small_blind', config.get('smallBlind', 10)),
            big_blind=config.get('big_blind', config.get('bigBlind', 20)),
            starting_chips=config.get('starting_chips', config.get('startingChips', 1000)),
            min_players=config.get('min_players', config.get('minPlayers', 2)),
            max_players=config.get('max_players', config.get('maxPlayers', 2)),
            seed=config.get('seed'),
        )
    
    def to_binding_config(self, seed: Optional[int] = None) -> 'poker_api_binding.GameConfig':
        """
        Convert to C++ poker_api_binding.GameConfig.
        
        This is the canonical way to create a C++ config from the Python config.
        Used by DirectGameSimulator and VecPokerEnv for consistent behavior.
        
        Args:
            seed: Optional seed override (uses random if None and self.seed is None)
            
        Returns:
            poker_api_binding.GameConfig instance
        """
        if not _BINDING_AVAILABLE:
            raise ImportError(
                "poker_api_binding not found. "
                "Please compile the binding with: cd api && make module"
            )
        
        gc = poker_api_binding.GameConfig()
        gc.smallBlind = self.small_blind
        gc.bigBlind = self.big_blind
        gc.startingChips = self.starting_chips
        gc.minPlayers = self.min_players
        gc.maxPlayers = self.max_players
        
        # Use provided seed, or self.seed, or generate random
        if seed is not None:
            gc.seed = seed
        elif self.seed is not None:
            gc.seed = self.seed
        else:
            gc.seed = random.randint(0, 1000000)
        
        return gc


# Default hand logging configuration
DEFAULT_HAND_LOGS_DIR = Path('/tmp/pokersim/hand_logs')
DEFAULT_HAND_LOG_FREQUENCY = 100  # Log 1 out of every N hands


class HandLogger:
    """
    Logger for recording poker hands with full action history and model predictions.
    Saves hands to disk in JSON format for analysis with the hand-viewer.
    
    This class is shared between training and ELO evaluation systems.
    """
    
    def __init__(
        self, 
        logs_dir: Optional[Path] = None, 
        log_frequency: int = DEFAULT_HAND_LOG_FREQUENCY
    ):
        """
        Initialize the hand logger.
        
        Args:
            logs_dir: Directory for saving hand logs (default: /tmp/pokersim/hand_logs)
            log_frequency: Log 1 out of every N hands (default: 100)
        """
        self.logs_dir = logs_dir or DEFAULT_HAND_LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_frequency = log_frequency
        self.hand_counter = 0
        self.current_hand: Optional[Dict] = None
    
    def should_log(self) -> bool:
        """Check if the current hand should be logged (1 in log_frequency)."""
        return self.hand_counter % self.log_frequency == 0
    
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


class PokerSimulator:
    """
    Core poker game simulation engine.
    
    This class provides methods to simulate poker hands using the C++ poker API.
    It abstracts away the details of API communication and provides a clean
    interface for agents to interact with the game.
    
    Features:
    - Stateless API-based simulation (history replay)
    - Support for multiple players (heads-up to full ring)
    - Handles action conversion and legal action masking
    - Raise limiting per betting round
    """
    
    MAX_RAISES_PER_ROUND = 4
    TERMINAL_STAGES = {'complete', 'showdown'}
    
    def __init__(self, config: Optional[GameConfig] = None):
        """
        Initialize the simulator.
        
        Args:
            config: Game configuration (uses defaults if not provided)
        """
        if not _BINDING_AVAILABLE:
            raise ImportError(
                "poker_api_binding not found. "
                "Please compile the binding with: cd api && make module"
            )
        
        self.config = config or GameConfig()
    
    def _call_api(
        self, 
        history: List[Dict], 
        config_override: Optional[Dict] = None
    ) -> Dict:
        """
        Call the poker API with the given history.
        
        Args:
            history: List of game events (addPlayer, playerAction)
            config_override: Optional config values to override
            
        Returns:
            API response dictionary
        """
        config = self.config.to_dict()
        if config_override:
            config.update(config_override)
        
        payload = {
            'config': config,
            'history': history
        }
        
        try:
            payload_str = json.dumps(payload)
            response_str = poker_api_binding.process_request(payload_str)
            return json.loads(response_str)
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def play_hand(
        self,
        agents: Dict[str, Any],
        starting_chips: Optional[Dict[str, int]] = None,
        on_action: Optional[callable] = None,
        max_steps: int = 200,
        hand_logger: Optional['HandLogger'] = None,
        agent_configs: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """
        Play a single poker hand with the given agents.
        
        Args:
            agents: Dict mapping player_id to agent instance.
                   Agents must have:
                   - name: str
                   - select_action(state, legal_actions) -> (action_type, amount, action_label)
                   - reset_hand() (optional)
                   - observe_action(player_id, action_type, amount, pot, stage) (optional)
            starting_chips: Optional dict mapping player_id to starting chips
                           (defaults to config.starting_chips for all)
            on_action: Optional callback called after each action:
                      on_action(player_id, action_type, amount, action_label, game_state)
            max_steps: Maximum number of actions before terminating
            hand_logger: Optional HandLogger for recording hands to disk
            agent_configs: Optional dict mapping player_id to agent config (for hand logging)
            
        Returns:
            Dict with:
            - success: bool
            - error: Optional error message
            - profits: Dict mapping player_id to chip profit/loss
            - hands_played: 1 if successful
            - final_state: Final game state
        """
        player_ids = list(agents.keys())
        
        # Determine effective stack for this hand
        if starting_chips:
            effective_stack = min(starting_chips.values())
            actual_stacks = starting_chips
        else:
            effective_stack = self.config.starting_chips
            actual_stacks = {pid: self.config.starting_chips for pid in player_ids}
        
        # Reset agents for new hand
        for agent in agents.values():
            if hasattr(agent, 'reset_hand'):
                agent.reset_hand()
        
        # Build initial history
        history = []
        for player_id, agent in agents.items():
            history.append({
                'type': 'addPlayer',
                'playerId': player_id,
                'playerName': agent.name
            })
        
        # Get initial game state
        #
        # IMPORTANT (correctness):
        # PokerSimulator uses the *stateless* API by repeatedly replaying `history`.
        # Therefore the RNG seed must be **stable across all API calls for this hand**.
        # If the seed changes between calls, the same history could replay with a
        # different deck, producing invalid transitions and biased eval results.
        config_override = {'startingChips': effective_stack}
        if 'seed' not in config_override or config_override.get('seed') is None:
            # Use a single seed for the entire hand.
            config_override['seed'] = self.config.seed if self.config.seed is not None else random.randint(0, 1_000_000)
        response = self._call_api(history, config_override)
        
        if not response.get('success'):
            if hand_logger:
                hand_logger.cancel_hand()
            return {
                'success': False,
                'error': response.get('error', 'Unknown error'),
                'profits': {pid: 0 for pid in player_ids},
                'hands_played': 0,
                'final_state': None
            }
        
        game_state = response['gameState']
        raises_this_round = 0
        current_betting_stage = None
        last_logged_stage = None
        
        # Start hand logging if enabled
        if hand_logger and len(player_ids) >= 2:
            p0, p1 = player_ids[0], player_ids[1]
            config_a = agent_configs.get(p0, {'type': 'unknown'}) if agent_configs else {'type': 'unknown'}
            config_b = agent_configs.get(p1, {'type': 'unknown'}) if agent_configs else {'type': 'unknown'}
            hand_logger.start_hand(
                player_a_id=p0,
                player_b_id=p1,
                player_a_name=agents[p0].name,
                player_b_name=agents[p1].name,
                player_a_config=config_a,
                player_b_config=config_b,
                starting_stack_a=actual_stacks.get(p0, effective_stack),
                starting_stack_b=actual_stacks.get(p1, effective_stack)
            )
            # Log hole cards and positions
            if hand_logger.current_hand:
                for p in game_state.get('players', []):
                    hand_logger.set_hole_cards(p['id'], p.get('holeCards', []))
                hand_logger.set_player_positions(game_state)
        
        # Helper to calculate pot from chip differences
        def calc_pot_from_chips(gs: Dict, eff_stack: int) -> int:
            total_chips_remaining = sum(p.get('chips', eff_stack) for p in gs.get('players', []))
            return (eff_stack * len(gs.get('players', []))) - total_chips_remaining
        
        # Main game loop
        for step in range(max_steps):
            stage = game_state.get('stage', '').lower()
            
            if stage in self.TERMINAL_STAGES:
                break
            
            # Track betting rounds for raise limiting
            if stage != current_betting_stage:
                current_betting_stage = stage
                raises_this_round = 0
            
            # Log community cards when stage changes
            if hand_logger and hand_logger.current_hand and stage != last_logged_stage:
                community_cards = game_state.get('communityCards', [])
                hand_logger.set_community_cards(stage, community_cards)
                last_logged_stage = stage
            
            # Get current player
            current_player_id = game_state.get('currentPlayerId')
            if not current_player_id or current_player_id == 'none':
                break
            
            agent = agents.get(current_player_id)
            if agent is None:
                break
            
            # Get legal actions
            legal_actions = game_state.get('actionConstraints', {}).get('legalActions', [])
            if not legal_actions:
                break
            
            # Limit raises per betting round
            if raises_this_round >= self.MAX_RAISES_PER_ROUND:
                legal_actions = [a for a in legal_actions if a not in ['raise', 'bet']]
                if not legal_actions:
                    legal_actions = ['fold']
            
            # Extract state for agent
            state_dict = extract_state(game_state, current_player_id)
            
            # Agent selects action - check if we need predictions for logging
            predictions = None
            if hand_logger and hand_logger.current_hand and hasattr(agent, 'select_action'):
                # Try to get predictions if agent is a ModelAgent
                try:
                    result = agent.select_action(state_dict, legal_actions, return_probs=True)
                    if len(result) == 4:
                        action_type, amount, action_label, predictions = result
                    else:
                        action_type, amount, action_label = result[:3]
                except TypeError:
                    # Agent doesn't support return_probs
                    result = agent.select_action(state_dict, legal_actions)
                    if len(result) >= 3:
                        action_type, amount, action_label = result[:3]
                    else:
                        action_type, amount = result[:2]
                        action_label = action_type
            else:
                result = agent.select_action(state_dict, legal_actions)
                if len(result) == 3:
                    action_type, amount, action_label = result
                elif len(result) == 4:
                    action_type, amount, action_label, _ = result
                else:
                    action_type, amount = result
                    action_label = action_type
            
            # Convert to all_in if needed
            player_chips = state_dict.get('player_chips', 0)
            to_call = state_dict.get('to_call', 0)
            
            original_action_type = action_type
            if action_type in ['bet', 'raise'] and amount >= player_chips:
                action_type, amount = 'all_in', 0
            elif action_type == 'call' and to_call >= player_chips > 0:
                action_type, amount = 'all_in', 0
            
            if original_action_type in ['raise', 'bet']:
                raises_this_round += 1
            
            # Notify agents of action (for opponent modeling)
            pot = state_dict.get('pot', 0)
            for other_id, other_agent in agents.items():
                if hasattr(other_agent, 'observe_action'):
                    other_agent.observe_action(
                        current_player_id, action_type, amount, pot, stage
                    )
            
            # Calculate pot before action for logging
            pot_before_action = calc_pot_from_chips(game_state, effective_stack)
            
            # Call action callback if provided
            if on_action:
                on_action(current_player_id, action_type, amount, action_label, game_state)
            
            # Apply action
            history.append({
                'type': 'playerAction',
                'playerId': current_player_id,
                'action': action_type,
                'amount': amount
            })
            
            response = self._call_api(history, config_override)
            if not response.get('success'):
                if hand_logger:
                    hand_logger.cancel_hand()
                return {
                    'success': False,
                    'error': response.get('error', 'Unknown error'),
                    'profits': {pid: 0 for pid in player_ids},
                    'hands_played': 0,
                    'final_state': None
                }
            
            game_state = response['gameState']
            
            # Log action with pot value
            if hand_logger and hand_logger.current_hand:
                new_stage = game_state.get('stage', '').lower()
                if new_stage in self.TERMINAL_STAGES:
                    pot_after = pot_before_action
                else:
                    pot_after = calc_pot_from_chips(game_state, effective_stack)
                
                hand_logger.log_action(
                    player_id=current_player_id,
                    player_name=agent.name,
                    action_type=action_type,
                    action_label=action_label or action_type,
                    amount=amount,
                    stage=stage,
                    pot=pot_after,
                    predictions=predictions
                )
        
        # Calculate profits
        profits = {}
        for player in game_state.get('players', []):
            player_id = player.get('id')
            if player_id:
                chips = player.get('chips', 0)
                profits[player_id] = chips - effective_stack
        
        # End hand logging
        if hand_logger and hand_logger.current_hand:
            # Determine winner
            winner_id = None
            winner_name = 'Draw'
            for pid, profit in profits.items():
                if profit > 0:
                    winner_id = pid
                    winner_name = agents[pid].name
                    break
            
            # Calculate final pot from actions
            actions = hand_logger.current_hand.get('actions', [])
            final_pot = max((a.get('pot_after', 0) for a in actions), default=0) if actions else 0
            
            hand_logger.end_hand(
                winner_id=winner_id or '',
                winner_name=winner_name,
                final_pot=final_pot,
                profits=profits
            )
        
        return {
            'success': True,
            'error': None,
            'profits': profits,
            'hands_played': 1,
            'final_state': game_state
        }
    
    def play_match(
        self,
        agents: Dict[str, Any],
        num_hands: int,
        alternate_positions: bool = True,
        on_hand_complete: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Play a match of multiple hands between agents.
        
        Args:
            agents: Dict mapping player_id to agent instance
            num_hands: Number of hands to play
            alternate_positions: If True, swap positions each hand (heads-up)
            on_hand_complete: Optional callback after each hand:
                             on_hand_complete(hand_num, result)
            
        Returns:
            Dict with:
            - success: bool
            - hands_played: int
            - total_profits: Dict mapping player_id to total chips won/lost
            - hand_wins: Dict mapping player_id to number of hands won
            - errors: int (number of errored hands)
        """
        player_ids = list(agents.keys())
        
        total_profits = {pid: 0 for pid in player_ids}
        hand_wins = {pid: 0 for pid in player_ids}
        hands_played = 0
        errors = 0
        
        # Create position-swapped agent dict for heads-up alternation
        if alternate_positions and len(player_ids) == 2:
            pid_a, pid_b = player_ids
            agents_normal = agents
            agents_swapped = {pid_a: agents[pid_b], pid_b: agents[pid_a]}
        else:
            agents_normal = agents
            agents_swapped = agents  # No swap for non-heads-up
        
        for hand_num in range(num_hands):
            # Alternate positions in heads-up
            swap = alternate_positions and (hand_num % 2 == 1) and len(player_ids) == 2
            current_agents = agents_swapped if swap else agents_normal
            
            # Map back to original player IDs for profit tracking
            if swap and len(player_ids) == 2:
                pid_a, pid_b = player_ids
                id_map = {'p0': pid_b, 'p1': pid_a}  # Swapped
            else:
                id_map = {pid: pid for pid in player_ids}
            
            result = self.play_hand(current_agents)
            
            if not result.get('success'):
                errors += 1
                continue
            
            hands_played += 1
            
            # Update profits (map back to original IDs if swapped)
            for api_id, profit in result.get('profits', {}).items():
                original_id = id_map.get(api_id, api_id)
                total_profits[original_id] = total_profits.get(original_id, 0) + profit
                
                if profit > 0:
                    hand_wins[original_id] = hand_wins.get(original_id, 0) + 1
            
            if on_hand_complete:
                on_hand_complete(hand_num, result)
        
        return {
            'success': True,
            'hands_played': hands_played,
            'total_profits': total_profits,
            'hand_wins': hand_wins,
            'errors': errors
        }


class DirectGameSimulator:
    """
    Direct poker simulation using C++ Game class.
    
    Unlike PokerSimulator which uses the stateless API, this class
    uses the direct C++ Game binding for better performance in
    vectorized training environments.
    
    This is the preferred simulator for high-performance training.
    """
    
    def __init__(self, config: Optional[Union[GameConfig, Dict[str, Any]]] = None):
        """
        Initialize the simulator.
        
        Args:
            config: Game configuration - either GameConfig instance or dict
                   (uses defaults if not provided)
        """
        if not _BINDING_AVAILABLE:
            raise ImportError(
                "poker_api_binding not found. "
                "Please compile the binding with: cd api && make module"
            )
        
        # Accept either GameConfig or dict
        if config is None:
            self.config = GameConfig()
        elif isinstance(config, dict):
            self.config = GameConfig.from_dict(config)
        else:
            self.config = config
        
        self._game_config = self.config.to_binding_config()
    
    def create_game(self, seed: Optional[int] = None) -> 'poker_api_binding.Game':
        """
        Create a new poker game instance.
        
        Args:
            seed: Optional random seed for the game
            
        Returns:
            poker_api_binding.Game instance
        """
        gc = self.config.to_binding_config(seed=seed)
        return poker_api_binding.Game(gc)
    
    def setup_players(
        self,
        game: 'poker_api_binding.Game',
        player_ids: List[str],
        player_names: Optional[List[str]] = None,
        starting_chips: Optional[int] = None
    ):
        """
        Add players to the game.
        
        Args:
            game: Game instance
            player_ids: List of player IDs
            player_names: Optional list of player names (defaults to Player 0, 1, ...)
            starting_chips: Starting chip count (defaults to config value)
        """
        chips = starting_chips or self.config.starting_chips
        
        for i, pid in enumerate(player_ids):
            name = player_names[i] if player_names and i < len(player_names) else f"Player {i}"
            game.add_player(pid, name, chips)
    
    def get_state_dict(self, game: 'poker_api_binding.Game') -> Dict[str, Any]:
        """
        Get the current game state as a dictionary.
        
        Args:
            game: Game instance
            
        Returns:
            Game state dictionary
        """
        return game.get_state_dict()
    
    def process_action(
        self,
        game: 'poker_api_binding.Game',
        player_id: str,
        action_type: str,
        amount: int = 0
    ) -> bool:
        """
        Process a player action.
        
        Args:
            game: Game instance
            player_id: Player ID taking the action
            action_type: Action type (fold, check, call, bet, raise, all_in)
            amount: Bet/raise amount (0 for fold/check/call/all_in)
            
        Returns:
            True if action was processed successfully
        """
        return game.process_action(player_id, action_type, amount)
    
    def advance_game(self, game: 'poker_api_binding.Game') -> bool:
        """
        Advance the game state (deal cards, etc.).
        
        Args:
            game: Game instance
            
        Returns:
            True if game advanced successfully
        """
        return game.advance_game()
    
    def start_hand(self, game: 'poker_api_binding.Game'):
        """
        Start a new hand.
        
        Args:
            game: Game instance
        """
        game.start_hand()
    
    def is_complete(self, game: 'poker_api_binding.Game') -> bool:
        """
        Check if the current hand is complete.
        
        Args:
            game: Game instance
            
        Returns:
            True if hand is complete
        """
        return game.get_stage_name() == "Complete"
    
    def get_current_player_id(self, game: 'poker_api_binding.Game') -> Optional[str]:
        """
        Get the current player's ID.
        
        Args:
            game: Game instance
            
        Returns:
            Player ID or None if no player to act
        """
        pid = game.get_current_player_id()
        return pid if pid else None


def play_hand_direct(
    agents: Dict[str, Any],
    config: Optional[Union[GameConfig, Dict[str, Any]]] = None,
    starting_chips: Optional[Union[int, Dict[str, int]]] = None,
    seed: Optional[int] = None,
    max_steps: int = 500,
) -> Dict[str, Any]:
    """
    Play a single hand using the *direct* C++ Game binding.

    This is the correct way to simulate hands when you need **per-player**
    starting stacks (e.g. freezeout / tournament-style matches). The stateless
    API-based `PokerSimulator.play_hand()` only supports a single global
    `startingChips`, so it cannot represent unequal stacks faithfully.

    Args:
        agents: Dict[player_id -> agent], where agent.select_action returns
                (action_type, amount, action_label) or (action_type, amount).
        config: GameConfig or dict (smallBlind/bigBlind/startingChips...). Defaults to GameConfig().
        starting_chips: Either a single int (same for all players) or a dict of per-player stacks.
        seed: Optional RNG seed for the C++ game.
        max_steps: Safety cap on actions.

    Returns:
        Dict with keys: success, profits, hands_played, final_state, error (optional).
    """
    if not _BINDING_AVAILABLE:
        return {"success": False, "error": "poker_api_binding not available", "hands_played": 0, "profits": {}}

    cfg = GameConfig.from_dict(config) if isinstance(config, dict) else (config or GameConfig())
    gc = cfg.to_binding_config(seed=seed)
    game = poker_api_binding.Game(gc)

    player_ids = list(agents.keys())
    # Resolve per-player starting stacks.
    if isinstance(starting_chips, dict):
        stacks = {pid: int(starting_chips.get(pid, cfg.starting_chips)) for pid in player_ids}
    else:
        chips = int(starting_chips or cfg.starting_chips)
        stacks = {pid: chips for pid in player_ids}

    # Reset agents for new hand.
    for agent in agents.values():
        if hasattr(agent, "reset_hand"):
            agent.reset_hand()

    for pid in player_ids:
        agent = agents[pid]
        game.add_player(pid, getattr(agent, "name", pid), stacks[pid])

    game.start_hand()

    for _ in range(max_steps):
        stage = game.get_stage_name().lower()
        if stage in {"complete", "showdown"}:
            break

        pid = game.get_current_player_id()
        if not pid:
            if not game.advance_game():
                break
            continue

        gs = game.get_state_dict()
        legal_actions = gs.get("actionConstraints", {}).get("legalActions", [])
        if not legal_actions:
            break

        agent = agents.get(pid)
        if agent is None:
            break

        state_dict = extract_state(gs, pid)

        try:
            result = agent.select_action(state_dict, legal_actions)
            if isinstance(result, tuple) and len(result) >= 3:
                action_type, amount, _ = result[:3]
            else:
                action_type, amount = result[:2]
        except Exception:
            action_type, amount = "fold", 0

        # Clamp to all-in if agent oversizes.
        player_chips = state_dict.get("player_chips", 0)
        to_call = state_dict.get("to_call", 0)
        if action_type in ("bet", "raise") and amount >= player_chips > 0:
            action_type, amount = "all_in", 0
        elif action_type == "call" and to_call >= player_chips > 0:
            action_type, amount = "all_in", 0

        # Notify agents (opponent modeling hooks).
        pot = state_dict.get("pot", 0)
        for other_agent in agents.values():
            if hasattr(other_agent, "observe_action"):
                try:
                    other_agent.observe_action(pid, action_type, amount, pot, stage)
                except Exception:
                    pass

        ok = game.process_action(pid, action_type, int(amount))
        if not ok:
            # Best-effort fallback to fold.
            try:
                game.process_action(pid, "fold", 0)
            except Exception:
                break

    final_state = game.get_state_dict()
    profits: Dict[str, int] = {}
    for p in final_state.get("players", []):
        pid = p.get("id")
        if pid in stacks:
            profits[pid] = int(p.get("chips", stacks[pid])) - int(stacks[pid])

    return {"success": True, "hands_played": 1, "profits": profits, "final_state": final_state}


def check_binding_available() -> bool:
    """Check if poker_api_binding is available."""
    return _BINDING_AVAILABLE


def call_poker_api(
    config: Union[GameConfig, Dict[str, Any]],
    history: List[Dict],
    seed: Optional[int] = None
) -> Dict:
    """
    Standalone helper to call the poker API with configuration and history.
    
    This is the canonical way to call the poker API. Use this function
    instead of directly calling poker_api_binding.process_request() to ensure
    consistent behavior across the codebase.
    
    Args:
        config: Game configuration - either GameConfig or dict
        history: List of game events (addPlayer, playerAction)
        seed: Optional seed for the game (uses random if not provided)
        
    Returns:
        API response dictionary with 'success', 'gameState', etc.
        
    Example:
        >>> from common import call_poker_api, GameConfig
        >>> cfg = GameConfig(small_blind=10, big_blind=20)
        >>> history = [{'type': 'addPlayer', 'playerId': 'p0', 'playerName': 'Player 0'}]
        >>> response = call_poker_api(cfg, history)
    """
    if not _BINDING_AVAILABLE:
        return {'success': False, 'error': 'poker_api_binding not available'}
    
    # Convert config to dict if needed
    if isinstance(config, GameConfig):
        config_dict = config.to_dict()
    else:
        config_dict = config.copy()
    
    # Set seed if provided, otherwise ensure one exists
    if seed is not None:
        config_dict['seed'] = seed
    elif 'seed' not in config_dict or config_dict['seed'] is None:
        config_dict['seed'] = random.randint(0, 1000000)
    
    payload = {
        'config': config_dict,
        'history': history
    }
    
    try:
        payload_str = json.dumps(payload)
        response_str = poker_api_binding.process_request(payload_str)
        return json.loads(response_str)
    except Exception as e:
        return {'success': False, 'error': str(e)}


def create_binding_config(
    config: Union[GameConfig, Dict[str, Any]],
    seed: Optional[int] = None
) -> 'poker_api_binding.GameConfig':
    """
    Create a C++ poker_api_binding.GameConfig from a GameConfig or dict.
    
    This is the canonical way to create a C++ config. Use this function
    instead of manually creating poker_api_binding.GameConfig to ensure
    consistent behavior across the codebase.
    
    Args:
        config: Either a GameConfig instance or a dict with config values
        seed: Optional seed override
        
    Returns:
        poker_api_binding.GameConfig instance
        
    Example:
        >>> from common import GameConfig, create_binding_config
        >>> cfg = GameConfig(small_blind=10, big_blind=20)
        >>> binding_cfg = create_binding_config(cfg)
        
        >>> # Or from a dict
        >>> binding_cfg = create_binding_config({'smallBlind': 10, 'bigBlind': 20})
    """
    if isinstance(config, dict):
        config = GameConfig.from_dict(config)
    
    return config.to_binding_config(seed=seed)

