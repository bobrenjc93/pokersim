"""
Gameplay module for running poker hands and matches.

This module provides reusable components for:
- Running single poker hands between agents
- Running freezeout rounds (play until bust)
- Running best-of-N matches with position alternation

These utilities are shared between training, ELO evaluation, and other gameplay contexts.
"""

import json
import random
from typing import Dict, Any, Tuple, List, Optional, Protocol, runtime_checkable

try:
    import poker_api_binding
except ImportError:
    poker_api_binding = None

from .utils import extract_state, convert_action_label


__all__ = [
    'PokerAgent',
    'GameConfig',
    'HandResult',
    'RoundResult',
    'MatchResult',
    'PokerGameRunner',
]


@runtime_checkable
class PokerAgent(Protocol):
    """Protocol defining the interface for poker agents."""
    player_id: str
    name: str
    
    def reset_hand(self) -> None:
        """Reset internal state for a new hand."""
        ...
    
    def observe_action(self, player_id: str, action_type: str, amount: int, 
                      pot: int, stage: str) -> None:
        """Observe an action (for opponent modeling)."""
        ...
    
    def select_action(self, state: Dict[str, Any], 
                     legal_actions: List[str]) -> Tuple[str, int, str]:
        """
        Select an action given the current state.
        
        Returns:
            Tuple of (action_type, amount, action_label)
        """
        ...


class GameConfig:
    """Configuration for poker games."""
    
    def __init__(
        self,
        small_blind: int = 10,
        big_blind: int = 20,
        starting_chips: int = 1000,
        min_players: int = 2,
        max_players: int = 2,
    ):
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.starting_chips = starting_chips
        self.min_players = min_players
        self.max_players = max_players
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            'num_players': 2,
            'smallBlind': self.small_blind,
            'bigBlind': self.big_blind,
            'startingChips': self.starting_chips,
            'minPlayers': self.min_players,
            'maxPlayers': self.max_players,
        }


class HandResult:
    """Result of a single poker hand."""
    
    def __init__(
        self,
        profit_p0: int,
        profit_p1: int,
        error: bool = False,
        final_state: Optional[Dict] = None,
    ):
        self.profit_p0 = profit_p0
        self.profit_p1 = profit_p1
        self.error = error
        self.final_state = final_state
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'profit_p0': self.profit_p0,
            'profit_p1': self.profit_p1,
            'error': self.error,
        }


class RoundResult:
    """Result of a freezeout round (play until bust or max hands)."""
    
    def __init__(
        self,
        winner: Optional[str],  # 'a', 'b', 'draw', or None on error
        hands_played: int,
        final_stack_a: int,
        final_stack_b: int,
        hand_wins_a: int,
        hand_wins_b: int,
        error: bool = False,
    ):
        self.winner = winner
        self.hands_played = hands_played
        self.final_stack_a = final_stack_a
        self.final_stack_b = final_stack_b
        self.hand_wins_a = hand_wins_a
        self.hand_wins_b = hand_wins_b
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'winner': self.winner,
            'hands_played': self.hands_played,
            'final_stack_a': self.final_stack_a,
            'final_stack_b': self.final_stack_b,
            'hand_wins_a': self.hand_wins_a,
            'hand_wins_b': self.hand_wins_b,
            'error': self.error,
        }


class MatchResult:
    """Result of a best-of-N match."""
    
    def __init__(
        self,
        rounds_played: int,
        round_wins_a: int,
        round_wins_b: int,
        total_hands: int,
        total_hand_wins_a: int,
        total_hand_wins_b: int,
        score_a: float,  # 1.0 = win, 0.5 = draw, 0.0 = loss
        error: bool = False,
    ):
        self.rounds_played = rounds_played
        self.round_wins_a = round_wins_a
        self.round_wins_b = round_wins_b
        self.total_hands = total_hands
        self.total_hand_wins_a = total_hand_wins_a
        self.total_hand_wins_b = total_hand_wins_b
        self.score_a = score_a
        self.error = error
    
    @property
    def winner(self) -> Optional[str]:
        if self.score_a == 1.0:
            return 'a'
        elif self.score_a == 0.0:
            return 'b'
        elif self.score_a == 0.5:
            return 'draw'
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rounds_played': self.rounds_played,
            'round_wins_a': self.round_wins_a,
            'round_wins_b': self.round_wins_b,
            'total_hands': self.total_hands,
            'total_hand_wins_a': self.total_hand_wins_a,
            'total_hand_wins_b': self.total_hand_wins_b,
            'score_a': self.score_a,
            'winner': self.winner,
            'error': self.error,
        }


class PokerGameRunner:
    """
    Runs poker games between agents using the C++ poker engine.
    
    Provides methods to run:
    - Single hands
    - Freezeout rounds (play until bust)
    - Best-of-N matches
    
    This class consolidates gameplay logic shared between training and evaluation.
    """
    
    def __init__(
        self,
        config: Optional[GameConfig] = None,
        max_raises_per_round: int = 4,
        max_steps_per_hand: int = 200,
        max_hands_per_round: int = 200,
    ):
        """
        Initialize the game runner.
        
        Args:
            config: Game configuration (blinds, starting chips, etc.)
            max_raises_per_round: Maximum raises allowed per betting round
            max_steps_per_hand: Maximum action steps per hand (safety limit)
            max_hands_per_round: Maximum hands per freezeout round
        """
        if poker_api_binding is None:
            raise ImportError("poker_api_binding not available")
        
        self.config = config or GameConfig()
        self.max_raises_per_round = max_raises_per_round
        self.max_steps_per_hand = max_steps_per_hand
        self.max_hands_per_round = max_hands_per_round
    
    def _call_api(self, history: List[Dict], starting_chips: Optional[int] = None) -> Dict:
        """Call the C++ poker API."""
        api_config = self.config.to_dict()
        api_config['seed'] = random.randint(0, 1000000)
        
        if starting_chips is not None:
            api_config['startingChips'] = starting_chips
        
        payload = {
            'config': api_config,
            'history': history,
        }
        
        try:
            payload_str = json.dumps(payload)
            response_str = poker_api_binding.process_request(payload_str)
            return json.loads(response_str)
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def play_hand(
        self,
        agent_p0: PokerAgent,
        agent_p1: PokerAgent,
        stack_p0: Optional[int] = None,
        stack_p1: Optional[int] = None,
        action_callback: Optional[callable] = None,
    ) -> HandResult:
        """
        Play a single poker hand between two agents.
        
        Args:
            agent_p0: Agent playing as p0
            agent_p1: Agent playing as p1
            stack_p0: Starting stack for p0 (defaults to config.starting_chips)
            stack_p1: Starting stack for p1 (defaults to config.starting_chips)
            action_callback: Optional callback called after each action with 
                           (player_id, action_type, amount, game_state)
        
        Returns:
            HandResult with profit/loss for each player
        """
        agents = {'p0': agent_p0, 'p1': agent_p1}
        
        # Reset agent state at start of hand
        for agent in agents.values():
            if hasattr(agent, 'reset_hand'):
                agent.reset_hand()
        
        # Determine effective stack (minimum of both stacks)
        actual_stack_p0 = stack_p0 if stack_p0 is not None else self.config.starting_chips
        actual_stack_p1 = stack_p1 if stack_p1 is not None else self.config.starting_chips
        effective_stack = min(actual_stack_p0, actual_stack_p1)
        
        # Initialize game
        history = [
            {'type': 'addPlayer', 'playerId': 'p0', 'playerName': agent_p0.name},
            {'type': 'addPlayer', 'playerId': 'p1', 'playerName': agent_p1.name},
        ]
        
        response = self._call_api(history, starting_chips=effective_stack)
        if not response.get('success'):
            return HandResult(0, 0, error=True)
        
        game_state = response['gameState']
        raises_this_round = 0
        current_betting_stage = None
        
        # Play the hand
        for _ in range(self.max_steps_per_hand):
            stage = game_state.get('stage', '').lower()
            if stage in ['complete', 'showdown']:
                break
            
            # Track betting rounds for raise limiting
            if stage != current_betting_stage:
                current_betting_stage = stage
                raises_this_round = 0
            
            current_player_id = game_state.get('currentPlayerId')
            if not current_player_id or current_player_id == 'none':
                break
            
            agent = agents[current_player_id]
            legal_actions = game_state.get('actionConstraints', {}).get('legalActions', [])
            if not legal_actions:
                break
            
            # Limit raises per round
            if raises_this_round >= self.max_raises_per_round:
                legal_actions = [a for a in legal_actions if a not in ['raise', 'bet']]
                if not legal_actions:
                    legal_actions = ['fold']
            
            # Get agent's action
            state_dict = extract_state(game_state, current_player_id)
            action_type, amount, action_label = agent.select_action(state_dict, legal_actions)
            
            # Convert to all_in if necessary
            player_chips = state_dict.get('player_chips', 0)
            to_call = state_dict.get('to_call', 0)
            
            if action_type in ['bet', 'raise'] and amount >= player_chips:
                action_type, amount = 'all_in', 0
            elif action_type == 'call' and to_call >= player_chips > 0:
                action_type, amount = 'all_in', 0
            
            if action_type in ['raise', 'bet']:
                raises_this_round += 1
            
            # Calculate pot before action for observe_action
            pot_before = self._calc_pot_from_chips(game_state, effective_stack)
            
            # Apply action
            history.append({
                'type': 'playerAction',
                'playerId': current_player_id,
                'action': action_type,
                'amount': amount,
            })
            
            response = self._call_api(history, starting_chips=effective_stack)
            if not response.get('success'):
                return HandResult(0, 0, error=True)
            
            game_state = response['gameState']
            
            # Notify all agents about the action (for opponent modeling)
            for pid, ag in agents.items():
                if hasattr(ag, 'observe_action'):
                    ag.observe_action(
                        current_player_id,
                        action_type,
                        amount,
                        pot_before,
                        stage.capitalize(),
                    )
            
            # Call action callback if provided
            if action_callback:
                action_callback(current_player_id, action_type, amount, game_state)
        
        # Calculate profits
        profit_p0 = profit_p1 = 0
        for p in game_state.get('players', []):
            profit = p['chips'] - effective_stack
            if p['id'] == 'p0':
                profit_p0 = profit
            else:
                profit_p1 = profit
        
        return HandResult(profit_p0, profit_p1, error=False, final_state=game_state)
    
    def _calc_pot_from_chips(self, game_state: Dict, effective_stack: int) -> int:
        """Calculate pot from chip differences."""
        total_chips_remaining = sum(
            p.get('chips', effective_stack) for p in game_state.get('players', [])
        )
        return (effective_stack * 2) - total_chips_remaining
    
    def play_freezeout_round(
        self,
        agent_a: PokerAgent,
        agent_b: PokerAgent,
        starting_stack: Optional[int] = None,
    ) -> RoundResult:
        """
        Play a freezeout round between two agents.
        
        Players start with starting_stack chips and play until one runs out.
        Positions alternate each hand.
        
        Args:
            agent_a: First agent
            agent_b: Second agent
            starting_stack: Starting chips (defaults to config.starting_chips)
        
        Returns:
            RoundResult with winner and statistics
        """
        stack = starting_stack if starting_stack is not None else self.config.starting_chips
        
        stack_a = stack
        stack_b = stack
        hands_played = 0
        hand_wins_a = 0
        hand_wins_b = 0
        
        while stack_a > 0 and stack_b > 0 and hands_played < self.max_hands_per_round:
            # Alternate positions each hand
            swap = (hands_played % 2 == 1)
            
            # Reset agent state between hands
            if hasattr(agent_a, 'reset_hand'):
                agent_a.reset_hand()
            if hasattr(agent_b, 'reset_hand'):
                agent_b.reset_hand()
            
            try:
                if swap:
                    # B is p0, A is p1
                    result = self.play_hand(agent_b, agent_a, stack_b, stack_a)
                    if result.error:
                        continue
                    profit_a, profit_b = result.profit_p1, result.profit_p0
                else:
                    # A is p0, B is p1
                    result = self.play_hand(agent_a, agent_b, stack_a, stack_b)
                    if result.error:
                        continue
                    profit_a, profit_b = result.profit_p0, result.profit_p1
                
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
            return RoundResult(
                winner=None,
                hands_played=0,
                final_stack_a=stack_a,
                final_stack_b=stack_b,
                hand_wins_a=0,
                hand_wins_b=0,
                error=True,
            )
        
        # Determine winner
        if stack_a > stack_b:
            winner = 'a'
        elif stack_b > stack_a:
            winner = 'b'
        else:
            winner = 'draw'
        
        return RoundResult(
            winner=winner,
            hands_played=hands_played,
            final_stack_a=stack_a,
            final_stack_b=stack_b,
            hand_wins_a=hand_wins_a,
            hand_wins_b=hand_wins_b,
            error=False,
        )
    
    def play_match(
        self,
        agent_a: PokerAgent,
        agent_b: PokerAgent,
        num_rounds: int = 51,
        starting_stack: Optional[int] = None,
    ) -> MatchResult:
        """
        Play a best-of-N match between two agents.
        
        Each match consists of multiple freezeout rounds. The winner is 
        whoever wins the most rounds.
        
        Args:
            agent_a: First agent
            agent_b: Second agent
            num_rounds: Number of rounds to play
            starting_stack: Starting chips per round
        
        Returns:
            MatchResult with match outcome and statistics
        """
        round_wins_a = 0
        round_wins_b = 0
        total_hands = 0
        total_hand_wins_a = 0
        total_hand_wins_b = 0
        rounds_played = 0
        
        for _ in range(num_rounds):
            result = self.play_freezeout_round(agent_a, agent_b, starting_stack)
            
            if result.error:
                continue
            
            rounds_played += 1
            total_hands += result.hands_played
            total_hand_wins_a += result.hand_wins_a
            total_hand_wins_b += result.hand_wins_b
            
            if result.winner == 'a':
                round_wins_a += 1
            elif result.winner == 'b':
                round_wins_b += 1
        
        if rounds_played == 0:
            return MatchResult(
                rounds_played=0,
                round_wins_a=0,
                round_wins_b=0,
                total_hands=0,
                total_hand_wins_a=0,
                total_hand_wins_b=0,
                score_a=0.5,
                error=True,
            )
        
        # Determine match score
        if round_wins_a > round_wins_b:
            score_a = 1.0
        elif round_wins_b > round_wins_a:
            score_a = 0.0
        else:
            score_a = 0.5
        
        return MatchResult(
            rounds_played=rounds_played,
            round_wins_a=round_wins_a,
            round_wins_b=round_wins_b,
            total_hands=total_hands,
            total_hand_wins_a=total_hand_wins_a,
            total_hand_wins_b=total_hand_wins_b,
            score_a=score_a,
            error=False,
        )

