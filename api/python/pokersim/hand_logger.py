"""
Hand logging module for recording poker hands with full action history.

This module provides utilities for saving detailed hand histories to disk
for later analysis with hand-viewer tools. Useful for debugging AI behavior
and training analysis.
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


__all__ = [
    'HandLogger',
    'HAND_LOGS_DIR',
    'HAND_LOG_FREQUENCY',
]


# Default configuration
HAND_LOGS_DIR = Path(os.environ.get('HAND_LOGS_DIR', '/tmp/pokersim/hand_logs'))
HAND_LOG_FREQUENCY = 100  # Log 1 out of every N hands


class HandLogger:
    """
    Logger for recording poker hands with full action history and model predictions.
    
    Saves hands to disk in JSON format for analysis with the hand-viewer.
    Supports configurable logging frequency (1 in N hands).
    """
    
    def __init__(
        self,
        logs_dir: Optional[Path] = None,
        log_frequency: int = HAND_LOG_FREQUENCY,
    ):
        """
        Initialize the hand logger.
        
        Args:
            logs_dir: Directory to save hand logs (defaults to HAND_LOGS_DIR)
            log_frequency: Log 1 out of every N hands (defaults to HAND_LOG_FREQUENCY)
        """
        self.logs_dir = logs_dir or HAND_LOGS_DIR
        self.log_frequency = log_frequency
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.hand_counter = 0
        self.current_hand: Optional[Dict[str, Any]] = None
    
    def should_log(self) -> bool:
        """Check if the current hand should be logged (1 in log_frequency)."""
        return self.hand_counter % self.log_frequency == 0
    
    @property
    def is_logging(self) -> bool:
        """Check if currently logging a hand."""
        return self.current_hand is not None
    
    def start_hand(
        self,
        player_a_id: str,
        player_b_id: str,
        player_a_name: str,
        player_b_name: str,
        player_a_config: Optional[Dict] = None,
        player_b_config: Optional[Dict] = None,
        starting_stack_a: int = 1000,
        starting_stack_b: int = 1000,
    ) -> bool:
        """
        Start logging a new hand.
        
        Args:
            player_a_id: ID of player A (usually 'p0')
            player_b_id: ID of player B (usually 'p1')
            player_a_name: Display name for player A
            player_b_name: Display name for player B
            player_a_config: Optional config dict for player A (for metadata)
            player_b_config: Optional config dict for player B (for metadata)
            starting_stack_a: Starting chip stack for player A
            starting_stack_b: Starting chip stack for player B
            
        Returns:
            True if logging this hand, False otherwise
        """
        self.hand_counter += 1
        
        if not self.should_log():
            self.current_hand = None
            return False
        
        config_a = player_a_config or {}
        config_b = player_b_config or {}
        
        self.current_hand = {
            'hand_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'players': [
                {
                    'player_id': player_a_id,
                    'name': player_a_name,
                    'model_path': config_a.get('path', ''),
                    'agent_type': config_a.get('type', 'unknown'),
                    'starting_stack': starting_stack_a,
                    'hole_cards': [],
                    'is_dealer': False,
                    'is_small_blind': False,
                    'is_big_blind': False,
                },
                {
                    'player_id': player_b_id,
                    'name': player_b_name,
                    'model_path': config_b.get('path', ''),
                    'agent_type': config_b.get('type', 'unknown'),
                    'starting_stack': starting_stack_b,
                    'hole_cards': [],
                    'is_dealer': False,
                    'is_small_blind': False,
                    'is_big_blind': False,
                },
            ],
            'community_cards': {
                'flop': [],
                'turn': None,
                'river': None,
            },
            'actions': [],
            'result': {},
        }
        
        return True
    
    def set_hole_cards(self, player_id: str, hole_cards: List) -> None:
        """Record hole cards for a player."""
        if not self.current_hand:
            return
        
        for p in self.current_hand['players']:
            if p['player_id'] == player_id:
                p['hole_cards'] = hole_cards
                break
    
    def set_player_positions(self, game_state: Dict) -> None:
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
    
    def set_community_cards(self, stage: str, cards: List) -> None:
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
        predictions: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log an action with optional model predictions.
        
        Args:
            player_id: ID of the acting player
            player_name: Display name of the acting player
            action_type: Type of action (fold, check, call, bet, raise, all_in)
            action_label: Full action label (e.g., 'raise_50%')
            amount: Bet/raise amount (0 for check/fold)
            stage: Current betting stage (Preflop, Flop, Turn, River)
            pot: Pot size after action
            predictions: Optional dict of action probabilities from model
        """
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
            'predictions': predictions or {},
        }
        
        self.current_hand['actions'].append(action_data)
    
    def end_hand(
        self,
        winner_id: str,
        winner_name: str,
        final_pot: int,
        profits: Dict[str, int],
    ) -> Optional[Path]:
        """
        Finalize and save the hand log.
        
        Args:
            winner_id: ID of the winning player (or empty string for draw)
            winner_name: Name of the winning player
            final_pot: Final pot size
            profits: Dict mapping player_id to profit/loss
            
        Returns:
            Path to saved file, or None if not logging
        """
        if not self.current_hand:
            return None
        
        self.current_hand['result'] = {
            'winner_id': winner_id,
            'winner_name': winner_name,
            'final_pot': final_pot,
            'profits': profits,
        }
        
        # Save to file
        filename = f"{self.current_hand['hand_id']}.json"
        filepath = self.logs_dir / filename
        
        try:
            # Ensure directory exists (handles parallel/multiprocess scenarios)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.current_hand, f, indent=2)
        except Exception as e:
            print(f"Failed to save hand log: {e}")
            filepath = None
        
        self.current_hand = None
        return filepath
    
    def cancel_hand(self) -> None:
        """Cancel current hand logging (e.g., on error)."""
        self.current_hand = None

