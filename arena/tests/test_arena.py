import sys
import unittest
import tempfile
import shutil
import json
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
import torch

# Add arena directory to path
ARENA_DIR = Path(__file__).parent.parent
sys.path.append(str(ARENA_DIR))

# Add training directory to path (needed for arena imports)
TRAINING_DIR = ARENA_DIR.parent / "training"
sys.path.append(str(TRAINING_DIR))

# Mock poker_api_binding before importing engine if it doesn't exist
try:
    import poker_api_binding
except ImportError:
    sys.modules['poker_api_binding'] = MagicMock()

from engine import Arena, parse_checkpoints


class TestArena(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.arena = Arena(device='cpu', output_dir=self.test_dir)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_arena_initialization(self):
        self.assertIsInstance(self.arena, Arena)
        self.assertEqual(self.arena.game_config['num_players'], 2)
        
    @patch('engine.poker_api_binding')
    def test_play_match_random_vs_random(self, mock_binding):
        """Test playing a match between two random agents"""
        
        # Mock the API response
        mock_response = {
            'success': True,
            'gameState': {
                'stage': 'Preflop',
                'players': [
                    {'id': 'p0', 'chips': 1000, 'holeCards': ['Ah', 'Kh'], 'isInHand': True},
                    {'id': 'p1', 'chips': 1000, 'holeCards': ['Qh', 'Jh'], 'isInHand': True}
                ],
                'pot': 30,
                'currentPlayerId': 'p0',
                'actionConstraints': {
                    'legalActions': ['fold', 'call', 'raise']
                }
            }
        }
        
        final_response = {
            'success': True,
            'gameState': {
                'stage': 'Complete',
                'players': [
                    {'id': 'p0', 'chips': 1010, 'isInHand': True},  # Won 10
                    {'id': 'p1', 'chips': 990, 'isInHand': True}    # Lost 10
                ]
            }
        }
        
        # Mock process_request sequence
        mock_binding.process_request.side_effect = [
            json.dumps(mock_response),
            json.dumps(final_response), 
            json.dumps(mock_response),  # Hand 2
            json.dumps(final_response)
        ]
        
        num_hands = 2
        
        config_a = {'type': 'random', 'name': 'RandomA'}
        config_b = {'type': 'random', 'name': 'RandomB'}
        
        results = self.arena.play_match(config_a, config_b, num_hands=num_hands)
        
        self.assertIn('agent_a', results)
        self.assertEqual(results['hands'], num_hands)
        # Hand 0: swap=False. p0=A, p1=B. p0 wins. A wins.
        # Hand 1: swap=True. p0=B, p1=A. p0 wins. B wins.
        # So A wins 1, B wins 1. Win rate 0.5.
        self.assertEqual(results['win_rate_a'], 0.5)

    @patch('engine.load_model_agent')
    @patch('engine.poker_api_binding')
    def test_play_hand_with_agents(self, mock_binding, mock_load_agent):
        """Test play_hand with mocked agents"""
        
        # Setup mock agents
        mock_agent_a = MagicMock()
        mock_agent_a.select_action.return_value = ('call', 10, 1)
        mock_agent_a.name = "ModelA"
        
        mock_agent_b = MagicMock()
        mock_agent_b.select_action.return_value = ('fold', 0, 0)
        mock_agent_b.name = "ModelB"
        
        # mock_load_agent should return these agents
        # It's called twice: once for A, once for B
        mock_load_agent.side_effect = [mock_agent_a, mock_agent_b]
        
        # Mock API
        mock_response_start = {
            'success': True,
            'gameState': {
                'stage': 'Preflop',
                'currentPlayerId': 'p0',
                'players': [{'id': 'p0', 'chips': 1000}, {'id': 'p1', 'chips': 1000}],
                'actionConstraints': {'legalActions': ['call']}
            }
        }
        
        mock_response_end = {
            'success': True,
            'gameState': {
                'stage': 'Complete',
                'players': [{'id': 'p0', 'chips': 1010}, {'id': 'p1', 'chips': 990}]
            }
        }
        
        mock_binding.process_request.side_effect = [
            json.dumps(mock_response_start),  # Init
            json.dumps(mock_response_end)     # After action
        ]
        
        # Dummy model
        model_a = MagicMock()
        model_b = MagicMock()
        
        config_a = {'type': 'model', 'model': model_a, 'name': 'Model A'}
        config_b = {'type': 'model', 'model': model_b, 'name': 'Model B'}
        
        res = self.arena.play_hand(config_a, config_b)
        
        # Verify result
        self.assertEqual(res['p0'], 0.5)  # (1010 - 1000) / 20 = 0.5 BB


class TestParseCheckpoints(unittest.TestCase):
    def test_parse_checkpoints_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoints = parse_checkpoints(Path(tmpdir))
            self.assertEqual(len(checkpoints), 0)
            
    def test_parse_checkpoints_with_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some checkpoint files
            Path(tmpdir, 'poker_rl_iter_100.pt').touch()
            Path(tmpdir, 'poker_rl_iter_200.pt').touch()
            Path(tmpdir, 'poker_rl_baseline.pt').touch()
            Path(tmpdir, 'other_file.txt').touch()
            
            checkpoints = parse_checkpoints(Path(tmpdir))
            
            self.assertEqual(len(checkpoints), 3)
            
            # Should be sorted by iteration
            self.assertEqual(checkpoints[0][0], 0)  # baseline
            self.assertEqual(checkpoints[1][0], 100)
            self.assertEqual(checkpoints[2][0], 200)


if __name__ == '__main__':
    unittest.main()
