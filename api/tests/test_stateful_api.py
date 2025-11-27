import json
import os
import sys
import unittest

# Ensure we can import the binding
# Adjust path as needed if running from different directories
try:
    import poker_api_binding
except ImportError:
    # Try to find it in expected locations
    # api/tests/../../training -> root/training
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../training'))
    try:
        import poker_api_binding
    except ImportError:
        print("Warning: poker_api_binding not found. Tests will fail if not in path.")

class TestStatefulAPI(unittest.TestCase):
    def setUp(self):
        self.config = poker_api_binding.GameConfig()
        self.config.smallBlind = 10
        self.config.bigBlind = 20
        self.config.startingChips = 1000
        self.config.minPlayers = 2
        self.config.maxPlayers = 2
        self.config.seed = 42
        
        self.game = poker_api_binding.Game(self.config)
        self.game.add_player("p1", "Player 1", 1000)
        self.game.add_player("p2", "Player 2", 1000)

    def test_initial_state(self):
        self.game.start_hand()
        stage = self.game.get_stage_name()
        self.assertEqual(stage, "Preflop")
        
        current = self.game.get_current_player_id()
        self.assertIsNotNone(current)
        
        state = self.game.get_state_dict()
        # Pot is 0 initially because bets are on table, not in pot
        self.assertEqual(state['pot'], 0) 
        self.assertEqual(state['currentBet'], 20) # BB amount
        self.assertEqual(len(state['players']), 2)

    def test_game_flow(self):
        self.game.start_hand()
        
        # Preflop: SB(p1) calls
        current = self.game.get_current_player_id()
        
        state = self.game.get_state_dict()
        legal_actions = state['actionConstraints']['legalActions']
        self.assertIn('call', legal_actions)
        
        success = self.game.process_action(current, "call", 10)
        self.assertTrue(success)
        
        # BB(p2) checks
        current = self.game.get_current_player_id()
        
        success = self.game.process_action(current, "check", 0)
        self.assertTrue(success)
        
        # Should be Flop now
        self.assertEqual(self.game.get_stage_name(), "Flop")
        
        state = self.game.get_state_dict()
        self.assertEqual(state['pot'], 40)
        self.assertEqual(len(state['communityCards']), 3)

    def test_state_json(self):
        self.game.start_hand()
        json_str = self.game.get_state_json()
        data = json.loads(json_str)
        self.assertIn('pot', data)
        self.assertIn('players', data)

if __name__ == '__main__':
    unittest.main()
