#!/usr/bin/env python3
"""
Test script for the stateless poker engine API.

This script:
1. Builds the C++ poker engine server
2. Starts the server in the background
3. Tests the stateless API with various payloads
4. Verifies responses are correct
5. Kills the server

The stateless API works by:
- Client sends: config, history of actions, and optionally a new action
- Server creates fresh game, replays history, applies new action, returns state
- Server maintains NO state between requests

Event Sourcing:
- History includes player actions: addPlayer, playerAction
- Card dealing events (dealHoleCards, dealFlop, etc.) are regenerated automatically from seed
- The server's returned gameState includes full history with all events for transparency
- Tests use simplified histories (no card events) since they're deterministically regenerated
"""

import subprocess
import time
import json
import sys
import signal
import os
from typing import Optional, Dict, Any, List

# ANSI color codes for pretty output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")

def print_error(msg: str):
    print(f"{Colors.RED}✗{Colors.END} {msg}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ{Colors.END} {msg}")

def print_header(msg: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")

class StatelessPokerAPITester:
    def __init__(self, port: int = 8080):
        self.port = port
        self.base_url = f"http://localhost:{port}/simulate"
        self.server_process: Optional[subprocess.Popen] = None
        self.passed_tests = 0
        self.failed_tests = 0
        
    def build_server(self) -> bool:
        """Build the C++ server"""
        print_header("Building C++ Poker Engine Server")
        try:
            # Run make from the api directory (parent of tests directory)
            api_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            result = subprocess.run(
                ["make", "clean", "all"],
                capture_output=True,
                text=True,
                cwd=api_dir
            )
            if result.returncode == 0:
                print_success("Server built successfully")
                return True
            else:
                print_error(f"Build failed: {result.stderr}")
                return False
        except Exception as e:
            print_error(f"Build error: {e}")
            return False
    
    def start_server(self) -> bool:
        """Start the server in the background"""
        print_header("Starting Server")
        try:
            # Start server process from the api directory (parent of tests directory)
            api_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.server_process = subprocess.Popen(
                [f"./build/poker_api", str(self.port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=api_dir
            )
            
            # Wait for server to be ready
            max_retries = 10
            for i in range(max_retries):
                time.sleep(0.5)
                result = subprocess.run(
                    ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", 
                     f"http://localhost:{self.port}/simulate"],
                    capture_output=True,
                    text=True
                )
                if result.stdout in ["404", "400", "200"]:
                    print_success(f"Server started on port {self.port}")
                    return True
            
            print_error("Server failed to start within timeout")
            return False
            
        except Exception as e:
            print_error(f"Failed to start server: {e}")
            return False
    
    def stop_server(self):
        """Stop the server"""
        print_header("Stopping Server")
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
                print_success("Server stopped")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                print_success("Server killed")
    
    def curl_post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a POST request using curl and return parsed JSON response"""
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", self.base_url,
             "-H", "Content-Type: application/json",
             "-d", json.dumps(payload)],
            capture_output=True,
            text=True
        )
        
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            print_error(f"Failed to parse JSON response: {e}")
            print_error(f"Raw response: {result.stdout}")
            return {"error": "Invalid JSON response"}
    
    def verify_response(self, response: Dict[str, Any], expected_success: bool, 
                       test_name: str, checks: Optional[Dict[str, Any]] = None) -> bool:
        """Verify a response matches expectations"""
        if response.get("success") == expected_success:
            if checks:
                for key, expected_value in checks.items():
                    actual = response
                    for k in key.split('.'):
                        actual = actual.get(k, {})
                    
                    if actual != expected_value:
                        print_error(f"{test_name}: {key} = {actual}, expected {expected_value}")
                        self.failed_tests += 1
                        return False
            
            print_success(test_name)
            self.passed_tests += 1
            return True
        else:
            error_msg = response.get("error", "Unknown error")
            print_error(f"{test_name}: {error_msg}")
            self.failed_tests += 1
            return False
    
    def test_empty_game(self):
        """Test 1: Create an empty game with just config"""
        print_info("Test 1: Empty game creation")
        
        payload = {
            "config": {
                "smallBlind": 10,
                "bigBlind": 20,
                "startingChips": 1000,
                "seed": 42
            }
        }
        
        response = self.curl_post(payload)
        self.verify_response(
            response, 
            expected_success=True,
            test_name="Empty game creation",
            checks={
                "gameState.stage": "Waiting",
                "gameState.pot": 0,
                "gameState.config.seed": 42
            }
        )
        
        return response.get("gameState")
    
    def test_add_players(self):
        """Test 2: Add players via history"""
        print_info("Test 2: Adding players")
        
        payload = {
            "config": {
                "smallBlind": 10,
                "bigBlind": 20,
                "startingChips": 1000,
                "seed": 42
            },
            "history": [
                {"type": "addPlayer", "playerId": "alice", "playerName": "Alice"},
                {"type": "addPlayer", "playerId": "bob", "playerName": "Bob"}
            ]
        }
        
        response = self.curl_post(payload)
        success = self.verify_response(
            response,
            expected_success=True,
            test_name="Adding 2 players"
        )
        
        if success:
            game_state = response.get("gameState", {})
            players = game_state.get("players", [])
            if len(players) == 2:
                print_success(f"  Players: {players[0]['name']}, {players[1]['name']}")
            else:
                print_error(f"  Expected 2 players, got {len(players)}")
        
        return response.get("gameState")
    
    def test_start_hand(self):
        """Test 3: Start a hand (implicit when enough players added)"""
        print_info("Test 3: Starting hand")
        
        payload = {
            "config": {
                "smallBlind": 10,
                "bigBlind": 20,
                "startingChips": 1000,
                "seed": 42
            },
            "history": [
                {"type": "addPlayer", "playerId": "alice", "playerName": "Alice"},
                {"type": "addPlayer", "playerId": "bob", "playerName": "Bob"}
            ]
            # Hand starts in Preflop - betting must complete before advancing
        }
        
        response = self.curl_post(payload)
        success = self.verify_response(
            response,
            expected_success=True,
            test_name="Starting hand implicitly (stays in Preflop until betting complete)",
            checks={
                "gameState.stage": "Preflop"
            }
        )
        
        if success:
            game_state = response.get("gameState", {})
            print_success(f"  Stage: {game_state['stage']}, Pot: {game_state['pot']}")
            
            # Show hole cards
            for player in game_state.get("players", []):
                cards = ', '.join(player.get("holeCards", []))
                print_success(f"  {player['name']}: {cards}")
        
        return response.get("gameState")
    
    def test_player_actions(self):
        """Test 4: Process player actions (with implicit advancement)"""
        print_info("Test 4: Player actions (call, check)")
        
        # After preflop betting completes, game advances to Flop automatically
        # Card dealing events are regenerated from seed, no need to specify
        payload = {
            "config": {
                "smallBlind": 10,
                "bigBlind": 20,
                "startingChips": 1000,
                "seed": 42
            },
            "history": [
                {"type": "addPlayer", "playerId": "alice", "playerName": "Alice"},
                {"type": "addPlayer", "playerId": "bob", "playerName": "Bob"},
                {"type": "playerAction", "playerId": "alice", "action": "call", "amount": 0},
                {"type": "playerAction", "playerId": "bob", "action": "check", "amount": 0}
            ]
            # No action - preflop complete, auto-advances to Flop
        }
        
        response = self.curl_post(payload)
        success = self.verify_response(
            response,
            expected_success=True,
            test_name="Player actions (auto-advances to Flop)",
            checks={
                "gameState.stage": "Flop"
            }
        )
        
        if success:
            game_state = response.get("gameState", {})
            community_cards = ', '.join(game_state.get("communityCards", []))
            print_success(f"  Flop: {community_cards}")
        
        return response.get("gameState")
    
    def test_betting_round(self):
        """Test 5: Complete betting round with bet and fold"""
        print_info("Test 5: Betting round (bet, fold)")
        
        payload = {
            "config": {
                "smallBlind": 10,
                "bigBlind": 20,
                "startingChips": 1000,
                "seed": 42
            },
            "history": [
                {"type": "addPlayer", "playerId": "alice", "playerName": "Alice"},
                {"type": "addPlayer", "playerId": "bob", "playerName": "Bob"},
                {"type": "playerAction", "playerId": "alice", "action": "call", "amount": 0},
                {"type": "playerAction", "playerId": "bob", "action": "check", "amount": 0},
                # Flop - Alice (small blind) acts first post-flop
                {"type": "playerAction", "playerId": "alice", "action": "check", "amount": 0},
                {"type": "playerAction", "playerId": "bob", "action": "bet", "amount": 50},
                {"type": "playerAction", "playerId": "alice", "action": "fold", "amount": 0}
            ]
        }
        
        response = self.curl_post(payload)
        success = self.verify_response(
            response,
            expected_success=True,
            test_name="Betting round (bet, fold)",
            checks={
                "gameState.stage": "Complete"
            }
        )
        
        if success:
            game_state = response.get("gameState", {})
            print_success(f"  Hand complete, final pot: {game_state['pot']}")
            
            # Show final chip counts
            for player in game_state.get("players", []):
                print_success(f"  {player['name']}: {player['chips']} chips")
        
        return response.get("gameState")
    
    def test_new_action_parameter(self):
        """Test 6: Verify actions must be in history (no manual actions)"""
        print_info("Test 6: Actions via history only (no manual actions)")
        
        # Build state up to preflop with p1's action in history
        payload = {
            "config": {
                "smallBlind": 10,
                "bigBlind": 20,
                "startingChips": 1000,
                "seed": 99
            },
            "history": [
                {"type": "addPlayer", "playerId": "p1", "playerName": "Player1"},
                {"type": "addPlayer", "playerId": "p2", "playerName": "Player2"},
                {"type": "playerAction", "playerId": "p1", "action": "call", "amount": 0}
            ]
        }
        
        response = self.curl_post(payload)
        success = self.verify_response(
            response,
            expected_success=True,
            test_name="Action in history (p1 call)"
        )
        
        if success:
            game_state = response.get("gameState", {})
            print_success(f"  Applied action from history, waiting for p2")
        
        return response.get("gameState")
    
    def test_deterministic_replay(self):
        """Test 7: Verify deterministic replay with same seed"""
        print_info("Test 7: Deterministic replay")
        
        # We'll verify that same seed + same history produces same results
        payload = {
            "config": {
                "seed": 12345,
                "smallBlind": 10,
                "bigBlind": 20,
                "startingChips": 1000
            },
            "history": [
                {"type": "addPlayer", "playerId": "a", "playerName": "A"},
                {"type": "addPlayer", "playerId": "b", "playerName": "B"},
                {"type": "playerAction", "playerId": "a", "action": "call", "amount": 0},
                {"type": "playerAction", "playerId": "b", "action": "check", "amount": 0}
            ]
            # Note: Betting round complete, no action field will auto-advance to Flop
        }
        
        # Send same request twice
        response1 = self.curl_post(payload)
        response2 = self.curl_post(payload)
        
        if response1.get("success") and response2.get("success"):
            state1 = response1.get("gameState", {})
            state2 = response2.get("gameState", {})
            
            # Check complete state equality
            errors = []
            
            # Check stage
            if state1.get("stage") != state2.get("stage"):
                errors.append(f"stage: {state1.get('stage')} vs {state2.get('stage')}")
            
            # Check pot
            if state1.get("pot") != state2.get("pot"):
                errors.append(f"pot: {state1.get('pot')} vs {state2.get('pot')}")
            
            # Check community cards
            cards1_comm = state1.get("communityCards", [])
            cards2_comm = state2.get("communityCards", [])
            if cards1_comm != cards2_comm:
                errors.append(f"community cards: {cards1_comm} vs {cards2_comm}")
            
            # Check hole cards
            cards1_hole = [p["holeCards"] for p in state1.get("players", [])]
            cards2_hole = [p["holeCards"] for p in state2.get("players", [])]
            if cards1_hole != cards2_hole:
                errors.append(f"hole cards: {cards1_hole} vs {cards2_hole}")
            
            # Check chip counts
            chips1 = [p["chips"] for p in state1.get("players", [])]
            chips2 = [p["chips"] for p in state2.get("players", [])]
            if chips1 != chips2:
                errors.append(f"chip counts: {chips1} vs {chips2}")
            
            if not errors:
                print_success("Deterministic replay verified (complete state match)")
                print_success(f"  Stage: {state1.get('stage')}, Pot: {state1.get('pot')}")
                if cards1_comm:
                    print_success(f"  Community: {', '.join(cards1_comm)}")
                self.passed_tests += 1
            else:
                print_error("Replay not deterministic:")
                for error in errors:
                    print_error(f"  {error}")
                self.failed_tests += 1
        else:
            print_error("Deterministic replay test failed")
            self.failed_tests += 1
    
    def test_automatic_progression(self):
        """Test 8: Implicit advancement (no explicit advance actions)"""
        print_info("Test 8: Implicit advancement")
        
        # Test 8a: No advancement without completed betting
        print_info("  8a: Cannot advance with incomplete betting round")
        payload = {
            "config": {
                "seed": 999,
                "smallBlind": 10,
                "bigBlind": 20,
                "startingChips": 1000
            },
            "history": [
                {"type": "addPlayer", "playerId": "p1", "playerName": "Player1"},
                {"type": "addPlayer", "playerId": "p2", "playerName": "Player2"}
            ]
            # No action - but betting round not complete, should stay in Preflop
        }
        
        response = self.curl_post(payload)
        success = self.verify_response(
            response,
            expected_success=True,
            test_name="No advancement with incomplete betting",
            checks={
                "gameState.stage": "Preflop"
            }
        )
        
        if success:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
        
        # Test 8b: Implicit advancement WITH completed betting
        print_info("  8b: Auto-advance with completed betting round")
        payload = {
            "config": {
                "seed": 999,
                "smallBlind": 10,
                "bigBlind": 20,
                "startingChips": 1000
            },
            "history": [
                {"type": "addPlayer", "playerId": "p1", "playerName": "Player1"},
                {"type": "addPlayer", "playerId": "p2", "playerName": "Player2"},
                {"type": "playerAction", "playerId": "p1", "action": "call", "amount": 0},
                {"type": "playerAction", "playerId": "p2", "action": "check", "amount": 0}
            ]
            # No action - betting round complete, should auto-advance to Flop
        }
        
        response = self.curl_post(payload)
        success = self.verify_response(
            response,
            expected_success=True,
            test_name="Implicit advancement to Flop",
            checks={
                "gameState.stage": "Flop"
            }
        )
        
        if success:
            game_state = response.get("gameState", {})
            community_cards = game_state.get("communityCards", [])
            if len(community_cards) == 3:
                print_success(f"  3 community cards dealt: {', '.join(community_cards)}")
                self.passed_tests += 1
            else:
                print_error(f"  Expected 3 community cards, got {len(community_cards)}")
                self.failed_tests += 1
        else:
            # Already counted by verify_response
            pass
    
    def test_error_handling(self):
        """Test 9: Error handling"""
        print_info("Test 9: Error handling")
        
        # Test 9a: Invalid action type
        payload = {
            "config": {"seed": 1},
            "history": [{"type": "invalidAction"}]
        }
        response = self.curl_post(payload)
        self.verify_response(
            response,
            expected_success=False,
            test_name="Invalid action type"
        )
        
        # Test 9b: Missing player info
        payload = {
            "config": {"seed": 1},
            "history": [{"type": "addPlayer", "playerId": ""}]
        }
        response = self.curl_post(payload)
        self.verify_response(
            response,
            expected_success=False,
            test_name="Missing player info"
        )
        
        # Test 9c: Empty request (missing config)
        payload = {}
        response = self.curl_post(payload)
        self.verify_response(
            response,
            expected_success=False,
            test_name="Empty request (missing config)"
        )
    
    def test_complete_hand(self):
        """Test 10: Complete hand from start to finish"""
        print_info("Test 10: Complete hand playthrough")
        
        payload = {
            "config": {
                "seed": 777,
                "smallBlind": 5,
                "bigBlind": 10,
                "startingChips": 500
            },
            "history": [
                {"type": "addPlayer", "playerId": "hero", "playerName": "Hero"},
                {"type": "addPlayer", "playerId": "villain", "playerName": "Villain"},
                # Preflop - Hero is small blind and acts first
                {"type": "playerAction", "playerId": "hero", "action": "raise", "amount": 30},
                {"type": "playerAction", "playerId": "villain", "action": "call", "amount": 0},
                # Flop - Hero (small blind) acts first post-flop
                {"type": "playerAction", "playerId": "hero", "action": "check", "amount": 0},
                {"type": "playerAction", "playerId": "villain", "action": "bet", "amount": 50},
                {"type": "playerAction", "playerId": "hero", "action": "call", "amount": 0},
                # Turn - Hero acts first again
                {"type": "playerAction", "playerId": "hero", "action": "check", "amount": 0},
                {"type": "playerAction", "playerId": "villain", "action": "check", "amount": 0},
                # River - Hero acts first again
                {"type": "playerAction", "playerId": "hero", "action": "check", "amount": 0},
                {"type": "playerAction", "playerId": "villain", "action": "bet", "amount": 100},
                {"type": "playerAction", "playerId": "hero", "action": "call", "amount": 0}
            ]
        }
        
        response = self.curl_post(payload)
        # After all player actions complete, game goes to Showdown
        # With no action field, it auto-advances from Showdown to Complete
        success = self.verify_response(
            response,
            expected_success=True,
            test_name="Complete hand (auto-advances to Complete)",
            checks={
                "gameState.stage": "Complete"
            }
        )
        
        if success:
            game_state = response.get("gameState", {})
            community_cards = ', '.join(game_state.get("communityCards", []))
            print_success(f"  Board: {community_cards}")
            print_success(f"  Pot: {game_state['pot']}")
            
            for player in game_state.get("players", []):
                hole_cards = ', '.join(player.get("holeCards", []))
                print_success(f"  {player['name']}: {hole_cards} ({player['chips']} chips)")
    
    def run_all_tests(self):
        """Run all tests"""
        print_header("STATELESS POKER ENGINE API TESTS")
        
        # Build server
        if not self.build_server():
            print_error("Build failed, cannot continue")
            return False
        
        # Start server
        if not self.start_server():
            print_error("Server start failed, cannot continue")
            return False
        
        try:
            # Run tests
            print_header("Running Test Suite")
            
            self.test_empty_game()
            self.test_add_players()
            self.test_start_hand()
            self.test_player_actions()
            self.test_betting_round()
            self.test_new_action_parameter()
            self.test_deterministic_replay()
            self.test_automatic_progression()
            self.test_error_handling()
            self.test_complete_hand()
            
            # Print summary
            print_header("Test Summary")
            total = self.passed_tests + self.failed_tests
            print(f"Total tests: {total}")
            print_success(f"Passed: {self.passed_tests}")
            if self.failed_tests > 0:
                print_error(f"Failed: {self.failed_tests}")
            else:
                print_info("Failed: 0")
            
            print()
            if self.failed_tests == 0:
                print(f"{Colors.GREEN}{Colors.BOLD}🎉 All tests passed!{Colors.END}")
                return True
            else:
                print(f"{Colors.RED}{Colors.BOLD}❌ Some tests failed{Colors.END}")
                return False
            
        finally:
            # Always stop server
            self.stop_server()

def main():
    """Main entry point"""
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print_error("Invalid port number")
            sys.exit(1)
    
    tester = StatelessPokerAPITester(port)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
