#!/usr/bin/env python3
"""
Snapshot tests for the stateless poker engine API.

This script:
1. Builds the C++ poker engine server
2. Starts the server in the background
3. Tests various poker scenarios with snapshot testing
4. Saves/compares full response snapshots to disk
5. Kills the server

Snapshot testing allows us to:
- Capture complete game state for complex scenarios
- Easily review changes when updating game logic
- Avoid brittle assertions on specific values
"""

import subprocess
import time
import json
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from difflib import unified_diff

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

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")

class SnapshotTester:
    def __init__(self, port: int = 8080, update_snapshots: bool = False):
        self.port = port
        self.base_url = f"http://localhost:{port}/simulate"
        self.server_process: Optional[subprocess.Popen] = None
        self.passed_tests = 0
        self.failed_tests = 0
        self.update_snapshots = update_snapshots
        
        # Create snapshots directory
        self.snapshot_dir = Path(__file__).parent / "snapshots"
        self.snapshot_dir.mkdir(exist_ok=True)
        
    def build_server(self) -> bool:
        """Build the C++ server"""
        print_header("Building C++ Poker Engine Server")
        try:
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
    
    def save_snapshot(self, name: str, data: Dict[str, Any]):
        """Save a snapshot to disk"""
        snapshot_path = self.snapshot_dir / f"{name}.json"
        with open(snapshot_path, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
        print_info(f"Saved snapshot: {snapshot_path}")
    
    def load_snapshot(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a snapshot from disk"""
        snapshot_path = self.snapshot_dir / f"{name}.json"
        if not snapshot_path.exists():
            return None
        with open(snapshot_path, 'r') as f:
            return json.load(f)
    
    def compare_snapshots(self, name: str, actual: Dict[str, Any]) -> bool:
        """Compare actual response with saved snapshot"""
        expected = self.load_snapshot(name)
        
        if expected is None:
            if self.update_snapshots:
                self.save_snapshot(name, actual)
                print_warning(f"Created new snapshot: {name}")
                return True
            else:
                print_error(f"Snapshot not found: {name}")
                print_info("Run with --update-snapshots to create it")
                return False
        
        # Compare JSON
        if actual == expected:
            return True
        
        # Show diff if they don't match
        if self.update_snapshots:
            self.save_snapshot(name, actual)
            print_warning(f"Updated snapshot: {name}")
            return True
        
        # Show diff
        actual_str = json.dumps(actual, indent=2, sort_keys=True).splitlines(keepends=True)
        expected_str = json.dumps(expected, indent=2, sort_keys=True).splitlines(keepends=True)
        
        diff = unified_diff(expected_str, actual_str, 
                          fromfile=f"expected/{name}.json",
                          tofile=f"actual/{name}.json",
                          lineterm='')
        
        print_error(f"Snapshot mismatch: {name}")
        print_info("Diff:")
        for line in list(diff)[:50]:  # Limit diff output
            if line.startswith('+'):
                print(f"{Colors.GREEN}{line}{Colors.END}", end='')
            elif line.startswith('-'):
                print(f"{Colors.RED}{line}{Colors.END}", end='')
            else:
                print(line, end='')
        
        print_info("\nRun with --update-snapshots to update the snapshot")
        return False
    
    def test_snapshot(self, name: str, description: str, payload: Dict[str, Any]) -> bool:
        """Run a test and compare with snapshot"""
        print_info(f"Test: {description}")
        
        response = self.curl_post(payload)
        
        # Check if response indicates success
        if not response.get("success", False):
            print_error(f"API returned error: {response.get('error', 'Unknown error')}")
            self.failed_tests += 1
            return False
        
        # Compare with snapshot
        if self.compare_snapshots(name, response):
            print_success(f"{description}")
            self.passed_tests += 1
            return True
        else:
            print_error(f"{description}")
            self.failed_tests += 1
            return False
    
    def test_side_pots(self):
        """Test 1: Side pots with multiple all-ins"""
        payload = {
            "config": {
                "smallBlind": 10,
                "bigBlind": 20,
                "startingChips": 1000,  # Default, but we override per player
                "minPlayers": 3,  # Need 3 players before hand starts
                "seed": 1001
            },
            "history": [
                # Add players with different stack sizes
                {"type": "addPlayer", "playerId": "short", "playerName": "ShortStack", "chips": 100},
                {"type": "addPlayer", "playerId": "medium", "playerName": "MediumStack", "chips": 500},
                {"type": "addPlayer", "playerId": "big", "playerName": "BigStack", "chips": 2000},
                
                # Preflop action order: medium (UTG), big (dealer/SB), short (BB)
                # medium acts first (UTG position), then big, then short, back to medium
                {"type": "playerAction", "playerId": "medium", "action": "raise", "amount": 90},  # Raise to 90
                {"type": "playerAction", "playerId": "big", "action": "raise", "amount": 500},    # Re-raise to 500
                {"type": "playerAction", "playerId": "short", "action": "call", "amount": 0},     # All-in for 100 total
                {"type": "playerAction", "playerId": "medium", "action": "call", "amount": 0},    # All-in for 500 total
            ]
        }
        
        self.test_snapshot("side_pots", "Side pots with multiple all-ins", payload)
    
    def test_all_in_preflop(self):
        """Test 2: All-in during preflop"""
        payload = {
            "config": {
                "smallBlind": 5,
                "bigBlind": 10,
                "startingChips": 200,
                "seed": 2001
            },
            "history": [
                {"type": "addPlayer", "playerId": "p1", "playerName": "Player1"},
                {"type": "addPlayer", "playerId": "p2", "playerName": "Player2"},
                
                # Heads-up: p1 is SB/dealer (acts first preflop), p2 is BB
                # p1 has posted SB (5), so has 195 remaining. Raises by 45 more to make total bet 50.
                {"type": "playerAction", "playerId": "p1", "action": "raise", "amount": 45},
                # p2 has posted BB (10), so has 190 remaining. Raises by 190 more to go all-in (total bet 200).
                {"type": "playerAction", "playerId": "p2", "action": "raise", "amount": 190},
                # p1 needs to call 150 more (200 - 50 already in)
                {"type": "playerAction", "playerId": "p1", "action": "call", "amount": 0},
            ]
        }
        
        self.test_snapshot("all_in_preflop", "All-in during preflop", payload)
    
    def test_all_in_on_flop(self):
        """Test 3: All-in on the flop"""
        payload = {
            "config": {
                "smallBlind": 10,
                "bigBlind": 20,
                "startingChips": 500,
                "seed": 3001
            },
            "history": [
                {"type": "addPlayer", "playerId": "hero", "playerName": "Hero"},
                {"type": "addPlayer", "playerId": "villain", "playerName": "Villain"},
                
                # Preflop: Both call
                {"type": "playerAction", "playerId": "hero", "action": "call", "amount": 0},
                {"type": "playerAction", "playerId": "villain", "action": "check", "amount": 0},
                
                # Flop: Hero checks, Villain shoves, Hero calls
                {"type": "playerAction", "playerId": "hero", "action": "check", "amount": 0},
                {"type": "playerAction", "playerId": "villain", "action": "bet", "amount": 480},
                {"type": "playerAction", "playerId": "hero", "action": "call", "amount": 0},
            ]
        }
        
        self.test_snapshot("all_in_on_flop", "All-in on the flop", payload)
    
    def test_all_in_on_turn(self):
        """Test 4: All-in on the turn"""
        payload = {
            "config": {
                "smallBlind": 10,
                "bigBlind": 20,
                "startingChips": 1000,
                "seed": 4001
            },
            "history": [
                {"type": "addPlayer", "playerId": "alice", "playerName": "Alice"},
                {"type": "addPlayer", "playerId": "bob", "playerName": "Bob"},
                
                # Preflop
                {"type": "playerAction", "playerId": "alice", "action": "call", "amount": 0},
                {"type": "playerAction", "playerId": "bob", "action": "check", "amount": 0},
                
                # Flop: Small bet and call
                {"type": "playerAction", "playerId": "alice", "action": "bet", "amount": 50},
                {"type": "playerAction", "playerId": "bob", "action": "call", "amount": 0},
                
                # Turn: Alice shoves, Bob calls
                {"type": "playerAction", "playerId": "alice", "action": "bet", "amount": 930},
                {"type": "playerAction", "playerId": "bob", "action": "call", "amount": 0},
            ]
        }
        
        self.test_snapshot("all_in_on_turn", "All-in on the turn", payload)
    
    def test_all_in_on_river(self):
        """Test 5: All-in on the river"""
        payload = {
            "config": {
                "smallBlind": 25,
                "bigBlind": 50,
                "startingChips": 2000,
                "seed": 5001
            },
            "history": [
                {"type": "addPlayer", "playerId": "pro", "playerName": "ProPlayer", "chips": 1500},
                {"type": "addPlayer", "playerId": "fish", "playerName": "FishPlayer", "chips": 2500},
                
                # Heads-up: pro is SB/dealer (acts first preflop), fish is BB
                {"type": "playerAction", "playerId": "pro", "action": "raise", "amount": 150},  # Raise to 150
                {"type": "playerAction", "playerId": "fish", "action": "call", "amount": 0},    # Call
                
                # Flop: pro acts first post-flop (SB acts first post-flop)
                {"type": "playerAction", "playerId": "pro", "action": "check", "amount": 0},
                {"type": "playerAction", "playerId": "fish", "action": "check", "amount": 0},
                
                # Turn: pro acts first
                {"type": "playerAction", "playerId": "pro", "action": "bet", "amount": 200},
                {"type": "playerAction", "playerId": "fish", "action": "call", "amount": 0},
                
                # River: Pro shoves remaining stack, Fish calls
                {"type": "playerAction", "playerId": "pro", "action": "bet", "amount": 1100},  # Remaining chips
                {"type": "playerAction", "playerId": "fish", "action": "call", "amount": 0},
            ]
        }
        
        self.test_snapshot("all_in_on_river", "All-in on the river", payload)
    
    def test_split_pot(self):
        """Test 6: Split pot - both players have the same hand"""
        # Use exact cards to guarantee a split pot
        # Board will have a royal flush: AD KD QD JD TD
        # Both players have low cards that don't play: p1 has 2H 3D, p2 has 4S 5C
        # Order: p1 hole (2), p2 hole (2), burn, flop (3), burn, turn (1), burn, river (1)
        payload = {
            "config": {
                "smallBlind": 10,
                "bigBlind": 20,
                "startingChips": 500,
                "exactCards": [
                    # Player 1 hole cards (dealer/SB)
                    "2H", "3D",
                    # Player 2 hole cards (BB)
                    "4S", "5C",
                    # Burn card before flop
                    "6H",
                    # Flop
                    "AD", "KD", "QD",
                    # Burn card before turn
                    "7S",
                    # Turn
                    "JD",
                    # Burn card before river
                    "8C",
                    # River
                    "TD"
                ]
            },
            "history": [
                {"type": "addPlayer", "playerId": "alice", "playerName": "Alice"},
                {"type": "addPlayer", "playerId": "bob", "playerName": "Bob"},
                
                # Preflop: alice is SB/dealer (acts first preflop), bob is BB
                {"type": "playerAction", "playerId": "alice", "action": "raise", "amount": 40},  # Raise to 50
                {"type": "playerAction", "playerId": "bob", "action": "call", "amount": 0},     # Call
                
                # Flop: alice acts first post-flop
                {"type": "playerAction", "playerId": "alice", "action": "bet", "amount": 100},
                {"type": "playerAction", "playerId": "bob", "action": "call", "amount": 0},
                
                # Turn: alice bets big
                {"type": "playerAction", "playerId": "alice", "action": "bet", "amount": 340},
                {"type": "playerAction", "playerId": "bob", "action": "call", "amount": 0},
                
                # River: both players check to trigger showdown
                {"type": "playerAction", "playerId": "alice", "action": "check", "amount": 0},
                {"type": "playerAction", "playerId": "bob", "action": "check", "amount": 0},
            ]
        }
        
        self.test_snapshot("split_pot", "Split pot with identical hands", payload)
    
    def test_three_way_all_in_split_pot(self):
        """Test 7: Three players all-in with two players splitting the pot"""
        # Use exact cards to create a scenario where two players tie for best hand
        # Alice (BB): AH AS - pocket aces (will tie with Charlie)
        # Bob (UTG): 2H 3H - low cards (will lose)
        # Charlie (dealer/SB): AC AD - pocket aces (will tie with Alice)
        # Board: KH KD KS QC JD - three kings
        # Result: Alice and Charlie both have AAAKK (full house, aces full of kings)
        #         Bob has KKK32 (three of a kind)
        # Card order: alice hole (2), bob hole (2), charlie hole (2), burn, flop (3), burn, turn (1), burn, river (1)
        payload = {
            "config": {
                "smallBlind": 10,
                "bigBlind": 20,
                "startingChips": 1000,
                "minPlayers": 3,
                "exactCards": [
                    # Alice hole cards (BB)
                    "AH", "AS",
                    # Bob hole cards (UTG)
                    "2H", "3H",
                    # Charlie hole cards (dealer/SB)
                    "AC", "AD",
                    # Burn card before flop
                    "6H",
                    # Flop
                    "KH", "KD", "KS",
                    # Burn card before turn
                    "7S",
                    # Turn
                    "QC",
                    # Burn card before river
                    "8C",
                    # River
                    "JD"
                ]
            },
            "history": [
                {"type": "addPlayer", "playerId": "alice", "playerName": "Alice"},
                {"type": "addPlayer", "playerId": "bob", "playerName": "Bob"},
                {"type": "addPlayer", "playerId": "charlie", "playerName": "Charlie"},
                
                # Preflop: charlie is dealer/SB, alice is BB, bob is UTG
                # Action order: bob (UTG), charlie (SB), alice (BB), back to bob
                {"type": "playerAction", "playerId": "bob", "action": "raise", "amount": 980},      # Bob raises to 1000
                {"type": "playerAction", "playerId": "charlie", "action": "raise", "amount": 990},  # Charlie re-raises all-in
                {"type": "playerAction", "playerId": "alice", "action": "call", "amount": 0},       # Alice calls all-in
                {"type": "playerAction", "playerId": "bob", "action": "call", "amount": 0},         # Bob calls all-in
            ]
        }
        
        self.test_snapshot("three_way_all_in_split_pot", "Three-way all-in with two-way split pot", payload)
    
    def run_all_tests(self):
        """Run all tests"""
        if self.update_snapshots:
            print_header("SNAPSHOT UPDATE MODE")
            print_warning("Snapshots will be created/updated")
        else:
            print_header("SNAPSHOT TEST MODE")
        
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
            print_header("Running Snapshot Tests")
            
            self.test_side_pots()
            self.test_all_in_preflop()
            self.test_all_in_on_flop()
            self.test_all_in_on_turn()
            self.test_all_in_on_river()
            self.test_split_pot()
            self.test_three_way_all_in_split_pot()
            
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
    update_snapshots = False
    
    # Parse arguments
    args = sys.argv[1:]
    for arg in args:
        if arg == "--update-snapshots" or arg == "-u":
            update_snapshots = True
        else:
            try:
                port = int(arg)
            except ValueError:
                print_error(f"Invalid argument: {arg}")
                print_info("Usage: test_stateless_api.py [port] [--update-snapshots]")
                sys.exit(1)
    
    tester = SnapshotTester(port, update_snapshots)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
