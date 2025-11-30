#!/usr/bin/env python3
"""
Benchmark script to compare rollout collection performance.

This script compares:
1. Old JSON-based stateless API (collect_episode)
2. New direct Game binding (collect_episode_fast)
3. Multi-process parallel collection

Run with: uv run python benchmark_rollouts.py
"""

import time
import os
import sys
import torch

# Add training directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import poker_api_binding
except ImportError:
    print("Error: 'poker_api_binding' not found. Please compile the binding (cd api && make module).")
    sys.exit(1)

from rl_model import create_actor_critic
from rl_state_encoder import RLStateEncoder


def benchmark_json_api(num_episodes: int = 100):
    """Benchmark the old JSON-based stateless API."""
    import random
    try:
        import orjson as json
    except ImportError:
        import json
    
    game_config = {
        'smallBlind': 10,
        'bigBlind': 20,
        'startingChips': 1000,
        'minPlayers': 2,
        'maxPlayers': 2,
        'num_players': 2,
    }
    
    start = time.perf_counter()
    
    for _ in range(num_episodes):
        # Simulate the old approach: build history and replay
        history = [
            {'type': 'addPlayer', 'playerId': 'p0', 'playerName': 'Player_p0'},
            {'type': 'addPlayer', 'playerId': 'p1', 'playerName': 'Player_p1'}
        ]
        
        # Initial call
        payload = {
            'config': {**game_config, 'seed': random.randint(0, 1000000)},
            'history': history
        }
        payload_bytes = json.dumps(payload)
        payload_str = payload_bytes.decode('utf-8') if isinstance(payload_bytes, bytes) else payload_bytes
        response_str = poker_api_binding.process_request(payload_str)
        response = json.loads(response_str)
        
        if not response.get('success'):
            continue
            
        # Simulate 6 more actions (typical hand has ~4-8 actions)
        game_state = response['gameState']
        for step in range(6):
            stage = game_state.get('stage', '').lower()
            if stage in {'complete', 'showdown'}:
                break
                
            current_player = game_state.get('currentPlayerId')
            if not current_player:
                break
            
            # Add action to history
            history.append({
                'type': 'playerAction',
                'playerId': current_player,
                'action': 'check' if 'check' in game_state.get('actionConstraints', {}).get('legalActions', []) else 'fold',
                'amount': 0
            })
            
            # Rebuild entire state from scratch (O(N²) behavior!)
            payload = {
                'config': {**game_config, 'seed': random.randint(0, 1000000)},
                'history': history
            }
            payload_bytes = json.dumps(payload)
            payload_str = payload_bytes.decode('utf-8') if isinstance(payload_bytes, bytes) else payload_bytes
            response_str = poker_api_binding.process_request(payload_str)
            response = json.loads(response_str)
            
            if not response.get('success'):
                break
            game_state = response['gameState']
    
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_direct_binding(num_episodes: int = 100):
    """Benchmark the new direct Game binding."""
    import random
    
    game_config = {
        'smallBlind': 10,
        'bigBlind': 20,
        'startingChips': 1000,
        'minPlayers': 2,
        'maxPlayers': 2,
    }
    
    start = time.perf_counter()
    
    for _ in range(num_episodes):
        # Create game directly (FAST!)
        config = poker_api_binding.GameConfig()
        config.smallBlind = game_config['smallBlind']
        config.bigBlind = game_config['bigBlind']
        config.startingChips = game_config['startingChips']
        config.minPlayers = game_config['minPlayers']
        config.maxPlayers = game_config['maxPlayers']
        config.seed = random.randint(0, 1000000)
        
        game = poker_api_binding.Game(config)
        game.add_player('p0', 'Player_p0', config.startingChips)
        game.add_player('p1', 'Player_p1', config.startingChips)
        game.start_hand()
        
        # Get initial state (no JSON parsing!)
        game_state = game.get_state_dict()
        
        # Simulate 6 actions
        for step in range(6):
            stage = game.get_stage_name().lower()
            if stage in {'complete', 'showdown'}:
                break
            
            current_player = game.get_current_player_id()
            if not current_player:
                if not game.advance_game():
                    break
                continue
            
            # Get state (fast!)
            game_state = game.get_state_dict()
            legal_actions = game_state.get('actionConstraints', {}).get('legalActions', [])
            
            if not legal_actions:
                break
            
            # Process action directly (O(1) per step!)
            action = 'check' if 'check' in legal_actions else 'fold'
            game.process_action(current_player, action, 0)
    
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_full_episode(num_episodes: int = 50):
    """Benchmark full episode collection with model inference."""
    from train import RLTrainingSession
    from ppo import PPOTrainer
    from pathlib import Path
    import tempfile
    
    device = torch.device('cpu')
    
    # Create a small model for benchmarking
    model = create_actor_critic(
        input_dim=256,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        gradient_checkpointing=False
    )
    model.to(device)
    model.eval()
    
    ppo_trainer = PPOTrainer(
        model=model,
        learning_rate=1e-4,
    )
    
    game_config = {
        'smallBlind': 10,
        'bigBlind': 20,
        'startingChips': 1000,
        'minPlayers': 2,
        'maxPlayers': 2,
        'num_players': 2,
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        session = RLTrainingSession(
            model=model,
            ppo_trainer=ppo_trainer,
            game_config=game_config,
            device=device,
            output_dir=Path(tmpdir),
            log_level=0,
            num_runouts=0,
            num_workers=0,
        )
        
        # Benchmark collect_episode (old JSON-based)
        print("  Benchmarking collect_episode (JSON-based)...")
        start = time.perf_counter()
        for _ in range(num_episodes):
            session.collect_episode(use_opponent_pool=False, verbose=False)
        old_time = time.perf_counter() - start
        
        # Benchmark collect_episode_fast (direct binding)
        print("  Benchmarking collect_episode_fast (direct binding)...")
        start = time.perf_counter()
        for _ in range(num_episodes):
            session.collect_episode_fast(use_opponent_pool=False, verbose=False)
        new_time = time.perf_counter() - start
        
        return old_time, new_time


def benchmark_threaded_collection(session, num_episodes: int = 100):
    """Benchmark threaded episode collection."""
    import concurrent.futures
    
    max_workers = min(num_episodes, max(1, (os.cpu_count() or 4) - 1), 20)
    
    start = time.perf_counter()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(session.collect_episode_fast, use_opponent_pool=False)
            for _ in range(num_episodes)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    elapsed = time.perf_counter() - start
    return elapsed, len(results)


def benchmark_multiprocess_collection(num_episodes: int = 100, num_workers: int = 4):
    """Benchmark multi-process episode collection."""
    from parallel_rollouts import ParallelRolloutManager
    from pathlib import Path
    import tempfile
    
    device = torch.device('cpu')
    
    # Create model
    model = create_actor_critic(
        input_dim=256,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        gradient_checkpointing=False
    )
    model.to(device)
    model.eval()
    
    game_config = {
        'smallBlind': 10,
        'bigBlind': 20,
        'startingChips': 1000,
        'minPlayers': 2,
        'maxPlayers': 2,
        'num_players': 2,
    }
    
    model_config = {
        'input_dim': 256,
        'hidden_dim': 256,
        'num_heads': 4,
        'num_layers': 2,
    }
    
    manager = ParallelRolloutManager(
        num_workers=num_workers,
        game_config=game_config,
        model_config=model_config,
        model=model,
        device='cpu',
        opponent_pool=[],
    )
    
    try:
        manager.start()
        
        start = time.perf_counter()
        episodes = manager.collect_episodes(
            num_episodes=num_episodes,
            use_opponent_pool=False,
            deterministic=False,
            verbose=False,
        )
        elapsed = time.perf_counter() - start
        
        return elapsed, len(episodes)
    finally:
        manager.shutdown()


def main():
    print("=" * 60)
    print("Poker Rollout Performance Benchmark")
    print("=" * 60)
    print()
    
    num_episodes = 200
    num_cpus = os.cpu_count() or 4
    
    # Test 1: Pure API overhead
    print(f"Test 1: API Overhead (no model inference, 100 episodes)")
    print("-" * 60)
    
    print("  Running JSON-based stateless API...")
    json_time = benchmark_json_api(100)
    print(f"  JSON API:      {json_time:.3f}s ({100/json_time:.1f} episodes/sec)")
    
    print("  Running direct Game binding...")
    direct_time = benchmark_direct_binding(100)
    print(f"  Direct binding: {direct_time:.3f}s ({100/direct_time:.1f} episodes/sec)")
    
    speedup = json_time / direct_time
    print(f"\n  Speedup: {speedup:.1f}x faster with direct binding!")
    print()
    
    # Test 2: Sequential episode collection (baseline)
    print(f"Test 2: Sequential Episode Collection ({num_episodes} episodes)")
    print("-" * 60)
    
    from train import RLTrainingSession
    from ppo import PPOTrainer
    from pathlib import Path
    import tempfile
    
    device = torch.device('cpu')
    
    model = create_actor_critic(
        input_dim=256,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        gradient_checkpointing=False
    )
    model.to(device)
    model.eval()
    
    ppo_trainer = PPOTrainer(model=model, learning_rate=1e-4)
    
    game_config = {
        'smallBlind': 10,
        'bigBlind': 20,
        'startingChips': 1000,
        'minPlayers': 2,
        'maxPlayers': 2,
        'num_players': 2,
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        session = RLTrainingSession(
            model=model,
            ppo_trainer=ppo_trainer,
            game_config=game_config,
            device=device,
            output_dir=Path(tmpdir),
            log_level=0,
            num_runouts=0,
            num_workers=0,
        )
        
        print("  Running sequential collection...")
        start = time.perf_counter()
        for _ in range(num_episodes):
            session.collect_episode_fast(use_opponent_pool=False)
        sequential_time = time.perf_counter() - start
        print(f"  Sequential: {sequential_time:.3f}s ({num_episodes/sequential_time:.1f} episodes/sec)")
        
        # Test 3: Threaded collection
        print()
        print(f"Test 3: Threaded Collection ({num_episodes} episodes, {num_cpus} threads)")
        print("-" * 60)
        
        print("  Running threaded collection...")
        threaded_time, _ = benchmark_threaded_collection(session, num_episodes)
        print(f"  Threaded:   {threaded_time:.3f}s ({num_episodes/threaded_time:.1f} episodes/sec)")
        
        thread_speedup = sequential_time / threaded_time
        print(f"\n  Speedup vs sequential: {thread_speedup:.1f}x")
        print("  (Limited by Python GIL for CPU-bound model inference)")
    
    # Test 4: Multi-process collection
    print()
    num_workers = max(4, num_cpus - 1)
    print(f"Test 4: Multi-Process Collection ({num_episodes} episodes, {num_workers} workers)")
    print("-" * 60)
    
    print("  Starting workers (this may take a moment)...")
    try:
        mp_time, mp_count = benchmark_multiprocess_collection(num_episodes, num_workers)
        print(f"  Multi-process: {mp_time:.3f}s ({mp_count/mp_time:.1f} episodes/sec)")
        
        mp_speedup = sequential_time / mp_time
        print(f"\n  Speedup vs sequential: {mp_speedup:.1f}x")
        print("  (True parallelism, bypasses GIL)")
    except Exception as e:
        print(f"  Multi-process benchmark failed: {e}")
        mp_speedup = 0
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  CPU cores available:     {num_cpus}")
    print(f"  API overhead reduction:  {json_time/direct_time:.1f}x")
    print(f"  Thread parallelism:      {thread_speedup:.1f}x")
    if mp_speedup > 0:
        print(f"  Multi-process speedup:   {mp_speedup:.1f}x")
    print()
    print("Key takeaways:")
    print("  - Model inference dominates episode collection time")
    print("  - Threading provides some parallelism but is GIL-limited")
    print("  - Multi-process provides true parallel speedup")
    print()
    print("Recommendation: Use --num-workers N for best performance")


if __name__ == '__main__':
    main()

