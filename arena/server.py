#!/usr/bin/env python3
"""
Poker AI Arena Web Server
Real-time streaming of match results via Server-Sent Events.
"""

import os
import sys
import json
import queue
import threading
import time
from pathlib import Path
import torch
from flask import Flask, render_template, request, jsonify, Response

# Add training directory to path
TRAINING_DIR = Path(__file__).parent.parent / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.append(str(TRAINING_DIR))

try:
    from engine import Arena, parse_checkpoints, RoundRobinTournament
    from config import DEFAULT_MODELS_DIR
except ImportError as e:
    try:
        from arena.engine import Arena, parse_checkpoints, RoundRobinTournament
        from arena.config import DEFAULT_MODELS_DIR
    except ImportError:
        print(f"Error importing modules: {e}")
        sys.exit(1)


def detect_device() -> str:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Auto-detect device at startup
DEVICE = detect_device()

app = Flask(__name__)


class StreamingArena(Arena):
    """Arena that yields results for each hand played."""
    
    def play_match_streaming(self, agent_a_config: dict, agent_b_config: dict, 
                             num_hands: int = 100, hand_callback=None) -> dict:
        """
        Play a match, calling callback after each hand for real-time updates.
        Returns final match result.
        """
        name_a = agent_a_config.get('name', 'Agent A')
        name_b = agent_b_config.get('name', 'Agent B')
        
        # Pre-load models
        config_a = agent_a_config.copy()
        config_b = agent_b_config.copy()
        
        if config_a.get('type') == 'model' and 'model' not in config_a and 'path' in config_a:
            config_a['model'] = self.load_model(Path(config_a['path']))
            
        if config_b.get('type') == 'model' and 'model' not in config_b and 'path' in config_b:
            config_b['model'] = self.load_model(Path(config_b['path']))
        
        wins_a = 0
        wins_b = 0
        total_rewards_a = 0
        total_rewards_b = 0
        hands_played = 0
        
        for j in range(num_hands):
            swap = (j % 2 == 1)
            
            try:
                if swap:
                    res = self.play_hand(config_b, config_a, capture_details=True)
                    reward_a = res.get('p1', 0)
                    reward_b = res.get('p0', 0)
                else:
                    res = self.play_hand(config_a, config_b, capture_details=True)
                    reward_a = res.get('p0', 0)
                    reward_b = res.get('p1', 0)
                
                if res.get('error'):
                    continue
                    
                hands_played += 1
                total_rewards_a += reward_a
                total_rewards_b += reward_b
                
                if reward_a > 0:
                    wins_a += 1
                if reward_b > 0:
                    wins_b += 1
                
                # Call callback for real-time updates
                if hand_callback:
                    # Extract and transform hand details for the callback
                    details = res.get('details')
                    hand_details = None
                    if details:
                        # Transform details to use consistent agent naming
                        hand_details = {
                            'hole_cards': {},
                            'community_cards': details.get('community_cards', []),
                            'actions': [],
                            'final_pot': details.get('final_state', {}).get('pot', 0),
                            'players': []
                        }
                        
                        # Map player IDs to agent names based on swap
                        if swap:
                            id_to_name = {'p0': name_b, 'p1': name_a}
                            id_to_agent = {'p0': 'b', 'p1': 'a'}
                        else:
                            id_to_name = {'p0': name_a, 'p1': name_b}
                            id_to_agent = {'p0': 'a', 'p1': 'b'}
                        
                        # Transform hole cards
                        for pid, cards in details.get('hole_cards', {}).items():
                            agent_key = id_to_agent.get(pid, pid)
                            hand_details['hole_cards'][agent_key] = cards
                        
                        # Transform actions with proper agent names
                        for action in details.get('actions', []):
                            pid = action.get('player_id')
                            hand_details['actions'].append({
                                'agent': id_to_agent.get(pid, pid),
                                'agent_name': id_to_name.get(pid, action.get('player_name', pid)),
                                'action': action.get('action'),
                                'amount': action.get('amount', 0),
                                'stage': action.get('stage'),
                                'pot_before': action.get('pot_before', 0)
                            })
                        
                        # Transform final player states
                        for p in details.get('final_state', {}).get('players', []):
                            pid = p.get('id')
                            hand_details['players'].append({
                                'agent': id_to_agent.get(pid, pid),
                                'name': id_to_name.get(pid, p.get('name', pid)),
                                'hole_cards': p.get('hole_cards', []),
                                'chips': p.get('chips', 0),
                                'profit_bb': p.get('profit_bb', 0)
                            })
                    
                    hand_callback({
                        'type': 'hand',
                        'hand_num': hands_played,
                        'total_hands': num_hands,
                        'reward_a': reward_a,
                        'reward_b': reward_b,
                        'winner': 'a' if reward_a > 0 else ('b' if reward_b > 0 else 'tie'),
                        'agent_a': name_a,
                        'agent_b': name_b,
                        'details': hand_details,
                        'cumulative': {
                            'wins_a': wins_a,
                            'wins_b': wins_b,
                            'win_rate_a': wins_a / hands_played if hands_played > 0 else 0,
                            'avg_bb_a': total_rewards_a / hands_played if hands_played > 0 else 0,
                            'bb_100_a': (total_rewards_a / hands_played) * 100 if hands_played > 0 else 0,
                        }
                    })
                
            except Exception as e:
                if hand_callback:
                    hand_callback({'type': 'error', 'message': str(e), 'hand_num': j + 1})
        
        # Return final match result
        n = hands_played
        if n > 0:
            return {
                'agent_a': name_a,
                'agent_b': name_b,
                'hands': n,
                'win_rate_a': wins_a / n,
                'win_rate_b': wins_b / n,
                'avg_bb_a': total_rewards_a / n,
                'avg_bb_b': total_rewards_b / n,
                'bb_100_a': (total_rewards_a / n) * 100
            }
        return {}


class JobManager:
    """Manages evaluation jobs with real-time streaming."""
    
    def __init__(self):
        self.thread = None
        self.stop_event = threading.Event()
        self.event_queue = queue.Queue()
        self.status = {
            "status": "idle",
            "current_match": "",
            "progress": 0,
            "results": [],
            "current_hands": []
        }
        self.arena = None
        self.subscribers = []
        self.lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        """Create a new subscriber queue for SSE."""
        q = queue.Queue()
        with self.lock:
            self.subscribers.append(q)
        return q
    
    def unsubscribe(self, q: queue.Queue):
        """Remove a subscriber."""
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def broadcast(self, event: dict):
        """Send event to all subscribers."""
        with self.lock:
            for q in self.subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    def start_job(self, models_dir: str = DEFAULT_MODELS_DIR):
        """Start an evaluation job that plays until one player wins all chips.
        
        Args:
            models_dir: Directory containing model checkpoints
        """
        if self.thread and self.thread.is_alive():
            return False, "Job already running"

        self.stop_event.clear()
        self.status = {
            "status": "running",
            "current_match": "Initializing...",
            "progress": 0,
            "hands_played": 0,
            "results": [],
        }
        
        self.broadcast({'type': 'job_started', 'mode': 'play_until_winner', 'device': DEVICE})
        
        self.thread = threading.Thread(
            target=self._run_evaluation,
            args=(models_dir,),
            daemon=True
        )
        self.thread.start()
        return True, "Started"

    def stop_job(self):
        """Stop the current job."""
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.broadcast({'type': 'job_stopped'})
            return True
        return False

    def _run_evaluation(self, models_dir_str: str):
        """Run round-robin tournament evaluation, streaming results."""
        try:
            self.arena = StreamingArena(device=DEVICE, output_dir="arena_results")
            models_dir = Path(models_dir_str)
            checkpoints = parse_checkpoints(models_dir)
            
            if not checkpoints:
                self.status["status"] = "error"
                self.status["current_match"] = f"No checkpoints found in {models_dir}"
                self.broadcast({
                    'type': 'error',
                    'message': f"No checkpoints found in {models_dir}"
                })
                return

            sorted_cps = sorted(checkpoints, key=lambda x: x[0])
            
            # Limit to max 4 checkpoints with evenly spaced sampling
            MAX_CHECKPOINTS = 4
            if len(sorted_cps) > MAX_CHECKPOINTS:
                n = len(sorted_cps)
                # Always include first and last, then evenly space the rest
                indices = [0]  # First
                step = (n - 1) / (MAX_CHECKPOINTS - 1)
                for i in range(1, MAX_CHECKPOINTS - 1):
                    indices.append(round(i * step))
                indices.append(n - 1)  # Last
                # Remove duplicates while preserving order
                seen = set()
                unique_indices = []
                for idx in indices:
                    if idx not in seen:
                        seen.add(idx)
                        unique_indices.append(idx)
                sorted_cps = [sorted_cps[i] for i in unique_indices]
                print(f"Sampled {len(sorted_cps)} checkpoints from {n} total (evenly spaced)")
            
            self._run_round_robin(sorted_cps)

            if not self.stop_event.is_set():
                self.status["status"] = "completed"
                self.status["progress"] = 100
                self.status["current_match"] = "Complete"
                self.broadcast({'type': 'job_completed', 'results': self.status['results']})
            else:
                self.status["status"] = "stopped"
                self.broadcast({'type': 'job_stopped'})

        except Exception as e:
            print(f"Job failed: {e}")
            import traceback
            traceback.print_exc()
            self.status["status"] = "error"
            self.status["current_match"] = f"Error: {str(e)}"
            self.broadcast({'type': 'error', 'message': str(e)})

    def _run_round_robin(self, sorted_cps):
        """
        Run bankroll survival tournament - each iteration starts with 10k chips.
        Plays until one player wins all chips (all others eliminated).
        """
        import random as rand
        
        STARTING_CHIPS = 250000
        BIG_BLIND = 40  # BB used for chip calculations
        
        total_checkpoints = len(sorted_cps)
        
        if total_checkpoints < 2:
            self.broadcast({
                'type': 'error',
                'message': 'Need at least 2 checkpoints for tournament'
            })
            return
        
        # Initialize results tracking for each checkpoint with starting chips
        results = {}
        chip_history = {}  # iter -> list of (hand_num, chips)
        
        for iter_num, path in sorted_cps:
            results[iter_num] = {
                'iteration': iter_num,
                'path': str(path),
                'chips': STARTING_CHIPS,
                'total_hands': 0,
                'total_wins': 0,
                'total_bb': 0.0,
                'eliminated': False,
                'eliminated_at_hand': None,
            }
            chip_history[iter_num] = [(0, STARTING_CHIPS)]  # Initial chips at hand 0
        
        # Pre-load all models to avoid reloading during tournament
        print(f"Pre-loading {total_checkpoints} checkpoints...")
        checkpoint_configs = {}
        for iter_num, path in sorted_cps:
            try:
                model = self.arena.load_model(path)
                checkpoint_configs[iter_num] = {
                    'type': 'model',
                    'model': model,
                    'path': str(path),
                    'name': f'Iter_{iter_num}'
                }
            except Exception as e:
                print(f"Error loading checkpoint {iter_num}: {e}")
                continue
        
        available_iters = list(checkpoint_configs.keys())
        active_iters = set(available_iters)  # Track which iterations are still alive
        
        if len(available_iters) < 2:
            self.broadcast({
                'type': 'error',
                'message': 'Need at least 2 successfully loaded checkpoints'
            })
            return
        
        # Broadcast tournament start with chip info
        self.broadcast({
            'type': 'tournament_started',
            'mode': 'play-until-winner',
            'checkpoints': len(available_iters),
            'starting_chips': STARTING_CHIPS,
            'initial_chip_history': {iter_num: STARTING_CHIPS for iter_num in available_iters}
        })
        
        # Play hands until only one player remains
        hand_num = 0
        while len(active_iters) > 1 and not self.stop_event.is_set():
            hand_num += 1
            
            # Randomly select two different ACTIVE checkpoints
            active_list = list(active_iters)
            iter_a, iter_b = rand.sample(active_list, 2)
            config_a = checkpoint_configs[iter_a]
            config_b = checkpoint_configs[iter_b]
            
            # Randomly decide who is p0 (position matters in poker)
            swap = rand.random() < 0.5
            
            self.status["current_match"] = f"Hand {hand_num}: {config_a['name']} vs {config_b['name']}"
            self.status["hands_played"] = hand_num
            self.status["active_players"] = len(active_iters)
            
            try:
                if swap:
                    res = self.arena.play_hand(config_b, config_a, capture_details=True)
                    reward_a = res.get('p1', 0)
                    reward_b = res.get('p0', 0)
                else:
                    res = self.arena.play_hand(config_a, config_b, capture_details=True)
                    reward_a = res.get('p0', 0)
                    reward_b = res.get('p1', 0)
                
                if res.get('error'):
                    continue
                
                # Convert BB rewards to chips
                chip_change_a = int(reward_a * BIG_BLIND)
                chip_change_b = int(reward_b * BIG_BLIND)
                
                # Update stats and chips for both checkpoints
                stats_a = results[iter_a]
                stats_a['total_hands'] += 1
                stats_a['total_bb'] += reward_a
                stats_a['chips'] += chip_change_a
                if reward_a > 0:
                    stats_a['total_wins'] += 1
                
                stats_b = results[iter_b]
                stats_b['total_hands'] += 1
                stats_b['total_bb'] += reward_b
                stats_b['chips'] += chip_change_b
                if reward_b > 0:
                    stats_b['total_wins'] += 1
                
                # Record chip history
                chip_history[iter_a].append((hand_num, max(0, stats_a['chips'])))
                chip_history[iter_b].append((hand_num, max(0, stats_b['chips'])))
                
                # Check for eliminations
                eliminated_this_hand = []
                
                if stats_a['chips'] <= 0 and not stats_a['eliminated']:
                    stats_a['eliminated'] = True
                    stats_a['eliminated_at_hand'] = hand_num
                    stats_a['chips'] = 0
                    active_iters.discard(iter_a)
                    eliminated_this_hand.append(iter_a)
                    print(f"Iteration {iter_a} ELIMINATED at hand {hand_num}")
                
                if stats_b['chips'] <= 0 and not stats_b['eliminated']:
                    stats_b['eliminated'] = True
                    stats_b['eliminated_at_hand'] = hand_num
                    stats_b['chips'] = 0
                    active_iters.discard(iter_b)
                    eliminated_this_hand.append(iter_b)
                    print(f"Iteration {iter_b} ELIMINATED at hand {hand_num}")
                
                # Build hand details for broadcast
                details = res.get('details')
                hand_details = None
                if details:
                    hand_details = {
                        'hole_cards': {},
                        'community_cards': details.get('community_cards', []),
                        'actions': [],
                        'final_pot': details.get('final_state', {}).get('pot', 0),
                        'players': []
                    }
                    
                    if swap:
                        id_to_name = {'p0': config_b['name'], 'p1': config_a['name']}
                        id_to_agent = {'p0': 'b', 'p1': 'a'}
                    else:
                        id_to_name = {'p0': config_a['name'], 'p1': config_b['name']}
                        id_to_agent = {'p0': 'a', 'p1': 'b'}
                    
                    for pid, cards in details.get('hole_cards', {}).items():
                        agent_key = id_to_agent.get(pid, pid)
                        hand_details['hole_cards'][agent_key] = cards
                    
                    for action in details.get('actions', []):
                        pid = action.get('player_id')
                        hand_details['actions'].append({
                            'agent': id_to_agent.get(pid, pid),
                            'agent_name': id_to_name.get(pid, action.get('player_name', pid)),
                            'action': action.get('action'),
                            'amount': action.get('amount', 0),
                            'stage': action.get('stage'),
                            'pot_before': action.get('pot_before', 0)
                        })
                    
                    for p in details.get('final_state', {}).get('players', []):
                        pid = p.get('id')
                        hand_details['players'].append({
                            'agent': id_to_agent.get(pid, pid),
                            'name': id_to_name.get(pid, p.get('name', pid)),
                            'hole_cards': p.get('hole_cards', []),
                            'chips': p.get('chips', 0),
                            'profit_bb': p.get('profit_bb', 0)
                        })
                
                # Calculate current stats for both players
                def calc_stats(stats):
                    th = stats['total_hands']
                    return {
                        'iteration': stats['iteration'],
                        'total_hands': th,
                        'win_rate': stats['total_wins'] / th if th > 0 else 0,
                        'bb_100': (stats['total_bb'] / th * 100) if th > 0 else 0,
                        'chips': stats['chips'],
                        'eliminated': stats['eliminated'],
                    }
                
                # Broadcast hand result with chip info
                self.broadcast({
                    'type': 'hand',
                    'hand_num': hand_num,
                    'agent_a': config_a['name'],
                    'agent_b': config_b['name'],
                    'iter_a': iter_a,
                    'iter_b': iter_b,
                    'reward_a': reward_a,
                    'reward_b': reward_b,
                    'chip_change_a': chip_change_a,
                    'chip_change_b': chip_change_b,
                    'winner': 'a' if reward_a > 0 else ('b' if reward_b > 0 else 'tie'),
                    'details': hand_details,
                    'stats_a': calc_stats(stats_a),
                    'stats_b': calc_stats(stats_b),
                    'eliminated': eliminated_this_hand,
                    'active_count': len(active_iters),
                })
                
                # Periodically broadcast full chip history and leaderboard (every 10 hands)
                if hand_num % 10 == 0:
                    self._broadcast_chip_update(results, chip_history, active_iters)
                    self._broadcast_leaderboard(results)
                
            except Exception as e:
                print(f"Error in hand {hand_num}: {e}")
                self.broadcast({'type': 'error', 'message': str(e)})
        
        # Tournament ended - broadcast winner if there is one
        if len(active_iters) == 1:
            winner_iter = list(active_iters)[0]
            self.broadcast({
                'type': 'tournament_winner',
                'winner_iteration': winner_iter,
                'winner_name': f'Iter_{winner_iter}',
                'final_chips': results[winner_iter]['chips'],
                'hands_played': hand_num
            })
            print(f"Tournament winner: Iter_{winner_iter} with {results[winner_iter]['chips']} chips after {hand_num} hands")
        elif len(active_iters) == 0:
            # Edge case: both eliminated simultaneously
            self.broadcast({
                'type': 'tournament_ended',
                'reason': 'draw',
                'hands_played': hand_num
            })
        
        # Final broadcasts
        self._broadcast_chip_update(results, chip_history, active_iters, final=True)
        self._broadcast_leaderboard(results, final=True)
    
    def _broadcast_chip_update(self, results, chip_history, active_iters, final=False):
        """Broadcast chip counts for all iterations."""
        chip_data = {}
        for iter_num in results:
            stats = results[iter_num]
            chip_data[iter_num] = {
                'chips': stats['chips'],
                'eliminated': stats['eliminated'],
                'eliminated_at_hand': stats['eliminated_at_hand'],
                'history': chip_history[iter_num],  # Full chip history
            }
        
        self.broadcast({
            'type': 'chip_update',
            'chip_data': chip_data,
            'active_count': len(active_iters),
            'final': final
        })
    
    def _broadcast_leaderboard(self, results, final=False):
        """Broadcast current leaderboard standings - sorted by chips (survival)."""
        leaderboard = []
        for iter_num in sorted(results.keys()):
            stats = results[iter_num]
            total_hands = stats['total_hands']
            row = {
                'iteration': iter_num,
                'total_hands': total_hands,
                'win_rate': stats['total_wins'] / total_hands if total_hands > 0 else 0,
                'bb_100': (stats['total_bb'] / total_hands * 100) if total_hands > 0 else 0,
                'chips': stats['chips'],
                'eliminated': stats['eliminated'],
                'eliminated_at_hand': stats['eliminated_at_hand'],
            }
            leaderboard.append(row)
        
        # Sort by chips (active players first, then by chip count)
        leaderboard.sort(key=lambda x: (not x['eliminated'], x['chips']), reverse=True)
        
        event_type = 'tournament_performance' if final else 'leaderboard_update'
        self.broadcast({
            'type': event_type,
            'leaderboard': leaderboard
        })


job_manager = JobManager()


@app.route('/')
def index():
    """Serve the main UI."""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Get current job status."""
    return jsonify(job_manager.status)


@app.route('/api/config')
def get_config():
    """Get server configuration (detected device, etc.)."""
    return jsonify({'device': DEVICE})


@app.route('/api/start', methods=['POST'])
def start_job():
    """Start an evaluation job that plays until one player wins all chips."""
    data = request.json or {}
    models_dir = data.get('models_dir', DEFAULT_MODELS_DIR)
    
    success, msg = job_manager.start_job(models_dir)
    return jsonify({'success': success, 'message': msg})


@app.route('/api/stop', methods=['POST'])
def stop_job():
    """Stop the current job."""
    success = job_manager.stop_job()
    return jsonify({'success': success})


@app.route('/api/stream')
def stream():
    """Server-Sent Events endpoint for real-time updates."""
    def generate():
        q = job_manager.subscribe()
        try:
            # Send current status immediately
            yield f"data: {json.dumps({'type': 'status', **job_manager.status})}\n\n"
            
            while True:
                try:
                    event = q.get(timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    # Send keepalive
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
        except GeneratorExit:
            pass
        finally:
            job_manager.unsubscribe(q)
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


def main():
    """Run the web server."""
    port = int(os.environ.get('PORT', 5050))
    print(f"Starting Poker Arena Server on http://localhost:{port}")
    print(f"Detected compute device: {DEVICE}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
