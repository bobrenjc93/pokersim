#!/usr/bin/env python3
"""
Poker AI ELO Rating Web Server
Real-time ELO rating simulation with Server-Sent Events.
"""

import os
import sys
import json
import queue
import threading
import time
import random
from pathlib import Path
import torch
from flask import Flask, render_template, request, jsonify, Response

# Add training directory to path
TRAINING_DIR = Path(__file__).parent.parent / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.append(str(TRAINING_DIR))

try:
    from engine import PokerEloArena, EloCalculator, parse_checkpoints
    from config import DEFAULT_MODELS_DIR
except ImportError as e:
    try:
        from elo.engine import PokerEloArena, EloCalculator, parse_checkpoints
        from elo.config import DEFAULT_MODELS_DIR
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


class EloJobManager:
    """Manages ELO simulation jobs with real-time streaming."""
    
    def __init__(self):
        self.thread = None
        self.stop_event = threading.Event()
        self.status = {
            "status": "idle",
            "current_match": "",
            "matches_played": 0,
            "total_matches": 0,
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
    
    def start_job(
        self, 
        models_dir: str = DEFAULT_MODELS_DIR,
        k_factor: float = 40.0
    ):
        """
        Start an ELO simulation job with freezeout matches (1000 chips, 40 BB).
        Runs indefinitely until explicitly stopped.
        
        Args:
            models_dir: Directory containing model checkpoints
            k_factor: K-factor for ELO calculations (default 40 for faster rating spread)
        """
        if self.thread and self.thread.is_alive():
            return False, "Job already running"
        
        self.stop_event.clear()
        self.status = {
            "status": "running",
            "current_match": "Initializing...",
            "matches_played": 0,
            "current_round": 0,
        }
        
        self.broadcast({'type': 'job_started', 'device': DEVICE})
        
        self.thread = threading.Thread(
            target=self._run_simulation,
            args=(models_dir, k_factor),
            daemon=True
        )
        self.thread.start()
        return True, "Started"
    
    def stop_job(self):
        """Stop the current job. The simulation thread will broadcast final results."""
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            return True
        return False
    
    def _run_simulation(
        self, 
        models_dir_str: str, 
        k_factor: float
    ):
        """Run ELO simulation with freezeout matches (1000 chips, 40 BB).
        Runs indefinitely until explicitly stopped."""
        try:
            self.arena = PokerEloArena(
                device=DEVICE,
                k_factor=k_factor
            )
            
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
            
            # Select 5 evenly spaced checkpoints
            num_to_select = 5
            if len(sorted_cps) > num_to_select:
                # Calculate evenly spaced indices
                indices = [int(i * (len(sorted_cps) - 1) / (num_to_select - 1)) for i in range(num_to_select)]
                sorted_cps = [sorted_cps[i] for i in indices]
                print(f"Selected {len(sorted_cps)} evenly spaced checkpoints from {len(checkpoints)} total")
            else:
                print(f"Using all {len(sorted_cps)} checkpoints as participants")
            
            # Build participant list
            participants = {}
            
            # Add model checkpoints
            for iter_num, path in sorted_cps:
                player_id = f"iter_{iter_num}"
                participants[player_id] = {
                    'type': 'model',
                    'path': str(path),
                    'name': f'Iter_{iter_num}',
                    'is_bot': False,
                    'iteration': iter_num
                }
                # Initialize rating
                self.arena.get_or_create_rating(player_id)
            
            player_ids = list(participants.keys())
            num_players = len(player_ids)
            
            if num_players < 2:
                self.broadcast({
                    'type': 'error',
                    'message': 'Need at least 2 participants'
                })
                return
            
            # Broadcast simulation start
            self.broadcast({
                'type': 'simulation_started',
                'num_checkpoints': len(participants),
                'total_participants': num_players,
                'k_factor': k_factor,
                'participants': {
                    pid: {
                        'name': p['name'],
                        'initial_rating': self.arena.ratings[pid].rating
                    }
                    for pid, p in participants.items()
                }
            })
            
            match_num = 0
            round_num = 0
            
            # Run indefinitely until stopped
            while not self.stop_event.is_set():
                round_num += 1
                self.status['current_round'] = round_num
                
                # Shuffle players for random pairings
                shuffled = player_ids.copy()
                random.shuffle(shuffled)
                
                # Pair up players
                for i in range(0, len(shuffled) - 1, 2):
                    if self.stop_event.is_set():
                        break
                    
                    player_a_id = shuffled[i]
                    player_b_id = shuffled[i + 1]
                    
                    config_a = participants[player_a_id]
                    config_b = participants[player_b_id]
                    
                    match_num += 1
                    self.status['matches_played'] = match_num
                    self.status['current_match'] = f"{config_a['name']} vs {config_b['name']}"
                    
                    self.broadcast({
                        'type': 'match_started',
                        'match_num': match_num,
                        'round': round_num,
                        'player_a': player_a_id,
                        'player_b': player_b_id,
                        'name_a': config_a['name'],
                        'name_b': config_b['name'],
                        'rating_a': self.arena.ratings[player_a_id].rating,
                        'rating_b': self.arena.ratings[player_b_id].rating,
                    })
                    
                    try:
                        result = self.arena.play_match(
                            player_a_id, player_b_id,
                            config_a, config_b
                        )
                        
                        if result.get('error'):
                            continue
                        
                        # Broadcast match result (best-of-9 rounds)
                        self.broadcast({
                            'type': 'match_complete',
                            'match_num': match_num,
                            'round': round_num,
                            'player_a': player_a_id,
                            'player_b': player_b_id,
                            'name_a': config_a['name'],
                            'name_b': config_b['name'],
                            'rounds_played': result['rounds_played'],
                            'round_wins_a': result['round_wins_a'],
                            'round_wins_b': result['round_wins_b'],
                            'hands_played': result['hands_played'],
                            'hand_wins_a': result['hand_wins_a'],
                            'hand_wins_b': result['hand_wins_b'],
                            'score_a': result['score_a'],
                            'winner': 'a' if result['score_a'] == 1.0 else ('b' if result['score_a'] == 0.0 else 'draw'),
                            'old_rating_a': result['old_rating_a'],
                            'old_rating_b': result['old_rating_b'],
                            'new_rating_a': result['new_rating_a'],
                            'new_rating_b': result['new_rating_b'],
                            'rating_change_a': result['rating_change_a'],
                            'rating_change_b': result['rating_change_b'],
                        })
                        
                    except Exception as e:
                        print(f"Error in match: {e}")
                        self.broadcast({'type': 'error', 'message': str(e)})
                
                # Broadcast round complete with full leaderboard
                self.broadcast({
                    'type': 'round_complete',
                    'round': round_num,
                    'leaderboard': self._get_leaderboard_data(participants),
                    'rating_histories': self._get_rating_histories()
                })
            
            # Stopped by user
            self.status["status"] = "stopped"
            self.status["current_match"] = "Stopped"
            self.broadcast({
                'type': 'job_stopped',
                'total_rounds': round_num,
                'total_matches': match_num,
                'leaderboard': self._get_leaderboard_data(participants),
                'rating_histories': self._get_rating_histories()
            })
                
        except Exception as e:
            print(f"Job failed: {e}")
            import traceback
            traceback.print_exc()
            self.status["status"] = "error"
            self.status["current_match"] = f"Error: {str(e)}"
            self.broadcast({'type': 'error', 'message': str(e)})
    
    def _get_leaderboard_data(self, participants: dict) -> list:
        """Get leaderboard data with participant info."""
        leaderboard = []
        for player_id, rating in self.arena.ratings.items():
            participant = participants.get(player_id, {})
            leaderboard.append({
                'player_id': player_id,
                'name': participant.get('name', player_id),
                'rating': rating.rating,
                'games_played': rating.games_played,
                'wins': rating.wins,
                'losses': rating.losses,
                'draws': rating.draws,
                'win_rate': rating.wins / rating.games_played if rating.games_played > 0 else 0
            })
        return sorted(leaderboard, key=lambda x: x['rating'], reverse=True)
    
    def _get_rating_histories(self) -> dict:
        """Get rating history for all players."""
        histories = {}
        for player_id, rating in self.arena.ratings.items():
            histories[player_id] = rating.rating_history
        return histories


job_manager = EloJobManager()


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
    """Get server configuration."""
    return jsonify({'device': DEVICE})


@app.route('/api/start', methods=['POST'])
def start_job():
    """Start an ELO simulation job with freezeout matches (1000 chips, 40 BB).
    Runs indefinitely until explicitly stopped."""
    data = request.json or {}
    models_dir = data.get('models_dir', DEFAULT_MODELS_DIR)
    k_factor = data.get('k_factor', 40.0)
    
    success, msg = job_manager.start_job(
        models_dir=models_dir,
        k_factor=k_factor
    )
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
            yield f"data: {json.dumps({'type': 'status', **job_manager.status})}\n\n"
            
            while True:
                try:
                    event = q.get(timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
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
    port = int(os.environ.get('PORT', 5051))
    print(f"Starting Poker ELO Server on http://localhost:{port}")
    print(f"Detected compute device: {DEVICE}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)


if __name__ == '__main__':
    main()

