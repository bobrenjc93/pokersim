#!/usr/bin/env python3
"""Poker AI ELO Rating Web Server with real-time SSE updates."""

import json
import os
import queue
import random
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import torch
from flask import Flask, Response, jsonify, render_template, request

from common import DEFAULT_MODELS_DIR, parse_checkpoints, select_spread_checkpoints, detect_device
from engine import PokerEloArena

MAX_CACHED_MODELS = int(os.environ.get('ELO_MAX_CACHED_MODELS', 2))
DEVICE = str(detect_device())
ELO_BASE_SEED = os.environ.get("ELO_SEED", "0")
# Pairing strategy has a big impact on variance and on how quickly ratings stabilize.
# Default to round-robin so "best checkpoint" is less likely to be an artifact of
# random opponent sampling / match order.
ELO_PAIRING = os.environ.get("ELO_PAIRING", "round_robin").lower()  # "random" | "round_robin"

app = Flask(__name__)

def _auto_models_dir(default_dir: str) -> str:
    """
    Choose a sensible models directory.

    If the provided directory exists and has checkpoints, keep it.
    Otherwise, fall back to the newest /tmp/pokersim/rl_models_v* directory
    that contains at least 2 iteration checkpoints.

    This avoids the common footgun where training wrote to v17 but the ELO server
    default points at v18 (or vice versa), which can make it look like "iter_1 is best"
    simply because you're evaluating a different run directory.
    """
    try:
        d = Path(default_dir)
        if d.exists() and parse_checkpoints(d):
            return str(d)
    except Exception:
        pass

    try:
        root = Path("/tmp/pokersim")
        if not root.exists():
            return default_dir
        candidates = []
        for p in root.glob("rl_models_v*"):
            if not p.is_dir():
                continue
            cps = parse_checkpoints(p)
            # Need at least 2 participants for ELO to be meaningful.
            if len(cps) >= 2:
                # Prefer newest by mtime.
                try:
                    candidates.append((p.stat().st_mtime, p))
                except Exception:
                    candidates.append((0.0, p))
        if not candidates:
            return default_dir
        candidates.sort(key=lambda x: x[0], reverse=True)
        return str(candidates[0][1])
    except Exception:
        return default_dir

def _fingerprint_model(model) -> dict:
    """Return a lightweight, stable-ish fingerprint for debug/verification."""
    try:
        params = [p.detach().float().cpu().view(-1) for p in model.parameters()]
        if not params:
            return {"n": 0, "l2": 0.0, "mean_abs": 0.0}
        v = torch.cat(params)
        return {"n": int(v.numel()), "l2": float(torch.linalg.vector_norm(v).item()), "mean_abs": float(v.abs().mean().item())}
    except Exception:
        return {"n": 0, "l2": 0.0, "mean_abs": 0.0}

def _file_identity(path: str) -> dict:
    try:
        st = os.stat(path)
        return {"size": int(st.st_size), "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))}
    except Exception:
        return {"size": None, "mtime_ns": None}

def _epoch_from_checkpoint(path: str) -> Optional[int]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "epoch" in raw:
            return int(raw.get("epoch"))
    except Exception:
        return None
    return None

def _as_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _as_float(x, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


class SSEBroadcaster:
    """Thread-safe SSE broadcaster for real-time updates."""
    
    def __init__(self):
        self._subs: List[queue.Queue] = []
        self._lock = threading.Lock()
    
    def subscribe(self) -> queue.Queue:
        # Keep this bounded so slow/disconnected clients can't grow memory forever.
        q = queue.Queue(maxsize=1000)
        with self._lock:
            self._subs.append(q)
        return q
    
    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)
    
    def send(self, event: dict):
        with self._lock:
            for q in self._subs:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    # Drop oldest event to make room (best-effort).
                    try:
                        q.get_nowait()
                        q.put_nowait(event)
                    except Exception:
                        pass


class EloSimulation:
    """Manages ELO simulation with background execution."""
    
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.status = {'status': 'idle', 'current_match': '', 'matches_played': 0}
        self.arena: Optional[PokerEloArena] = None
        self.broadcaster = SSEBroadcaster()
        self._participants: Dict = {}
    
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
    
    def start(self, models_dir: str = DEFAULT_MODELS_DIR, k_factor: float = 40.0, num_models: int = 8) -> tuple:
        if self.is_running:
            return False, 'Job already running'

        # Only auto-fallback when the caller is using the default models dir.
        # If the user explicitly requested a directory, do not silently switch
        # to a different run (that can create the illusion that "iter_1 is best").
        requested_models_dir = str(models_dir)
        resolved_models_dir = requested_models_dir
        if requested_models_dir == str(DEFAULT_MODELS_DIR):
            resolved_models_dir = _auto_models_dir(requested_models_dir)
        else:
            try:
                cps = parse_checkpoints(Path(requested_models_dir))
                if len(cps) < 2:
                    return False, f"Need at least 2 iter_*.pt checkpoints in {requested_models_dir} (found {len(cps)})"
            except Exception:
                return False, f"Invalid models_dir: {requested_models_dir}"

        models_dir = resolved_models_dir
        self._stop.clear()
        self.status = {'status': 'running', 'current_match': 'Initializing...', 'matches_played': 0}
        self.broadcaster.send({
            'type': 'job_started',
            'device': DEVICE,
            'models_dir': models_dir,
            'requested_models_dir': requested_models_dir,
        })
        
        self._thread = threading.Thread(target=self._run, args=(models_dir, k_factor, num_models), daemon=True)
        self._thread.start()
        return True, 'Started'
    
    def stop(self) -> bool:
        if self.is_running:
            self._stop.set()
            # Best-effort join so a subsequent start doesn't keep stale state around.
            try:
                self._thread.join(timeout=2.0)
            except Exception:
                pass
            return True
        return False
    
    def _load_participants(self, models_dir: str, num_models: int) -> bool:
        """Load checkpoints as participants."""
        checkpoints = parse_checkpoints(Path(models_dir))
        if not checkpoints:
            self.broadcaster.send({'type': 'error', 'message': f'No checkpoints in {models_dir}'})
            return False
        
        sorted_cps = sorted(checkpoints, key=lambda x: x[0])
        if len(sorted_cps) > num_models:
            sorted_cps = select_spread_checkpoints(sorted_cps, num_models)
        
        self._participants = {}
        load_failures: List[Dict[str, str]] = []
        warnings: List[Dict[str, str]] = []
        for iter_num, path in sorted_cps:
            pid = 'baseline' if iter_num == -1 else f'iter_{iter_num}'
            name = 'Baseline' if iter_num == -1 else f'Iter_{iter_num}'
            cfg = {
                'type': 'model',
                'path': str(path),
                'name': name,
                'iteration': iter_num,
                # Default to deterministic action selection for stable ELO.
                'deterministic': True,
            }
            # Proactively verify checkpoint loads; drop it if not.
            model, err = self.arena._cache.get_with_error(cfg['path'])
            if model is None:
                load_failures.append({'player_id': pid, 'path': cfg['path'], 'error': err or 'Unknown load error'})
                continue
            # Attach identity + fingerprint so the UI/logs can confirm we're evaluating distinct checkpoints.
            cfg["file"] = _file_identity(cfg["path"])
            cfg["fingerprint"] = _fingerprint_model(model)
            # Optional: warn if checkpoint metadata epoch doesn't match filename iteration (iter checkpoints only).
            if iter_num >= 0:
                epoch = _epoch_from_checkpoint(cfg["path"])
                if epoch is not None and epoch not in (-1, iter_num):
                    warnings.append({
                        "player_id": pid,
                        "path": cfg["path"],
                        "message": f"epoch mismatch: ckpt['epoch']={epoch} but filename iter={iter_num}",
                    })
            cfg['model'] = model
            self._participants[pid] = cfg
            self.arena.get_or_create_rating(pid)

        if load_failures:
            # Surface to UI (and logs) so silent load failures don't bias results.
            self.broadcaster.send({'type': 'checkpoint_load_failures', 'failures': load_failures})
        if warnings:
            self.broadcaster.send({'type': 'checkpoint_warnings', 'warnings': warnings})
        
        return len(self._participants) >= 2
    
    def _get_leaderboard(self) -> list:
        return sorted([
            {'player_id': pid, 'name': self._participants.get(pid, {}).get('name', pid), **self.arena.ratings[pid].to_dict()}
            for pid in self.arena.ratings
        ], key=lambda x: x['rating'], reverse=True)
    
    def _run_match(self, pid_a: str, pid_b: str, match_num: int, round_num: int) -> bool:
        """Play a single match and broadcast results."""
        cfg_a, cfg_b = self._participants[pid_a], self._participants[pid_b]
        
        self.status['matches_played'] = match_num
        self.status['current_match'] = f"{cfg_a['name']} vs {cfg_b['name']}"
        
        self.broadcaster.send({
            'type': 'match_started', 'match_num': match_num, 'round': round_num,
            'player_a': pid_a, 'player_b': pid_b,
            'name_a': cfg_a['name'], 'name_b': cfg_b['name'],
            'rating_a': self.arena.ratings[pid_a].rating,
            'rating_b': self.arena.ratings[pid_b].rating,
        })
        
        try:
            result = self.arena.play_match(pid_a, pid_b, cfg_a, cfg_b)
            if result.get('error'):
                return False
            
            winner = 'a' if result['score_a'] == 1.0 else ('b' if result['score_a'] == 0.0 else 'draw')
            self.broadcaster.send({
                'type': 'match_complete', 'match_num': match_num, 'round': round_num,
                'player_a': pid_a, 'player_b': pid_b,
                'name_a': cfg_a['name'], 'name_b': cfg_b['name'], 'winner': winner,
                'leaderboard': self._get_leaderboard(),
                **{k: result[k] for k in [
                    'rounds_played', 'round_wins_a', 'round_wins_b', 'hands_played',
                    'hand_wins_a', 'hand_wins_b', 'score_a', 'old_rating_a', 'old_rating_b',
                    'new_rating_a', 'new_rating_b', 'rating_change_a', 'rating_change_b'
                ]}
            })
            return True
        except Exception as e:
            print(f'Match error: {e}')
            return False
    
    def _run(self, models_dir: str, k_factor: float, num_models: int):
        """Main simulation loop."""
        try:
            base_seed = None
            try:
                # Default to deterministic evaluation unless explicitly disabled.
                base_seed = None if str(ELO_BASE_SEED).lower() in ("none", "off", "false") else int(ELO_BASE_SEED)
            except Exception:
                base_seed = 0
            self.arena = PokerEloArena(
                device=DEVICE,
                k_factor=k_factor,
                max_cached_models=MAX_CACHED_MODELS,
                base_seed=base_seed,
            )
            
            if not self._load_participants(models_dir, num_models):
                self.broadcaster.send({'type': 'error', 'message': 'Need at least 2 participants'})
                return
            
            player_ids = list(self._participants.keys())
            self.broadcaster.send({
                'type': 'simulation_started', 'num_checkpoints': len(self._participants), 'k_factor': k_factor,
                'participants': {
                    pid: {
                        'name': p['name'],
                        'iteration': p.get('iteration'),
                        'initial_rating': self.arena.ratings[pid].rating,
                        'fingerprint': p.get('fingerprint'),
                        'file': p.get('file'),
                    }
                    for pid, p in self._participants.items()
                }
            })
            
            match_num = round_num = 0
            
            while not self._stop.is_set():
                round_num += 1
                self.status['current_round'] = round_num
                
                if ELO_PAIRING == "round_robin":
                    # Every pair plays once per round (order is stable for reproducibility).
                    for i in range(len(player_ids)):
                        for j in range(i + 1, len(player_ids)):
                            if self._stop.is_set():
                                break
                            match_num += 1
                            self._run_match(player_ids[i], player_ids[j], match_num, round_num)
                        if self._stop.is_set():
                            break
                else:
                    # Default: random pairings each round (faster, but higher variance).
                    shuffled = player_ids.copy()
                    random.shuffle(shuffled)
                    for i in range(0, len(shuffled) - 1, 2):
                        if self._stop.is_set():
                            break
                        match_num += 1
                        self._run_match(shuffled[i], shuffled[i + 1], match_num, round_num)
                
                self.broadcaster.send({
                    'type': 'round_complete', 'round': round_num,
                    'leaderboard': self._get_leaderboard(),
                    'rating_histories': {pid: r.history for pid, r in self.arena.ratings.items()}
                })
            
            self.status.update({'status': 'stopped', 'current_match': 'Stopped'})
            self.broadcaster.send({
                'type': 'job_stopped', 'total_rounds': round_num, 'total_matches': match_num,
                'leaderboard': self._get_leaderboard(),
                'rating_histories': {pid: r.history for pid, r in self.arena.ratings.items()}
            })
                
        except Exception as e:
            traceback.print_exc()
            self.status.update({'status': 'error', 'current_match': f'Error: {e}'})
            self.broadcaster.send({'type': 'error', 'message': str(e)})


# Global simulation instance
simulation = EloSimulation()


# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    return jsonify(simulation.status)


@app.route('/api/config')
def get_config():
    return jsonify({'device': DEVICE, 'default_models_dir': str(_auto_models_dir(DEFAULT_MODELS_DIR))})


@app.route('/api/start', methods=['POST'])
def start_simulation():
    data = request.json or {}
    models_dir = str(data.get('models_dir', DEFAULT_MODELS_DIR))
    k_factor = _as_float(data.get('k_factor', 40.0), 40.0)
    num_models = _as_int(data.get('num_models', 8), 8)

    if k_factor <= 0:
        return jsonify({'success': False, 'message': 'k_factor must be > 0'}), 400
    if num_models < 2:
        return jsonify({'success': False, 'message': 'num_models must be >= 2'}), 400

    success, msg = simulation.start(
        models_dir=models_dir,
        k_factor=k_factor,
        num_models=num_models
    )
    return jsonify({'success': success, 'message': msg})


@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    return jsonify({'success': simulation.stop()})


@app.route('/api/stream')
def stream():
    """SSE endpoint for real-time updates."""
    def generate():
        q = simulation.broadcaster.subscribe()
        try:
            yield f"data: {json.dumps({'type': 'status', **simulation.status})}\n\n"
            while True:
                try:
                    event = q.get(timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
        except GeneratorExit:
            pass
        finally:
            simulation.broadcaster.unsubscribe(q)
    
    return Response(generate(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'X-Accel-Buffering': 'no'})


def main():
    port = int(os.environ.get('PORT', 5051))
    print(f'Starting Poker ELO Server on http://localhost:{port}')
    print(f'Device: {DEVICE}, Max cached models: {MAX_CACHED_MODELS}')
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
