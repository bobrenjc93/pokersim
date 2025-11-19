#!/usr/bin/env python3
"""
Hand Viewer Web Server

Displays poker hand histories logged during ELO evaluation,
including model predictions and action sequences.
"""

import os
import json
import time
from pathlib import Path
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Default hand logs directory
HAND_LOGS_DIR = Path(os.environ.get('HAND_LOGS_DIR', '/tmp/pokersim/hand_logs'))

# Simple cache for hand summaries (avoids reading 500 files per request)
_hand_cache = {
    'summaries': [],
    'total': 0,
    'last_update': 0,
    'file_count': 0
}
CACHE_TTL_SECONDS = 10


def get_hand_files():
    """Get all hand log files sorted by modification time (newest first)."""
    if not HAND_LOGS_DIR.exists():
        return []
    
    hand_files = []
    for f in HAND_LOGS_DIR.glob('*.json'):
        try:
            stat = f.stat()
            hand_files.append({
                'filename': f.name,
                'path': str(f),
                'mtime': stat.st_mtime,
                'size': stat.st_size
            })
        except Exception:
            continue
    
    # Sort by modification time, newest first
    return sorted(hand_files, key=lambda x: x['mtime'], reverse=True)


def load_hand(filename: str):
    """Load a hand log file."""
    filepath = HAND_LOGS_DIR / filename
    if not filepath.exists():
        return None
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {'error': str(e)}


@app.route('/')
def index():
    """Serve the main UI."""
    return render_template('index.html')


@app.route('/api/hands')
def list_hands():
    """Get list of all logged hands (cached to avoid reading many files per request)."""
    hands = get_hand_files()
    now = time.time()
    
    # Use cache if valid and file count hasn't changed
    if (now - _hand_cache['last_update'] < CACHE_TTL_SECONDS 
        and _hand_cache['file_count'] == len(hands)
        and _hand_cache['summaries']):
        return jsonify({
            'hands': _hand_cache['summaries'],
            'total': _hand_cache['total'],
            'logs_dir': str(HAND_LOGS_DIR)
        })
    
    # Rebuild cache - load summary info for each hand
    hand_summaries = []
    for h in hands[:500]:  # Limit to 500 most recent
        hand_data = load_hand(h['filename'])
        if hand_data and 'error' not in hand_data:
            summary = {
                'filename': h['filename'],
                'hand_id': hand_data.get('hand_id', h['filename']),
                'timestamp': hand_data.get('timestamp', ''),
                'players': [p.get('name', p.get('player_id', 'Unknown')) for p in hand_data.get('players', [])],
                'winner': hand_data.get('result', {}).get('winner_name', 'Unknown'),
                'total_pot': hand_data.get('result', {}).get('final_pot', 0),
                'num_actions': len(hand_data.get('actions', [])),
            }
            hand_summaries.append(summary)
    
    # Update cache
    _hand_cache['summaries'] = hand_summaries
    _hand_cache['total'] = len(hands)
    _hand_cache['last_update'] = now
    _hand_cache['file_count'] = len(hands)
    
    return jsonify({
        'hands': hand_summaries,
        'total': len(hands),
        'logs_dir': str(HAND_LOGS_DIR)
    })


@app.route('/api/hand/<filename>')
def get_hand(filename: str):
    """Get detailed hand data."""
    hand_data = load_hand(filename)
    if hand_data is None:
        return jsonify({'error': 'Hand not found'}), 404
    
    return jsonify(hand_data)


@app.route('/api/stats')
def get_stats():
    """Get overall statistics."""
    hands = get_hand_files()
    
    if not hands:
        return jsonify({
            'total_hands': 0,
            'logs_dir': str(HAND_LOGS_DIR),
            'exists': HAND_LOGS_DIR.exists()
        })
    
    # Calculate stats from recent hands
    recent_hands = hands[:100]
    player_stats = {}
    
    for h in recent_hands:
        hand_data = load_hand(h['filename'])
        if hand_data and 'error' not in hand_data:
            for p in hand_data.get('players', []):
                name = p.get('name', p.get('player_id', 'Unknown'))
                if name not in player_stats:
                    player_stats[name] = {'hands': 0, 'wins': 0}
                player_stats[name]['hands'] += 1
            
            winner = hand_data.get('result', {}).get('winner_name')
            if winner and winner in player_stats:
                player_stats[winner]['wins'] += 1
    
    return jsonify({
        'total_hands': len(hands),
        'recent_analyzed': len(recent_hands),
        'player_stats': player_stats,
        'logs_dir': str(HAND_LOGS_DIR),
        'exists': HAND_LOGS_DIR.exists()
    })


def main():
    """Run the web server."""
    port = int(os.environ.get('PORT', 5052))
    print(f"Starting Hand Viewer on http://localhost:{port}")
    print(f"Reading hands from: {HAND_LOGS_DIR}")
    app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == '__main__':
    main()

