#!/usr/bin/env python3
"""
Poker Playground - Play against trained AI models

This Flask app provides an interactive poker game where you can:
- Select a trained model checkpoint as your opponent
- Play Texas Hold'em heads-up against the AI
- See the AI's action probabilities and reasoning
"""

import sys
import os
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, render_template, request, jsonify
import uuid

# Add training directory to path
TRAINING_DIR = Path(__file__).parent.parent / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import torch
from model_agent import (
    load_model_agent, extract_state, ModelAgent,
    ACTION_NAMES, convert_action_label
)
import poker_api_binding
from config import DEFAULT_MODELS_DIR

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'pokersim-playground-dev-key')

# Game configuration
GAME_CONFIG = {
    'num_players': 2,
    'smallBlind': 10,
    'bigBlind': 20,
    'startingChips': 1000,
    'minPlayers': 2,
    'maxPlayers': 2
}

# In-memory game sessions (in production, use Redis or similar)
game_sessions: Dict[str, Dict[str, Any]] = {}

# Cache for loaded models
model_cache: Dict[str, ModelAgent] = {}

# Device detection
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

DEVICE = get_device()


def get_available_checkpoints() -> List[Dict[str, Any]]:
    """Get list of available model checkpoints"""
    checkpoints = []
    models_dir = Path(DEFAULT_MODELS_DIR)
    
    if not models_dir.exists():
        return checkpoints
    
    pattern = re.compile(r"poker_rl_iter_(\d+)\.pt")
    
    for f in sorted(models_dir.glob("*.pt")):
        if f.name == "poker_rl_baseline.pt":
            checkpoints.append({
                'iteration': 0,
                'name': 'Baseline (Iter 0)',
                'path': str(f),
                'filename': f.name
            })
            continue
        
        match = pattern.match(f.name)
        if match:
            iteration = int(match.group(1))
            checkpoints.append({
                'iteration': iteration,
                'name': f'Iteration {iteration}',
                'path': str(f),
                'filename': f.name
            })
    
    # Sort by iteration
    checkpoints.sort(key=lambda x: x['iteration'])
    return checkpoints


def load_opponent(checkpoint_path: str) -> ModelAgent:
    """Load or retrieve cached model agent"""
    if checkpoint_path in model_cache:
        return model_cache[checkpoint_path]
    
    agent = load_model_agent(
        player_id='ai',
        name='AI Opponent',
        model_path=checkpoint_path,
        device=DEVICE,
        deterministic=False  # Use sampling for more varied play
    )
    model_cache[checkpoint_path] = agent
    return agent


def call_api(history: List[Dict]) -> Dict:
    """Call the poker API binding"""
    payload = {
        'config': {
            **GAME_CONFIG,
            'seed': random.randint(0, 1000000)
        },
        'history': history
    }
    
    try:
        payload_str = json.dumps(payload)
        response_str = poker_api_binding.process_request(payload_str)
        return json.loads(response_str)
    except Exception as e:
        return {'success': False, 'error': str(e)}


def format_card(card: str) -> Dict[str, str]:
    """Format card for display"""
    if not card or len(card) < 2:
        return {'rank': '?', 'suit': '?', 'display': '??'}
    
    rank = card[0]
    suit = card[1]
    
    # Convert rank display
    rank_display = {
        'T': '10', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'
    }.get(rank, rank)
    
    # Convert suit to symbol
    suit_symbols = {'C': '♣', 'D': '♦', 'H': '♥', 'S': '♠'}
    suit_colors = {'C': 'black', 'D': 'red', 'H': 'red', 'S': 'black'}
    
    return {
        'rank': rank_display,
        'suit': suit_symbols.get(suit, suit),
        'color': suit_colors.get(suit, 'black'),
        'display': f"{rank_display}{suit_symbols.get(suit, suit)}",
        'raw': card
    }


def get_game_state_for_client(session_data: Dict, player_perspective: str = 'human') -> Dict:
    """Convert game state to client-friendly format"""
    game_state = session_data.get('game_state', {})
    
    # Find players
    human_player = None
    ai_player = None
    
    for p in game_state.get('players', []):
        if p['id'] == 'human':
            human_player = p
        else:
            ai_player = p
    
    # Format cards
    human_cards = [format_card(c) for c in (human_player.get('holeCards', []) if human_player else [])]
    ai_cards = [format_card(c) for c in (ai_player.get('holeCards', []) if ai_player else [])]
    community_cards = [format_card(c) for c in game_state.get('communityCards', [])]
    
    # Determine whose turn it is
    current_player = game_state.get('currentPlayerId', '')
    is_human_turn = current_player == 'human'
    
    # Get legal actions
    action_constraints = game_state.get('actionConstraints', {})
    legal_actions = action_constraints.get('legalActions', [])
    
    # Get stage
    stage = game_state.get('stage', 'Preflop')
    is_complete = stage.lower() in ['complete', 'showdown']
    
    # Hide AI cards unless showdown
    show_ai_cards = is_complete or session_data.get('show_ai_cards', False)
    
    return {
        'session_id': session_data.get('session_id'),
        'stage': stage,
        'pot': game_state.get('pot', 0),
        'current_bet': game_state.get('currentBet', 0),
        'human': {
            'chips': human_player.get('chips', 0) if human_player else 0,
            'bet': human_player.get('bet', 0) if human_player else 0,
            'cards': human_cards,
            'is_dealer': human_player.get('isDealer', False) if human_player else False,
            'in_hand': human_player.get('isInHand', True) if human_player else False,
        },
        'ai': {
            'chips': ai_player.get('chips', 0) if ai_player else 0,
            'bet': ai_player.get('bet', 0) if ai_player else 0,
            'cards': ai_cards if show_ai_cards else [{'rank': '?', 'suit': '?', 'display': '??', 'hidden': True}] * 2,
            'is_dealer': ai_player.get('isDealer', False) if ai_player else False,
            'in_hand': ai_player.get('isInHand', True) if ai_player else False,
            'name': session_data.get('opponent_name', 'AI')
        },
        'community_cards': community_cards,
        'is_human_turn': is_human_turn,
        'is_complete': is_complete,
        'legal_actions': legal_actions,
        'action_constraints': action_constraints,
        'last_action': session_data.get('last_action'),
        'hand_result': session_data.get('hand_result'),
        'hands_played': session_data.get('hands_played', 0),
        'human_total_chips': session_data.get('human_total_chips', GAME_CONFIG['startingChips']),
        'ai_total_chips': session_data.get('ai_total_chips', GAME_CONFIG['startingChips']),
    }


@app.route('/')
def index():
    """Serve the main playground page"""
    return render_template('index.html')


@app.route('/api/checkpoints', methods=['GET'])
def get_checkpoints():
    """Get available model checkpoints"""
    checkpoints = get_available_checkpoints()
    return jsonify({
        'success': True,
        'checkpoints': checkpoints,
        'models_dir': DEFAULT_MODELS_DIR,
        'device': str(DEVICE)
    })


@app.route('/api/start', methods=['POST'])
def start_game():
    """Start a new game session"""
    data = request.get_json() or {}
    checkpoint_path = data.get('checkpoint_path')
    
    if not checkpoint_path:
        # Use latest checkpoint if none specified
        checkpoints = get_available_checkpoints()
        if not checkpoints:
            return jsonify({
                'success': False,
                'error': 'No checkpoints available'
            }), 400
        checkpoint_path = checkpoints[-1]['path']
    
    # Verify checkpoint exists
    if not Path(checkpoint_path).exists():
        return jsonify({
            'success': False,
            'error': f'Checkpoint not found: {checkpoint_path}'
        }), 404
    
    # Create new session
    session_id = str(uuid.uuid4())
    
    # Load opponent model
    try:
        opponent = load_opponent(checkpoint_path)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to load model: {str(e)}'
        }), 500
    
    # Initialize game history
    history = [
        {'type': 'addPlayer', 'playerId': 'human', 'playerName': 'You'},
        {'type': 'addPlayer', 'playerId': 'ai', 'playerName': opponent.name}
    ]
    
    # Get initial state
    response = call_api(history)
    if not response.get('success'):
        return jsonify({
            'success': False,
            'error': response.get('error', 'Failed to initialize game')
        }), 500
    
    # Store session
    game_sessions[session_id] = {
        'session_id': session_id,
        'checkpoint_path': checkpoint_path,
        'opponent_name': f'AI ({Path(checkpoint_path).stem})',
        'history': history,
        'game_state': response['gameState'],
        'hands_played': 0,
        'human_total_chips': GAME_CONFIG['startingChips'],
        'ai_total_chips': GAME_CONFIG['startingChips'],
        'last_action': None,
        'hand_result': None,
    }
    
    # Reset opponent's action history
    opponent.reset_hand()
    
    # Check if AI needs to act first
    game_state = response['gameState']
    current_player = game_state.get('currentPlayerId', '')
    
    if current_player == 'ai':
        # AI acts first
        session_data = game_sessions[session_id]
        process_ai_turn(session_data, opponent)
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'game_state': get_game_state_for_client(game_sessions[session_id])
    })


def process_ai_turn(session_data: Dict, opponent: ModelAgent) -> Optional[Dict]:
    """Process AI's turn and return action info"""
    game_state = session_data['game_state']
    
    stage = game_state.get('stage', '').lower()
    if stage in ['complete', 'showdown']:
        return None
    
    current_player = game_state.get('currentPlayerId', '')
    if current_player != 'ai':
        return None
    
    # Get legal actions
    legal_actions = game_state.get('actionConstraints', {}).get('legalActions', [])
    if not legal_actions:
        return None
    
    # Extract state for AI
    state_dict = extract_state(game_state, 'ai')
    
    # Get AI action with probabilities
    action_type, amount, action_label, probs = opponent.select_action(
        state_dict, legal_actions, return_probs=True
    )
    
    # Handle all-in conversion
    ai_player = next((p for p in game_state['players'] if p['id'] == 'ai'), None)
    if ai_player:
        ai_chips = ai_player.get('chips', 0)
        to_call = state_dict.get('to_call', 0)
        
        if action_type in ['bet', 'raise'] and amount >= ai_chips:
            action_type = 'all_in'
            amount = 0
        elif action_type == 'call' and to_call >= ai_chips and ai_chips > 0:
            action_type = 'all_in'
            amount = 0
    
    # Apply action
    session_data['history'].append({
        'type': 'playerAction',
        'playerId': 'ai',
        'action': action_type,
        'amount': amount
    })
    
    # Update game state
    response = call_api(session_data['history'])
    if response.get('success'):
        session_data['game_state'] = response['gameState']
    
    # Record action
    action_info = {
        'player': 'ai',
        'action': action_type,
        'amount': amount,
        'label': action_label,
        'probabilities': probs
    }
    session_data['last_action'] = action_info
    
    # Notify opponent of its own action (for history tracking)
    opponent.observe_action('ai', action_type, amount, game_state.get('pot', 0), game_state.get('stage', 'Preflop'))
    
    return action_info


@app.route('/api/action', methods=['POST'])
def player_action():
    """Handle player action"""
    data = request.get_json() or {}
    session_id = data.get('session_id')
    action_type = data.get('action')
    amount = data.get('amount', 0)
    
    if not session_id or session_id not in game_sessions:
        return jsonify({
            'success': False,
            'error': 'Invalid session'
        }), 400
    
    session_data = game_sessions[session_id]
    game_state = session_data['game_state']
    
    # Check if it's human's turn
    current_player = game_state.get('currentPlayerId', '')
    if current_player != 'human':
        return jsonify({
            'success': False,
            'error': 'Not your turn'
        }), 400
    
    # Validate action
    legal_actions = game_state.get('actionConstraints', {}).get('legalActions', [])
    
    # Map action to legal action type
    base_action = action_type
    if action_type.startswith('bet_') or action_type.startswith('raise_'):
        base_action = 'bet' if action_type.startswith('bet_') else 'raise'
    
    if base_action not in legal_actions and action_type not in legal_actions:
        return jsonify({
            'success': False,
            'error': f'Illegal action: {action_type}'
        }), 400
    
    # Convert action label to action type and amount
    state_dict = extract_state(game_state, 'human')
    final_action, final_amount = convert_action_label(action_type, state_dict)
    
    # Override with explicit amount if provided
    if amount > 0:
        final_amount = amount
    
    # Handle all-in conversion
    human_player = next((p for p in game_state['players'] if p['id'] == 'human'), None)
    if human_player:
        human_chips = human_player.get('chips', 0)
        to_call = state_dict.get('to_call', 0)
        
        if final_action in ['bet', 'raise'] and final_amount >= human_chips:
            final_action = 'all_in'
            final_amount = 0
        elif final_action == 'call' and to_call >= human_chips and human_chips > 0:
            final_action = 'all_in'
            final_amount = 0
    
    # Build the action to add
    new_action = {
        'type': 'playerAction',
        'playerId': 'human',
        'action': final_action,
        'amount': final_amount
    }
    
    # Test the action before committing to history
    test_history = session_data['history'] + [new_action]
    response = call_api(test_history)
    if not response.get('success'):
        return jsonify({
            'success': False,
            'error': response.get('error', 'Action failed')
        }), 500
    
    # Action succeeded, commit to history
    session_data['history'].append(new_action)
    
    session_data['game_state'] = response['gameState']
    session_data['last_action'] = {
        'player': 'human',
        'action': final_action,
        'amount': final_amount,
        'label': action_type
    }
    
    # Load opponent and notify of human action
    opponent = load_opponent(session_data['checkpoint_path'])
    opponent.observe_action('human', final_action, final_amount, 
                           game_state.get('pot', 0), game_state.get('stage', 'Preflop'))
    
    # Check if hand is complete
    new_stage = session_data['game_state'].get('stage', '').lower()
    if new_stage in ['complete', 'showdown']:
        # Hand is over
        handle_hand_complete(session_data)
    else:
        # Check if AI needs to act
        new_current_player = session_data['game_state'].get('currentPlayerId', '')
        if new_current_player == 'ai':
            process_ai_turn(session_data, opponent)
            
            # Check again if hand is complete after AI action
            new_stage = session_data['game_state'].get('stage', '').lower()
            if new_stage in ['complete', 'showdown']:
                handle_hand_complete(session_data)
    
    return jsonify({
        'success': True,
        'game_state': get_game_state_for_client(session_data)
    })


def handle_hand_complete(session_data: Dict):
    """Handle hand completion"""
    game_state = session_data['game_state']
    
    # Find players
    human_player = None
    ai_player = None
    
    for p in game_state.get('players', []):
        if p['id'] == 'human':
            human_player = p
        else:
            ai_player = p
    
    # Calculate results
    starting_chips = GAME_CONFIG['startingChips']
    human_profit = (human_player.get('chips', 0) - starting_chips) if human_player else 0
    ai_profit = (ai_player.get('chips', 0) - starting_chips) if ai_player else 0
    
    # Determine winner
    if human_profit > 0:
        winner = 'human'
        winner_text = 'You win!'
    elif ai_profit > 0:
        winner = 'ai'
        winner_text = f'{session_data.get("opponent_name", "AI")} wins!'
    else:
        winner = 'tie'
        winner_text = 'Tie!'
    
    session_data['hand_result'] = {
        'winner': winner,
        'winner_text': winner_text,
        'human_profit': human_profit,
        'ai_profit': ai_profit,
        'human_cards': human_player.get('holeCards', []) if human_player else [],
        'ai_cards': ai_player.get('holeCards', []) if ai_player else [],
        'community_cards': game_state.get('communityCards', []),
        'final_pot': game_state.get('pot', 0)
    }
    
    session_data['show_ai_cards'] = True
    session_data['hands_played'] += 1
    session_data['human_total_chips'] += human_profit
    session_data['ai_total_chips'] += ai_profit


@app.route('/api/new_hand', methods=['POST'])
def new_hand():
    """Start a new hand in the current session"""
    data = request.get_json() or {}
    session_id = data.get('session_id')
    
    if not session_id or session_id not in game_sessions:
        return jsonify({
            'success': False,
            'error': 'Invalid session'
        }), 400
    
    session_data = game_sessions[session_id]
    
    # Reset for new hand
    history = [
        {'type': 'addPlayer', 'playerId': 'human', 'playerName': 'You'},
        {'type': 'addPlayer', 'playerId': 'ai', 'playerName': session_data.get('opponent_name', 'AI')}
    ]
    
    # Get new game state
    response = call_api(history)
    if not response.get('success'):
        return jsonify({
            'success': False,
            'error': response.get('error', 'Failed to start new hand')
        }), 500
    
    session_data['history'] = history
    session_data['game_state'] = response['gameState']
    session_data['last_action'] = None
    session_data['hand_result'] = None
    session_data['show_ai_cards'] = False
    
    # Reset opponent's action history
    opponent = load_opponent(session_data['checkpoint_path'])
    opponent.reset_hand()
    
    # Check if AI needs to act first
    current_player = response['gameState'].get('currentPlayerId', '')
    if current_player == 'ai':
        process_ai_turn(session_data, opponent)
    
    return jsonify({
        'success': True,
        'game_state': get_game_state_for_client(session_data)
    })


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get current game state"""
    session_id = request.args.get('session_id')
    
    if not session_id or session_id not in game_sessions:
        return jsonify({
            'success': False,
            'error': 'Invalid session'
        }), 400
    
    return jsonify({
        'success': True,
        'game_state': get_game_state_for_client(game_sessions[session_id])
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting Poker Playground on port {port}")
    print(f"Device: {DEVICE}")
    print(f"Models directory: {DEFAULT_MODELS_DIR}")
    
    checkpoints = get_available_checkpoints()
    print(f"Available checkpoints: {len(checkpoints)}")
    
    app.run(host='0.0.0.0', port=port, debug=True)

