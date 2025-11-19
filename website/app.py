from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)

# Configuration for C++ API server
API_HOST = os.environ.get('API_HOST', 'localhost')
API_PORT = os.environ.get('API_PORT', '8080')
API_URL = f'http://{API_HOST}:{API_PORT}'


@app.route('/')
def index():
    """Serve the main poker simulation page."""
    return render_template('index.html')


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """
    Proxy endpoint to the C++ API server.
    Expects JSON with 'gameState' and 'seed' fields.
    """
    try:
        data = request.get_json()
        
        if not data or 'gameState' not in data or 'seed' not in data:
            return jsonify({
                'error': 'Missing required fields: gameState and seed'
            }), 400
        
        # Forward request to C++ API server
        response = requests.post(
            f'{API_URL}/simulate',
            json=data,
            timeout=10
        )
        
        return jsonify(response.json()), response.status_code
        
    except requests.exceptions.ConnectionError:
        return jsonify({
            'error': 'Could not connect to API server. Is it running?'
        }), 503
    except requests.exceptions.Timeout:
        return jsonify({
            'error': 'API server request timed out'
        }), 504
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

