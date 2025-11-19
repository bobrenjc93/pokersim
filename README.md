# Poker Simulator Monorepo

A full-stack poker simulation system with a Python web interface and high-performance C++ API backend.

## Project Overview

This monorepo contains:
- **Website**: Python Flask web server with modern UI for poker simulations
- **API**: Pure C++ API server for fast game state simulations

## Architecture

```
┌─────────────────┐
│   Web Browser   │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│  Python Flask   │  Port 5000
│  Web Server     │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│   C++ API       │  Port 8080
│   Server        │
└─────────────────┘
```

The Python website serves the user interface and proxies requests to the C++ backend, which handles the computational work of simulating poker game states.

## Quick Start

### 1. Start the C++ API Server

```bash
cd api
make
./poker_api
```

You should see: `🚀 C++ API Server started on http://localhost:8080`

### 2. Start the Python Web Server

In a new terminal:

```bash
cd website
uv venv  # Create virtual environment
uv pip install -r requirements.txt  # Install dependencies
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python app.py
```

**Don't have uv?** Install it first:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

You should see: `Running on http://0.0.0.0:5000`

### 3. Open the Website

Navigate to `http://localhost:5000` in your browser and start simulating!

## Project Structure

```
pokersim/
├── README.md              # This file
├── website/               # Python Flask web server
│   ├── app.py            # Flask application
│   ├── requirements.txt  # Python dependencies
│   ├── templates/
│   │   └── index.html   # Web UI
│   └── README.md        # Website documentation
└── api/                  # C++ API server
    ├── main.cpp         # Server implementation
    ├── CMakeLists.txt   # CMake configuration
    ├── Makefile         # Make configuration
    └── README.md        # API documentation
```

## Features

### Current Features
- ✅ RESTful API for game state simulation
- ✅ Deterministic simulation with seeded RNG
- ✅ Modern, responsive web interface
- ✅ JSON-based communication
- ✅ CORS support for development
- ✅ Error handling and validation

### Ready to Extend
- 🎯 Implement full poker game rules
- 🎯 Add multi-player support
- 🎯 Track hand histories
- 🎯 Calculate win probabilities
- 🎯 Add AI opponents
- 🎯 Visualization of game progression

## Example Usage

### Via Web Interface

1. Open `http://localhost:5000`
2. Enter a game state:
   ```json
   {"players": 2, "pot": 100, "cards": ["AS", "KD"]}
   ```
3. Enter a seed: `42`
4. Click "Simulate Next State"

### Via API (curl)

```bash
curl -X POST http://localhost:8080/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "gameState": {
      "players": 2,
      "pot": 100,
      "cards": ["AS", "KD"]
    },
    "seed": 42
  }'
```

Response:
```json
{
  "success": true,
  "nextGameState": {
    "players": 2,
    "pot": 130,
    "cards": ["AS", "KD", "QH"],
    "lastAction": "bet",
    "lastBetAmount": 30,
    "simulated": true,
    "timestamp": 1700000000
  }
}
```

## Development

### Prerequisites

#### For Website
- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)

#### For API
- C++ compiler with C++17 support
- Make or CMake 3.10+

### Running in Development Mode

#### Website (with auto-reload)
```bash
cd website
FLASK_ENV=development python app.py
```

#### API (rebuild after changes)
```bash
cd api
make clean && make
./poker_api
```

## Configuration

### Environment Variables

#### Website
- `PORT`: Web server port (default: 5000)
- `API_HOST`: C++ API hostname (default: localhost)
- `API_PORT`: C++ API port (default: 8080)

Example:
```bash
PORT=3000 API_HOST=localhost API_PORT=8080 python app.py
```

#### API
Pass port as command line argument:
```bash
./poker_api 9000
```

## Extending the Poker Logic

The current implementation includes basic example logic. To implement real poker rules:

1. Edit `api/main.cpp` → `GameSimulator::simulateNextState()`
2. Add your poker game logic:
   - Hand evaluation
   - Betting rounds
   - Card dealing
   - Winner determination
3. Rebuild the API server
4. Update the website UI as needed

## Testing

### Test API Server
```bash
cd api
# Build and run
make && ./poker_api &

# Test endpoint
curl -X POST http://localhost:8080/simulate \
  -H "Content-Type: application/json" \
  -d '{"gameState": {"pot": 100}, "seed": 42}'
```

### Test Website
```bash
cd website
python app.py &
curl http://localhost:5000
```

## Troubleshooting

### "Connection refused" errors
- Ensure both servers are running
- Check ports aren't already in use
- Verify environment variables are set correctly

### Build errors (C++)
- Verify C++17 compiler support: `g++ --version` (need 7.0+)
- The first build downloads dependencies (requires internet)

### Python import errors
- Activate virtual environment if using one
- Install dependencies: `uv pip install -r requirements.txt`

## Performance

The C++ backend provides:
- Sub-millisecond response times for simulations
- Efficient memory usage
- Deterministic results with seeded RNG

## Contributing

When extending this project:
1. Keep the website and API loosely coupled
2. Use JSON for all communication
3. Add error handling for edge cases
4. Update documentation as you add features

## License

This project is provided as-is for poker simulation purposes.

## Next Steps

1. **Implement Poker Rules**: Add real poker game logic to `api/main.cpp`
2. **Enhance UI**: Add visualizations for cards, chips, and players
3. **Add Database**: Store hand histories and statistics
4. **Implement AI**: Add computer opponents with different strategies
5. **Add Authentication**: Support multiple users and sessions
6. **Deploy**: Containerize with Docker for easy deployment

Happy simulating! 🃏

