# Poker Simulator Monorepo

A full-stack poker simulation system with a Python web interface and high-performance C++ API backend.

## Project Overview

This monorepo contains:
- **Website**: Python Flask web server with modern UI for poker simulations
- **API**: Pure C++ API server for fast game state simulations

## Architecture

```
┌─────────────────────┐
│   Web Browser       │
└──────────┬──────────┘
           │ HTTP
           ▼
┌─────────────────────┐
│  Python Flask       │  Port 5000
│  Web Server         │
└──────────┬──────────┘
           │ HTTP/JSON
           ▼
┌─────────────────────┐
│  C++ Poker Engine   │  Port 8080
│  (Stateless API)    │
└─────────────────────┘
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
- ✅ Complete Texas Hold'em poker engine in C++
- ✅ Full game management (2-10 players)
- ✅ Proper hand evaluation (High Card to Royal Flush)
- ✅ Side pot management for all-in scenarios
- ✅ Automatic game flow management (preflop, flop, turn, river, showdown)
- ✅ RESTful HTTP API with comprehensive game commands
- ✅ Deterministic simulation with seeded RNG
- ✅ Modern, responsive web interface
- ✅ JSON-based communication
- ✅ CORS support for development
- ✅ Error handling and validation
- ✅ Comprehensive test suites

### Ready to Extend
- 🎯 Track hand histories and statistics
- 🎯 Calculate win probabilities and equity
- 🎯 Add AI opponents with different strategies
- 🎯 Enhanced visualization of game progression
- 🎯 Tournament mode with increasing blinds
- 🎯 Multi-table support

## Example Usage

### Via Web Interface

1. Open `http://localhost:5000`
2. Use the poker game interface to create games and play hands
3. The website communicates with the C++ poker engine backend

### Via API (curl)

The API supports complete poker game management with a **stateless architecture**. Here's a quick example:

```bash
# Add players and start hand (hand starts automatically!)
curl -X POST http://localhost:8080/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "config": {"seed": 42, "smallBlind": 10, "bigBlind": 20, "startingChips": 1000},
    "history": [
      {"type": "addPlayer", "playerId": "alice", "playerName": "Alice"},
      {"type": "addPlayer", "playerId": "bob", "playerName": "Bob"}
    ]
  }'
# Result: Stage = "Preflop", hole cards dealt, blinds posted

# Complete preflop betting to automatically advance to Flop
curl -X POST http://localhost:8080/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "config": {"seed": 42, "smallBlind": 10, "bigBlind": 20, "startingChips": 1000},
    "history": [
      {"type": "addPlayer", "playerId": "alice", "playerName": "Alice"},
      {"type": "addPlayer", "playerId": "bob", "playerName": "Bob"},
      {"type": "playerAction", "playerId": "alice", "action": "call", "amount": 0},
      {"type": "playerAction", "playerId": "bob", "action": "check", "amount": 0}
    ]
  }'
# Result: Stage = "Flop", 3 community cards dealt (automatic advancement!)
```

**For complete API documentation, see [`api/docs/README.md`](api/docs/README.md)**

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

## Poker Engine Details

The poker engine is fully implemented with the following components:

- **Card**: Playing card with rank (2-A) and suit (♣♦♥♠)
- **Deck**: 52-card deck with shuffle, deal, and burn operations
- **Hand**: Complete hand evaluation (High Card through Royal Flush)
- **Player**: Player state, chips, hole cards, and actions
- **Pot**: Main pot and side pot management for all-in scenarios
- **Game**: Complete game orchestrator managing game flow

For detailed documentation:
- **Complete API & Engine Reference**: See [`api/docs/README.md`](api/docs/README.md)

## Testing

### Test Poker Engine
Test the core poker engine classes:
```bash
cd api
make test
```

This runs comprehensive unit tests for Card, Deck, Hand, Player, and Game classes.

### Test API Integration
Test the complete HTTP API with integrated poker engine:

**Option 1: Bash script**
```bash
cd api
./build/poker_api &  # Start server in background
./test_api.sh        # Run comprehensive API tests
```

**Option 2: Python script**
```bash
cd api
./build/poker_api &  # Start server in background
pip3 install requests
./test_api.py        # Run comprehensive API tests
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

The poker engine is fully integrated! Here are some ideas for further enhancement:

1. **Enhance Website UI**: Add rich visualizations for cards, chips, and players
2. **Add Database**: Store hand histories and statistics  
3. **Implement AI**: Add computer opponents with different strategies
4. **Monte Carlo Simulations**: Calculate equity and win probabilities
5. **Tournament Mode**: Add support for increasing blinds and multi-table
6. **Add Authentication**: Support multiple users and sessions
7. **Deploy**: Containerize with Docker for easy deployment

The foundation is solid - build something amazing! 🃏🎰

