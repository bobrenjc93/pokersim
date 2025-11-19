# Poker Simulation C++ API Server

A high-performance C++ API server with a complete Texas Hold'em poker engine.

## Features

- ✅ Complete poker engine with Card, Deck, Hand, Player, Pot, and Game classes
- ✅ Proper hand evaluation (High Card to Royal Flush)
- ✅ Side pot management for all-in scenarios
- ✅ Automatic game flow management
- ✅ 2-10 player support
- ✅ Deterministic simulation with seeded RNG
- ✅ HTTP API server for remote game simulation

## Prerequisites

Choose one of the following build methods:

### Option 1: Using Make (Simpler)
- C++ compiler with C++17 support (g++ 7+ or clang 5+)
- Make
- curl (for downloading dependencies)

### Option 2: Using CMake (More Flexible)
- C++ compiler with C++17 support
- CMake 3.10 or higher
- Make or Ninja

## Installation

### Option 1: Build with Make

1. Navigate to the api directory:
```bash
cd api
```

2. Build the project:
```bash
make
```

The Makefile will automatically download the required nlohmann/json library and compile the server.

### Option 2: Build with CMake

1. Navigate to the api directory:
```bash
cd api
```

2. Create a build directory:
```bash
mkdir build
cd build
```

3. Configure and build:
```bash
cmake ..
make
```

The executable will be in `build/bin/poker_api`

## Running the Server

### Default Port (8080)

If you built with Make:
```bash
./poker_api
```

If you built with CMake:
```bash
./build/bin/poker_api
```

The server will start at `http://localhost:8080`

### Custom Port

To run on a different port, pass it as an argument:
```bash
./poker_api 9000
```

You should see output like:
```
🚀 C++ API Server started on http://localhost:8080
📡 Listening for POST requests to /simulate
```

## API Endpoint

### `POST /simulate`

Simulates the next game state based on the current state and a random seed.

**Request:**
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

**Request Body Schema:**
```json
{
  "gameState": {
    // Your game state object
    // Can contain any valid JSON structure
  },
  "seed": 12345  // Integer seed for random number generation
}
```

**Response (Success):**
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

**Response (Error):**
```json
{
  "error": "Missing required fields: gameState and seed"
}
```

## Poker Engine

The project includes a complete poker engine with the following classes:

- **Card** - Playing card with rank and suit
- **Deck** - 52-card deck with shuffle and deal
- **Hand** - Poker hand evaluation and comparison
- **Player** - Player state, chips, and actions
- **Pot** - Pot management with side pots
- **Game** - Complete Texas Hold'em game orchestrator

For detailed documentation, see [POKER_ENGINE.md](POKER_ENGINE.md).

### Quick Example

```cpp
#include "Game.h"

int main() {
    Game game;
    game.addPlayer("p1", "Alice");
    game.addPlayer("p2", "Bob");
    
    game.startHand();
    game.processAction("p1", Player::Action::RAISE, 60);
    game.processAction("p2", Player::Action::CALL);
    
    // Game automatically progresses through stages
    return 0;
}
```

## Testing

Test the poker engine:

```bash
make test
```

This runs the test suite which validates:
- Card operations
- Deck shuffling and dealing
- Hand evaluation
- Player actions
- Complete game flow

## Project Structure

```
api/
├── main.cpp           # HTTP API server
├── Card.h/cpp         # Playing card class
├── Deck.h/cpp         # Deck management
├── Hand.h/cpp         # Hand evaluation
├── Player.h/cpp       # Player state and actions
├── Pot.h/cpp          # Pot and side pot management
├── Game.h/cpp         # Game orchestrator
├── test_*.cpp         # Individual test files
├── CMakeLists.txt     # CMake build configuration
├── Makefile           # Simple Make build configuration
├── README.md          # This file
└── POKER_ENGINE.md    # Detailed poker engine documentation
```

## Dependencies

- **nlohmann/json**: Modern C++ JSON library
  - Automatically downloaded during build
  - Header-only library
  - [GitHub Repository](https://github.com/nlohmann/json)

## Performance

This is a bare-metal C++ HTTP server optimized for:
- Low latency response times
- Efficient memory usage
- Deterministic simulation with seeded RNG

## Troubleshooting

### Port Already in Use

If port 8080 is busy, specify a different port:
```bash
./poker_api 8081
```

### Compilation Errors

Make sure you have a C++17 compatible compiler:
```bash
g++ --version  # Should be 7.0 or higher
```

### Connection Refused

If the website can't connect:
1. Verify the server is running
2. Check the port number matches
3. Make sure no firewall is blocking the connection

## Testing

Test the server with curl:

```bash
# Simple test
curl -X POST http://localhost:8080/simulate \
  -H "Content-Type: application/json" \
  -d '{"gameState": {"pot": 100}, "seed": 42}'

# Test with invalid data
curl -X POST http://localhost:8080/simulate \
  -H "Content-Type: application/json" \
  -d '{"invalid": "data"}'
```

## Development Tips

1. **Enable compiler warnings** for better code quality:
   - Already enabled in Makefile with `-Wall -Wextra`

2. **Use a debugger** like gdb:
   ```bash
   g++ -g -std=c++17 main.cpp -o poker_api
   gdb ./poker_api
   ```

3. **Profile performance** if needed:
   ```bash
   g++ -pg -std=c++17 main.cpp -o poker_api
   ```

## Cleaning Up

To remove build artifacts:

With Make:
```bash
make clean
```

With CMake:
```bash
rm -rf build
```

