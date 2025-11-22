# Stateless Poker Engine C++ API Server

A high-performance, **truly stateless** C++ API server with a complete Texas Hold'em poker engine.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [API Documentation](#api-documentation)
  - [Key Concepts](#-key-concepts-implicit-advancement--stateless-architecture)
  - [API Endpoint](#api-endpoint)
  - [Request Format](#request-format)
  - [Response Format](#response-format)
  - [Workflow Examples](#workflow-examples)
  - [Player Actions](#player-actions)
  - [Game Stages](#game-stages)
  - [Error Handling](#error-handling)
- [Poker Engine Documentation](#poker-engine-documentation)
  - [Architecture Overview](#architecture-overview)
  - [Class Reference](#class-reference)
  - [Integration Examples](#integration-examples)
- [Quick Reference](#quick-reference)
  - [Class Hierarchy](#class-hierarchy)
  - [Code Examples](#quick-reference-code-examples)
  - [Enums and Constants](#enums-and-constants)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## Features

- ✅ **Truly stateless architecture** - no game IDs, no server persistence
- ✅ **Deterministic replay** - same seed + history = same game state
- ✅ Complete poker engine with Card, Deck, Hand, Player, Pot, and Game classes
- ✅ Proper hand evaluation (High Card to Royal Flush)
- ✅ Side pot management for all-in scenarios
- ✅ Automatic game flow management
- ✅ 2-10 player support
- ✅ Seeded RNG for reproducible games
- ✅ HTTP API server for remote game simulation

---

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

---

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

---

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

---

## API Documentation

### 🎯 Key Concepts: Implicit Advancement & Stateless Architecture

This poker engine features **implicit game advancement** and is **truly stateless**:

#### Automatic Advancement
- ✅ **Fully deterministic** - Game state is solely determined by seed + history
- ✅ Game advances automatically: Preflop → Flop → Turn → River → Showdown → Complete
- ✅ No need to manually specify `{"type": "advance"}` - it's automatic!
- ✅ All actions must be in history - no manual actions allowed
- ✅ Everything is driven by the seed for complete determinism

#### Stateless Architecture
Unlike traditional game servers, this poker engine **maintains NO state** between requests:

- ❌ No game IDs stored on the server
- ❌ No session management
- ❌ No server-side persistence

Instead:

- ✅ Client sends full game state (via config + history)
- ✅ Server reconstructs state deterministically using seed
- ✅ Server applies new action and returns updated state
- ✅ **Same seed + same history = same game state** (deterministic replay)

### API Endpoint

**Base URL:** `http://localhost:8080`

**Endpoint:** `POST /simulate`

**Content-Type:** `application/json`

### Request Format

Every request contains one required field and one optional field:

#### 1. `config` (required)

Game configuration including the seed for deterministic randomness:

```json
{
  "config": {
    "smallBlind": 10,
    "bigBlind": 20,
    "startingChips": 1000,
    "minPlayers": 2,
    "maxPlayers": 10,
    "seed": 42,
    "exactCards": ["AS", "KH", "2D", "3C", "4H", ...]
  }
}
```

**Fields:**
- `seed`: Integer seed for RNG (use same seed for reproducibility)
- `smallBlind`, `bigBlind`: Blind amounts
- `startingChips`: Initial chip stack for each player
- `minPlayers`, `maxPlayers`: Player limits
- `exactCards`: (Optional) Array of card strings in exact order for deterministic testing. If provided, these cards are used instead of shuffling. Useful for testing specific scenarios like split pots or specific board textures.

#### 2. `history` (optional)

Array of actions that have occurred so far. Server replays these to reconstruct game state:

```json
{
  "history": [
    {"type": "addPlayer", "playerId": "alice", "playerName": "Alice"},
    {"type": "addPlayer", "playerId": "bob", "playerName": "Bob"}
  ]
}
```

**Action Types:**
- `addPlayer`: Add a player to the game
- `playerAction`: Process a player's action (call, bet, raise, fold, check, all-in)

**Note:** The API does **not** support explicit `startHand`, `advance`, or `next` actions. Starting hands, dealing cards, and determining winners all happen automatically based on the game state! All actions must be provided in the `history` array - the API is fully deterministic based on seed + history.

### Response Format

All responses follow this structure:

```json
{
  "success": true,
  "gameState": {
    "stage": "PREFLOP",
    "pot": 30,
    "currentBet": 20,
    "minRaise": 20,
    "dealerPosition": 0,
    "handNumber": 1,
    "config": {
      "smallBlind": 10,
      "bigBlind": 20,
      "startingChips": 1000,
      "minPlayers": 2,
      "maxPlayers": 10,
      "seed": 42
    },
    "communityCards": [],
    "currentPlayerId": "alice",
    "players": [
      {
        "id": "alice",
        "name": "Alice",
        "chips": 990,
        "bet": 10,
        "totalBet": 10,
        "state": "Active",
        "lastAction": "none",
        "position": 0,
        "isDealer": true,
        "isSmallBlind": true,
        "isBigBlind": false,
        "canAct": true,
        "isInHand": true,
        "holeCards": ["AS", "KH"]
      }
    ]
  }
}
```

**Notes:**
- `success`: Boolean indicating if request was processed successfully
- `gameState`: Full game state including all players' hole cards
- Server always returns complete state - client maintains history

### Workflow Examples

#### Quick Example: Auto-advance Through All Stages

```bash
# 1. Add players and auto-start hand (hand starts automatically!)
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

# 2. Complete preflop betting to advance to Flop
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
# Result: Stage = "Flop", 3 community cards dealt
```

**Key Points:**
- 📌 The game automatically advances after replaying history **only if betting round is complete**
- 📌 If betting round is not complete, the game waits at current state for more actions
- 📌 History grows as the game progresses - include all completed betting rounds
- 📌 Same seed + same history = same result (deterministic replay)
- 📌 All game state is determined by seed + history - no manual actions allowed

#### Testing with Exact Cards

For precise testing scenarios, you can override random shuffling by providing an `exactCards` array in the config:

```bash
curl -X POST http://localhost:8080/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "smallBlind": 10,
      "bigBlind": 20,
      "startingChips": 500,
      "exactCards": [
        "2H", "3D",   # Player 1 hole cards
        "4S", "5C",   # Player 2 hole cards
        "6H",         # Burn before flop
        "AD", "KD", "QD",  # Flop
        "7S",         # Burn before turn
        "JD",         # Turn
        "8C",         # Burn before river
        "TD"          # River
      ]
    },
    "history": [
      {"type": "addPlayer", "playerId": "alice", "playerName": "Alice"},
      {"type": "addPlayer", "playerId": "bob", "playerName": "Bob"}
    ]
  }'
# Result: Royal flush on board - guaranteed split pot!
```

**Card Order:**
1. Hole cards for each player (2 cards × number of players)
2. Burn card (1)
3. Flop (3 cards)
4. Burn card (1)
5. Turn (1 card)
6. Burn card (1)
7. River (1 card)

This is especially useful for testing edge cases like split pots, specific hand matchups, or board textures.

### Player Actions

Available actions when it's a player's turn:

- **fold** - Fold hand and forfeit pot
- **check** - Pass (only when current bet is 0)
- **call** - Match current bet
- **bet** - Make first bet in round (requires `amount`)
- **raise** - Increase current bet (requires `amount`)
- **all_in** - Go all-in with remaining chips

### Game Stages

The game progresses through these stages automatically:

1. **WAITING** - Waiting for players to join
2. **PREFLOP** - After hole cards dealt, before flop
3. **FLOP** - 3 community cards on board
4. **TURN** - 4 community cards on board  
5. **RIVER** - 5 community cards on board
6. **SHOWDOWN** - Revealing hands and determining winner
7. **COMPLETE** - Hand complete, ready for next hand

### Player States

- **Active** - In the hand, can act
- **Folded** - Folded this hand
- **All-In** - All-in, no more actions
- **Out** - No chips remaining

### Error Handling

All errors return JSON with `success: false` and an `error` field:

```json
{
  "success": false,
  "error": "Must provide config with seed"
}
```

**Common Errors:**
- `"Must provide config with seed"` - Config missing
- `"Failed to replay history action"` - Invalid action in history
- `"Failed to apply action"` - New action invalid or wrong player's turn
- `"Exception: ..."` - Internal error

### CORS Support

The server includes CORS headers for cross-origin requests:

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: POST, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

---

## Poker Engine Documentation

### Architecture Overview

The poker engine is built with a modular architecture consisting of six main classes:

```
┌─────────────────────────────────────────────────┐
│                     Game                        │
│  (Main orchestrator - manages game flow)        │
└─────────────────┬───────────────────────────────┘
                  │
         ┌────────┴─────────┐
         │                  │
    ┌────▼─────┐      ┌────▼─────┐
    │   Pot    │      │  Player  │
    │          │      │ (1-10)   │
    └────┬─────┘      └────┬─────┘
         │                  │
         │            ┌─────▼──────┐
         │            │    Hand    │
         │            │(Evaluation)│
         │            └─────┬──────┘
         │                  │
    ┌────▼──────────────────▼───┐
    │         Deck              │
    │   (52 cards, shuffle)     │
    └──────────────┬─────────────┘
                   │
              ┌────▼─────┐
              │   Card   │
              │ (Basic)  │
              └──────────┘
```

### Class Reference

#### 1. Card (`Card.h`)

Represents a single playing card with rank and suit.

**Features:**
- **Ranks**: 2-10, J, Q, K, A
- **Suits**: Clubs (♣), Diamonds (♦), Hearts (♥), Spades (♠)
- String parsing (e.g., "AS" for Ace of Spades)
- Unicode suit symbols
- Comparison operators

**Example Usage:**
```cpp
Card aceSpades("AS");
Card kingHearts("KH");

std::cout << aceSpades.toString();        // "AS"
std::cout << aceSpades.getSuitSymbol();   // "♠"
std::cout << aceSpades.getRankValue();    // 14

bool equal = (aceSpades == kingHearts);   // false
bool less = (kingHearts < aceSpades);     // true
```

**API:**
```cpp
// Constructors
Card()                              // Default: 2 of Clubs
Card(Rank r, Suit s)               // By rank and suit
Card(const std::string& str)       // Parse "AS", "7H", etc.

// Getters
Rank getRank() const
Suit getSuit() const
int getRankValue() const           // 2-14 (Ace=14)
int getSuitValue() const           // 0-3
std::string toString() const       // "AS", "7H", etc.
std::string getSuitSymbol() const  // "♠", "♥", etc.

// Operators
bool operator==(const Card& other) const
bool operator!=(const Card& other) const
bool operator<(const Card& other) const
```

#### 2. Deck (`Deck.h`)

Manages a standard 52-card deck with shuffle and deal operations.

**Features:**
- Fisher-Yates shuffle algorithm
- Seeded RNG for reproducible games
- Burn cards (remove top card)
- Remove specific cards (for known boards)
- Track remaining cards

**Example Usage:**
```cpp
Deck deck(12345);  // Seed for reproducibility
deck.shuffle();

Card card1 = deck.dealCard();
std::vector<Card> flop = deck.dealCards(3);

deck.burn();  // Burn before turn
Card turn = deck.dealCard();

std::cout << "Cards remaining: " << deck.cardsRemaining();
```

**API:**
```cpp
// Constructors
Deck()                          // Random seed
Deck(unsigned int seed)         // Fixed seed

// Operations
void reset()                    // Reset to full 52 cards
void shuffle()                  // Shuffle with current seed
void shuffle(unsigned int seed) // Shuffle with new seed
Card dealCard()                 // Deal one card
std::vector<Card> dealCards(size_t count)
void burn()                     // Burn top card
void removeCards(const std::vector<Card>& toRemove)

// Info
size_t cardsRemaining() const
size_t size() const             // Total cards (52)
```

#### 3. Hand (`Hand.h`)

Evaluates and compares poker hands according to standard Texas Hold'em rules.

**Features:**
- Evaluates all standard poker hands (High Card to Royal Flush)
- Compares hands with proper tiebreakers
- Finds best 5-card combination from 7 cards
- Handles wheel straight (A-2-3-4-5)

**Hand Rankings (Low to High):**
0. High Card
1. One Pair
2. Two Pair
3. Three of a Kind
4. Straight
5. Flush
6. Full House
7. Four of a Kind
8. Straight Flush
9. Royal Flush

**Example Usage:**
```cpp
// Evaluate 5 cards
std::vector<Card> cards = {
    Card("AS"), Card("AH"), Card("KD"), Card("QC"), Card("JH")
};
Hand::EvaluatedHand hand = Hand::evaluate(cards);
std::cout << hand.getRankingName();  // "One Pair"

// Evaluate with hole cards and community cards
std::vector<Card> holeCards = {Card("AS"), Card("KS")};
std::vector<Card> board = {Card("QS"), Card("JS"), Card("TS"), Card("7H"), Card("2D")};
Hand::EvaluatedHand bestHand = Hand::evaluate(holeCards, board);
std::cout << bestHand.getRankingName();  // "Royal Flush"

// Compare hands
if (hand1 > hand2) {
    std::cout << "Hand 1 wins!";
}
```

**API:**
```cpp
// Evaluation
static EvaluatedHand evaluate(const std::vector<Card>& cards)
static EvaluatedHand evaluate(const std::vector<Card>& holeCards,
                              const std::vector<Card>& communityCards)

// EvaluatedHand structure
struct EvaluatedHand {
    Ranking ranking;
    std::vector<int> tiebreakers;
    std::vector<Card> bestFive;
    
    int compare(const EvaluatedHand& other) const
    bool operator>(const EvaluatedHand& other) const
    bool operator<(const EvaluatedHand& other) const
    bool operator==(const EvaluatedHand& other) const
    std::string getRankingName() const
}
```

#### 4. Player (`Player.h`)

Represents a poker player with chips, cards, and game state.

**Features:**
- Chip management
- Hole cards
- Betting actions (fold, check, call, bet, raise, all-in)
- Player states (active, folded, all-in, out)
- Position tracking (dealer, small blind, big blind)
- Hand evaluation

**Example Usage:**
```cpp
Player player("p1", "Alice", 1000);

// Deal hole cards
std::vector<Card> holeCards = {Card("AS"), Card("KS")};
player.dealHoleCards(holeCards);

// Actions
player.makeBet(100);     // Bet 100
player.call(50);         // Call 50
player.raise(200);       // Raise to 200
player.fold();           // Fold
player.goAllIn();        // All-in

// Info
std::cout << player.getName();          // "Alice"
std::cout << player.getChips();         // 1000
std::cout << player.getBet();           // Current bet this round
std::cout << player.getTotalBet();      // Total bet this hand
std::cout << player.getActionName();    // "Bet", "Call", etc.
std::cout << player.getStateName();     // "Active", "Folded", etc.

// State checks
bool canAct = player.canAct();
bool inHand = player.isInHand();
```

**API:**
```cpp
// Constructor
Player(const std::string& id, const std::string& name, int startingChips)

// Getters
std::string getId() const
std::string getName() const
int getChips() const
int getBet() const
int getTotalBet() const
const std::vector<Card>& getHoleCards() const
State getState() const
Action getLastAction() const
int getPosition() const
bool getIsDealer() const
bool getIsSmallBlind() const
bool getIsBigBlind() const

// Actions
void dealHoleCards(const std::vector<Card>& cards)
void clearHoleCards()
bool placeBet(int amount)
bool postBlind(int amount)
void fold()
bool check(int currentBet)
bool call(int amountToCall)
bool makeBet(int amount)
bool raise(int raiseAmount, int totalAmount)
void goAllIn()
void winChips(int amount)
void resetBet()
void resetLastAction()
void resetForNewHand()

// State checks
bool canAct() const
bool isInHand() const

// Hand evaluation
Hand::EvaluatedHand evaluateHand(const std::vector<Card>& communityCards) const
```

#### 5. Pot (`Pot.h`)

Manages the pot and side pots, handling complex all-in scenarios.

**Features:**
- Main pot and side pot creation
- Automatic side pot calculation for all-in situations
- Pot distribution to winners
- Split pots for tied hands
- Betting round management

**Example Usage:**
```cpp
Pot pot;

// Set betting parameters
pot.setCurrentBet(20);
pot.setMinRaise(20);

// After betting round, collect bets
std::vector<Player*> players = game.getPlayers();
pot.collectBets(players);  // Creates side pots if needed

std::cout << "Total pot: " << pot.getTotalPot();
std::cout << "Side pots: " << pot.getSidePotCount();

// At showdown, distribute to winners
std::vector<Card> communityCards = game.getCommunityCards();
auto results = pot.distributePots(players, communityCards);

for (const auto& result : results) {
    std::cout << result.playerId << " wins " << result.amountWon 
              << " chips with " << result.handRanking;
}
```

**API:**
```cpp
// Constructor
Pot()

// Getters
int getTotalPot() const
int getMainPot() const
size_t getSidePotCount() const
const std::vector<SidePot>& getPots() const
int getCurrentBet() const
int getMinRaise() const

// Setters
void setCurrentBet(int bet)
void setMinRaise(int raise)

// Operations
void reset()
void collectBets(std::vector<Player*>& players)
std::vector<ShowdownResult> distributePots(
    const std::vector<Player*>& players,
    const std::vector<Card>& communityCards)
void startNewRound()
void updateBet(int newBet, int previousBet)

// Structures
struct SidePot {
    int amount;
    std::vector<std::string> eligiblePlayerIds;
}

struct ShowdownResult {
    std::string playerId;
    std::string handRanking;
    std::vector<std::string> bestFive;  // Best 5 card hand as strings
    int amountWon;
}
```

#### 6. Game (`Game.h`)

Main game orchestrator that manages the complete game flow for Texas Hold'em.

**Features:**
- Complete game state management
- Automatic dealer button rotation
- Blind posting
- Betting round management
- Community card dealing (flop, turn, river)
- Automatic stage progression
- Winner determination
- Multi-player support (2-10 players)

**Example Usage:**
```cpp
// Configure game
Game::GameConfig config;
config.smallBlind = 10;
config.bigBlind = 20;
config.startingChips = 1000;
config.minPlayers = 2;
config.maxPlayers = 10;
config.seed = 12345;  // For reproducible games

Game game(config);

// Add players
game.addPlayer("p1", "Alice");
game.addPlayer("p2", "Bob");
game.addPlayer("p3", "Charlie");

// Start a hand
if (game.startHand()) {
    std::cout << "Hand started! Stage: " << game.getStageName();
    
    // Game loop
    while (game.getStage() != Game::Stage::COMPLETE) {
        Player* currentPlayer = game.getCurrentPlayer();
        
        // Get player action (from UI, AI, etc.)
        Player::Action action = getPlayerAction(currentPlayer);
        int amount = getActionAmount(action);
        
        // Process action
        game.processAction(currentPlayer->getId(), action, amount);
    }
    
    std::cout << "Hand complete!";
}
```

**API:**
```cpp
// Constructor
Game(const GameConfig& config = GameConfig())

// Configuration
struct GameConfig {
    int smallBlind;                    // Default: 10
    int bigBlind;                      // Default: 20
    int startingChips;                 // Default: 1000
    int minPlayers;                    // Default: 2
    int maxPlayers;                    // Default: 10
    unsigned int seed;                 // Default: 0 (random)
    std::vector<std::string> exactCards; // Optional: exact card order for testing
}

// Player management
bool addPlayer(const std::string& id, const std::string& name)
bool removePlayer(const std::string& id)
Player* getPlayer(const std::string& id)
std::vector<Player*> getPlayers()
std::vector<Player*> getActivePlayers()

// Game flow
bool startHand()
bool processAction(const std::string& playerId, Player::Action action, int amount = 0)
bool advanceGame()  // Advances game to next stage deterministically
Player* getCurrentPlayer()

// Getters
Stage getStage() const
const std::vector<Card>& getCommunityCards() const
int getPotSize() const
int getCurrentBet() const
int getDealerPosition() const
int getHandNumber() const
const GameConfig& getConfig() const
std::string getStageName() const
```

### Integration Examples

#### Example 1: Simple Two-Player Game

```cpp
#include "Game.h"

int main() {
    Game game;
    game.addPlayer("p1", "Alice");
    game.addPlayer("p2", "Bob");
    
    game.startHand();
    
    // Preflop: Alice is big blind, Bob is small blind
    game.processAction("p1", Player::Action::RAISE, 60);
    game.processAction("p2", Player::Action::CALL);
    
    // Flop
    game.processAction("p2", Player::Action::CHECK);
    game.processAction("p1", Player::Action::BET, 100);
    game.processAction("p2", Player::Action::FOLD);
    
    // Alice wins
    return 0;
}
```

#### Example 2: Multi-Player with Side Pots

```cpp
Game game;
game.addPlayer("p1", "Alice");  // 1000 chips
game.addPlayer("p2", "Bob");    // 500 chips
game.addPlayer("p3", "Charlie"); // 250 chips

game.startHand();

// Everyone all-in preflop
game.processAction("p1", Player::Action::ALL_IN);  // 1000
game.processAction("p2", Player::Action::ALL_IN);  // 500
game.processAction("p3", Player::Action::ALL_IN);  // 250

// Main pot: 750 (250 * 3) - all three eligible
// Side pot 1: 500 (250 * 2) - Alice and Bob eligible
// Side pot 2: 500 (500 * 1) - Alice only eligible

// Pots distributed automatically at showdown
```

---

## Quick Reference

### Class Hierarchy

```
Game
├── Deck (manages cards)
├── Pot (manages betting)
└── Player[] (1-10 players)
    └── Hand (evaluates cards)
        └── Card (basic card)
```

### Quick Reference Code Examples

#### Card.h - Playing Card
```cpp
Card aceSpades("AS");
std::cout << aceSpades.toString();     // "AS"
std::cout << aceSpades.getSuitSymbol(); // "♠"
int rank = aceSpades.getRankValue();    // 14 (Ace)
```

#### Deck.h - Card Deck
```cpp
Deck deck(12345);           // Seeded for reproducibility
deck.shuffle();
Card card = deck.dealCard();
auto flop = deck.dealCards(3);
deck.burn();                // Burn before turn
```

#### Hand.h - Hand Evaluation
```cpp
std::vector<Card> cards = {Card("AS"), Card("KS"), /*...*/};
Hand::EvaluatedHand hand = Hand::evaluate(cards);
std::cout << hand.getRankingName();  // "Royal Flush"

if (hand1 > hand2) {
    std::cout << "Hand 1 wins!";
}
```

#### Player.h - Player State
```cpp
Player player("p1", "Alice", 1000);
player.dealHoleCards({Card("AS"), Card("KS")});
player.makeBet(100);
player.call(50);
player.raise(200);
player.fold();
player.goAllIn();

std::cout << player.getChips();      // 1000
std::cout << player.getActionName(); // "Bet"
```

#### Pot.h - Pot Management
```cpp
Pot pot;
pot.setCurrentBet(20);
pot.collectBets(players);  // Creates side pots if needed

std::cout << pot.getTotalPot();      // Total chips in pot
std::cout << pot.getSidePotCount();  // Number of side pots

// At showdown
auto results = pot.distributePots(players, communityCards);
```

#### Game.h - Game Orchestrator
```cpp
Game::GameConfig config;
config.smallBlind = 10;
config.bigBlind = 20;
config.startingChips = 1000;

Game game(config);
game.addPlayer("p1", "Alice");
game.addPlayer("p2", "Bob");

game.startHand();
game.processAction("p1", Player::Action::RAISE, 60);
game.processAction("p2", Player::Action::CALL);

std::cout << game.getStageName();           // "Flop"
std::cout << game.getPotSize();             // 120
auto cards = game.getCommunityCards();      // {Card, Card, Card}
```

### Enums and Constants

#### Action Types
```cpp
enum class Player::Action {
    NONE,
    FOLD,
    CHECK,
    CALL,
    BET,
    RAISE,
    ALL_IN
};
```

#### Player States
```cpp
enum class Player::State {
    WAITING,    // Waiting for game to start
    ACTIVE,     // Active in current hand
    FOLDED,     // Folded this hand
    ALL_IN,     // All-in
    OUT         // No chips left
};
```

#### Game Stages
```cpp
enum class Game::Stage {
    WAITING,    // Waiting for players
    PREFLOP,    // Before flop
    FLOP,       // 3 community cards
    TURN,       // 4 community cards
    RIVER,      // 5 community cards
    SHOWDOWN,   // Revealing hands
    COMPLETE    // Hand complete
};
```

#### Hand Rankings (Low to High)
```cpp
enum class Hand::Ranking {
    HIGH_CARD = 0,
    ONE_PAIR,
    TWO_PAIR,
    THREE_OF_A_KIND,
    STRAIGHT,
    FLUSH,
    FULL_HOUSE,
    FOUR_OF_A_KIND,
    STRAIGHT_FLUSH,
    ROYAL_FLUSH
};
```

---

## Testing

### Test the Poker Engine

Test the core poker engine classes:

```bash
make test
```

This runs the test suite which validates:
- Card operations
- Deck shuffling and dealing
- Hand evaluation
- Player actions
- Complete game flow

### Test the Stateless API

A comprehensive test suite is provided that:
1. Builds the C++ server
2. Starts it in the background
3. Tests with various payloads
4. Verifies responses are correct
5. Kills the server

**Recommended: Use uv (modern Python package manager)**
```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run tests with uv (automatically manages dependencies)
./run_tests.sh

# Or directly
uv run tests/test_stateless_api.py

# Specify custom port
uv run tests/test_stateless_api.py 9000
```

**Alternative: Use Python directly**
```bash
python3 tests/test_stateless_api.py
```

The test script validates:
- Empty game creation
- Adding players via history
- Starting hands and dealing cards
- Player actions (call, check, bet, raise, fold)
- Complete betting rounds through all stages
- Deterministic replay with same seed
- Error handling for invalid requests
- Complete hand playthrough to showdown
- Implicit advancement through all stages

---

## Project Structure

```
api/
├── src/                    # Source files
│   ├── main.cpp            # HTTP API server with stateless poker engine
│   ├── Card.h/cpp          # Playing card class
│   ├── Deck.h/cpp          # Deck management
│   ├── Hand.h/cpp          # Hand evaluation
│   ├── Player.h/cpp        # Player state and actions
│   ├── Pot.h/cpp           # Pot and side pot management
│   ├── Game.h/cpp          # Game orchestrator
│   ├── HTTPServer.h/cpp    # HTTP server implementation
│   ├── PokerEngineAPI.h/cpp # Stateless API interface
│   ├── JsonSerializer.h/cpp # JSON serialization utilities
│   └── json.hpp            # nlohmann/json library (auto-downloaded)
├── tests/                  # Test files
│   ├── test_card.cpp       # Card class tests
│   ├── test_deck.cpp       # Deck class tests
│   ├── test_hand.cpp       # Hand evaluation tests
│   ├── test_player.cpp     # Player class tests
│   ├── test_game.cpp       # Game integration tests
│   ├── test_stateless_api.py # Comprehensive API test suite
│   └── snapshots/          # Test snapshots for regression testing
├── build/                  # Build artifacts (generated)
├── README.md               # This file - complete API & engine documentation
├── CMakeLists.txt          # CMake build configuration
├── Makefile                # Simple Make build configuration
├── pyproject.toml          # Python project configuration for uv
└── uv.lock                 # uv dependency lock file
```

---

## Performance

### Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| Card creation | O(1) | Constant time |
| Deck shuffle | O(n) | Fisher-Yates, n=52 |
| Deal card | O(1) | Array access |
| Hand evaluation | O(21) | Fixed (C(7,5) = 21 combinations) |
| Player action | O(1) | State update |
| Pot collection | O(p) | p = number of players |
| Pot distribution | O(p × s) | p = players, s = side pots |

Memory usage: ~1-2KB per game instance + ~200 bytes per player

### API Performance Notes

- **Truly Stateless:** No memory used between requests
- **Single-threaded:** Uses blocking I/O
- **Deterministic:** Same seed always produces same results
- **Fast Replay:** History replay is efficient even with long games

### Thread Safety

**Not thread-safe**: The poker engine is designed for single-threaded use. For multi-threaded applications:
1. Use one Game instance per thread, or
2. Add external synchronization (mutexes) around Game methods

---

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

### Cleaning Up

To remove build artifacts:

With Make:
```bash
make clean
```

With CMake:
```bash
rm -rf build
```

---

## Dependencies

- **nlohmann/json**: Modern C++ JSON library
  - Automatically downloaded during build
  - Header-only library
  - [GitHub Repository](https://github.com/nlohmann/json)

---

## Key Benefits

### ✅ Automatic Advancement
- **Simpler API**: No need to specify `{"type": "advance"}` - it's automatic!
- **Fully deterministic**: Game state is solely determined by seed + history
- **No manual card dealing**: Server handles all card dealing automatically
- **Same seed = same cards**: Always produces same cards in same order
- **Perfect for simulations**: Quickly run thousands of hands for analysis
- **Less error-prone**: No need to track when to advance or which cards to deal

### ✅ Deterministic Replay
Same seed + same history = exactly same game state every time. Perfect for:
- Game replays and analysis
- Testing and debugging
- Multiplayer synchronization
- Monte Carlo simulations

### ✅ No Server State
- No memory leaks from abandoned games
- Restarts don't lose game data
- Easy to scale horizontally

### ✅ Client Controls History
- Client maintains full game history
- Can replay from any point
- Can fork game states for "what if" scenarios

### ✅ Simple Integration
- No session management needed
- No need to track game IDs
- Stateless = simpler architecture

---

## Security Considerations

⚠️ **This is a development server.** For production:

1. Add authentication/authorization
2. Validate all inputs thoroughly
3. Add rate limiting
4. Use HTTPS
5. Use a production HTTP server (nginx, etc.)
6. Implement proper error logging
7. Consider adding request signing to prevent tampering

---

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

---

## Quick API Tests

Test the server with curl:

```bash
# Create empty game
curl -X POST http://localhost:8080/simulate \
  -H "Content-Type: application/json" \
  -d '{"config": {"seed": 42, "smallBlind": 10, "bigBlind": 20}}'

# Add players and start hand
curl -X POST http://localhost:8080/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "config": {"seed": 42, "smallBlind": 10, "bigBlind": 20, "startingChips": 1000},
    "history": [
      {"type": "addPlayer", "playerId": "p1", "playerName": "Alice"},
      {"type": "addPlayer", "playerId": "p2", "playerName": "Bob"}
    ]
  }'

# Test error handling (missing config)
curl -X POST http://localhost:8080/simulate \
  -H "Content-Type: application/json" \
  -d '{}'
```

For comprehensive testing, use: `./run_tests.sh` or `uv run tests/test_stateless_api.py`

---

## License

See project LICENSE file.

---

## Support

For issues, questions, or contributions, see the main project README.
