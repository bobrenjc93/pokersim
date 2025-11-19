# Poker Engine Documentation

A complete, production-ready Texas Hold'em poker engine implementation in C++17.

## Architecture Overview

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

## Class Reference

### 1. Card (`Card.h`)

Represents a single playing card with rank and suit.

#### Features
- **Ranks**: 2-10, J, Q, K, A
- **Suits**: Clubs (♣), Diamonds (♦), Hearts (♥), Spades (♠)
- String parsing (e.g., "AS" for Ace of Spades)
- Unicode suit symbols
- Comparison operators

#### Example Usage
```cpp
Card aceSpades("AS");
Card kingHearts("KH");

std::cout << aceSpades.toString();        // "AS"
std::cout << aceSpades.getSuitSymbol();   // "♠"
std::cout << aceSpades.getRankValue();    // 14

bool equal = (aceSpades == kingHearts);   // false
bool less = (kingHearts < aceSpades);     // true
```

#### API
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

---

### 2. Deck (`Deck.h`)

Manages a standard 52-card deck with shuffle and deal operations.

#### Features
- Fisher-Yates shuffle algorithm
- Seeded RNG for reproducible games
- Burn cards (remove top card)
- Remove specific cards (for known boards)
- Track remaining cards

#### Example Usage
```cpp
Deck deck(12345);  // Seed for reproducibility
deck.shuffle();

Card card1 = deck.dealCard();
std::vector<Card> flop = deck.dealCards(3);

deck.burn();  // Burn before turn
Card turn = deck.dealCard();

std::cout << "Cards remaining: " << deck.cardsRemaining();
```

#### API
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

---

### 3. Hand (`Hand.h`)

Evaluates and compares poker hands according to standard Texas Hold'em rules.

#### Features
- Evaluates all standard poker hands (High Card to Royal Flush)
- Compares hands with proper tiebreakers
- Finds best 5-card combination from 7 cards
- Handles wheel straight (A-2-3-4-5)

#### Hand Rankings (Low to High)
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

#### Example Usage
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

#### API
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

---

### 4. Player (`Player.h`)

Represents a poker player with chips, cards, and game state.

#### Features
- Chip management
- Hole cards
- Betting actions (fold, check, call, bet, raise, all-in)
- Player states (active, folded, all-in, out)
- Position tracking (dealer, small blind, big blind)
- Hand evaluation

#### Example Usage
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

#### API
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
bool check()
bool call(int amountToCall)
bool makeBet(int amount)
bool raise(int raiseAmount, int totalAmount)
void goAllIn()
void winChips(int amount)
void resetBet()
void resetForNewHand()

// State checks
bool canAct() const
bool isInHand() const

// Hand evaluation
Hand::EvaluatedHand evaluateHand(const std::vector<Card>& communityCards) const
```

---

### 5. Pot (`Pot.h`)

Manages the pot and side pots, handling complex all-in scenarios.

#### Features
- Main pot and side pot creation
- Automatic side pot calculation for all-in situations
- Pot distribution to winners
- Split pots for tied hands
- Betting round management

#### Example Usage
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
auto winnings = pot.distributePots(players, communityCards);

for (const auto& [playerId, amount] : winnings) {
    std::cout << playerId << " wins " << amount << " chips";
}
```

#### API
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
std::map<std::string, int> distributePots(
    const std::vector<Player*>& players,
    const std::vector<Card>& communityCards)
void startNewRound()
void updateBet(int newBet, int previousBet)

// SidePot structure
struct SidePot {
    int amount;
    std::vector<std::string> eligiblePlayerIds;
}
```

---

### 6. Game (`Game.h`)

Main game orchestrator that manages the complete game flow for Texas Hold'em.

#### Features
- Complete game state management
- Automatic dealer button rotation
- Blind posting
- Betting round management
- Community card dealing (flop, turn, river)
- Automatic stage progression
- Winner determination
- Multi-player support (2-10 players)

#### Game Stages
1. **WAITING** - Waiting for players to join
2. **PREFLOP** - After hole cards dealt, before flop
3. **FLOP** - 3 community cards on board
4. **TURN** - 4 community cards on board
5. **RIVER** - 5 community cards on board
6. **SHOWDOWN** - Revealing hands
7. **COMPLETE** - Hand complete, ready for next hand

#### Example Usage
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
        
        // Show community cards if stage changed
        if (game.getCommunityCards().size() > 0) {
            for (const auto& card : game.getCommunityCards()) {
                std::cout << card.toString() << " ";
            }
        }
    }
    
    std::cout << "Hand complete!";
}
```

#### Complete Game Example
```cpp
Game game;
game.addPlayer("p1", "Alice");
game.addPlayer("p2", "Bob");

// Play a hand
game.startHand();

// Preflop
game.processAction("p1", Player::Action::CALL);
game.processAction("p2", Player::Action::CHECK);

// Flop dealt automatically
game.processAction("p2", Player::Action::CHECK);
game.processAction("p1", Player::Action::BET, 50);
game.processAction("p2", Player::Action::CALL);

// Turn dealt automatically
game.processAction("p2", Player::Action::CHECK);
game.processAction("p1", Player::Action::BET, 100);
game.processAction("p2", Player::Action::RAISE, 200);
game.processAction("p1", Player::Action::CALL);

// River dealt automatically
game.processAction("p2", Player::Action::BET, 200);
game.processAction("p1", Player::Action::FOLD);

// Winner determined automatically
// Start next hand
game.startHand();
```

#### API
```cpp
// Constructor
Game(const GameConfig& config = GameConfig())

// Configuration
struct GameConfig {
    int smallBlind;      // Default: 10
    int bigBlind;        // Default: 20
    int startingChips;   // Default: 1000
    int minPlayers;      // Default: 2
    int maxPlayers;      // Default: 10
    unsigned int seed;   // Default: 0 (random)
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

---

## Integration Examples

### Example 1: Simple Two-Player Game

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

### Example 2: Multi-Player with Side Pots

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

### Example 3: API Server Integration

```cpp
#include "Game.h"
#include "json.hpp"

using json = nlohmann::json;

class PokerAPI {
private:
    Game game;
    
public:
    json processGameAction(const json& request) {
        std::string playerId = request["playerId"];
        std::string actionStr = request["action"];
        int amount = request.value("amount", 0);
        
        Player::Action action = parseAction(actionStr);
        bool success = game.processAction(playerId, action, amount);
        
        json response;
        response["success"] = success;
        response["stage"] = game.getStageName();
        response["pot"] = game.getPotSize();
        response["currentBet"] = game.getCurrentBet();
        
        // Add community cards
        json cardsJson = json::array();
        for (const auto& card : game.getCommunityCards()) {
            cardsJson.push_back(card.toString());
        }
        response["communityCards"] = cardsJson;
        
        // Add players info
        json playersJson = json::array();
        for (auto* player : game.getPlayers()) {
            json playerJson;
            playerJson["id"] = player->getId();
            playerJson["name"] = player->getName();
            playerJson["chips"] = player->getChips();
            playerJson["bet"] = player->getBet();
            playerJson["state"] = player->getStateName();
            playersJson.push_back(playerJson);
        }
        response["players"] = playersJson;
        
        return response;
    }
};
```

---

## Testing

The poker engine includes comprehensive test suites in individual `test_*.cpp` files:

```bash
# Build and run all tests
make test

# Or build and run separately
make build_tests
./build/test_card
./build/test_deck
./build/test_hand
./build/test_player
./build/test_game
```

### Test Coverage
- ✅ Card parsing and comparison
- ✅ Deck shuffling and dealing
- ✅ Hand evaluation (all rankings)
- ✅ Player actions and state management
- ✅ Complete game flow
- ✅ Multiple betting rounds
- ✅ Automatic stage progression

---

## Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| Card creation | O(1) | Constant time |
| Deck shuffle | O(n) | Fisher-Yates, n=52 |
| Deal card | O(1) | Array access |
| Hand evaluation | O(21) | Fixed (C(7,5) = 21 combinations) |
| Player action | O(1) | State update |
| Pot collection | O(p) | p = number of players |
| Pot distribution | O(p * s) | p = players, s = side pots |

Memory usage: ~1-2KB per game instance + ~200 bytes per player

---

## Thread Safety

**Not thread-safe**: The poker engine is designed for single-threaded use. For multi-threaded applications:
1. Use one Game instance per thread, or
2. Add external synchronization (mutexes) around Game methods

---

## Future Enhancements

Potential additions for future versions:
- [ ] Tournament support (increasing blinds)
- [ ] Omaha Hold'em variant
- [ ] Hand history logging
- [ ] Replay functionality
- [ ] AI player integration
- [ ] Pot odds calculator
- [ ] Expected value (EV) calculations
- [ ] Monte Carlo simulation for equity

---

## License

See project LICENSE file.

---

## Support

For issues, questions, or contributions, see the main project README.

