# Poker Engine Classes Summary

This document provides a quick reference for all poker engine classes.

## Class Hierarchy

```
Game
├── Deck (manages cards)
├── Pot (manages betting)
└── Player[] (1-10 players)
    └── Hand (evaluates cards)
        └── Card (basic card)
```

## Quick Reference

### Card.h - Playing Card
```cpp
Card aceSpades("AS");
std::cout << aceSpades.toString();     // "AS"
std::cout << aceSpades.getSuitSymbol(); // "♠"
int rank = aceSpades.getRankValue();    // 14 (Ace)
```

### Deck.h - Card Deck
```cpp
Deck deck(12345);           // Seeded for reproducibility
deck.shuffle();
Card card = deck.dealCard();
auto flop = deck.dealCards(3);
deck.burn();                // Burn before turn
```

### Hand.h - Hand Evaluation
```cpp
std::vector<Card> cards = {Card("AS"), Card("KS"), /*...*/};
Hand::EvaluatedHand hand = Hand::evaluate(cards);
std::cout << hand.getRankingName();  // "Royal Flush"

if (hand1 > hand2) {
    std::cout << "Hand 1 wins!";
}
```

### Player.h - Player State
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

### Pot.h - Pot Management
```cpp
Pot pot;
pot.setCurrentBet(20);
pot.collectBets(players);  // Creates side pots if needed

std::cout << pot.getTotalPot();      // Total chips in pot
std::cout << pot.getSidePotCount();  // Number of side pots

// At showdown
auto winnings = pot.distributePots(players, communityCards);
```

### Game.h - Game Orchestrator
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

## Complete Game Flow Example

```cpp
#include "Game.h"
#include <iostream>

int main() {
    // Setup
    Game::GameConfig config;
    config.seed = 42;  // Reproducible
    Game game(config);
    
    // Add players
    game.addPlayer("alice", "Alice");
    game.addPlayer("bob", "Bob");
    game.addPlayer("charlie", "Charlie");
    
    // Start hand
    game.startHand();
    std::cout << "Stage: " << game.getStageName() << "\n";  // "Preflop"
    
    // Preflop betting
    auto* current = game.getCurrentPlayer();
    game.processAction(current->getId(), Player::Action::RAISE, 60);
    
    current = game.getCurrentPlayer();
    game.processAction(current->getId(), Player::Action::CALL);
    
    current = game.getCurrentPlayer();
    game.processAction(current->getId(), Player::Action::FOLD);
    
    // Flop (dealt automatically)
    std::cout << "Stage: " << game.getStageName() << "\n";  // "Flop"
    std::cout << "Community cards: ";
    for (const auto& card : game.getCommunityCards()) {
        std::cout << card.toString() << " ";
    }
    std::cout << "\n";
    
    // Flop betting
    current = game.getCurrentPlayer();
    game.processAction(current->getId(), Player::Action::CHECK);
    
    current = game.getCurrentPlayer();
    game.processAction(current->getId(), Player::Action::BET, 100);
    
    current = game.getCurrentPlayer();
    game.processAction(current->getId(), Player::Action::CALL);
    
    // Turn (dealt automatically)
    std::cout << "Stage: " << game.getStageName() << "\n";  // "Turn"
    
    // Continue betting...
    
    return 0;
}
```

## Action Types

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

## Player States

```cpp
enum class Player::State {
    WAITING,    // Waiting for game to start
    ACTIVE,     // Active in current hand
    FOLDED,     // Folded this hand
    ALL_IN,     // All-in
    OUT         // No chips left
};
```

## Game Stages

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

## Hand Rankings (Low to High)

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

## Building and Testing

```bash
# Build everything
make

# Run tests
make test

# Clean build artifacts
make clean
```

## Files

| File | Purpose |
|------|---------|
| `Card.h/cpp` | Playing card representation |
| `Deck.h/cpp` | Deck management (52 cards) |
| `Hand.h/cpp` | Hand evaluation and comparison |
| `Player.h/cpp` | Player state and actions |
| `Pot.h/cpp` | Pot and side pot management |
| `Game.h/cpp` | Game orchestrator |
| `test_*.cpp` | Individual test files |
| `main.cpp` | HTTP API server |

## Key Features

✅ **Complete Texas Hold'em** - Full game implementation  
✅ **Hand Evaluation** - Proper ranking of all poker hands  
✅ **Side Pots** - Automatic side pot creation for all-in scenarios  
✅ **2-10 Players** - Support for heads-up to full ring games  
✅ **Seeded RNG** - Deterministic games for testing/replay  
✅ **Production Ready** - Clean, well-documented, tested code  
✅ **Header-only** - Easy to integrate into any project  
✅ **C++17** - Modern C++ with no external dependencies (except JSON for API)  

## Performance

- **Hand Evaluation**: O(21) - Evaluates all 21 combinations of 5 cards from 7
- **Deck Shuffle**: O(52) - Fisher-Yates algorithm
- **Player Action**: O(1) - Constant time state updates
- **Pot Distribution**: O(players × side_pots) - Typically very fast

## Integration

All classes are header-only and can be easily integrated:

```cpp
#include "Card.h"
#include "Deck.h"
#include "Hand.h"
#include "Player.h"
#include "Pot.h"
#include "Game.h"

// Ready to use!
```

For detailed documentation, see [POKER_ENGINE.md](POKER_ENGINE.md).

