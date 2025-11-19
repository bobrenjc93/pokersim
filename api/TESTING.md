# Poker Engine Test Suite

This document describes the testing structure for the poker engine.

## Test Files

The test suite has been organized into granular, component-specific test files:

### 1. `test_card.cpp` - Card Class Tests
Tests the fundamental Card class functionality:
- Card creation from string notation (e.g., "AS", "KH")
- Rank values (Ace=14, King=13, etc.)
- Suit symbols (♠, ♥, ♦, ♣)
- Card display formatting

**Run:** `./test_card`

### 2. `test_deck.cpp` - Deck Class Tests
Tests deck management and shuffling:
- Deck initialization (52 cards)
- Shuffling functionality
- Dealing cards and tracking remaining count
- Deck reset
- Reproducible shuffles with seeds

**Run:** `./test_deck`

### 3. `test_hand.cpp` - Hand Evaluation Tests
Tests poker hand ranking and comparison:
- All hand rankings (Royal Flush through High Card)
- Hand comparison logic
- Proper detection of:
  - Royal Flush
  - Straight Flush
  - Four of a Kind
  - Full House
  - Flush
  - Straight
  - Three of a Kind
  - Two Pair
  - One Pair
  - High Card

**Run:** `./test_hand`

### 4. `test_player.cpp` - Player Class Tests
Tests player state management:
- Player creation and initialization
- Dealing hole cards
- Betting mechanics (bet, call, raise)
- All-in functionality
- Folding
- Action tracking
- Chip management
- Player reset for new hands

**Run:** `./test_player`

### 5. `test_game.cpp` - Game Integration Tests
Tests the complete game flow:
- Game initialization with configuration
- Adding players
- Starting a hand
- Preflop betting rounds
- Flop betting rounds
- Player folding
- Current player tracking
- Pot management

**Run:** `./test_game`

## Running Tests

### Run All Tests
```bash
make test
```

This will:
1. Build all test executables
2. Run each test suite in sequence
3. Display results for each component
4. Report overall success/failure

### Run Individual Tests
```bash
# Build specific test
make test_card

# Run it
./test_card
```

### Build All Tests Without Running
```bash
make build_tests
```

### Clean Up
```bash
make clean
```

This removes all test executables and the main poker_api binary.

## Test Output Format

Each test suite displays:
- Suite name with decorative header
- Individual test names and results
- ✓ checkmarks for passing tests
- Detailed error messages for failures
- Summary at the end

Example:
```
🃏 Card Test Suite
==================
Testing card creation...
  ✓ Card creation successful
Testing card ranks...
  ✓ Card rank values correct
...
✅ All Card tests passed!
```

## Adding New Tests

To add a new test file:

1. Create the test file (e.g., `test_pot.cpp`)
2. Include necessary headers
3. Implement test functions
4. Create a main() function that calls all tests
5. Add to Makefile:
   ```makefile
   test_pot: test_pot.cpp $(POKER_SOURCES)
       $(CXX) $(CXXFLAGS) test_pot.cpp $(POKER_SOURCES) -o test_pot
   ```
6. Add `test_pot` to `TEST_TARGETS` list
7. Update the test chain in the `test:` target

## Test Design Philosophy

- **Granular**: Each test file focuses on a single component
- **Isolated**: Tests don't depend on each other
- **Descriptive**: Test names clearly indicate what's being tested
- **Comprehensive**: Cover normal cases, edge cases, and error conditions
- **Fast**: All tests run in under a second
- **Deterministic**: Using seeds for reproducible results

## Continuous Integration

These tests are designed to be run as part of a CI/CD pipeline:
- All tests must pass before merging
- Exit code 0 indicates success
- Non-zero exit code indicates failure
- Tests can run in parallel if needed (no shared state)

## Coverage

Current test coverage includes:
- ✅ Card creation and properties
- ✅ Deck management and shuffling
- ✅ All poker hand rankings
- ✅ Player state and actions
- ✅ Game flow and betting rounds
- ✅ Pot management
- ⚠️  Side pot logic (basic coverage)
- ⚠️  Showdown and winner determination (basic coverage)
- ❌ Network/API layer (not yet tested)

## Future Enhancements

Potential improvements to the test suite:
- Add `test_pot.cpp` for detailed pot/side pot testing
- Add stress tests for edge cases
- Add performance benchmarks
- Add memory leak detection
- Add fuzzing tests for card/hand parsing
- Add integration tests for full game scenarios

