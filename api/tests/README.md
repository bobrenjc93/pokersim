# Poker Engine API Tests

This directory contains snapshot tests for the stateless poker engine API.

## Overview

The test suite uses **snapshot testing** to validate complex poker scenarios. Instead of manually asserting on specific values, we capture the complete game state and save it to disk as JSON files. This makes tests easier to maintain and allows us to review changes when game logic is updated.

## Test Cases

The suite includes the following test scenarios:

1. **Side pots** - Multiple players with different stack sizes going all-in, creating side pots
2. **All-in preflop** - Players going all-in during the preflop betting round
3. **All-in on flop** - All-in after the flop cards are dealt
4. **All-in on turn** - All-in after the turn card is dealt
5. **All-in on river** - All-in on the final betting round

## Running Tests

### Normal Mode (Compare with Snapshots)

```bash
python3 test_stateless_api.py
```

This runs the tests and compares the actual responses with saved snapshots. Tests fail if there are any differences.

### Update Mode (Create/Update Snapshots)

```bash
python3 test_stateless_api.py --update-snapshots
```

Use this mode to:
- Create snapshots for new tests
- Update existing snapshots after intentional changes to game logic
- Review what changed when snapshots don't match

### Custom Port

```bash
python3 test_stateless_api.py 9000
python3 test_stateless_api.py 9000 --update-snapshots
```

## Snapshot Files

Snapshots are stored in `snapshots/` directory as JSON files. Each snapshot contains:

- Complete game state (stage, pot, community cards, etc.)
- All player states (chips, cards, position, etc.)
- Full event history (player actions, card dealing, etc.)
- Configuration used for the test

### Example Snapshot Structure

```json
{
  "gameState": {
    "stage": "Complete",
    "pot": 0,
    "communityCards": ["JC", "3C", "4D"],
    "players": [
      {
        "id": "player1",
        "chips": 1500,
        "holeCards": ["AS", "KS"],
        ...
      }
    ],
    "history": [...],
    "config": {...}
  },
  "success": true
}
```

## Key Features

### Per-Player Chip Amounts

Tests can now specify different starting chip amounts for each player:

```json
{
  "type": "addPlayer",
  "playerId": "short",
  "playerName": "ShortStack",
  "chips": 100
}
```

If `chips` is not specified, the player gets `config.startingChips` by default.

### Deterministic Replays

All tests use fixed seeds to ensure:
- Card dealing is deterministic
- Same history always produces same result
- Tests are reproducible

### Stateless API

The API is fully stateless:
- Each request includes complete history
- Server maintains no state between requests
- History is replayed to reconstruct game state

## Adding New Tests

1. Create a new test method in `SnapshotTester` class
2. Define the payload with config and history
3. Call `self.test_snapshot(name, description, payload)`
4. Run with `--update-snapshots` to create the snapshot
5. Run normally to verify the snapshot works

Example:

```python
def test_my_scenario(self):
    """Test: My custom scenario"""
    payload = {
        "config": {
            "seed": 1234,
            "smallBlind": 10,
            "bigBlind": 20,
            "startingChips": 1000
        },
        "history": [
            {"type": "addPlayer", "playerId": "p1", "playerName": "Player1"},
            {"type": "addPlayer", "playerId": "p2", "playerName": "Player2"},
            # ... more actions
        ]
    }
    self.test_snapshot("my_scenario", "My custom scenario", payload)
```

## Troubleshooting

### Snapshot Mismatch

If a test fails with snapshot mismatch:

1. Review the diff shown in the output
2. If the change is expected (e.g., you updated game logic):
   - Run with `--update-snapshots` to accept the changes
3. If the change is unexpected:
   - Fix your code or test case
   - Re-run tests to verify

### Player Action Order

In heads-up play:
- The dealer is also the small blind
- Small blind acts first preflop
- Small blind acts first post-flop

With 3+ players:
- Dealer is separate from blinds
- Small blind is to dealer's left
- Big blind is to small blind's left
- First to act preflop is to big blind's left
- First to act post-flop is first active player after dealer

### Raise Amounts

The `amount` parameter for raises is the **additional amount to add** to your current bet, not the total bet:

```json
// If you've bet 10 and want to raise to 100 total:
{"type": "playerAction", "playerId": "p1", "action": "raise", "amount": 90}
```

## Benefits of Snapshot Testing

1. **Complete Coverage** - Captures entire game state, not just specific fields
2. **Easy Maintenance** - Update snapshots instead of rewriting assertions
3. **Visual Diffs** - Easy to see what changed when tests fail
4. **Regression Testing** - Ensures game logic doesn't change unexpectedly
5. **Documentation** - Snapshots serve as examples of expected game states

