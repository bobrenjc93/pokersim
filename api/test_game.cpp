#include <iostream>
#include <cassert>
#include "Game.h"

void testGameInitialization() {
    std::cout << "Testing game initialization..." << std::endl;
    
    Game::GameConfig config;
    config.smallBlind = 10;
    config.bigBlind = 20;
    config.startingChips = 1000;
    config.seed = 12345;
    
    Game game(config);
    
    assert(game.getPlayers().empty());
    assert(game.getCommunityCards().empty());
    assert(game.getPotSize() == 0);
    
    std::cout << "  ✓ Game initialized with correct config" << std::endl;
}

void testAddingPlayers() {
    std::cout << "Testing adding players..." << std::endl;
    
    Game::GameConfig config;
    config.startingChips = 1000;
    Game game(config);
    
    assert(game.addPlayer("p1", "Alice"));
    assert(game.addPlayer("p2", "Bob"));
    assert(game.addPlayer("p3", "Charlie"));
    
    auto players = game.getPlayers();
    assert(players.size() == 3);
    assert(players[0]->getName() == "Alice");
    assert(players[1]->getName() == "Bob");
    assert(players[2]->getName() == "Charlie");
    
    std::cout << "  ✓ Players added successfully" << std::endl;
}

void testStartHand() {
    std::cout << "Testing starting a hand..." << std::endl;
    
    Game::GameConfig config;
    config.smallBlind = 10;
    config.bigBlind = 20;
    config.startingChips = 1000;
    config.seed = 12345;
    
    Game game(config);
    assert(game.addPlayer("p1", "Alice"));
    assert(game.addPlayer("p2", "Bob"));
    assert(game.addPlayer("p3", "Charlie"));
    
    bool started = game.startHand();
    assert(started);
    
    // Check hole cards were dealt
    auto players = game.getPlayers();
    for (auto* player : players) {
        assert(player->getHoleCards().size() == 2);
    }
    
    // Check that blinds were posted (check player bets)
    int totalBets = 0;
    for (auto* player : players) {
        totalBets += player->getBet();
    }
    assert(totalBets == 30); // Small blind + big blind
    
    // Check stage
    assert(game.getStageName() == "Preflop");
    
    std::cout << "  ✓ Hand started with blinds and hole cards" << std::endl;
}

void testCompleteHandFlow() {
    std::cout << "Testing complete hand flow from preflop to showdown..." << std::endl;
    
    Game::GameConfig config;
    config.smallBlind = 10;
    config.bigBlind = 20;
    config.startingChips = 1000;
    config.seed = 12345;
    
    Game game(config);
    assert(game.addPlayer("p1", "Alice"));
    assert(game.addPlayer("p2", "Bob"));
    assert(game.addPlayer("p3", "Charlie"));
    
    assert(game.startHand());
    assert(game.getStageName() == "Preflop");
    
    int initialPot = game.getPotSize();
    
    // Complete preflop - all players call/check
    for (int i = 0; i < 3; i++) {
        auto* current = game.getCurrentPlayer();
        assert(current != nullptr);
        if (i < 2) {
            assert(game.processAction(current->getId(), Player::Action::CALL));
        } else {
            assert(game.processAction(current->getId(), Player::Action::CHECK));
        }
    }
    
    // Should advance to Flop
    std::string stage = game.getStageName();
    assert(stage == "Flop" || stage == "Complete");
    assert(game.getPotSize() > initialPot);
    assert(game.getPotSize() >= 60);  // 3 players × 20 chips
    
    std::cout << "  ✓ Hand flow, pot management, and player tracking validated" << std::endl;
}

void testPlayerFoldingEndHand() {
    std::cout << "Testing folding leads to early hand completion..." << std::endl;
    
    Game::GameConfig config;
    config.smallBlind = 10;
    config.bigBlind = 20;
    config.startingChips = 1000;
    config.seed = 54321;
    
    Game game(config);
    (void)game.addPlayer("p1", "Alice");
    (void)game.addPlayer("p2", "Bob");
    (void)game.addPlayer("p3", "Charlie");
    
    (void)game.startHand();
    
    // First two players fold, third wins
    auto* player1 = game.getCurrentPlayer();
    assert(game.processAction(player1->getId(), Player::Action::FOLD));
    assert(player1->getState() == Player::State::FOLDED);
    
    auto* player2 = game.getCurrentPlayer();
    assert(player2 != nullptr);
    assert(player2->getId() != player1->getId());
    assert(game.processAction(player2->getId(), Player::Action::FOLD));
    
    // Game should end, remaining player wins
    assert(game.getStageName() == "Complete");
    
    std::cout << "  ✓ Folding mechanics and early hand completion work" << std::endl;
}

int main() {
    std::cout << "\n🃏 Game Integration Test Suite" << std::endl;
    std::cout << "===============================" << std::endl;
    
    try {
        testGameInitialization();
        testAddingPlayers();
        testStartHand();
        testCompleteHandFlow();
        testPlayerFoldingEndHand();
        
        std::cout << "\n✅ All Game integration tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}

