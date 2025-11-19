#include <iostream>
#include <cassert>
#include "Player.h"
#include "Card.h"

void testPlayerCreation() {
    std::cout << "Testing player creation..." << std::endl;
    
    Player player("p1", "Alice", 1000);
    
    assert(player.getId() == "p1");
    assert(player.getName() == "Alice");
    assert(player.getChips() == 1000);
    assert(player.getBet() == 0);
    assert(player.getState() == Player::State::WAITING);
    
    std::cout << "  ✓ Player created with correct initial state" << std::endl;
}

void testDealingHoleCards() {
    std::cout << "Testing dealing hole cards..." << std::endl;
    
    Player player("p1", "Alice", 1000);
    
    std::vector<Card> holeCards = {Card("AS"), Card("KS")};
    player.dealHoleCards(holeCards);
    
    auto cards = player.getHoleCards();
    assert(cards.size() == 2);
    assert(cards[0].toString() == "AS");
    assert(cards[1].toString() == "KS");
    
    std::cout << "  ✓ Hole cards dealt correctly" << std::endl;
}

void testBettingAndActions() {
    std::cout << "Testing betting mechanics and player actions..." << std::endl;
    
    Player player("p1", "Alice", 1000);
    
    // Test basic bet
    (void)player.makeBet(100);
    assert(player.getChips() == 900);
    assert(player.getBet() == 100);
    assert(player.getActionName() == "Bet");
    
    // Test accumulating bets
    (void)player.makeBet(50);
    assert(player.getChips() == 850);
    assert(player.getBet() == 150);
    
    // Test check action
    Player player2("p2", "Bob", 1000);
    (void)player2.check();
    assert(player2.getActionName() == "Check");
    assert(player2.getBet() == 0);
    
    // Test call action
    (void)player2.call(150);
    assert(player2.getActionName() == "Call");
    assert(player2.getChips() == 850);
    assert(player2.getBet() == 150);
    
    // Test fold action
    player2.fold();
    assert(player2.getState() == Player::State::FOLDED);
    assert(player2.getActionName() == "Fold");
    
    std::cout << "  ✓ Betting, actions, and chip tracking work correctly" << std::endl;
}

void testAllInAndChipManagement() {
    std::cout << "Testing all-in and winning chips..." << std::endl;
    
    Player player("p1", "Alice", 100);
    
    // Test all-in
    (void)player.makeBet(100);
    assert(player.getChips() == 0);
    assert(player.getBet() == 100);
    
    // Test winning chips
    player.winChips(250);
    assert(player.getChips() == 250);
    assert(player.getBet() == 100);  // Bet amount unchanged by winnings
    
    std::cout << "  ✓ All-in and chip winning work correctly" << std::endl;
}

void testResetForNewHand() {
    std::cout << "Testing reset for new hand..." << std::endl;
    
    Player player("p1", "Alice", 1000);
    
    // Setup player state
    player.setState(Player::State::ACTIVE);
    (void)player.makeBet(100);
    player.fold();
    player.dealHoleCards({Card("AS"), Card("KS")});
    
    // Reset
    player.resetForNewHand();
    
    assert(player.getBet() == 0);
    assert(player.getState() != Player::State::FOLDED);
    assert(player.getHoleCards().empty());
    
    std::cout << "  ✓ Player reset correctly for new hand" << std::endl;
}

int main() {
    std::cout << "\n🃏 Player Test Suite" << std::endl;
    std::cout << "====================" << std::endl;
    
    try {
        testPlayerCreation();
        testDealingHoleCards();
        testBettingAndActions();
        testAllInAndChipManagement();
        testResetForNewHand();
        
        std::cout << "\n✅ All Player tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}

