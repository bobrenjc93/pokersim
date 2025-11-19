#include <iostream>
#include <cassert>
#include "Card.h"

void testCardCreation() {
    std::cout << "Testing card creation..." << std::endl;
    
    Card aceSpades("AS");
    Card kingHearts("KH");
    Card queenDiamonds("QD");
    
    assert(aceSpades.toString() == "AS");
    assert(kingHearts.toString() == "KH");
    assert(queenDiamonds.toString() == "QD");
    
    std::cout << "  ✓ Card creation successful" << std::endl;
}

void testCardRanks() {
    std::cout << "Testing card ranks..." << std::endl;
    
    Card ace("AS");
    Card king("KH");
    Card queen("QD");
    Card jack("JC");
    Card ten("TS");
    Card two("2H");
    
    assert(ace.getRankValue() == 14);
    assert(king.getRankValue() == 13);
    assert(queen.getRankValue() == 12);
    assert(jack.getRankValue() == 11);
    assert(ten.getRankValue() == 10);
    assert(two.getRankValue() == 2);
    
    std::cout << "  ✓ Card rank values correct" << std::endl;
}

void testCardSuits() {
    std::cout << "Testing card suits..." << std::endl;
    
    Card spade("AS");
    Card heart("KH");
    Card diamond("QD");
    Card club("JC");
    
    assert(spade.getSuitSymbol() == "♠");
    assert(heart.getSuitSymbol() == "♥");
    assert(diamond.getSuitSymbol() == "♦");
    assert(club.getSuitSymbol() == "♣");
    
    std::cout << "  ✓ Card suit symbols correct" << std::endl;
}

int main() {
    std::cout << "\n🃏 Card Test Suite" << std::endl;
    std::cout << "==================" << std::endl;
    
    try {
        testCardCreation();
        testCardRanks();
        testCardSuits();
        
        std::cout << "\n✅ All Card tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}

