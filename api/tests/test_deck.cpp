#include <iostream>
#include <cassert>
#include "Deck.h"

void testDeckInitialization() {
    std::cout << "Testing deck initialization..." << std::endl;
    
    Deck deck(42);
    assert(deck.size() == 52);
    assert(deck.cardsRemaining() == 52);
    
    std::cout << "  ✓ Deck initialized with 52 cards" << std::endl;
}

void testDeckShuffle() {
    std::cout << "Testing deck shuffle..." << std::endl;
    
    Deck deck(42);
    deck.shuffle();
    
    // Deck should still have 52 cards after shuffle
    assert(deck.cardsRemaining() == 52);
    
    std::cout << "  ✓ Deck shuffled successfully" << std::endl;
}

void testDealingCards() {
    std::cout << "Testing dealing cards..." << std::endl;
    
    Deck deck(42);
    deck.shuffle();
    
    Card card1 = deck.dealCard();
    assert(deck.cardsRemaining() == 51);
    
    Card card2 = deck.dealCard();
    assert(deck.cardsRemaining() == 50);
    
    Card card3 = deck.dealCard();
    assert(deck.cardsRemaining() == 49);
    
    std::cout << "  Dealt 3 cards: " << card1.toString() << ", " 
              << card2.toString() << ", " << card3.toString() << std::endl;
    std::cout << "  ✓ Cards dealt correctly, count updated" << std::endl;
}

void testDeckReset() {
    std::cout << "Testing deck reset..." << std::endl;
    
    Deck deck(42);
    deck.shuffle();
    
    // Deal some cards
    for (int i = 0; i < 10; i++) {
        (void)deck.dealCard();
    }
    assert(deck.cardsRemaining() == 42);
    
    // Reset deck
    deck.reset();
    assert(deck.cardsRemaining() == 52);
    
    std::cout << "  ✓ Deck reset successfully" << std::endl;
}

void testDeckReproducibility() {
    std::cout << "Testing deck reproducibility with seed..." << std::endl;
    
    Deck deck1(12345);
    deck1.shuffle();
    Card card1 = deck1.dealCard();
    
    Deck deck2(12345);
    deck2.shuffle();
    Card card2 = deck2.dealCard();
    
    // Same seed should produce same first card after shuffle
    assert(card1.toString() == card2.toString());
    
    std::cout << "  ✓ Same seed produces same shuffle" << std::endl;
}

int main() {
    std::cout << "\n🃏 Deck Test Suite" << std::endl;
    std::cout << "==================" << std::endl;
    
    try {
        testDeckInitialization();
        testDeckShuffle();
        testDealingCards();
        testDeckReset();
        testDeckReproducibility();
        
        std::cout << "\n✅ All Deck tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}

