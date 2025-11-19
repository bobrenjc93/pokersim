#ifndef DECK_H
#define DECK_H

#include "Card.h"
#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>

/**
 * Represents a standard 52-card deck with shuffle and deal operations.
 */
class Deck {
private:
    std::vector<Card> cards;
    size_t currentCard;
    std::mt19937 rng;
    
public:
    Deck();
    
    /**
     * Constructor with seed for deterministic shuffles
     */
    explicit Deck(unsigned int seed);
    
    /**
     * Resets the deck to a full 52-card deck in order
     */
    void reset();
    
    /**
     * Shuffles the deck using Fisher-Yates algorithm
     */
    void shuffle();
    
    /**
     * Shuffles with a specific seed
     */
    void shuffle(unsigned int seed);
    
    /**
     * Deals a single card from the top of the deck
     */
    [[nodiscard]] Card dealCard();
    
    /**
     * Deals multiple cards
     */
    [[nodiscard]] std::vector<Card> dealCards(size_t count);
    
    /**
     * Returns the number of cards remaining in the deck
     */
    size_t cardsRemaining() const noexcept {
        return cards.size() - currentCard;
    }
    
    /**
     * Returns the total number of cards in the deck
     */
    size_t size() const noexcept {
        return cards.size();
    }
    
    /**
     * Burns a card (removes the top card without returning it)
     */
    void burn();
};

#endif // DECK_H

