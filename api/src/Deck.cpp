#include "Deck.h"

Deck::Deck() : currentCard(0) {
    reset();
}

Deck::Deck(unsigned int seed) : currentCard(0), rng(seed) {
    reset();
}

void Deck::reset() {
    cards.clear();
    cards.reserve(52);  // Reserve space for all cards upfront
    currentCard = 0;
    
    // Create all 52 cards with compile-time constants
    static constexpr int suitCount = 4;
    static constexpr int minRank = 2;
    static constexpr int maxRank = 14;
    
    for (int s = 0; s < suitCount; s++) {
        const Card::Suit suit = static_cast<Card::Suit>(s);
        for (int r = minRank; r <= maxRank; r++) {
            const Card::Rank rank = static_cast<Card::Rank>(r);
            cards.emplace_back(rank, suit);
        }
    }
}

void Deck::shuffle() {
    currentCard = 0;
    std::shuffle(cards.begin(), cards.end(), rng);
}

void Deck::shuffle(unsigned int seed) {
    rng.seed(seed);
    shuffle();
}

Card Deck::dealCard() {
    if (currentCard >= cards.size()) {
        throw std::runtime_error("No cards left in deck");
    }
    return cards[currentCard++];
}

std::vector<Card> Deck::dealCards(size_t count) {
    std::vector<Card> dealt;
    dealt.reserve(count);
    
    for (size_t i = 0; i < count; i++) {
        dealt.push_back(dealCard());
    }
    
    return dealt;
}

void Deck::burn() {
    if (currentCard < cards.size()) {
        currentCard++;
    }
}

