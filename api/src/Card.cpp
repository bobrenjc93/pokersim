#include "Card.h"
#include <stdexcept>

Card::Card(const std::string& str) {
    if (str.length() != 2) {
        throw std::invalid_argument("Card string must be 2 characters");
    }
    
    // Parse rank
    switch (str[0]) {
        case '2': rank = Rank::TWO; break;
        case '3': rank = Rank::THREE; break;
        case '4': rank = Rank::FOUR; break;
        case '5': rank = Rank::FIVE; break;
        case '6': rank = Rank::SIX; break;
        case '7': rank = Rank::SEVEN; break;
        case '8': rank = Rank::EIGHT; break;
        case '9': rank = Rank::NINE; break;
        case 'T': rank = Rank::TEN; break;
        case 'J': rank = Rank::JACK; break;
        case 'Q': rank = Rank::QUEEN; break;
        case 'K': rank = Rank::KING; break;
        case 'A': rank = Rank::ACE; break;
        default: throw std::invalid_argument("Invalid rank: " + std::string(1, str[0]));
    }
    
    // Parse suit
    switch (str[1]) {
        case 'C': case 'c': suit = Suit::CLUBS; break;
        case 'D': case 'd': suit = Suit::DIAMONDS; break;
        case 'H': case 'h': suit = Suit::HEARTS; break;
        case 'S': case 's': suit = Suit::SPADES; break;
        default: throw std::invalid_argument("Invalid suit: " + std::string(1, str[1]));
    }
}

std::string Card::toString() const {
    // Use compile-time lookup tables for zero-cost abstraction
    static constexpr const char rankChars[] = "??23456789TJQKA";
    static constexpr const char suitChars[] = "CDHS";
    
    std::string result;
    result.reserve(2);
    result += rankChars[static_cast<int>(rank)];
    result += suitChars[static_cast<int>(suit)];
    
    return result;
}

std::string Card::getSuitSymbol() const {
    // Compile-time lookup table for zero overhead
    static constexpr const char* const suitSymbols[] = {"♣", "♦", "♥", "♠"};
    return suitSymbols[static_cast<int>(suit)];
}

