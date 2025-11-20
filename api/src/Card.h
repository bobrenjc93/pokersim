#ifndef CARD_H
#define CARD_H

#include <string>
#include <stdexcept>

/**
 * Represents a single playing card with rank and suit.
 */
class Card {
public:
    enum class Rank {
        TWO = 2, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN,
        JACK, QUEEN, KING, ACE
    };
    
    enum class Suit {
        CLUBS = 0, DIAMONDS, HEARTS, SPADES
    };

private:
    Rank rank;
    Suit suit;

public:
    Card() : rank(Rank::TWO), suit(Suit::CLUBS) {}
    
    Card(Rank r, Suit s) : rank(r), suit(s) {}
    
    /**
     * Constructs a card from a string representation like "AS" (Ace of Spades)
     * Format: [Rank][Suit] where Rank is 2-9,T,J,Q,K,A and Suit is C,D,H,S
     */
    Card(const std::string& str);
    
    Rank getRank() const noexcept { return rank; }
    Suit getSuit() const noexcept { return suit; }
    
    int getRankValue() const noexcept { return static_cast<int>(rank); }
    int getSuitValue() const noexcept { return static_cast<int>(suit); }
    
    /**
     * Returns string representation like "AS" or "7H"
     */
    std::string toString() const;
    
    /**
     * Returns Unicode suit symbol for display
     */
    std::string getSuitSymbol() const;
    
    bool operator==(const Card& other) const noexcept {
        return rank == other.rank && suit == other.suit;
    }
    
    bool operator!=(const Card& other) const noexcept {
        return !(*this == other);
    }
    
    bool operator<(const Card& other) const noexcept {
        if (rank != other.rank) {
            return rank < other.rank;
        }
        return suit < other.suit;
    }
};

#endif // CARD_H

