#ifndef HAND_H
#define HAND_H

#include "Card.h"
#include <vector>
#include <algorithm>
#include <map>
#include <string>

/**
 * Evaluates poker hands and compares them.
 * Supports standard poker hand rankings from High Card to Royal Flush.
 */
class Hand {
public:
    enum class Ranking {
        HIGH_CARD = 0,
        ONE_PAIR,
        TWO_PAIR,
        THREE_OF_A_KIND,
        STRAIGHT,
        FLUSH,
        FULL_HOUSE,
        FOUR_OF_A_KIND,
        STRAIGHT_FLUSH,
        ROYAL_FLUSH
    };
    
    struct EvaluatedHand {
        Ranking ranking;
        std::vector<int> tiebreakers; // High cards for comparing hands of same rank
        std::vector<Card> bestFive;   // The best 5 cards that make this hand
        
        EvaluatedHand() : ranking(Ranking::HIGH_CARD) {}
        
        /**
         * Compares two hands. Returns:
         * > 0 if this hand wins
         * < 0 if other hand wins
         * = 0 if hands are tied
         */
        int compare(const EvaluatedHand& other) const;
        
        bool operator>(const EvaluatedHand& other) const {
            return compare(other) > 0;
        }
        
        bool operator<(const EvaluatedHand& other) const {
            return compare(other) < 0;
        }
        
        bool operator==(const EvaluatedHand& other) const {
            return compare(other) == 0;
        }
        
        std::string getRankingName() const;
    };

private:
    /**
     * Counts cards by rank
     */
    static std::map<int, int> countRanks(const std::vector<Card>& cards);
    
    /**
     * Counts cards by suit
     */
    static std::map<int, int> countSuits(const std::vector<Card>& cards);
    
    /**
     * Checks for a straight in the given ranks
     */
    static bool isStraight(const std::vector<int>& ranks, int& highCard);
    
    /**
     * Gets the best 5 cards from 7 cards
     */
    static EvaluatedHand evaluateFive(const std::vector<Card>& fiveCards);

public:
    /**
     * Evaluates the best 5-card poker hand from 7 cards (5 community + 2 hole)
     */
    [[nodiscard]] static EvaluatedHand evaluate(const std::vector<Card>& cards);
    
    /**
     * Evaluates hand from hole cards and community cards
     */
    [[nodiscard]] static EvaluatedHand evaluate(const std::vector<Card>& holeCards, 
                                  const std::vector<Card>& communityCards);
};

#endif // HAND_H

