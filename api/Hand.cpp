#include "Hand.h"
#include <functional>

int Hand::EvaluatedHand::compare(const EvaluatedHand& other) const {
    if (ranking != other.ranking) {
        return static_cast<int>(ranking) - static_cast<int>(other.ranking);
    }
    
    // Same ranking, compare tiebreakers
    for (size_t i = 0; i < std::min(tiebreakers.size(), other.tiebreakers.size()); i++) {
        if (tiebreakers[i] != other.tiebreakers[i]) {
            return tiebreakers[i] - other.tiebreakers[i];
        }
    }
    
    return 0; // Exact tie
}

std::string Hand::EvaluatedHand::getRankingName() const {
    // Compile-time lookup table with bounds safety
    static constexpr const char* const rankingNames[] = {
        "High Card",
        "One Pair",
        "Two Pair",
        "Three of a Kind",
        "Straight",
        "Flush",
        "Full House",
        "Four of a Kind",
        "Straight Flush",
        "Royal Flush"
    };
    static constexpr size_t nameCount = sizeof(rankingNames) / sizeof(rankingNames[0]);
    
    const auto idx = static_cast<size_t>(ranking);
    if (idx < nameCount) {
        return rankingNames[idx];
    }
    return "Unknown";
}

std::map<int, int> Hand::countRanks(const std::vector<Card>& cards) {
    std::map<int, int> counts;
    for (const auto& card : cards) {
        ++counts[card.getRankValue()]; // Prefix increment is slightly faster
    }
    return counts;
}

std::map<int, int> Hand::countSuits(const std::vector<Card>& cards) {
    std::map<int, int> counts;
    for (const auto& card : cards) {
        ++counts[card.getSuitValue()]; // Prefix increment is slightly faster
    }
    return counts;
}

bool Hand::isStraight(const std::vector<int>& ranks, int& highCard) {
    std::vector<int> uniqueRanks = ranks;
    std::sort(uniqueRanks.begin(), uniqueRanks.end());
    uniqueRanks.erase(std::unique(uniqueRanks.begin(), uniqueRanks.end()), uniqueRanks.end());
    
    if (uniqueRanks.size() < 5) return false;
    
    // Check for regular straight
    for (size_t i = 0; i <= uniqueRanks.size() - 5; i++) {
        if (uniqueRanks[i+4] - uniqueRanks[i] == 4) {
            highCard = uniqueRanks[i+4];
            return true;
        }
    }
    
    // Check for A-2-3-4-5 (wheel)
    // uniqueRanks is sorted, so check if we have 2,3,4,5 and Ace (14)
    if (uniqueRanks.size() >= 5 && uniqueRanks.back() == 14) {
        // Check if we have consecutive 2,3,4,5 at the start
        if (uniqueRanks[0] == 2 && uniqueRanks[1] == 3 && 
            uniqueRanks[2] == 4 && uniqueRanks[3] == 5) {
            highCard = 5; // In A-2-3-4-5, the 5 is the high card
            return true;
        }
    }
    
    return false;
}

Hand::EvaluatedHand Hand::evaluateFive(const std::vector<Card>& fiveCards) {
    EvaluatedHand result;
    result.bestFive = fiveCards;
    
    // Collect ranks once and sort
    std::vector<int> ranks;
    ranks.reserve(5);
    for (const auto& card : fiveCards) {
        ranks.push_back(card.getRankValue());
    }
    std::sort(ranks.begin(), ranks.end(), std::greater<int>());
    
    auto rankCounts = countRanks(fiveCards);
    auto suitCounts = countSuits(fiveCards);
    
    // Check for flush
    bool isFlush = false;
    for (const auto& [suit, count] : suitCounts) {
        if (count == 5) {
            isFlush = true;
            break;
        }
    }
    
    // Check for straight
    int straightHigh = 0;
    bool isStraightHand = isStraight(ranks, straightHigh);
    
    // Royal Flush: A-K-Q-J-T of same suit
    if (isFlush && isStraightHand && straightHigh == 14) {
        result.ranking = Ranking::ROYAL_FLUSH;
        result.tiebreakers = {14};
        return result;
    }
    
    // Straight Flush
    if (isFlush && isStraightHand) {
        result.ranking = Ranking::STRAIGHT_FLUSH;
        result.tiebreakers = {straightHigh};
        return result;
    }
    
    // Find pairs, trips, quads
    std::vector<std::pair<int, int>> ranksWithCounts;
    for (const auto& [rank, count] : rankCounts) {
        ranksWithCounts.push_back({rank, count});
    }
    std::sort(ranksWithCounts.begin(), ranksWithCounts.end(), 
              [](const auto& a, const auto& b) {
                  if (a.second != b.second) return a.second > b.second;
                  return a.first > b.first;
              });
    
    // Four of a Kind
    if (ranksWithCounts[0].second == 4) {
        result.ranking = Ranking::FOUR_OF_A_KIND;
        result.tiebreakers = {ranksWithCounts[0].first, ranksWithCounts[1].first};
        return result;
    }
    
    // Full House
    if (ranksWithCounts[0].second == 3 && ranksWithCounts[1].second == 2) {
        result.ranking = Ranking::FULL_HOUSE;
        result.tiebreakers = {ranksWithCounts[0].first, ranksWithCounts[1].first};
        return result;
    }
    
    // Flush
    if (isFlush) {
        result.ranking = Ranking::FLUSH;
        result.tiebreakers = ranks;
        return result;
    }
    
    // Straight
    if (isStraightHand) {
        result.ranking = Ranking::STRAIGHT;
        result.tiebreakers = {straightHigh};
        return result;
    }
    
    // Three of a Kind
    if (ranksWithCounts[0].second == 3) {
        result.ranking = Ranking::THREE_OF_A_KIND;
        result.tiebreakers = {ranksWithCounts[0].first, ranksWithCounts[1].first, ranksWithCounts[2].first};
        return result;
    }
    
    // Two Pair
    if (ranksWithCounts[0].second == 2 && ranksWithCounts[1].second == 2) {
        result.ranking = Ranking::TWO_PAIR;
        result.tiebreakers = {ranksWithCounts[0].first, ranksWithCounts[1].first, ranksWithCounts[2].first};
        return result;
    }
    
    // One Pair
    if (ranksWithCounts[0].second == 2) {
        result.ranking = Ranking::ONE_PAIR;
        result.tiebreakers = {ranksWithCounts[0].first, ranksWithCounts[1].first, 
                            ranksWithCounts[2].first, ranksWithCounts[3].first};
        return result;
    }
    
    // High Card
    result.ranking = Ranking::HIGH_CARD;
    result.tiebreakers = ranks;
    return result;
}

Hand::EvaluatedHand Hand::evaluate(const std::vector<Card>& cards) {
    if (cards.size() < 5) {
        throw std::invalid_argument("Need at least 5 cards to evaluate");
    }
    
    if (cards.size() == 5) {
        return evaluateFive(cards);
    }
    
    // For 7 cards, use recursive combination generation (C(7,5) = 21 combinations)
    // This is more efficient than checking all 128 bit patterns
    EvaluatedHand bestHand;
    const size_t n = cards.size();
    std::vector<Card> fiveCards;
    fiveCards.reserve(5);
    
    // Recursive lambda to generate combinations
    std::function<void(size_t, size_t)> generateCombinations = [&](size_t start, size_t chosen) {
        if (chosen == 5) {
            EvaluatedHand hand = evaluateFive(fiveCards);
            if (hand > bestHand) {
                bestHand = hand;
            }
            return;
        }
        
        // Need to choose (5 - chosen) more cards from remaining (n - start) cards
        // Stop when there aren't enough cards left
        for (size_t i = start; i <= n - (5 - chosen); ++i) {
            fiveCards.push_back(cards[i]);
            generateCombinations(i + 1, chosen + 1);
            fiveCards.pop_back();
        }
    };
    
    generateCombinations(0, 0);
    return bestHand;
}

Hand::EvaluatedHand Hand::evaluate(const std::vector<Card>& holeCards, 
                              const std::vector<Card>& communityCards) {
    std::vector<Card> allCards = holeCards;
    allCards.insert(allCards.end(), communityCards.begin(), communityCards.end());
    return evaluate(allCards);
}

