#include "HandStrength.h"
#include <array>
#include <set>

int HandStrengthEstimator::parseRank(char c) {
    switch (c) {
        case '2': return 0;
        case '3': return 1;
        case '4': return 2;
        case '5': return 3;
        case '6': return 4;
        case '7': return 5;
        case '8': return 6;
        case '9': return 7;
        case 'T': return 8;
        case 'J': return 9;
        case 'Q': return 10;
        case 'K': return 11;
        case 'A': return 12;
        default: return 0;
    }
}

int HandStrengthEstimator::parseSuit(char c) {
    switch (c) {
        case 'C': case 'c': return 0;
        case 'D': case 'd': return 1;
        case 'H': case 'h': return 2;
        case 'S': case 's': return 3;
        default: return 0;
    }
}

float HandStrengthEstimator::estimatePreflop(const std::vector<std::string>& holeCards) {
    if (holeCards.size() < 2) return 0.3f;
    
    int r1 = parseRank(holeCards[0][0]);
    int r2 = parseRank(holeCards[1][0]);
    int s1 = parseSuit(holeCards[0][1]);
    int s2 = parseSuit(holeCards[1][1]);
    
    int highRank = std::max(r1, r2);
    int lowRank = std::min(r1, r2);
    bool isSuited = (s1 == s2);
    bool isPair = (r1 == r2);
    int gap = highRank - lowRank;
    
    // Base strength from high cards (normalized)
    float baseStrength = (highRank + lowRank) / 48.0f;
    
    // Pair bonus
    if (isPair) {
        // AA=1.0, KK=0.95, ..., 22=0.55
        return 0.55f + (highRank / 12.0f) * 0.45f;
    }
    
    // Premium unpaired hands
    // Ace-high
    if (highRank == 12) {  // Ace
        if (lowRank >= 10) {  // AT+
            return 0.70f + (lowRank - 10) * 0.05f + (isSuited ? 0.03f : 0.0f);
        } else if (lowRank >= 7) {  // A7-A9
            return 0.50f + (lowRank - 7) * 0.03f + (isSuited ? 0.05f : 0.0f);
        } else {  // A2-A6
            return 0.35f + (lowRank * 0.02f) + (isSuited ? 0.07f : 0.0f);
        }
    }
    
    // King-high
    if (highRank == 11) {
        if (lowRank >= 10) {
            return 0.55f + (lowRank - 10) * 0.05f + (isSuited ? 0.04f : 0.0f);
        } else if (lowRank >= 7) {
            return 0.40f + (lowRank - 7) * 0.03f + (isSuited ? 0.05f : 0.0f);
        } else {
            return 0.25f + (isSuited ? 0.06f : 0.0f);
        }
    }
    
    // Queen-high
    if (highRank == 10) {
        if (lowRank >= 9) {
            return 0.48f + (lowRank - 9) * 0.04f + (isSuited ? 0.04f : 0.0f);
        } else {
            return 0.25f + (isSuited ? 0.05f : 0.0f);
        }
    }
    
    // Connected cards (potential straights)
    float connectivityBonus = 0.0f;
    if (gap == 1) {
        connectivityBonus = 0.08f;
    } else if (gap == 2) {
        connectivityBonus = 0.04f;
    } else if (gap == 3) {
        connectivityBonus = 0.02f;
    }
    
    // Suited bonus
    float suitedBonus = isSuited ? 0.07f : 0.0f;
    
    // Final calculation for other hands
    float strength = baseStrength + connectivityBonus + suitedBonus;
    
    // Clamp to reasonable range
    return std::max(0.15f, std::min(0.55f, strength));
}

float HandStrengthEstimator::estimate(
    const std::vector<std::string>& holeCards,
    const std::vector<std::string>& communityCards
) {
    if (holeCards.size() < 2) return 0.3f;
    
    // Preflop
    if (communityCards.empty()) {
        return estimatePreflop(holeCards);
    }
    
    // Parse all cards
    std::vector<int> holeRanks, holeSuits;
    for (const auto& card : holeCards) {
        if (card.size() >= 2) {
            holeRanks.push_back(parseRank(card[0]));
            holeSuits.push_back(parseSuit(card[1]));
        }
    }
    
    std::vector<int> commRanks, commSuits;
    for (const auto& card : communityCards) {
        if (card.size() >= 2) {
            commRanks.push_back(parseRank(card[0]));
            commSuits.push_back(parseSuit(card[1]));
        }
    }
    
    // Combine all cards
    std::vector<int> allRanks = holeRanks;
    std::vector<int> allSuits = holeSuits;
    allRanks.insert(allRanks.end(), commRanks.begin(), commRanks.end());
    allSuits.insert(allSuits.end(), commSuits.begin(), commSuits.end());
    
    // Count ranks
    std::array<int, 13> rankCounts = {0};
    for (int r : allRanks) {
        rankCounts[r]++;
    }
    
    // Count suits
    std::array<int, 4> suitCounts = {0};
    for (int s : allSuits) {
        suitCounts[s]++;
    }
    
    // Get unique sorted ranks for straight detection
    std::set<int> uniqueRankSet(allRanks.begin(), allRanks.end());
    std::vector<int> uniqueRanks(uniqueRankSet.begin(), uniqueRankSet.end());
    std::sort(uniqueRanks.begin(), uniqueRanks.end());
    
    // Determine made hand strength
    int madeHandRank = 0;  // 0=high card, 1=pair, ... 8=straight flush
    
    // Check for flush
    bool hasFlush = false;
    int flushSuit = -1;
    for (int s = 0; s < 4; s++) {
        if (suitCounts[s] >= 5) {
            hasFlush = true;
            flushSuit = s;
            break;
        }
    }
    
    // Check if hole cards contribute to flush
    bool holeContributesToFlush = false;
    if (hasFlush && flushSuit >= 0) {
        for (int s : holeSuits) {
            if (s == flushSuit) {
                holeContributesToFlush = true;
                break;
            }
        }
    }
    
    // Check for straight
    bool hasStraight = false;
    int straightHigh = 0;
    
    // Check for 5 consecutive ranks
    if (uniqueRanks.size() >= 5) {
        for (size_t i = 0; i <= uniqueRanks.size() - 5; i++) {
            if (uniqueRanks[i + 4] - uniqueRanks[i] == 4) {
                hasStraight = true;
                straightHigh = uniqueRanks[i + 4];
                break;
            }
        }
    }
    
    // Check for wheel (A-2-3-4-5)
    if (!hasStraight) {
        bool hasWheel = (uniqueRankSet.count(12) && uniqueRankSet.count(0) &&
                        uniqueRankSet.count(1) && uniqueRankSet.count(2) &&
                        uniqueRankSet.count(3));
        if (hasWheel) {
            hasStraight = true;
            straightHigh = 3;  // 5-high
        }
    }
    
    // Check if hole cards contribute to straight
    bool holeContributesToStraight = false;
    if (hasStraight) {
        for (int r : holeRanks) {
            if (r >= std::max(0, straightHigh - 4) && r <= straightHigh) {
                holeContributesToStraight = true;
                break;
            }
            // Wheel case
            if (straightHigh <= 4 && r == 12) {
                holeContributesToStraight = true;
                break;
            }
        }
    }
    
    // Check for pairs, trips, quads
    int maxCount = 0;
    for (int i = 0; i < 13; i++) {
        maxCount = std::max(maxCount, rankCounts[i]);
    }
    
    // Check if hole cards contribute to pairs
    bool holeContributesPairs = false;
    for (int r : holeRanks) {
        if (rankCounts[r] >= 2) {
            holeContributesPairs = true;
            break;
        }
    }
    
    // Two pair check
    int pairCount = 0;
    for (int i = 0; i < 13; i++) {
        if (rankCounts[i] >= 2) pairCount++;
    }
    bool hasTwoPair = (pairCount >= 2);
    
    // Full house check
    bool hasTrips = (maxCount >= 3);
    bool hasFullHouse = (hasTrips && pairCount >= 2);
    
    // Determine made hand rank
    if (maxCount == 4 && holeContributesPairs) {
        madeHandRank = 7;  // Quads
    } else if (hasFullHouse && holeContributesPairs) {
        madeHandRank = 6;  // Full house
    } else if (hasFlush && holeContributesToFlush) {
        madeHandRank = 5;  // Flush
    } else if (hasStraight && holeContributesToStraight) {
        madeHandRank = 4;  // Straight
    } else if (maxCount == 3 && holeContributesPairs) {
        madeHandRank = 3;  // Trips
    } else if (hasTwoPair && holeContributesPairs) {
        madeHandRank = 2;  // Two pair
    } else if (holeContributesPairs) {
        madeHandRank = 1;  // Pair
    } else {
        madeHandRank = 0;  // High card
    }
    
    // Convert to strength score
    static const float madeHandStrengths[] = {
        0.15f,  // High card
        0.35f,  // Pair
        0.50f,  // Two pair
        0.60f,  // Trips
        0.70f,  // Straight
        0.75f,  // Flush
        0.85f,  // Full house
        0.95f,  // Quads
        1.00f   // Straight flush
    };
    
    float baseStrength = madeHandStrengths[madeHandRank];
    
    // Kicker bonus for high card, pair, two pair
    float kickerBonus = 0.0f;
    if (madeHandRank <= 2) {
        int highHole = std::max(holeRanks[0], holeRanks[1]);
        kickerBonus = (highHole / 12.0f) * 0.10f;
    }
    
    // Draw potential (when not already made strong hand)
    float drawBonus = 0.0f;
    if (madeHandRank < 4) {
        // Flush draw
        for (int s = 0; s < 4; s++) {
            int myCount = suitCounts[s];
            int holeInSuit = 0;
            for (int hs : holeSuits) {
                if (hs == s) holeInSuit++;
            }
            if (myCount == 4 && holeInSuit >= 1) {
                drawBonus += 0.08f;
            }
        }
        
        // Straight draw (simplified)
        if (uniqueRanks.size() >= 4) {
            for (size_t i = 0; i <= uniqueRanks.size() - 4; i++) {
                if (uniqueRanks[i + 3] - uniqueRanks[i] <= 4) {
                    // Check if hole cards in range
                    for (int r : holeRanks) {
                        if (r >= uniqueRanks[i] && r <= uniqueRanks[i + 3]) {
                            drawBonus += 0.05f;
                            break;
                        }
                    }
                    break;
                }
            }
        }
    }
    
    float finalStrength = baseStrength + kickerBonus + drawBonus;
    return std::min(1.0f, finalStrength);
}

