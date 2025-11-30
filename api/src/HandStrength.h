#pragma once

#include <vector>
#include <string>

/**
 * Fast hand strength estimation for RL training.
 * 
 * This provides a quick heuristic estimate of hand strength (0.0-1.0)
 * without doing full hand evaluation. Used during episode collection
 * to guide action shaping rewards.
 * 
 * Performance: ~10x faster than Python equivalent due to:
 * - No Python interpreter overhead
 * - Static memory allocation
 * - Optimized data structures
 */
class HandStrengthEstimator {
public:
    /**
     * Estimate hand strength given hole cards and community cards.
     * 
     * @param holeCards Vector of 2 card strings (e.g., ["AH", "KS"])
     * @param communityCards Vector of 0-5 card strings
     * @return Strength estimate in range [0.0, 1.0]
     */
    static float estimate(
        const std::vector<std::string>& holeCards,
        const std::vector<std::string>& communityCards
    );
    
    /**
     * Estimate preflop hand strength.
     * Based on Sklansky-Chubukov rankings.
     */
    static float estimatePreflop(const std::vector<std::string>& holeCards);
    
private:
    // Card parsing helpers
    static int parseRank(char c);
    static int parseSuit(char c);
};

