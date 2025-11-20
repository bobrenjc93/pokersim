#ifndef POT_H
#define POT_H

#include "Player.h"
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

/**
 * Manages the pot and side pots in a poker game.
 * Handles betting rounds and pot distribution.
 */
class Pot {
public:
    struct SidePot {
        int amount;
        std::vector<std::string> eligiblePlayerIds;
        
        SidePot(int amt) : amount(amt) {}
    };

private:
    std::vector<SidePot> pots;
    int currentBet;
    int minRaise;

public:
    Pot();
    
    /**
     * Gets the total pot amount (main pot + side pots)
     */
    int getTotalPot() const noexcept;
    
    /**
     * Gets the main pot amount
     */
    int getMainPot() const noexcept;
    
    /**
     * Gets the number of side pots
     */
    size_t getSidePotCount() const noexcept;
    
    /**
     * Gets all pots
     */
    const std::vector<SidePot>& getPots() const noexcept {
        return pots;
    }
    
    /**
     * Gets current bet to call
     */
    int getCurrentBet() const noexcept {
        return currentBet;
    }
    
    /**
     * Gets minimum raise amount
     */
    int getMinRaise() const noexcept {
        return minRaise;
    }
    
    /**
     * Sets current bet
     */
    void setCurrentBet(int bet) {
        currentBet = bet;
    }
    
    /**
     * Sets minimum raise
     */
    void setMinRaise(int raise) {
        minRaise = raise;
    }
    
    /**
     * Resets pot for new hand
     */
    void reset();
    
    /**
     * Collects bets from all players and creates side pots if needed
     */
    void collectBets(std::vector<Player*>& players);
    
    /**
     * Distributes pots to winners
     * Returns an unordered_map of player ID to amount won (O(1) lookups)
     */
    std::unordered_map<std::string, int> distributePots(
        const std::vector<Player*>& players,
        const std::vector<Card>& communityCards);
    
    /**
     * Starts a new betting round
     */
    void startNewRound();
    
    /**
     * Updates bet when a player raises
     */
    void updateBet(int newBet, int previousBet);
};

#endif // POT_H

