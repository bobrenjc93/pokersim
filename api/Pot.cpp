#include "Pot.h"

Pot::Pot() : currentBet(0), minRaise(0) {}

int Pot::getTotalPot() const noexcept {
    int total = 0;
    for (const auto& pot : pots) {
        total += pot.amount;
    }
    return total;
}

int Pot::getMainPot() const noexcept {
    return pots.empty() ? 0 : pots[0].amount;
}

size_t Pot::getSidePotCount() const noexcept {
    return pots.size() > 1 ? pots.size() - 1 : 0;
}

void Pot::reset() {
    pots.clear();
    currentBet = 0;
    minRaise = 0;
}

void Pot::collectBets(std::vector<Player*>& players) {
    // Build a list of player bets (only those who bet)
    std::vector<std::pair<std::string, int>> playerBets;
    playerBets.reserve(players.size());  // Worst case: all players bet
    
    for (auto* player : players) {
        if (player->getBet() > 0) {
            playerBets.emplace_back(player->getId(), player->getBet());
        }
    }
    
    if (playerBets.empty()) {
        return;
    }
    
    // Sort by bet amount
    std::sort(playerBets.begin(), playerBets.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    int previousBetLevel = 0;
    
    // Create pots for each bet level
    for (size_t i = 0; i < playerBets.size(); ) {
        int currentBetLevel = playerBets[i].second;
        int potAmount = 0;
        
        SidePot pot(0);
        pot.eligiblePlayerIds.reserve(playerBets.size() - i);  // Reserve space for eligible players
        
        // Collect from all players still in
        for (size_t j = i; j < playerBets.size(); j++) {
            int contribution = std::min(currentBetLevel - previousBetLevel, 
                                       playerBets[j].second - previousBetLevel);
            potAmount += contribution;
            pot.eligiblePlayerIds.push_back(playerBets[j].first);
        }
        
        pot.amount = potAmount;
        pots.push_back(std::move(pot));  // Move instead of copy
        
        // Move to next bet level
        previousBetLevel = currentBetLevel;
        while (i < playerBets.size() && playerBets[i].second == currentBetLevel) {
            i++;
        }
    }
    
    // Reset player bets
    for (auto* player : players) {
        player->resetBet();
    }
    
    // Reset current bet for next round
    currentBet = 0;
}

std::unordered_map<std::string, int> Pot::distributePots(
    const std::vector<Player*>& players,
    const std::vector<Card>& communityCards) {
    
    std::unordered_map<std::string, int> winnings;
    
    for (const auto& pot : pots) {
        // Find eligible players who are still in the hand
        // Use unordered_set for O(1) lookup instead of O(n) std::find
        std::unordered_set<std::string> eligibleSet(
            pot.eligiblePlayerIds.begin(), 
            pot.eligiblePlayerIds.end()
        );
        
        std::vector<Player*> eligiblePlayers;
        eligiblePlayers.reserve(pot.eligiblePlayerIds.size());  // Avoid reallocation
        for (auto* player : players) {
            if (player->isInHand() && eligibleSet.count(player->getId())) {
                eligiblePlayers.push_back(player);
            }
        }
        
        if (eligiblePlayers.empty()) {
            continue;
        }
        
        // Evaluate all hands
        std::vector<std::pair<Player*, Hand::EvaluatedHand>> evaluatedHands;
        for (auto* player : eligiblePlayers) {
            evaluatedHands.push_back({
                player,
                player->evaluateHand(communityCards)
            });
        }
        
        // Find the best hand(s)
        Hand::EvaluatedHand bestHand = evaluatedHands[0].second;
        for (const auto& [player, hand] : evaluatedHands) {
            if (hand > bestHand) {
                bestHand = hand;
            }
        }
        
        // Find all players with the best hand (for splits)
        std::vector<Player*> winners;
        for (const auto& [player, hand] : evaluatedHands) {
            if (hand == bestHand) {
                winners.push_back(player);
            }
        }
        
        // Split pot among winners
        int amountPerWinner = pot.amount / winners.size();
        int remainder = pot.amount % winners.size();
        
        for (size_t i = 0; i < winners.size(); i++) {
            int winAmount = amountPerWinner + (i == 0 ? remainder : 0);
            winners[i]->winChips(winAmount);
            winnings[winners[i]->getId()] += winAmount;
        }
    }
    
    return winnings;
}

void Pot::startNewRound() {
    currentBet = 0;
    minRaise = 0;
}

void Pot::updateBet(int newBet, int previousBet) {
    int raiseAmount = newBet - previousBet;
    if (raiseAmount > minRaise) {
        minRaise = raiseAmount;
    }
    currentBet = newBet;
}

