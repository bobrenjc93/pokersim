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
    
    // Only create pots if there are actual bets to collect
    if (!playerBets.empty()) {
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
    }
    
    // ALWAYS reset player bets and current bet, even if no bets were collected.
    // This ensures a clean state for the next betting round.
    for (auto* player : players) {
        player->resetBet();
    }
    
    // Reset current bet for next round
    currentBet = 0;
}

std::vector<Pot::ShowdownResult> Pot::distributePots(
    const std::vector<Player*>& players,
    const std::vector<Card>& communityCards) {
    
    // Pre-evaluate all hands once to avoid duplicate evaluations
    std::unordered_map<std::string, Hand::EvaluatedHand> evaluatedHands;
    evaluatedHands.reserve(players.size());
    
    std::unordered_map<std::string, ShowdownResult> resultsMap;
    
    // Initialize all players in the hand with their evaluated hands
    for (auto* player : players) {
        if (player->isInHand()) {
            ShowdownResult result;
            result.playerId = player->getId();
            
            // Evaluate the player's hand once and cache it
            Hand::EvaluatedHand evalHand = player->evaluateHand(communityCards);
            evaluatedHands[player->getId()] = evalHand;
            result.handRanking = evalHand.getRankingName();
            
            // Convert best five cards to strings
            for (const auto& card : evalHand.bestFive) {
                result.bestFive.push_back(card.toString());
            }
            
            result.amountWon = 0;
            resultsMap[player->getId()] = result;
        }
    }
    
    for (const auto& pot : pots) {
        // Find eligible players who are still in the hand
        // Use unordered_set for O(1) lookup instead of O(n) std::find
        std::unordered_set<std::string> eligibleSet(
            pot.eligiblePlayerIds.begin(), 
            pot.eligiblePlayerIds.end()
        );
        
        std::vector<std::pair<Player*, Hand::EvaluatedHand>> eligiblePlayersWithHands;
        eligiblePlayersWithHands.reserve(pot.eligiblePlayerIds.size());  // Avoid reallocation
        for (auto* player : players) {
            if (player->isInHand() && eligibleSet.count(player->getId())) {
                // Use pre-evaluated hand from cache
                eligiblePlayersWithHands.push_back({player, evaluatedHands[player->getId()]});
            }
        }
        
        if (eligiblePlayersWithHands.empty()) {
            continue;
        }
        
        // Find the best hand(s)
        Hand::EvaluatedHand bestHand = eligiblePlayersWithHands[0].second;
        for (const auto& [player, hand] : eligiblePlayersWithHands) {
            if (hand > bestHand) {
                bestHand = hand;
            }
        }
        
        // Find all players with the best hand (for splits)
        std::vector<Player*> winners;
        for (const auto& [player, hand] : eligiblePlayersWithHands) {
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
            resultsMap[winners[i]->getId()].amountWon += winAmount;
        }
    }
    
    // Convert map to vector
    std::vector<ShowdownResult> results;
    results.reserve(resultsMap.size());
    for (const auto& [playerId, result] : resultsMap) {
        results.push_back(result);
    }
    
    return results;
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


