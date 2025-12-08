#include "ActionUtils.h"
#include <unordered_map>
#include <algorithm>

namespace ActionUtils {

// Static action name array
static const std::vector<std::string> ACTION_NAMES = {
    "fold", "check", "call",
    "raise_10%", "raise_25%", "raise_33%", "raise_50%", "raise_75%",
    "raise_100%", "raise_150%", "raise_200%", "raise_300%",
    "all_in"
};

// Static action name to index map
static const std::unordered_map<std::string, int> ACTION_NAME_TO_INDEX = {
    {"fold", FOLD}, {"check", CHECK}, {"call", CALL},
    {"raise_10%", RAISE_10}, {"raise_25%", RAISE_25}, {"raise_33%", RAISE_33},
    {"raise_50%", RAISE_50}, {"raise_75%", RAISE_75}, {"raise_100%", RAISE_100},
    {"raise_150%", RAISE_150}, {"raise_200%", RAISE_200}, {"raise_300%", RAISE_300},
    {"all_in", ALL_IN}
};

// Raise size fractions (as pot fraction)
static const std::unordered_map<int, float> RAISE_SIZE_MAP = {
    {RAISE_10, 0.10f}, {RAISE_25, 0.25f}, {RAISE_33, 0.33f},
    {RAISE_50, 0.50f}, {RAISE_75, 0.75f}, {RAISE_100, 1.00f},
    {RAISE_150, 1.50f}, {RAISE_200, 2.00f}, {RAISE_300, 3.00f}
};

// Raise size from label string
static const std::unordered_map<std::string, float> RAISE_SIZE_FROM_LABEL = {
    {"raise_10%", 0.10f}, {"raise_25%", 0.25f}, {"raise_33%", 0.33f},
    {"raise_50%", 0.50f}, {"raise_75%", 0.75f}, {"raise_100%", 1.00f},
    {"raise_150%", 1.50f}, {"raise_200%", 2.00f}, {"raise_300%", 3.00f}
};


std::string getActionName(int actionIndex) {
    if (actionIndex >= 0 && actionIndex < NUM_ACTIONS) {
        return ACTION_NAMES[actionIndex];
    }
    return "unknown";
}

int getActionIndex(const std::string& actionName) {
    auto it = ACTION_NAME_TO_INDEX.find(actionName);
    if (it != ACTION_NAME_TO_INDEX.end()) {
        return it->second;
    }
    return -1;
}

float getRaiseSize(int actionIndex) {
    auto it = RAISE_SIZE_MAP.find(actionIndex);
    if (it != RAISE_SIZE_MAP.end()) {
        return it->second;
    }
    return 0.0f;
}


ConvertedAction convertActionLabel(const std::string& actionLabel, 
                                   const nlohmann::json& state) {
    ConvertedAction result;
    result.actionType = "check";
    result.amount = 0;
    
    int playerChips = state.value("player_chips", 0);
    
    // Simple actions
    if (actionLabel == "fold" || actionLabel == "check" || 
        actionLabel == "call" || actionLabel == "all_in" ||
        actionLabel == "bet" || actionLabel == "raise") {
        result.actionType = actionLabel;
        result.amount = 0;
        return result;
    }
    
    // Unified raise actions - convert to bet or raise based on game context
    if (actionLabel.substr(0, 6) == "raise_") {
        int pot = state.value("pot", 0);
        int playerBet = state.value("player_bet", 0);
        int currentBet = state.value("current_bet", 0);
        int toCall = std::max(0, currentBet - playerBet);
        int minBet = state.value("min_bet", state.value("big_blind", 20));
        int minRaiseTotal = state.value("min_raise_total", state.value("big_blind", 20));
        
        // Get size fraction
        float sizeFraction = 0.5f;  // Default
        auto it = RAISE_SIZE_FROM_LABEL.find(actionLabel);
        if (it != RAISE_SIZE_FROM_LABEL.end()) {
            sizeFraction = it->second;
        }
        
        int sizingAmount = (pot > 0) ? static_cast<int>(pot * sizeFraction) : minBet;
        
        // Determine if this should be a bet or raise based on whether there's a bet to face
        if (toCall == 0) {
            // No bet to face - this is a BET
            int amount = std::max(minBet, sizingAmount);
            
            // If bet would use all chips, go all-in
            if (amount >= playerChips) {
                result.actionType = "all_in";
                result.amount = 0;
                return result;
            }
            
            result.actionType = "bet";
            result.amount = amount;
            return result;
        } else {
            // There's a bet to face - this is a RAISE
            // Amount is call + raise increment
            int amount = toCall + sizingAmount;
            
            // Ensure we meet minimum raise requirement
            if (amount < minRaiseTotal) {
                amount = minRaiseTotal;
            }
            
            // If raise would use all chips, go all-in
            if (amount >= playerChips) {
                result.actionType = "all_in";
                result.amount = 0;
                return result;
            }
            
            // If we can't afford the minimum raise, go all-in instead
            if (playerChips < minRaiseTotal) {
                result.actionType = "all_in";
                result.amount = 0;
                return result;
            }
            
            result.actionType = "raise";
            result.amount = amount;
            return result;
        }
    }
    
    // Legacy support for bet_X% actions (convert to raise_X% logic)
    if (actionLabel.substr(0, 4) == "bet_") {
        std::string raiseLabel = "raise_" + actionLabel.substr(4);
        return convertActionLabel(raiseLabel, state);
    }
    
    // Fallback
    return result;
}


std::array<bool, NUM_ACTIONS> createLegalActionsMask(
    const std::vector<std::string>& legalActions) {
    
    std::array<bool, NUM_ACTIONS> mask = {false};
    
    for (const auto& action : legalActions) {
        int idx = getActionIndex(action);
        if (idx >= 0 && idx < NUM_ACTIONS) {
            // Simple action (fold, check, call, all_in)
            mask[idx] = true;
        } else if (action == "bet" || action == "raise") {
            // Enable all raise_X% sizing actions
            // They become bet or raise based on game context in convertActionLabel
            for (int i = RAISE_10; i <= RAISE_300; i++) {
                mask[i] = true;
            }
        }
    }
    
    return mask;
}


std::vector<float> createLegalActionsMaskFloat(
    const std::vector<std::string>& legalActions) {
    
    auto boolMask = createLegalActionsMask(legalActions);
    std::vector<float> floatMask(NUM_ACTIONS);
    
    for (int i = 0; i < NUM_ACTIONS; i++) {
        floatMask[i] = boolMask[i] ? 1.0f : 0.0f;
    }
    
    return floatMask;
}


nlohmann::json extractState(const nlohmann::json& gameState, 
                            const std::string& playerId) {
    nlohmann::json result;
    
    // Find the player
    if (!gameState.contains("players") || !gameState["players"].is_array()) {
        return result;  // Empty
    }
    
    const nlohmann::json* playerPtr = nullptr;
    for (const auto& p : gameState["players"]) {
        if (p.value("id", "") == playerId) {
            playerPtr = &p;
            break;
        }
    }
    
    if (!playerPtr) {
        return result;  // Player not found
    }
    
    const auto& player = *playerPtr;
    
    // Get config
    nlohmann::json config;
    if (gameState.contains("config") && gameState["config"].is_object()) {
        config = gameState["config"];
    }
    
    // Get action constraints
    nlohmann::json actionConstraints;
    if (gameState.contains("actionConstraints") && gameState["actionConstraints"].is_object()) {
        actionConstraints = gameState["actionConstraints"];
    }
    
    // Compute max stack across all players for relative normalization
    int maxStack = 0;
    for (const auto& p : gameState["players"]) {
        int chips = p.value("chips", 0);
        maxStack = std::max(maxStack, chips);
    }
    
    // Use at least starting_chips to avoid division by zero
    int startingChips = config.value("startingChips", 1000);
    maxStack = std::max(maxStack, startingChips);
    
    // Build result
    result["player_id"] = playerId;
    
    // Hole cards
    if (player.contains("holeCards") && player["holeCards"].is_array()) {
        result["hole_cards"] = player["holeCards"];
    } else {
        result["hole_cards"] = nlohmann::json::array();
    }
    
    // Community cards
    if (gameState.contains("communityCards") && gameState["communityCards"].is_array()) {
        result["community_cards"] = gameState["communityCards"];
    } else {
        result["community_cards"] = nlohmann::json::array();
    }
    
    result["pot"] = gameState.value("pot", 0);
    result["current_bet"] = gameState.value("currentBet", 0);
    result["player_chips"] = player.value("chips", 0);
    result["player_bet"] = player.value("bet", 0);
    result["player_total_bet"] = player.value("totalBet", 0);
    result["stage"] = gameState.value("stage", "Preflop");
    result["num_players"] = static_cast<int>(gameState["players"].size());
    
    // Count active players
    int numActive = 0;
    for (const auto& p : gameState["players"]) {
        if (p.value("isInHand", false)) {
            numActive++;
        }
    }
    result["num_active"] = numActive;
    
    result["position"] = player.value("position", 0);
    result["is_dealer"] = player.value("isDealer", false);
    result["is_small_blind"] = player.value("isSmallBlind", false);
    result["is_big_blind"] = player.value("isBigBlind", false);
    result["big_blind"] = config.value("bigBlind", 20);
    result["small_blind"] = config.value("smallBlind", 10);
    result["starting_chips"] = startingChips;
    result["max_stack"] = maxStack;
    result["to_call"] = actionConstraints.value("toCall", 0);
    result["min_bet"] = actionConstraints.value("minBet", 20);
    result["min_raise_total"] = actionConstraints.value("minRaiseTotal", 20);
    
    return result;
}

} // namespace ActionUtils





