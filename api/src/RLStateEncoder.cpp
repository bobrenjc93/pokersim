#include "RLStateEncoder.h"
#include "HandStrength.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// ============================================================================
// ActionHistory Implementation
// ============================================================================

ActionType ActionHistory::parseActionType(const std::string& actionStr) {
    if (actionStr == "fold") return ActionType::FOLD;
    if (actionStr == "check") return ActionType::CHECK;
    if (actionStr == "call") return ActionType::CALL;
    if (actionStr == "bet") return ActionType::BET;
    if (actionStr == "raise") return ActionType::RAISE;
    if (actionStr == "all_in") return ActionType::ALL_IN;
    return ActionType::UNKNOWN;
}

int ActionHistory::parseStage(const std::string& stageStr) {
    if (stageStr == "Preflop" || stageStr == "preflop") return 0;
    if (stageStr == "Flop" || stageStr == "flop") return 1;
    if (stageStr == "Turn" || stageStr == "turn") return 2;
    if (stageStr == "River" || stageStr == "river") return 3;
    return 0;
}

void ActionHistory::addAction(const std::string& playerId, const std::string& actionTypeStr,
                              int amount, int pot, const std::string& stage) {
    ActionRecord record;
    record.playerId = playerId;
    record.actionType = parseActionType(actionTypeStr);
    record.amount = amount;
    record.pot = pot;
    record.stage = parseStage(stage);
    record.potFraction = (amount > 0 && pot > 0) ? 
                         static_cast<float>(amount) / static_cast<float>(pot) : 0.0f;
    
    actions_.push_back(record);
    
    // Keep only recent history
    while (actions_.size() > MAX_HISTORY) {
        actions_.pop_front();
    }
}

std::array<float, ActionHistory::OPPONENT_FEATURES_DIM> ActionHistory::getOpponentFeatures(
    const std::string& currentPlayerId) const {
    
    std::array<float, OPPONENT_FEATURES_DIM> features = {0};
    
    // Count opponent actions
    std::array<int, 6> actionCounts = {0};  // fold, check, call, bet, raise, all_in
    std::vector<float> betSizes;
    int totalActions = 0;
    
    // Collect last 3 opponent actions for history encoding
    std::vector<ActionType> lastActions;
    
    for (const auto& action : actions_) {
        if (action.playerId != currentPlayerId) {
            int typeIdx = static_cast<int>(action.actionType);
            if (typeIdx >= 0 && typeIdx < 6) {
                actionCounts[typeIdx]++;
                totalActions++;
            }
            
            if (action.amount > 0) {
                betSizes.push_back(action.potFraction);
            }
            
            lastActions.push_back(action.actionType);
        }
    }
    
    if (totalActions > 0) {
        // Normalized action counts (indices 0-5)
        for (int i = 0; i < 6; i++) {
            features[i] = static_cast<float>(actionCounts[i]) / static_cast<float>(totalActions);
        }
        
        // Average bet size (index 6)
        if (!betSizes.empty()) {
            float sum = 0.0f;
            for (float bs : betSizes) sum += bs;
            features[6] = sum / static_cast<float>(betSizes.size());
        }
        
        // Aggression frequency (index 7)
        features[7] = static_cast<float>(actionCounts[3] + actionCounts[4]) / 
                      static_cast<float>(totalActions);
    }
    
    // Last 3 opponent actions (one-hot, indices 8-25)
    // Pad to ensure we have 3 actions
    while (lastActions.size() < 3) {
        lastActions.insert(lastActions.begin(), ActionType::UNKNOWN);
    }
    
    // Take last 3
    int historyStart = std::max(0, static_cast<int>(lastActions.size()) - 3);
    for (int i = 0; i < 3; i++) {
        int actionIdx = static_cast<int>(lastActions[historyStart + i]);
        if (actionIdx >= 0 && actionIdx < 6) {
            features[8 + i * 6 + actionIdx] = 1.0f;
        }
    }
    
    return features;
}

void ActionHistory::clear() {
    actions_.clear();
}


// ============================================================================
// RLStateEncoder Implementation
// ============================================================================

RLStateEncoder::RLStateEncoder() {
    buffer_.fill(0.0f);
}

int RLStateEncoder::parseRank(char c) {
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

int RLStateEncoder::parseSuit(char c) {
    switch (c) {
        case 'C': case 'c': return 0;
        case 'D': case 'd': return 1;
        case 'H': case 'h': return 2;
        case 'S': case 's': return 3;
        default: return 0;
    }
}

int RLStateEncoder::parseStage(const std::string& stage) {
    if (stage == "Preflop" || stage == "preflop") return 0;
    if (stage == "Flop" || stage == "flop") return 1;
    if (stage == "Turn" || stage == "turn") return 2;
    if (stage == "River" || stage == "river") return 3;
    if (stage == "Complete" || stage == "complete") return 4;
    return 0;
}

void RLStateEncoder::encodeCard(int idx, const std::string& card) {
    if (card.size() >= 2) {
        int rank = parseRank(card[0]);
        int suit = parseSuit(card[1]);
        buffer_[idx + rank] = 1.0f;        // Rank one-hot (13 values)
        buffer_[idx + 13 + suit] = 1.0f;   // Suit one-hot (4 values)
    }
    // If card is empty/invalid, all zeros (already zeroed)
}

void RLStateEncoder::encodePotInfo(int idx, const nlohmann::json& state, float maxStack) {
    if (maxStack <= 0) maxStack = 1000.0f;
    
    buffer_[idx + 0] = state.value("pot", 0) / maxStack;
    buffer_[idx + 1] = state.value("current_bet", 0) / maxStack;
    buffer_[idx + 2] = state.value("player_chips", 0) / maxStack;
    buffer_[idx + 3] = state.value("player_bet", 0) / maxStack;
    buffer_[idx + 4] = state.value("player_total_bet", 0) / maxStack;
}

void RLStateEncoder::encodeStage(int idx, const std::string& stage) {
    int stageIdx = parseStage(stage);
    if (stageIdx >= 0 && stageIdx < STAGE_DIM) {
        buffer_[idx + stageIdx] = 1.0f;
    }
}

void RLStateEncoder::encodePosition(int idx, const nlohmann::json& state) {
    buffer_[idx + 0] = state.value("num_players", 2) / 10.0f;
    buffer_[idx + 1] = state.value("num_active", 2) / 10.0f;
    buffer_[idx + 2] = state.value("position", 0) / 10.0f;
    buffer_[idx + 3] = state.value("is_dealer", false) ? 1.0f : 0.0f;
    buffer_[idx + 4] = state.value("is_small_blind", false) ? 1.0f : 0.0f;
    buffer_[idx + 5] = state.value("is_big_blind", false) ? 1.0f : 0.0f;
}

void RLStateEncoder::encodeGameTheory(int idx, const nlohmann::json& state, float maxStack) {
    int toCall = state.value("to_call", 0);
    int potSize = state.value("pot", 0);
    int playerChips = state.value("player_chips", 0);
    int bigBlind = state.value("big_blind", 20);
    int playerTotalBet = state.value("player_total_bet", 0);
    
    // Pot odds (probability needed to call profitably)
    float potOdds = (toCall > 0 && potSize + toCall > 0) ? 
                    static_cast<float>(toCall) / static_cast<float>(potSize + toCall) : 0.0f;
    
    // Stack to pot ratio
    float stackToPot = (potSize > 0) ? 
                       static_cast<float>(playerChips) / static_cast<float>(potSize) : 1.0f;
    
    // Effective stack depth (normalized by big blind, capped at 100 BBs)
    float effectiveStack = (bigBlind > 0) ? 
                           static_cast<float>(playerChips) / static_cast<float>(bigBlind) : 50.0f;
    float effectiveStackNorm = std::min(1.0f, effectiveStack / 100.0f);
    
    // Call amount as fraction of remaining chips
    float callFraction = (playerChips > 0) ? 
                         static_cast<float>(toCall) / static_cast<float>(playerChips) : 0.0f;
    
    // Pot commitment
    float playerTotalBetNorm = (maxStack > 0) ? playerTotalBet / maxStack : 0.0f;
    float playerChipsNorm = (maxStack > 0) ? playerChips / maxStack : 0.0f;
    float potCommitment = (playerTotalBetNorm + playerChipsNorm > 0.001f) ?
                          playerTotalBetNorm / (playerTotalBetNorm + playerChipsNorm) : 0.0f;
    
    buffer_[idx + 0] = potOdds;
    buffer_[idx + 1] = stackToPot;
    buffer_[idx + 2] = effectiveStackNorm;
    buffer_[idx + 3] = callFraction;
    buffer_[idx + 4] = potCommitment;
}

void RLStateEncoder::encodeOpponentFeatures(int idx, const std::string& playerId) {
    auto features = actionHistory_.getOpponentFeatures(playerId);
    for (int i = 0; i < OPPONENT_DIM; i++) {
        buffer_[idx + i] = features[i];
    }
}

float RLStateEncoder::encodeHandStrength(const std::vector<std::string>& holeCards,
                                         const std::vector<std::string>& communityCards) {
    return HandStrengthEstimator::estimate(holeCards, communityCards);
}

std::vector<float> RLStateEncoder::encodeState(const nlohmann::json& state) {
    // Zero out buffer
    buffer_.fill(0.0f);
    
    // Determine max_stack for normalization
    float maxStack = state.value("max_stack", 
                                 state.value("starting_chips", 1000.0f));
    if (maxStack <= 0) {
        maxStack = state.value("starting_chips", 1000.0f);
    }
    
    int idx = 0;
    
    // 1. Hole cards (34 features: 2 cards × 17)
    std::vector<std::string> holeCards;
    if (state.contains("hole_cards") && state["hole_cards"].is_array()) {
        for (const auto& card : state["hole_cards"]) {
            if (card.is_string()) {
                holeCards.push_back(card.get<std::string>());
            }
        }
    }
    
    for (int i = 0; i < 2; i++) {
        if (i < static_cast<int>(holeCards.size())) {
            encodeCard(idx, holeCards[i]);
        }
        idx += HOLE_CARD_DIM;
    }
    
    // 2. Community cards (85 features: 5 cards × 17)
    std::vector<std::string> communityCards;
    if (state.contains("community_cards") && state["community_cards"].is_array()) {
        for (const auto& card : state["community_cards"]) {
            if (card.is_string()) {
                communityCards.push_back(card.get<std::string>());
            }
        }
    }
    
    for (int i = 0; i < 5; i++) {
        if (i < static_cast<int>(communityCards.size())) {
            encodeCard(idx, communityCards[i]);
        }
        idx += COMMUNITY_CARD_DIM;
    }
    
    // 3. Pot/betting info (5 features)
    encodePotInfo(idx, state, maxStack);
    idx += POT_DIM;
    
    // 4. Stage (5 features)
    std::string stage = state.value("stage", "Preflop");
    encodeStage(idx, stage);
    idx += STAGE_DIM;
    
    // 5. Position (6 features)
    encodePosition(idx, state);
    idx += POSITION_DIM;
    
    // 6. Game theory (5 features)
    encodeGameTheory(idx, state, maxStack);
    idx += GAMETHEORY_DIM;
    
    // 7. Opponent modeling (26 features)
    std::string playerId = state.value("player_id", "unknown");
    encodeOpponentFeatures(idx, playerId);
    idx += OPPONENT_DIM;
    
    // 8. Hand strength (1 feature)
    buffer_[idx] = encodeHandStrength(holeCards, communityCards);
    
    // Return a copy of the buffer
    return std::vector<float>(buffer_.begin(), buffer_.end());
}

void RLStateEncoder::resetHistory() {
    actionHistory_.clear();
}

void RLStateEncoder::addAction(const std::string& playerId, const std::string& actionType,
                               int amount, int pot, const std::string& stage) {
    actionHistory_.addAction(playerId, actionType, amount, pot, stage);
}


// ============================================================================
// ActionSpace Implementation
// ============================================================================

static const std::vector<std::string> ACTION_NAMES = {
    "fold", "check", "call",
    "raise_10%", "raise_25%", "raise_33%", "raise_50%", "raise_75%",
    "raise_100%", "raise_150%", "raise_200%", "raise_300%",
    "all_in"
};

static const std::unordered_map<std::string, int> ACTION_TO_INDEX = {
    {"fold", 0}, {"check", 1}, {"call", 2},
    {"raise_10%", 3}, {"raise_25%", 4}, {"raise_33%", 5}, {"raise_50%", 6}, {"raise_75%", 7},
    {"raise_100%", 8}, {"raise_150%", 9}, {"raise_200%", 10}, {"raise_300%", 11},
    {"all_in", 12}
};

static const std::unordered_map<int, float> RAISE_SIZES = {
    {3, 0.10f}, {4, 0.25f}, {5, 0.33f}, {6, 0.50f}, {7, 0.75f},
    {8, 1.00f}, {9, 1.50f}, {10, 2.00f}, {11, 3.00f}
};

int ActionSpace::actionToIndex(const std::string& actionName) {
    auto it = ACTION_TO_INDEX.find(actionName);
    if (it != ACTION_TO_INDEX.end()) {
        return it->second;
    }
    return -1;  // Invalid action
}

std::string ActionSpace::indexToAction(int index) {
    if (index >= 0 && index < NUM_ACTIONS) {
        return ACTION_NAMES[index];
    }
    return "unknown";
}

float ActionSpace::getRaiseSize(int actionIndex) {
    auto it = RAISE_SIZES.find(actionIndex);
    if (it != RAISE_SIZES.end()) {
        return it->second;
    }
    return 0.0f;
}

float ActionSpace::getRaiseSizeFromLabel(const std::string& label) {
    int idx = actionToIndex(label);
    return getRaiseSize(idx);
}





