#include "HeuristicAgents.h"
#include <algorithm>
#include <cmath>

namespace HeuristicAgents {

// ============================================================================
// BaseAgent Implementation
// ============================================================================

ParsedCard BaseAgent::parseCard(const std::string& card) {
    ParsedCard result = {0, 0};
    if (card.size() < 2) return result;
    
    // Parse rank
    char r = card[0];
    switch (r) {
        case '2': result.rank = 0; break;
        case '3': result.rank = 1; break;
        case '4': result.rank = 2; break;
        case '5': result.rank = 3; break;
        case '6': result.rank = 4; break;
        case '7': result.rank = 5; break;
        case '8': result.rank = 6; break;
        case '9': result.rank = 7; break;
        case 'T': result.rank = 8; break;
        case 'J': result.rank = 9; break;
        case 'Q': result.rank = 10; break;
        case 'K': result.rank = 11; break;
        case 'A': result.rank = 12; break;
        default: result.rank = 0; break;
    }
    
    // Parse suit
    char s = card[1];
    switch (s) {
        case 'C': case 'c': result.suit = 0; break;
        case 'D': case 'd': result.suit = 1; break;
        case 'H': case 'h': result.suit = 2; break;
        case 'S': case 's': result.suit = 3; break;
        default: result.suit = 0; break;
    }
    
    return result;
}

bool BaseAgent::hasAction(const std::vector<std::string>& actions, const std::string& action) {
    return std::find(actions.begin(), actions.end(), action) != actions.end();
}

ActionResult BaseAgent::makeAction(const std::string& label, const nlohmann::json& state) {
    auto converted = ActionUtils::convertActionLabel(label, state);
    return {converted.actionType, converted.amount, label};
}

float BaseAgent::randomFloat() {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng_);
}


// ============================================================================
// RandomAgent Implementation
// ============================================================================

ActionResult RandomAgent::selectAction(const nlohmann::json& state,
                                       const std::vector<std::string>& legalActions) {
    if (legalActions.empty()) {
        return {"fold", 0, "fold"};
    }
    
    std::uniform_int_distribution<size_t> dist(0, legalActions.size() - 1);
    std::string action = legalActions[dist(rng_)];
    
    std::string actionLabel = action;
    
    if (action == "bet" || action == "raise") {
        // Pick a random size
        static const std::vector<std::string> sizes = {"raise_50%", "raise_75%", "raise_100%"};
        std::uniform_int_distribution<size_t> sizeDist(0, sizes.size() - 1);
        actionLabel = sizes[sizeDist(rng_)];
    }
    
    return makeAction(actionLabel, state);
}


// ============================================================================
// TightAgent Implementation
// ============================================================================

float TightAgent::getPreflopStrength(const std::vector<std::string>& holeCards) {
    if (holeCards.size() < 2) return 0.0f;
    
    ParsedCard c1 = parseCard(holeCards[0]);
    ParsedCard c2 = parseCard(holeCards[1]);
    
    int h1 = std::max(c1.rank, c2.rank);
    int h2 = std::min(c1.rank, c2.rank);
    bool isSuited = (c1.suit == c2.suit);
    bool isPair = (c1.rank == c2.rank);
    
    // Premium pairs: AA, KK, QQ, JJ
    if (isPair && h1 >= 9) {  // JJ+
        return 0.9f + (h1 - 9) * 0.025f;
    }
    
    // Medium pairs: TT-77
    if (isPair && h1 >= 5) {
        return 0.6f + (h1 - 5) * 0.05f;
    }
    
    // Small pairs: 66-22
    if (isPair) {
        return 0.35f + h1 * 0.03f;
    }
    
    // Big Ace (AK, AQ, AJ, AT)
    if (h1 == 12) {  // Ace
        if (h2 >= 8) {  // AT+
            float base = 0.65f + (h2 - 8) * 0.05f;
            return base + (isSuited ? 0.05f : 0.0f);
        } else if (h2 >= 5) {
            return 0.45f + (isSuited ? 0.05f : 0.0f);
        } else {
            return 0.30f + (isSuited ? 0.06f : 0.0f);
        }
    }
    
    // Broadway cards
    if (h1 >= 8 && h2 >= 8) {
        return 0.50f + (isSuited ? 0.05f : 0.0f);
    }
    
    // Suited connectors
    if (isSuited && std::abs(h1 - h2) == 1) {
        return 0.35f + (h1 / 12.0f) * 0.1f;
    }
    
    // Other suited cards
    if (isSuited) {
        return 0.25f + (h1 / 12.0f) * 0.1f;
    }
    
    // Offsuit connected
    if (std::abs(h1 - h2) <= 2) {
        return 0.20f + (h1 / 12.0f) * 0.1f;
    }
    
    // Trash
    return 0.15f;
}

ActionResult TightAgent::selectAction(const nlohmann::json& state,
                                      const std::vector<std::string>& legalActions) {
    // Extract cards
    std::vector<std::string> holeCards;
    if (state.contains("hole_cards") && state["hole_cards"].is_array()) {
        for (const auto& card : state["hole_cards"]) {
            if (card.is_string()) {
                holeCards.push_back(card.get<std::string>());
            }
        }
    }
    
    std::vector<std::string> communityCards;
    if (state.contains("community_cards") && state["community_cards"].is_array()) {
        for (const auto& card : state["community_cards"]) {
            if (card.is_string()) {
                communityCards.push_back(card.get<std::string>());
            }
        }
    }
    
    bool canCheck = hasAction(legalActions, "check");
    bool canCall = hasAction(legalActions, "call");
    bool canBet = hasAction(legalActions, "bet");
    bool canRaise = hasAction(legalActions, "raise");
    
    std::string actionLabel;
    
    // Preflop - tight hand selection
    if (communityCards.empty()) {
        float strength = getPreflopStrength(holeCards);
        
        // Only play premium hands (top ~15%)
        if (strength >= 0.65f) {
            // Premium - raise/3-bet
            if (canRaise || canBet) {
                actionLabel = "raise_75%";
            } else if (canCall) {
                actionLabel = "call";
            } else {
                actionLabel = canCheck ? "check" : "fold";
            }
        } else if (strength >= 0.50f) {
            // Strong but not premium - call or small raise
            if (canCall) {
                actionLabel = "call";
            } else if (canCheck) {
                actionLabel = "check";
            } else {
                actionLabel = "fold";
            }
        } else if (strength >= 0.35f) {
            // Medium - only check if free
            actionLabel = canCheck ? "check" : "fold";
        } else {
            // Weak/trash - always fold if facing bet
            actionLabel = canCheck ? "check" : "fold";
        }
    } else {
        // Postflop - simplified
        float strength = getPreflopStrength(holeCards);  // Simplified
        
        int toCall = state.value("to_call", 0);
        int pot = state.value("pot", 0);
        float potOdds = (toCall > 0 && pot + toCall > 0) ? 
                        static_cast<float>(toCall) / static_cast<float>(pot + toCall) : 0.0f;
        
        // Strong made hand - bet for value
        if (strength >= 0.65f) {
            if (canBet || canRaise) {
                actionLabel = "raise_50%";
            } else if (canCall) {
                actionLabel = "call";
            } else {
                actionLabel = canCheck ? "check" : "fold";
            }
        }
        // Medium - play passively
        else if (strength >= 0.45f) {
            if (canCheck) {
                actionLabel = "check";
            } else if (canCall && potOdds < 0.3f) {
                actionLabel = "call";
            } else {
                actionLabel = "fold";
            }
        }
        // Weak - check/fold
        else {
            actionLabel = canCheck ? "check" : "fold";
        }
    }
    
    return makeAction(actionLabel, state);
}


// ============================================================================
// LoosePassiveAgent Implementation
// ============================================================================

ActionResult LoosePassiveAgent::selectAction(const nlohmann::json& state,
                                             const std::vector<std::string>& legalActions) {
    bool canCheck = hasAction(legalActions, "check");
    bool canCall = hasAction(legalActions, "call");
    bool canBet = hasAction(legalActions, "bet");
    
    std::string actionLabel;
    float roll = randomFloat();
    
    // Can check - usually check
    if (canCheck) {
        if (roll < 0.85f) {
            actionLabel = "check";
        } else if (canBet) {
            actionLabel = "raise_33%";  // Small bet sometimes
        } else {
            actionLabel = "check";
        }
    }
    // Facing bet - call most of the time
    else if (canCall) {
        if (roll < 0.75f) {
            actionLabel = "call";
        } else if (roll < 0.80f && hasAction(legalActions, "raise")) {
            actionLabel = "raise_50%";  // Rare raise
        } else {
            actionLabel = "fold";  // Sometimes fold
        }
    } else {
        actionLabel = "fold";
    }
    
    return makeAction(actionLabel, state);
}


// ============================================================================
// AggressiveAgent Implementation
// ============================================================================

ActionResult AggressiveAgent::selectAction(const nlohmann::json& state,
                                           const std::vector<std::string>& legalActions) {
    bool canCheck = hasAction(legalActions, "check");
    bool canCall = hasAction(legalActions, "call");
    bool canBet = hasAction(legalActions, "bet");
    bool canRaise = hasAction(legalActions, "raise");
    
    std::string actionLabel;
    float roll = randomFloat();
    
    static const std::vector<std::string> sizes = {"raise_50%", "raise_75%", "raise_100%"};
    
    // Aggressive: bets and raises frequently
    if (canBet) {
        if (roll < 0.65f) {
            // Bet most of the time when we can
            std::uniform_int_distribution<size_t> dist(0, sizes.size() - 1);
            actionLabel = sizes[dist(rng_)];
        } else if (roll < 0.85f) {
            actionLabel = canCheck ? "check" : sizes[0];
        } else {
            actionLabel = canCheck ? "check" : "raise_50%";
        }
    } else if (canRaise) {
        if (roll < 0.55f) {
            // Raise frequently
            std::uniform_int_distribution<size_t> dist(0, sizes.size() - 1);
            actionLabel = sizes[dist(rng_)];
        } else if (roll < 0.85f) {
            actionLabel = "call";
        } else {
            actionLabel = "fold";
        }
    } else if (canCall) {
        if (roll < 0.70f) {
            actionLabel = "call";
        } else {
            actionLabel = "fold";
        }
    } else if (canCheck) {
        actionLabel = "check";
    } else {
        actionLabel = "fold";
    }
    
    return makeAction(actionLabel, state);
}


// ============================================================================
// CallingStationAgent Implementation
// ============================================================================

float CallingStationAgent::getHandStrength(const std::vector<std::string>& holeCards) {
    if (holeCards.size() < 2) return 0.3f;
    
    ParsedCard c1 = parseCard(holeCards[0]);
    ParsedCard c2 = parseCard(holeCards[1]);
    
    int h1 = std::max(c1.rank, c2.rank);
    int h2 = std::min(c1.rank, c2.rank);
    bool isSuited = (c1.suit == c2.suit);
    bool isPair = (c1.rank == c2.rank);
    
    // Pairs
    if (isPair) {
        return 0.5f + (h1 / 12.0f) * 0.4f;
    }
    
    // High cards
    float base = (h1 / 12.0f) * 0.4f + (h2 / 12.0f) * 0.2f;
    if (isSuited) base += 0.08f;
    if (std::abs(h1 - h2) <= 2) base += 0.05f;
    
    return std::max(0.15f, std::min(0.85f, base));
}

ActionResult CallingStationAgent::selectAction(const nlohmann::json& state,
                                               const std::vector<std::string>& legalActions) {
    bool canCheck = hasAction(legalActions, "check");
    bool canCall = hasAction(legalActions, "call");
    bool canBet = hasAction(legalActions, "bet");
    
    std::vector<std::string> holeCards;
    if (state.contains("hole_cards") && state["hole_cards"].is_array()) {
        for (const auto& card : state["hole_cards"]) {
            if (card.is_string()) {
                holeCards.push_back(card.get<std::string>());
            }
        }
    }
    
    float handStrength = getHandStrength(holeCards);
    int toCall = state.value("to_call", 0);
    int playerChips = state.value("player_chips", 0);
    
    // Calculate if this is an all-in situation
    bool isFacingAllIn = (toCall >= playerChips * 0.5f);
    
    float roll = randomFloat();
    std::string actionLabel;
    
    // Check if we can
    if (canCheck) {
        if (roll < 0.10f && canBet) {
            // Occasionally bet (10%)
            actionLabel = "raise_33%";
        } else {
            actionLabel = "check";
        }
    }
    // Facing bet or raise - CALL FREQUENTLY
    else if (canCall) {
        // CRITICAL: Call even all-ins with any decent hand
        if (isFacingAllIn) {
            // Facing all-in: call if hand is remotely decent
            if (handStrength >= 0.25f) {
                actionLabel = "call";
            } else if (roll < 0.40f) {  // Even call some trash (40% of time)
                actionLabel = "call";
            } else {
                actionLabel = "fold";
            }
        } else {
            // Regular bet: almost always call
            if (handStrength >= 0.20f) {
                actionLabel = "call";
            } else if (roll < 0.60f) {  // Call 60% of trash too
                actionLabel = "call";
            } else {
                actionLabel = "fold";
            }
        }
    } else {
        actionLabel = "fold";
    }
    
    return makeAction(actionLabel, state);
}


// ============================================================================
// HeroCallerAgent Implementation
// ============================================================================

void HeroCallerAgent::reset() {
    opponentAggressionCount_ = 0;
}

void HeroCallerAgent::observeAction(const std::string& playerId, const std::string& actionType,
                                    int amount, int pot, const std::string& stage) {
    // Track opponent aggression (assumes 2-player game, so any other player is opponent)
    if (actionType == "bet" || actionType == "raise" || actionType == "all_in") {
        opponentAggressionCount_++;
    }
}

float HeroCallerAgent::getHandStrength(const std::vector<std::string>& holeCards,
                                       const std::vector<std::string>& communityCards) {
    if (holeCards.size() < 2) return 0.3f;
    
    ParsedCard c1 = parseCard(holeCards[0]);
    ParsedCard c2 = parseCard(holeCards[1]);
    
    int h1 = std::max(c1.rank, c2.rank);
    int h2 = std::min(c1.rank, c2.rank);
    bool isSuited = (c1.suit == c2.suit);
    bool isPair = (c1.rank == c2.rank);
    
    // Preflop strength
    if (communityCards.empty()) {
        if (isPair) {
            return 0.5f + (h1 / 12.0f) * 0.45f;
        }
        float base = (h1 / 12.0f) * 0.4f + (h2 / 12.0f) * 0.2f;
        if (isSuited) base += 0.08f;
        return std::max(0.15f, std::min(0.75f, base));
    }
    
    // Postflop: simplified - add pair bonus if we hit
    std::vector<int> commRanks;
    for (const auto& card : communityCards) {
        commRanks.push_back(parseCard(card).rank);
    }
    
    bool paired = (std::find(commRanks.begin(), commRanks.end(), c1.rank) != commRanks.end() ||
                   std::find(commRanks.begin(), commRanks.end(), c2.rank) != commRanks.end());
    bool twoPair = (std::find(commRanks.begin(), commRanks.end(), c1.rank) != commRanks.end() &&
                   std::find(commRanks.begin(), commRanks.end(), c2.rank) != commRanks.end());
    
    float base = (h1 / 12.0f) * 0.25f + (h2 / 12.0f) * 0.1f;
    if (isPair) base += 0.20f;
    if (paired) base += 0.25f;
    if (twoPair) base += 0.35f;
    
    return std::max(0.15f, std::min(0.90f, base));
}

ActionResult HeroCallerAgent::selectAction(const nlohmann::json& state,
                                           const std::vector<std::string>& legalActions) {
    bool canCheck = hasAction(legalActions, "check");
    bool canCall = hasAction(legalActions, "call");
    bool canBet = hasAction(legalActions, "bet");
    
    std::vector<std::string> holeCards;
    if (state.contains("hole_cards") && state["hole_cards"].is_array()) {
        for (const auto& card : state["hole_cards"]) {
            if (card.is_string()) {
                holeCards.push_back(card.get<std::string>());
            }
        }
    }
    
    std::vector<std::string> communityCards;
    if (state.contains("community_cards") && state["community_cards"].is_array()) {
        for (const auto& card : state["community_cards"]) {
            if (card.is_string()) {
                communityCards.push_back(card.get<std::string>());
            }
        }
    }
    
    float handStrength = getHandStrength(holeCards, communityCards);
    
    // Adjust call threshold based on opponent aggression
    // More aggressive opponent = lighter calls
    float callThreshold = std::max(0.20f, 0.45f - (opponentAggressionCount_ * 0.10f));
    
    float roll = randomFloat();
    std::string actionLabel;
    
    if (canCheck) {
        if (handStrength >= 0.55f && canBet && roll < 0.50f) {
            actionLabel = "raise_50%";
        } else {
            actionLabel = "check";
        }
    } else if (canCall) {
        // Hero call logic: call more loosely when opponent is aggressive
        if (handStrength >= callThreshold) {
            actionLabel = "call";
        } else if (roll < 0.20f) {  // Occasional light call
            actionLabel = "call";
        } else {
            actionLabel = "fold";
        }
    } else {
        actionLabel = "fold";
    }
    
    return makeAction(actionLabel, state);
}


// ============================================================================
// HeuristicAgent Implementation
// ============================================================================

float HeuristicAgent::getHandStrength(const std::vector<std::string>& holeCards,
                                      const std::vector<std::string>& communityCards) {
    if (holeCards.size() < 2) return 0.0f;
    
    ParsedCard c1 = parseCard(holeCards[0]);
    ParsedCard c2 = parseCard(holeCards[1]);
    
    int h1 = std::max(c1.rank, c2.rank);
    int h2 = std::min(c1.rank, c2.rank);
    bool isPair = (c1.rank == c2.rank);
    
    // Preflop heuristic
    if (communityCards.empty()) {
        // Pocket pair
        if (isPair) {
            return 0.5f + (h1 / 12.0f) * 0.5f;
        }
        // High cards
        return (h1 / 12.0f) * 0.6f + (h2 / 12.0f) * 0.2f;
    }
    
    // Postflop - random noise mixed with preflop strength
    return (h1 / 12.0f) * 0.4f + (h2 / 12.0f) * 0.1f + randomFloat() * 0.5f;
}

ActionResult HeuristicAgent::selectAction(const nlohmann::json& state,
                                          const std::vector<std::string>& legalActions) {
    std::vector<std::string> holeCards;
    if (state.contains("hole_cards") && state["hole_cards"].is_array()) {
        for (const auto& card : state["hole_cards"]) {
            if (card.is_string()) {
                holeCards.push_back(card.get<std::string>());
            }
        }
    }
    
    std::vector<std::string> communityCards;
    if (state.contains("community_cards") && state["community_cards"].is_array()) {
        for (const auto& card : state["community_cards"]) {
            if (card.is_string()) {
                communityCards.push_back(card.get<std::string>());
            }
        }
    }
    
    float strength = getHandStrength(holeCards, communityCards);
    
    bool canCheck = hasAction(legalActions, "check");
    bool canCall = hasAction(legalActions, "call");
    bool canBet = hasAction(legalActions, "bet");
    bool canRaise = hasAction(legalActions, "raise");
    
    std::string actionLabel;
    
    // Very strong hand -> Bet/Raise
    if (strength > 0.8f) {
        if (canRaise) {
            actionLabel = "raise_100%";
        } else if (canBet) {
            actionLabel = "raise_75%";
        } else if (canCall) {
            actionLabel = "call";
        } else {
            actionLabel = canCheck ? "check" : "fold";
        }
    }
    // Strong hand -> Bet small / Call
    else if (strength > 0.6f) {
        if (canBet) {
            actionLabel = "raise_50%";
        } else if (canCall) {
            actionLabel = "call";
        } else {
            actionLabel = canCheck ? "check" : "fold";
        }
    }
    // Medium hand -> Check/Call if cheap
    else if (strength > 0.4f) {
        if (canCheck) {
            actionLabel = "check";
        } else if (canCall) {
            actionLabel = "call";
        } else {
            actionLabel = "fold";
        }
    }
    // Weak hand -> Check/Fold (bluff occasionally)
    else {
        if (randomFloat() < 0.1f && canBet) {  // Bluff 10%
            actionLabel = "raise_50%";
        } else if (canCheck) {
            actionLabel = "check";
        } else {
            actionLabel = "fold";
        }
    }
    
    return makeAction(actionLabel, state);
}


// ============================================================================
// Simple Agents Implementation
// ============================================================================

ActionResult AlwaysRaiseAgent::selectAction(const nlohmann::json& state,
                                            const std::vector<std::string>& legalActions) {
    std::string actionLabel;
    
    if (hasAction(legalActions, "raise")) {
        actionLabel = "raise_100%";
    } else if (hasAction(legalActions, "bet")) {
        actionLabel = "raise_100%";
    } else if (hasAction(legalActions, "all_in")) {
        actionLabel = "all_in";
    } else if (hasAction(legalActions, "call")) {
        actionLabel = "call";
    } else if (hasAction(legalActions, "check")) {
        actionLabel = "check";
    } else {
        actionLabel = "fold";
    }
    
    return makeAction(actionLabel, state);
}

ActionResult AlwaysCallAgent::selectAction(const nlohmann::json& state,
                                           const std::vector<std::string>& legalActions) {
    std::string actionLabel;
    
    if (hasAction(legalActions, "check")) {
        actionLabel = "check";
    } else if (hasAction(legalActions, "call")) {
        actionLabel = "call";
    } else {
        actionLabel = "fold";
    }
    
    return makeAction(actionLabel, state);
}

ActionResult AlwaysFoldAgent::selectAction(const nlohmann::json& state,
                                           const std::vector<std::string>& legalActions) {
    std::string actionLabel;
    
    if (hasAction(legalActions, "check")) {
        actionLabel = "check";
    } else {
        actionLabel = "fold";
    }
    
    return makeAction(actionLabel, state);
}


// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<BaseAgent> createAgent(const std::string& agentType, unsigned int seed) {
    if (agentType == "random") {
        return seed ? std::make_unique<RandomAgent>(seed) : std::make_unique<RandomAgent>();
    } else if (agentType == "tight") {
        return seed ? std::make_unique<TightAgent>(seed) : std::make_unique<TightAgent>();
    } else if (agentType == "loose_passive") {
        return seed ? std::make_unique<LoosePassiveAgent>(seed) : std::make_unique<LoosePassiveAgent>();
    } else if (agentType == "aggressive") {
        return seed ? std::make_unique<AggressiveAgent>(seed) : std::make_unique<AggressiveAgent>();
    } else if (agentType == "calling_station") {
        return seed ? std::make_unique<CallingStationAgent>(seed) : std::make_unique<CallingStationAgent>();
    } else if (agentType == "hero_caller") {
        return seed ? std::make_unique<HeroCallerAgent>(seed) : std::make_unique<HeroCallerAgent>();
    } else if (agentType == "heuristic") {
        return seed ? std::make_unique<HeuristicAgent>(seed) : std::make_unique<HeuristicAgent>();
    } else if (agentType == "always_raise") {
        return seed ? std::make_unique<AlwaysRaiseAgent>(seed) : std::make_unique<AlwaysRaiseAgent>();
    } else if (agentType == "always_call") {
        return seed ? std::make_unique<AlwaysCallAgent>(seed) : std::make_unique<AlwaysCallAgent>();
    } else if (agentType == "always_fold") {
        return seed ? std::make_unique<AlwaysFoldAgent>(seed) : std::make_unique<AlwaysFoldAgent>();
    }
    
    return nullptr;
}

std::vector<std::string> getAgentTypes() {
    return {
        "random",
        "tight",
        "loose_passive",
        "aggressive",
        "calling_station",
        "hero_caller",
        "heuristic",
        "always_raise",
        "always_call",
        "always_fold"
    };
}

} // namespace HeuristicAgents




