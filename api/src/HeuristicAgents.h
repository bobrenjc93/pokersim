#pragma once

#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <random>
#include "json.hpp"
#include "ActionUtils.h"
#include "HandStrength.h"

/**
 * C++ implementation of heuristic agents for RL training.
 * 
 * These agents are used as opponents during self-play training.
 * Implementing them in C++ provides ~10-20x speedup over Python
 * during episode collection.
 * 
 * All agents implement a common interface:
 * - reset(): Reset state for new hand
 * - observeAction(): Track opponent actions (for opponent modeling)
 * - selectAction(): Choose an action given game state and legal actions
 */

namespace HeuristicAgents {

/**
 * Result of action selection.
 */
struct ActionResult {
    std::string actionType;   // "fold", "check", "call", "bet", "raise", "all_in"
    int amount;               // Bet/raise amount (0 for fold/check/call/all_in)
    std::string actionLabel;  // Original label (e.g., "raise_50%")
};

/**
 * Parsed card for hand evaluation.
 */
struct ParsedCard {
    int rank;  // 2=0, 3=1, ..., A=12
    int suit;  // 0-3
};

/**
 * Base class for heuristic agents.
 */
class BaseAgent {
public:
    virtual ~BaseAgent() = default;
    
    /**
     * Reset state for new hand.
     */
    virtual void reset() {}
    
    /**
     * Observe an action (for opponent modeling).
     */
    virtual void observeAction(const std::string& playerId, const std::string& actionType,
                               int amount, int pot, const std::string& stage) {}
    
    /**
     * Select an action given game state and legal actions.
     * 
     * @param state Extracted player state (from ActionUtils::extractState)
     * @param legalActions List of legal action strings
     * @return ActionResult with action type, amount, and label
     */
    virtual ActionResult selectAction(const nlohmann::json& state,
                                      const std::vector<std::string>& legalActions) = 0;

protected:
    std::mt19937 rng_;
    
    // Helper methods
    static ParsedCard parseCard(const std::string& card);
    static bool hasAction(const std::vector<std::string>& actions, const std::string& action);
    ActionResult makeAction(const std::string& label, const nlohmann::json& state);
    float randomFloat();
    
    BaseAgent() : rng_(std::random_device{}()) {}
    explicit BaseAgent(unsigned int seed) : rng_(seed) {}
};


/**
 * Agent that selects random valid actions.
 * Useful for baselines and initial training diversity.
 */
class RandomAgent : public BaseAgent {
public:
    RandomAgent() = default;
    explicit RandomAgent(unsigned int seed) : BaseAgent(seed) {}
    
    ActionResult selectAction(const nlohmann::json& state,
                              const std::vector<std::string>& legalActions) override;
};


/**
 * Tight-aggressive agent that only plays premium hands.
 * 
 * Critical for training:
 * - Teaches model that not all aggression works
 * - Shows bluffing has diminishing returns against tight players
 * - Requires actual hand strength to win
 */
class TightAgent : public BaseAgent {
public:
    TightAgent() = default;
    explicit TightAgent(unsigned int seed) : BaseAgent(seed) {}
    
    ActionResult selectAction(const nlohmann::json& state,
                              const std::vector<std::string>& legalActions) override;

private:
    float getPreflopStrength(const std::vector<std::string>& holeCards);
};


/**
 * Loose-passive agent that calls too much but rarely raises.
 * 
 * Useful for training:
 * - Teaches value of thin value bets
 * - Shows passive play is exploitable
 */
class LoosePassiveAgent : public BaseAgent {
public:
    LoosePassiveAgent() = default;
    explicit LoosePassiveAgent(unsigned int seed) : BaseAgent(seed) {}
    
    ActionResult selectAction(const nlohmann::json& state,
                              const std::vector<std::string>& legalActions) override;
};


/**
 * Loose-aggressive agent that bets and raises frequently.
 * 
 * Useful for training:
 * - Tests model's ability to call down with marginal hands
 * - Shows not all aggression has real hands behind it
 */
class AggressiveAgent : public BaseAgent {
public:
    AggressiveAgent() = default;
    explicit AggressiveAgent(unsigned int seed) : BaseAgent(seed) {}
    
    ActionResult selectAction(const nlohmann::json& state,
                              const std::vector<std::string>& legalActions) override;
};


/**
 * Calling Station - calls almost everything including all-ins.
 * 
 * CRITICAL FOR TRAINING:
 * - Teaches model that all-in with weak hands LOSES MONEY
 * - Shows trash hands lose at showdown
 */
class CallingStationAgent : public BaseAgent {
public:
    CallingStationAgent() = default;
    explicit CallingStationAgent(unsigned int seed) : BaseAgent(seed) {}
    
    ActionResult selectAction(const nlohmann::json& state,
                              const std::vector<std::string>& legalActions) override;

private:
    float getHandStrength(const std::vector<std::string>& holeCards);
};


/**
 * Hero Caller - calls down suspected bluffs with medium-strength hands.
 * 
 * Teaches model that bluffing and aggressive all-ins don't always work.
 */
class HeroCallerAgent : public BaseAgent {
public:
    HeroCallerAgent() = default;
    explicit HeroCallerAgent(unsigned int seed) : BaseAgent(seed) {}
    
    void reset() override;
    void observeAction(const std::string& playerId, const std::string& actionType,
                       int amount, int pot, const std::string& stage) override;
    ActionResult selectAction(const nlohmann::json& state,
                              const std::vector<std::string>& legalActions) override;

private:
    int opponentAggressionCount_ = 0;
    std::string ownPlayerId_;
    
    float getHandStrength(const std::vector<std::string>& holeCards,
                          const std::vector<std::string>& communityCards);
};


/**
 * Rule-based heuristic agent for baseline comparison.
 * Implements simple aggressive strategy based on hand strength.
 */
class HeuristicAgent : public BaseAgent {
public:
    HeuristicAgent() = default;
    explicit HeuristicAgent(unsigned int seed) : BaseAgent(seed) {}
    
    ActionResult selectAction(const nlohmann::json& state,
                              const std::vector<std::string>& legalActions) override;

private:
    float getHandStrength(const std::vector<std::string>& holeCards,
                          const std::vector<std::string>& communityCards);
};


/**
 * Always raises/bets when possible, otherwise call/check.
 */
class AlwaysRaiseAgent : public BaseAgent {
public:
    AlwaysRaiseAgent() = default;
    explicit AlwaysRaiseAgent(unsigned int seed) : BaseAgent(seed) {}
    
    ActionResult selectAction(const nlohmann::json& state,
                              const std::vector<std::string>& legalActions) override;
};


/**
 * Always checks or calls, never bets/raises.
 */
class AlwaysCallAgent : public BaseAgent {
public:
    AlwaysCallAgent() = default;
    explicit AlwaysCallAgent(unsigned int seed) : BaseAgent(seed) {}
    
    ActionResult selectAction(const nlohmann::json& state,
                              const std::vector<std::string>& legalActions) override;
};


/**
 * Always folds when facing a bet, checks if free.
 */
class AlwaysFoldAgent : public BaseAgent {
public:
    AlwaysFoldAgent() = default;
    explicit AlwaysFoldAgent(unsigned int seed) : BaseAgent(seed) {}
    
    ActionResult selectAction(const nlohmann::json& state,
                              const std::vector<std::string>& legalActions) override;
};


/**
 * Factory function to create agent by name.
 * 
 * @param agentType Agent type name (e.g., "tight", "calling_station")
 * @param seed Optional random seed
 * @return Unique pointer to agent, or nullptr if type unknown
 */
std::unique_ptr<BaseAgent> createAgent(const std::string& agentType, unsigned int seed = 0);

/**
 * Get list of available agent types.
 */
std::vector<std::string> getAgentTypes();

} // namespace HeuristicAgents




