#pragma once

#include <vector>
#include <string>
#include <array>
#include <unordered_map>
#include <deque>
#include "json.hpp"

/**
 * Action types for opponent modeling.
 */
enum class ActionType {
    FOLD = 0,
    CHECK = 1,
    CALL = 2,
    BET = 3,
    RAISE = 4,
    ALL_IN = 5,
    UNKNOWN = 6
};

/**
 * Tracks action history for opponent modeling in RL training.
 */
class ActionHistory {
public:
    struct ActionRecord {
        std::string playerId;
        ActionType actionType;
        int amount;
        int pot;
        int stage;  // 0=preflop, 1=flop, 2=turn, 3=river
        float potFraction;
    };
    
    static constexpr int MAX_HISTORY = 20;
    static constexpr int OPPONENT_FEATURES_DIM = 26;
    
    ActionHistory() = default;
    
    /**
     * Add an action to the history.
     */
    void addAction(const std::string& playerId, const std::string& actionTypeStr,
                   int amount, int pot, const std::string& stage);
    
    /**
     * Get opponent modeling features (26 dimensions).
     * 
     * Features:
     * - Action counts (fold, check, call, bet, raise, all_in) -> 6
     * - Average bet size (as fraction of pot) -> 1
     * - Aggression frequency -> 1
     * - Last 3 actions (one-hot) -> 3 * 6 = 18
     */
    std::array<float, OPPONENT_FEATURES_DIM> getOpponentFeatures(
        const std::string& currentPlayerId) const;
    
    /**
     * Clear all history.
     */
    void clear();
    
private:
    std::deque<ActionRecord> actions_;
    
    static ActionType parseActionType(const std::string& actionStr);
    static int parseStage(const std::string& stageStr);
};


/**
 * High-performance state encoder for RL training.
 * 
 * Encodes poker game state into a fixed-size feature vector (167 dimensions)
 * for the neural network. This C++ implementation is ~10-20x faster than
 * the Python equivalent.
 * 
 * Feature breakdown:
 * 1. Hole cards: 2 × 17 = 34 (13 rank one-hot + 4 suit one-hot per card)
 * 2. Community cards: 5 × 17 = 85
 * 3. Pot/betting info: 5 (pot, current_bet, player_chips, player_bet, total_bet)
 * 4. Stage: 5 (one-hot: preflop, flop, turn, river, complete)
 * 5. Position: 6 (num_players, num_active, position, is_dealer, is_sb, is_bb)
 * 6. Game theory: 5 (pot_odds, stack_to_pot, eff_stack, call_fraction, pot_commitment)
 * 7. Opponent modeling: 26 (action distributions + recent history)
 * 8. Hand strength: 1
 * 
 * Total: 167 features
 */
class RLStateEncoder {
public:
    static constexpr int FEATURE_DIM = 167;
    
    // Component dimensions
    static constexpr int HOLE_CARD_DIM = 17;
    static constexpr int COMMUNITY_CARD_DIM = 17;
    static constexpr int POT_DIM = 5;
    static constexpr int STAGE_DIM = 5;
    static constexpr int POSITION_DIM = 6;
    static constexpr int GAMETHEORY_DIM = 5;
    static constexpr int OPPONENT_DIM = 26;
    static constexpr int HANDSTRENGTH_DIM = 1;
    
    RLStateEncoder();
    
    /**
     * Encode a game state into a feature vector.
     * 
     * @param state JSON object or dict with game state fields:
     *   - hole_cards: List of card strings ["AH", "KS"]
     *   - community_cards: List of card strings
     *   - pot: int
     *   - current_bet: int
     *   - player_chips: int
     *   - player_bet: int
     *   - player_total_bet: int
     *   - stage: string ("Preflop", "Flop", etc.)
     *   - num_players: int
     *   - num_active: int
     *   - position: int
     *   - is_dealer: bool
     *   - is_small_blind: bool
     *   - is_big_blind: bool
     *   - big_blind: int
     *   - to_call: int
     *   - player_id: string
     *   - max_stack: int (optional, defaults to starting_chips)
     *   - starting_chips: int (optional, defaults to 1000)
     * 
     * @return Vector of FEATURE_DIM floats
     */
    std::vector<float> encodeState(const nlohmann::json& state);
    
    /**
     * Reset action history (call at start of new hand).
     */
    void resetHistory();
    
    /**
     * Add an action to the history for opponent modeling.
     */
    void addAction(const std::string& playerId, const std::string& actionType,
                   int amount, int pot, const std::string& stage);
    
    /**
     * Get the feature dimension (167).
     */
    static constexpr int getFeatureDim() { return FEATURE_DIM; }
    
private:
    ActionHistory actionHistory_;
    std::array<float, FEATURE_DIM> buffer_;  // Pre-allocated buffer
    
    // Card parsing
    static int parseRank(char c);
    static int parseSuit(char c);
    
    // Stage parsing
    static int parseStage(const std::string& stage);
    
    // Encode components
    void encodeCard(int idx, const std::string& card);
    void encodePotInfo(int idx, const nlohmann::json& state, float maxStack);
    void encodeStage(int idx, const std::string& stage);
    void encodePosition(int idx, const nlohmann::json& state);
    void encodeGameTheory(int idx, const nlohmann::json& state, float maxStack);
    void encodeOpponentFeatures(int idx, const std::string& playerId);
    float encodeHandStrength(const std::vector<std::string>& holeCards,
                            const std::vector<std::string>& communityCards);
};


/**
 * Map string action names to indices in the unified action space.
 */
class ActionSpace {
public:
    // Unified action space: fold, check, call, raise_10%, raise_25%, raise_33%, 
    // raise_50%, raise_75%, raise_100%, raise_150%, raise_200%, raise_300%, all_in
    static constexpr int NUM_ACTIONS = 13;
    
    static constexpr int FOLD = 0;
    static constexpr int CHECK = 1;
    static constexpr int CALL = 2;
    static constexpr int RAISE_10 = 3;
    static constexpr int RAISE_25 = 4;
    static constexpr int RAISE_33 = 5;
    static constexpr int RAISE_50 = 6;
    static constexpr int RAISE_75 = 7;
    static constexpr int RAISE_100 = 8;
    static constexpr int RAISE_150 = 9;
    static constexpr int RAISE_200 = 10;
    static constexpr int RAISE_300 = 11;
    static constexpr int ALL_IN = 12;
    
    /**
     * Get action index from action name.
     */
    static int actionToIndex(const std::string& actionName);
    
    /**
     * Get action name from index.
     */
    static std::string indexToAction(int index);
    
    /**
     * Get raise size as fraction of pot.
     * Returns 0 for non-raise actions.
     */
    static float getRaiseSize(int actionIndex);
    
    /**
     * Get raise size from action label (e.g., "raise_50%").
     */
    static float getRaiseSizeFromLabel(const std::string& label);
};





