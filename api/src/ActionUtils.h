#pragma once

#include <string>
#include <vector>
#include <utility>
#include <array>
#include "json.hpp"

/**
 * Utility functions for action conversion, legal action masking, and state extraction.
 * 
 * These functions are used in RL training for:
 * - Converting model action labels to game actions
 * - Creating legal action masks for the policy network
 * - Extracting player state from raw game state
 */
namespace ActionUtils {

/**
 * Number of actions in the unified action space.
 */
constexpr int NUM_ACTIONS = 13;

/**
 * Action indices in the unified action space.
 */
enum ActionIndex {
    FOLD = 0,
    CHECK = 1,
    CALL = 2,
    RAISE_10 = 3,
    RAISE_25 = 4,
    RAISE_33 = 5,
    RAISE_50 = 6,
    RAISE_75 = 7,
    RAISE_100 = 8,
    RAISE_150 = 9,
    RAISE_200 = 10,
    RAISE_300 = 11,
    ALL_IN = 12
};

/**
 * Result of action conversion.
 */
struct ConvertedAction {
    std::string actionType;  // "fold", "check", "call", "bet", "raise", "all_in"
    int amount;              // Bet/raise amount (0 for fold/check/call/all_in)
};

/**
 * Convert an action label to (action_type, amount).
 * 
 * Unified raise_X% actions are converted to either 'bet' or 'raise' based on
 * game context (whether there's already a bet to face).
 * 
 * @param actionLabel Action label (e.g., 'raise_50%', 'call', 'fold')
 * @param state Game state JSON with keys:
 *   - player_chips: int
 *   - pot: int
 *   - player_bet: int
 *   - current_bet: int
 *   - min_bet: int (or big_blind)
 *   - min_raise_total: int (or big_blind)
 * 
 * @return ConvertedAction with action_type and amount
 */
ConvertedAction convertActionLabel(const std::string& actionLabel, 
                                   const nlohmann::json& state);

/**
 * Create a boolean mask for legal actions.
 * 
 * With the unified action space, 'bet' and 'raise' both enable the same raise_X% actions.
 * The convertActionLabel function handles converting to the correct game action.
 * 
 * @param legalActions Vector of legal action strings (e.g., ["fold", "call", "raise"])
 * @return Array of NUM_ACTIONS bools (true = legal, false = illegal)
 */
std::array<bool, NUM_ACTIONS> createLegalActionsMask(
    const std::vector<std::string>& legalActions);

/**
 * Create a legal actions mask as a vector of floats (0.0 or 1.0).
 * This format is convenient for direct use with PyTorch tensors.
 * 
 * @param legalActions Vector of legal action strings
 * @return Vector of NUM_ACTIONS floats (1.0 = legal, 0.0 = illegal)
 */
std::vector<float> createLegalActionsMaskFloat(
    const std::vector<std::string>& legalActions);

/**
 * Extract state for a specific player from the raw API game state.
 * 
 * @param gameState Raw game state from C++ API (JSON)
 * @param playerId Player ID to extract state for
 * @return JSON object with extracted state features, or empty object if player not found
 * 
 * Output keys:
 *   - player_id: string
 *   - hole_cards: array of strings
 *   - community_cards: array of strings
 *   - pot: int
 *   - current_bet: int
 *   - player_chips: int
 *   - player_bet: int
 *   - player_total_bet: int
 *   - stage: string
 *   - num_players: int
 *   - num_active: int
 *   - position: int
 *   - is_dealer: bool
 *   - is_small_blind: bool
 *   - is_big_blind: bool
 *   - big_blind: int
 *   - small_blind: int
 *   - starting_chips: int
 *   - max_stack: int
 *   - to_call: int
 *   - min_bet: int
 *   - min_raise_total: int
 */
nlohmann::json extractState(const nlohmann::json& gameState, 
                            const std::string& playerId);

/**
 * Get action name from index.
 */
std::string getActionName(int actionIndex);

/**
 * Get action index from name.
 * Returns -1 if action name is not found.
 */
int getActionIndex(const std::string& actionName);

/**
 * Get raise size as fraction of pot for a given action index.
 * Returns 0.0 for non-raise actions.
 */
float getRaiseSize(int actionIndex);

} // namespace ActionUtils





