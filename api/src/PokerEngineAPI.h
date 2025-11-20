#ifndef POKER_ENGINE_API_H
#define POKER_ENGINE_API_H

#include <string>
#include "json.hpp"
#include "Game.h"

using json = nlohmann::json;

/**
 * Truly Stateless Poker Engine API
 * 
 * The server maintains NO state between requests. No game IDs stored!
 * 
 * Each request contains:
 *   - config: game configuration with seed (for deterministic randomness)
 *   - history: array of all actions taken so far (optional)
 *   - action: new action to apply (optional)
 * 
 * The server:
 *   1. Creates a fresh game with the given seed and config
 *   2. Replays all history actions to reconstruct state
 *   3. Applies the new action (if provided)
 *   4. If no action provided, automatically advances game state (starts hand, deals cards, determines winners)
 *   5. Returns the resulting game state
 * 
 * This approach is fully stateless and deterministic:
 *   - Same seed + same history = same game state
 *   - Client is responsible for maintaining history
 *   - Hand starting and card dealing are implicit when appropriate
 */
class PokerEngineAPI {
public:
    /**
     * Process a stateless API request and return JSON response
     * @param request JSON request containing config, history, and optional action
     * @return JSON response with success flag and game state or error
     */
    [[nodiscard]] json processRequest(const json& request);
    
private:
    /**
     * Applies a single action to the game
     */
    bool applyAction(Game* game, const json& actionJson, std::string& errorMsg);
    
    /**
     * Creates an error response JSON object
     */
    static json errorResponse(const std::string& message);
};

#endif // POKER_ENGINE_API_H

