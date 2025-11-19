#include "PokerEngineAPI.h"
#include "JsonSerializer.h"
#include <memory>

json PokerEngineAPI::processRequest(const json& request) {
    try {
        // Extract configuration with seed
        Game::GameConfig config;
        if (request.contains("config")) {
            const auto& configJson = request["config"];
            config.smallBlind = configJson.value("smallBlind", 10);
            config.bigBlind = configJson.value("bigBlind", 20);
            config.startingChips = configJson.value("startingChips", 1000);
            config.minPlayers = configJson.value("minPlayers", 2);
            config.maxPlayers = configJson.value("maxPlayers", 10);
            config.seed = configJson.value("seed", 0);
            
            // Extract exactCards if provided
            if (configJson.contains("exactCards") && configJson["exactCards"].is_array()) {
                for (const auto& card : configJson["exactCards"]) {
                    if (card.is_string()) {
                        config.exactCards.push_back(card.get<std::string>());
                    }
                }
            }
        } else {
            return errorResponse("Must provide config with seed");
        }
        
        // Create a fresh game with the seed
        auto game = std::make_unique<Game>(config);
        
        // Replay history to reconstruct state
        // The game will automatically record all events (including card dealing)
        if (request.contains("history") && request["history"].is_array()) {
            for (const auto& actionJson : request["history"]) {
                std::string errorMsg;
                // Skip card dealing events - they will be regenerated automatically
                std::string eventType = actionJson.value("type", "");
                if (eventType == "dealHoleCards" || eventType == "dealFlop" || 
                    eventType == "dealTurn" || eventType == "dealRiver") {
                    continue;  // Skip these, they'll be regenerated deterministically
                }
                
                if (!applyAction(game.get(), actionJson, errorMsg)) {
                    return errorResponse("Failed to replay history: " + errorMsg);
                }
                
                // After applying action, try implicit advancement
                // For addPlayer: start hand when enough players
                // For playerAction: advance to next stage when betting complete
                Game::Stage stage = game->getStage();
                if (stage == Game::Stage::WAITING) {
                    // Try to auto-start hand if we have enough players
                    (void)game->startHand();
                } else if (stage != Game::Stage::COMPLETE && stage != Game::Stage::SHOWDOWN) {
                    // Try to advance to next stage if betting is complete
                    (void)game->advanceGame();
                }
            }
        }
        
        // Automatic Advancement: Always advance the game after replaying history
        // 
        // Design rationale:
        // - Makes API fully deterministic: seed + history uniquely determines state
        // - No manual actions allowed - everything must go through history
        // - Stateless and simple: just send seed + history and get next state
        // 
        // Behavior:
        // - If WAITING + enough players → auto-start hand
        // - If hand in progress and betting complete → auto-advance: PREFLOP→FLOP→TURN→RIVER→SHOWDOWN→COMPLETE
        // - If betting not complete → just return current state (waiting for action in history)
        // - Keep advancing until can't advance anymore (useful for all-in situations)
        
        Game::Stage stage = game->getStage();
        
        // If in WAITING stage with enough players, auto-start the hand
        if (stage == Game::Stage::WAITING) {
            if (game->startHand()) {
                // Hand started successfully, nothing more to do
            } else {
                // Not enough players yet, that's okay - just return current state
            }
        }
        
        // Keep advancing until we can't anymore (for all-in situations)
        // This ensures we reach showdown when all players are all-in
        while (game->advanceGame()) {
            // Continue advancing
        }
        
        // Return the resulting game state
        json response;
        response["success"] = true;
        response["gameState"] = JsonSerializer::gameToJson(*game);
        return response;
        
    } catch (const std::exception& e) {
        return errorResponse(std::string("Exception: ") + e.what());
    }
}

bool PokerEngineAPI::applyAction(Game* game, const json& actionJson, std::string& errorMsg) {
    std::string type = actionJson.value("type", "");
    
    if (type == "addPlayer") {
        std::string playerId = actionJson.value("playerId", "");
        std::string playerName = actionJson.value("playerName", "");
        int chips = actionJson.value("chips", 0);  // 0 means use config default
        if (playerId.empty() || playerName.empty()) {
            errorMsg = "addPlayer: missing playerId or playerName";
            return false;
        }
        if (!game->addPlayer(playerId, playerName, chips)) {
            errorMsg = "addPlayer: failed to add player " + playerId;
            return false;
        }
        return true;
        
    } else if (type == "playerAction") {
        std::string playerId = actionJson.value("playerId", "");
        std::string actionStr = actionJson.value("action", "");
        int amount = actionJson.value("amount", 0);
        
        if (playerId.empty() || actionStr.empty()) {
            errorMsg = "playerAction: missing playerId or action";
            return false;
        }
        
        Player::Action action = JsonSerializer::stringToAction(actionStr);
        if (action == Player::Action::NONE && actionStr != "none") {
            errorMsg = "playerAction: invalid action '" + actionStr + "'";
            return false;
        }
        
        // Get current player for better error message
        Player* currentPlayer = game->getCurrentPlayer();
        std::string currentPlayerId = currentPlayer ? currentPlayer->getId() : "none";
        
        if (!game->processAction(playerId, action, amount)) {
            errorMsg = "playerAction: " + playerId + " cannot " + actionStr + 
                      " (current player is " + currentPlayerId + 
                      ", stage=" + game->getStageName() + ")";
            return false;
        }
        return true;
        
    } else if (type == "advance" || type == "next") {
        errorMsg = "Action type '" + type + "' is deprecated. Remove the 'action' field to trigger implicit advancement.";
        return false;
    } else if (type.empty()) {
        // Empty action type is allowed (just return current state)
        return true;
    }
    
    // Unknown action type
    errorMsg = "Unknown action type: " + type;
    return false;
}

json PokerEngineAPI::errorResponse(const std::string& message) {
    json response;
    response["success"] = false;
    response["error"] = message;
    return response;
}

