#ifndef JSON_SERIALIZER_H
#define JSON_SERIALIZER_H

#include <string>
#include "json.hpp"
#include "Player.h"
#include "Game.h"

using json = nlohmann::json;

/**
 * JsonSerializer - Utility class for converting between game objects and JSON
 * 
 * Provides static methods for serialization/deserialization of:
 * - Player::Action enum to/from strings
 * - Player objects to JSON
 * - Game objects to JSON
 */
class JsonSerializer {
public:
    /**
     * Converts a Player::Action enum to string
     */
    [[nodiscard]] static std::string actionToString(Player::Action action);
    
    /**
     * Converts a string to Player::Action enum
     * Uses static map for O(1) lookup
     * Takes string_view to avoid copies from literals and substrings
     */
    [[nodiscard]] static Player::Action stringToAction(std::string_view str);
    
    /**
     * Converts player state to JSON (includes hole cards)
     */
    [[nodiscard]] static json playerToJson(const Player* player);
    
    /**
     * Converts game state to JSON (full serialization)
     */
    [[nodiscard]] static json gameToJson(const Game& game);
};

#endif // JSON_SERIALIZER_H

