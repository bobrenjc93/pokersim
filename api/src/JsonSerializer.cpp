#include "JsonSerializer.h"
#include <unordered_map>

std::string JsonSerializer::actionToString(Player::Action action) {
    switch (action) {
        case Player::Action::NONE: return "none";
        case Player::Action::FOLD: return "fold";
        case Player::Action::CHECK: return "check";
        case Player::Action::CALL: return "call";
        case Player::Action::BET: return "bet";
        case Player::Action::RAISE: return "raise";
        case Player::Action::ALL_IN: return "all_in";
        default: return "unknown";
    }
}

Player::Action JsonSerializer::stringToAction(std::string_view str) {
    static const std::unordered_map<std::string_view, Player::Action> actionMap = {
        {"fold", Player::Action::FOLD},
        {"check", Player::Action::CHECK},
        {"call", Player::Action::CALL},
        {"bet", Player::Action::BET},
        {"raise", Player::Action::RAISE},
        {"all_in", Player::Action::ALL_IN}
    };
    
    auto it = actionMap.find(str);
    return (it != actionMap.end()) ? it->second : Player::Action::NONE;
}

json JsonSerializer::playerToJson(const Player* player) {
    // Build hole cards array (typically 2 cards)
    const auto& holeCards = player->getHoleCards();
    json holeCardsJson = json::array();
    for (const auto& card : holeCards) {
        holeCardsJson.push_back(card.toString());
    }
    
    // Use initializer list for single allocation
    return json{
        {"id", player->getId()},
        {"name", player->getName()},
        {"chips", player->getChips()},
        {"bet", player->getBet()},
        {"totalBet", player->getTotalBet()},
        {"state", player->getStateName()},
        {"lastAction", actionToString(player->getLastAction())},
        {"position", player->getPosition()},
        {"isDealer", player->getIsDealer()},
        {"isSmallBlind", player->getIsSmallBlind()},
        {"isBigBlind", player->getIsBigBlind()},
        {"canAct", player->canAct()},
        {"isInHand", player->isInHand()},
        {"holeCards", std::move(holeCardsJson)}
    };
}

json JsonSerializer::gameToJson(const Game& game) {
    // Build community cards array (max 5 cards: flop + turn + river)
    const auto& communityCards = game.getCommunityCards();
    json cardsJson = json::array();
    for (const auto& card : communityCards) {
        cardsJson.push_back(card.toString());
    }
    
    // Build players array
    const auto players = game.getPlayers();
    json playersJson = json::array();
    for (const auto* player : players) {
        playersJson.push_back(playerToJson(player));
    }
    
    // Get current player ID
    const Player* currentPlayer = game.getCurrentPlayer();
    json currentPlayerId = currentPlayer ? json(currentPlayer->getId()) : json(nullptr);
    
    // Get config
    const auto& config = game.getConfig();
    
    // Get history
    const auto& historyEvents = game.getHistory();
    json historyJson = json::array();
    for (const auto& event : historyEvents) {
        historyJson.push_back(event);
    }
    
    // Get action constraints
    json actionConstraints = game.getActionConstraints();
    
    // Use initializer list for single allocation
    return json{
        {"stage", game.getStageName()},
        {"pot", game.getPotSize()},
        {"currentBet", game.getCurrentBet()},
        {"minRaise", game.getMinRaise()},
        {"dealerPosition", game.getDealerPosition()},
        {"handNumber", game.getHandNumber()},
        {"config", {
            {"smallBlind", config.smallBlind},
            {"bigBlind", config.bigBlind},
            {"startingChips", config.startingChips},
            {"minPlayers", config.minPlayers},
            {"maxPlayers", config.maxPlayers},
            {"seed", config.seed}
        }},
        {"communityCards", std::move(cardsJson)},
        {"currentPlayerId", std::move(currentPlayerId)},
        {"players", std::move(playersJson)},
        {"history", std::move(historyJson)},
        {"actionConstraints", std::move(actionConstraints)}
    };
}
