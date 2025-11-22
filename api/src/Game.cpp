#include "Game.h"
#include "JsonSerializer.h"

Game::Game(const GameConfig& cfg) 
    : stage(Stage::WAITING), config(cfg), dealerPosition(0), 
      currentPlayerIndex(0), lastRaiserIndex(-1), 
      currentSeed(cfg.seed), handNumber(0) {
    
    if (currentSeed == 0) {
        currentSeed = std::random_device{}();
    }
    deck = Deck(currentSeed);
}

void Game::recordEvent(const json& event) {
    history.push_back(event);
}

bool Game::addPlayer(std::string_view id, std::string_view name, int chips) {
    if (players.size() >= static_cast<size_t>(config.maxPlayers)) {
        return false;
    }
    
    if (stage != Stage::WAITING) {
        return false;
    }
    
    // Check for duplicate ID using O(1) lookup (C++20 heterogeneous lookup)
    if (playerLookup.contains(id)) {
        return false;
    }
    
    // Use provided chips or fall back to config default
    int startingChips = (chips > 0) ? chips : config.startingChips;
    
    auto player = std::make_unique<Player>(id, name, startingChips);
    player->setPosition(players.size());
    Player* playerPtr = player.get();
    playerLookup[player->getId()] = playerPtr;  // Store with actual string key
    players.push_back(std::move(player));
    
    // Record the event
    json event = {
        {"type", "addPlayer"},
        {"playerId", std::string(id)},
        {"playerName", std::string(name)}
    };
    if (chips > 0) {
        event["chips"] = chips;
    }
    recordEvent(event);
    
    return true;
}

bool Game::removePlayer(std::string_view id) {
    if (stage != Stage::WAITING) {
        return false;
    }
    
    auto it = std::find_if(players.begin(), players.end(),
        [id](const auto& p) { return p->getId() == id; });
    
    if (it != players.end()) {
        playerLookup.erase((*it)->getId());
        players.erase(it);
        
        // Update positions and rebuild lookup map
        playerLookup.clear();
        for (size_t i = 0; i < players.size(); i++) {
            players[i]->setPosition(i);
            playerLookup[players[i]->getId()] = players[i].get();
        }
        
        return true;
    }
    
    return false;
}

Player* Game::getPlayer(std::string_view id) {
    // O(1) lookup with C++20 heterogeneous lookup (no string copy needed!)
    auto it = playerLookup.find(id);
    return (it != playerLookup.end()) ? it->second : nullptr;
}

const Player* Game::getPlayer(std::string_view id) const {
    // O(1) lookup with C++20 heterogeneous lookup (no string copy needed!)
    auto it = playerLookup.find(id);
    return (it != playerLookup.end()) ? it->second : nullptr;
}

std::vector<Player*> Game::getPlayers() {
    std::vector<Player*> result;
    result.reserve(players.size());  // Avoid reallocation
    for (auto& player : players) {
        result.push_back(player.get());
    }
    return result;
}

std::vector<const Player*> Game::getPlayers() const {
    std::vector<const Player*> result;
    result.reserve(players.size());  // Avoid reallocation
    for (const auto& player : players) {
        result.push_back(player.get());
    }
    return result;
}

std::vector<Player*> Game::getActivePlayers() {
    std::vector<Player*> active;
    active.reserve(players.size());  // Worst case: all players active
    for (auto& player : players) {
        if (player->isInHand()) {
            active.push_back(player.get());
        }
    }
    return active;
}

bool Game::startHand() {
    if (players.size() < static_cast<size_t>(config.minPlayers)) {
        return false;
    }
    
    // Reset for new hand
    handNumber++;
    communityCards.clear();
    pot.reset();
    deck.reset();
    
    // Use exact cards if provided, otherwise shuffle with seed
    if (!config.exactCards.empty()) {
        deck.setExactOrder(config.exactCards);
    } else {
        deck.shuffle(currentSeed + handNumber);
    }
    
    // Reset all players
    for (auto& player : players) {
        player->resetForNewHand();
    }
    
    // Move dealer button
    dealerPosition = (dealerPosition + 1) % players.size();
    
    // Assign blinds
    int sbPos = (dealerPosition + 1) % players.size();
    int bbPos = (dealerPosition + 2) % players.size();
    
    players[dealerPosition]->setDealer(true);
    players[sbPos]->setSmallBlind(true);
    players[bbPos]->setBigBlind(true);
    
    // Post blinds (always succeeds, but check for consistency)
    (void)players[sbPos]->postBlind(config.smallBlind);
    (void)players[bbPos]->postBlind(config.bigBlind);
    
    pot.setCurrentBet(config.bigBlind);
    pot.setMinRaise(config.bigBlind);
    
    // Deal hole cards
    for (auto& player : players) {
        std::vector<Card> holeCards = deck.dealCards(2);
        player->dealHoleCards(holeCards);
        
        // Record the event
        json cardsJson = json::array();
        for (const auto& card : holeCards) {
            cardsJson.push_back(card.toString());
        }
        recordEvent({
            {"type", "dealHoleCards"},
            {"playerId", player->getId()},
            {"cards", cardsJson}
        });
    }
    
    stage = Stage::PREFLOP;
    currentPlayerIndex = (bbPos + 1) % players.size();
    lastRaiserIndex = bbPos; // Big blind is last to act preflop
    
    return true;
}

bool Game::processAction(std::string_view playerId, Player::Action action, int amount) {
    Player* player = getPlayer(playerId);
    if (!player || !player->canAct()) {
        return false;
    }
    
    if (players[currentPlayerIndex]->getId() != playerId) {
        return false; // Not this player's turn
    }
    
    bool success = false;
    
    switch (action) {
        case Player::Action::FOLD:
            player->fold();
            success = true;
            break;
            
        case Player::Action::CHECK:
            success = player->check(pot.getCurrentBet());
            break;
            
        case Player::Action::CALL: {
            int toCall = pot.getCurrentBet() - player->getBet();
            success = player->call(toCall);
            break;
        }
            
        case Player::Action::BET:
            if (pot.getCurrentBet() == 0 && amount >= config.bigBlind) {
                success = player->makeBet(amount);
                if (success) {
                    pot.updateBet(amount, 0);
                    lastRaiserIndex = currentPlayerIndex;
                }
            }
            break;
            
        case Player::Action::RAISE: {
            int totalBet = player->getBet() + amount;
            if (totalBet > pot.getCurrentBet() && 
                amount >= pot.getMinRaise()) {
                success = player->raise(totalBet);
                if (success) {
                    pot.updateBet(totalBet, pot.getCurrentBet());
                    lastRaiserIndex = currentPlayerIndex;
                }
            }
            break;
        }
            
        case Player::Action::ALL_IN:
            player->goAllIn();
            if (player->getBet() > pot.getCurrentBet()) {
                pot.updateBet(player->getBet(), pot.getCurrentBet());
                lastRaiserIndex = currentPlayerIndex;
            }
            success = true;
            break;
            
        default:
            break;
    }
    
    if (success) {
        // Record the event
        recordEvent({
            {"type", "playerAction"},
            {"playerId", std::string(playerId)},
            {"action", JsonSerializer::actionToString(action)},
            {"amount", amount}
        });
        
        advanceToNextPlayer();
        checkBettingRoundComplete();
    }
    
    return success;
}

Player* Game::getCurrentPlayer() {
    if (currentPlayerIndex >= 0 && 
        currentPlayerIndex < static_cast<int>(players.size())) {
        return players[currentPlayerIndex].get();
    }
    return nullptr;
}

const Player* Game::getCurrentPlayer() const {
    if (currentPlayerIndex >= 0 && 
        currentPlayerIndex < static_cast<int>(players.size())) {
        return players[currentPlayerIndex].get();
    }
    return nullptr;
}

std::string Game::getStageName() const {
    // Compile-time lookup table with bounds safety
    static constexpr const char* const stageNames[] = {
        "Waiting", "Preflop", "Flop", "Turn", "River", "Showdown", "Complete"
    };
    static constexpr size_t nameCount = sizeof(stageNames) / sizeof(stageNames[0]);
    
    const auto idx = static_cast<size_t>(stage);
    if (idx < nameCount) {
        return stageNames[idx];
    }
    return "Unknown";
}

void Game::advanceToNextPlayer() {
    int startIndex = currentPlayerIndex;
    
    do {
        currentPlayerIndex = (currentPlayerIndex + 1) % players.size();
        
        if (players[currentPlayerIndex]->canAct()) {
            return;
        }
        
    } while (currentPlayerIndex != startIndex);
}

bool Game::isBettingRoundComplete() const {
    // Two-pass algorithm for clarity and performance:
    // Pass 1: Quick count of active players (early exit if ≤1)
    // Pass 2: Check if betting round is complete
    // 
    // A round is complete when:
    // 1. All active players have acted (lastAction != NONE), AND
    // 2. All active players have matching bets (or are all-in)
    // 
    // Special case: The player who raised last (lastRaiserIndex) has completed
    // the round when action returns to them and they've had a chance to act again.
    
    // Pass 1: Count active players (fast path for fold-outs)
    int activePlayers = 0;
    for (const auto& player : players) {
        if (player->isInHand()) {
            activePlayers++;
            if (activePlayers > 1) {
                break; // Early exit - we know there are at least 2 players
            }
        }
    }
    
    // If only one player left, hand is over (everyone else folded)
    if (activePlayers <= 1) {
        return true; // Betting is "complete" since hand should end
    }
    
    // Pass 2: Check if all players have acted and bets are equal
    for (size_t i = 0; i < players.size(); i++) {
        const Player* player = players[i].get();
        
        if (player->isInHand() && player->canAct()) {
            // Condition 1: Check if bets are equal
            if (player->getBet() < pot.getCurrentBet()) {
                return false; // Round not complete
            }
            
            // Condition 2: Check if player has acted
            // Special handling for the last raiser: they've "completed" the round
            // when action returns to them after their raise. Skip the NONE check for them.
            if (static_cast<int>(i) == lastRaiserIndex &&
                player->getLastAction() != Player::Action::NONE) {
                continue; // Last raiser has acted, they're done
            }
            
            // For everyone else, check if they've acted at all this round
            if (player->getLastAction() == Player::Action::NONE) {
                return false; // Round not complete
            }
        }
    }
    
    return true; // All conditions met - betting round is complete
}

void Game::checkBettingRoundComplete() {
    // Check if betting is done (using the shared logic)
    if (!isBettingRoundComplete()) {
        return; // Not done yet
    }
    
    // Count active players to handle fold-outs
    int activePlayers = 0;
    for (const auto& player : players) {
        if (player->isInHand()) {
            activePlayers++;
            if (activePlayers > 1) {
                break;
            }
        }
    }
    
    // If only one player left, hand is over (everyone else folded)
    if (activePlayers <= 1) {
        endHand();
        return;
    }
    
    // Otherwise advance to next stage
    advanceStage();
}

bool Game::performStageTransition() {
    // Performs the actual stage transition and card dealing
    // Returns true if successfully transitioned, false if at terminal state
    switch (stage) {
        case Stage::PREFLOP:
            dealFlop();
            stage = Stage::FLOP;
            return true;
            
        case Stage::FLOP:
            dealTurn();
            stage = Stage::TURN;
            return true;
            
        case Stage::TURN:
            dealRiver();
            stage = Stage::RIVER;
            return true;
            
        case Stage::RIVER:
            stage = Stage::SHOWDOWN;
            endHand();
            return true;
            
        default:
            return false;
    }
}

void Game::advanceStage() {
    // Collect bets into pot
    auto playerPtrs = getPlayers();
    pot.collectBets(playerPtrs);
    pot.startNewRound();
    
    // Perform stage transition
    if (!performStageTransition()) {
        return;
    }
    
    // If we just ended the hand (reached SHOWDOWN), no need to reset player state
    if (stage == Stage::SHOWDOWN) {
        return;
    }
    
    // Reset to first player after dealer for new betting round
    currentPlayerIndex = (dealerPosition + 1) % players.size();
    while (!players[currentPlayerIndex]->canAct() && 
           currentPlayerIndex != dealerPosition) {
        currentPlayerIndex = (currentPlayerIndex + 1) % players.size();
    }
    
    lastRaiserIndex = -1;
    
    // Reset lastAction for all active players for the new betting round
    for (auto& player : players) {
        if (player->isInHand() && player->canAct()) {
            player->resetLastAction();
        }
    }
}

void Game::dealFlop() {
    deck.burn();
    json cardsJson = json::array();
    for (int i = 0; i < 3; i++) {
        Card card = deck.dealCard();
        communityCards.push_back(card);
        cardsJson.push_back(card.toString());
    }
    
    // Record the event
    recordEvent({
        {"type", "dealFlop"},
        {"cards", cardsJson}
    });
}

void Game::dealTurn() {
    deck.burn();
    Card card = deck.dealCard();
    communityCards.push_back(card);
    
    // Record the event
    recordEvent({
        {"type", "dealTurn"},
        {"card", card.toString()}
    });
}

void Game::dealRiver() {
    deck.burn();
    Card card = deck.dealCard();
    communityCards.push_back(card);
    
    // Record the event
    recordEvent({
        {"type", "dealRiver"},
        {"card", card.toString()}
    });
}

void Game::endHand() {
    // Collect any remaining bets
    auto playerPtrs = getPlayers();
    pot.collectBets(playerPtrs);
    
    // Check if only one player remains (everyone else folded)
    auto activePlayers = getActivePlayers();
    if (activePlayers.size() == 1) {
        // Award entire pot to remaining player without showdown
        int totalPot = pot.getTotalPot();
        if (totalPot > 0) {
            activePlayers[0]->winChips(totalPot);
            
            // Add win without showdown to history
            history.push_back({
                {"type", "handResult"},
                {"winnerIds", json::array({activePlayers[0]->getId()})},
                {"amountWon", totalPot},
                {"showdown", false}
            });
        }
        pot.reset();
        stage = Stage::COMPLETE;
    } else {
        // Normal showdown - distribute pots to winners based on hand evaluation
        auto showdownResults = pot.distributePots(playerPtrs, communityCards);
        
        // Add showdown results to history
        json playersJson = json::array();
        for (const auto& result : showdownResults) {
            playersJson.push_back({
                {"playerId", result.playerId},
                {"handRanking", result.handRanking},
                {"bestFive", result.bestFive},
                {"amountWon", result.amountWon}
            });
        }
        
        history.push_back({
            {"type", "showdown"},
            {"players", playersJson}
        });
        
        // Stage handling: If we reached here from advanceGame() after the River,
        // stage is already SHOWDOWN (set by advanceGame). Keep it as SHOWDOWN.
        // If we reached here from processAction() when a player folds mid-hand,
        // stage might be PREFLOP/FLOP/TURN/RIVER. Set it to COMPLETE in that case.
        if (stage != Stage::SHOWDOWN) {
            stage = Stage::COMPLETE;
        }
        // If stage == SHOWDOWN, leave it as SHOWDOWN to indicate hand went to showdown
    }
}

bool Game::advanceGame() {
    // Can't advance if waiting for players or already complete
    if (stage == Stage::WAITING || stage == Stage::COMPLETE) {
        return false;
    }
    
    // If at showdown, end the hand and move to complete
    if (stage == Stage::SHOWDOWN) {
        stage = Stage::COMPLETE;
        return true;
    }
    
    // Check if betting round is complete before allowing advancement
    // This prevents skipping player actions during PREFLOP, FLOP, TURN, or RIVER
    if (!isBettingRoundComplete()) {
        return false; // Cannot advance - betting round not finished
    }
    
    // Collect any bets from current round
    auto playerPtrs = getPlayers();
    pot.collectBets(playerPtrs);
    pot.startNewRound();
    
    // Perform stage transition using shared logic
    return performStageTransition();
}

json Game::getActionConstraints() const {
    json constraints;
    
    // Get current player
    const Player* player = getCurrentPlayer();
    if (!player || !player->canAct()) {
        // No current player or player can't act
        constraints["canAct"] = false;
        constraints["legalActions"] = json::array();
        return constraints;
    }
    
    constraints["canAct"] = true;
    
    int currentBet = pot.getCurrentBet();
    int playerBet = player->getBet();
    int playerChips = player->getChips();
    int minRaise = pot.getMinRaise();
    
    // Calculate key amounts
    int toCall = currentBet - playerBet;
    int minRaiseTotal = toCall + minRaise; // Total amount needed to raise (call + min raise)
    
    // Determine legal actions
    json legalActions = json::array();
    
    // Can always fold
    legalActions.push_back("fold");
    
    // Check or call
    if (toCall == 0) {
        legalActions.push_back("check");
        // Can bet ONLY if no current bet exists in the round AND have enough for min bet
        if (playerChips >= config.bigBlind && currentBet == 0) {
            legalActions.push_back("bet");
        }
    } else {
        // There's a bet to call
        if (playerChips >= toCall) {
            legalActions.push_back("call");
            // Can raise if have enough chips beyond call
            if (playerChips > toCall) {
                legalActions.push_back("raise");
            }
        }
    }
    
    // Can always go all-in if have chips
    if (playerChips > 0) {
        legalActions.push_back("all_in");
    }
    
    constraints["legalActions"] = legalActions;
    
    // Add amount constraints
    constraints["toCall"] = toCall;
    constraints["minBet"] = config.bigBlind;
    constraints["minRaiseTotal"] = minRaiseTotal; // Total amount to add for minimum raise
    constraints["playerChips"] = playerChips;
    constraints["playerBet"] = playerBet;
    constraints["currentBet"] = currentBet;
    constraints["minRaise"] = minRaise; // Minimum raise increment
    
    return constraints;
}

