#include "Game.h"

Game::Game(const GameConfig& cfg) 
    : stage(Stage::WAITING), config(cfg), dealerPosition(0), 
      currentPlayerIndex(0), lastRaiserIndex(-1), 
      currentSeed(cfg.seed), handNumber(0) {
    
    if (currentSeed == 0) {
        currentSeed = std::random_device{}();
    }
    deck = Deck(currentSeed);
}

bool Game::addPlayer(std::string_view id, std::string_view name) {
    if (players.size() >= static_cast<size_t>(config.maxPlayers)) {
        return false;
    }
    
    if (stage != Stage::WAITING) {
        return false;
    }
    
    // Check for duplicate ID using O(1) lookup
    if (playerLookup.count(std::string(id))) {
        return false;
    }
    
    auto player = std::make_unique<Player>(id, name, config.startingChips);
    player->setPosition(players.size());
    Player* playerPtr = player.get();
    playerLookup[player->getId()] = playerPtr;  // Store with actual string key
    players.push_back(std::move(player));
    
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
    // O(1) lookup (requires string copy in C++17; C++20 would support heterogeneous lookup)
    auto it = playerLookup.find(std::string(id));
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
    deck.shuffle(currentSeed + handNumber);
    
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
            if (player->getBet() == pot.getCurrentBet()) {
                success = player->check();
            }
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

void Game::checkBettingRoundComplete() {
    // Check if all players have acted and bets are equal
    bool allActed = true;
    int activePlayers = 0;
    
    for (size_t i = 0; i < players.size(); i++) {
        Player* player = players[i].get();
        
        if (player->isInHand()) {
            activePlayers++;
            
            // Check if player needs to act
            if (player->canAct()) {
                if (player->getBet() < pot.getCurrentBet()) {
                    allActed = false;
                    break;
                }
                
                // Check if they've had a chance to act
                if (static_cast<int>(i) == lastRaiserIndex) {
                    continue; // Raiser has acted
                }
                
                if (player->getLastAction() == Player::Action::NONE) {
                    allActed = false;
                    break;
                }
            }
        }
    }
    
    // If only one player left, hand is over
    if (activePlayers <= 1) {
        endHand();
        return;
    }
    
    if (allActed) {
        advanceStage();
    }
}

void Game::advanceStage() {
    // Collect bets into pot
    auto playerPtrs = getPlayers();
    pot.collectBets(playerPtrs);
    pot.startNewRound();
    
    switch (stage) {
        case Stage::PREFLOP:
            dealFlop();
            stage = Stage::FLOP;
            break;
            
        case Stage::FLOP:
            dealTurn();
            stage = Stage::TURN;
            break;
            
        case Stage::TURN:
            dealRiver();
            stage = Stage::RIVER;
            break;
            
        case Stage::RIVER:
            stage = Stage::SHOWDOWN;
            endHand();
            return;
            
        default:
            return;
    }
    
    // Reset to first player after dealer
    currentPlayerIndex = (dealerPosition + 1) % players.size();
    while (!players[currentPlayerIndex]->canAct() && 
           currentPlayerIndex != dealerPosition) {
        currentPlayerIndex = (currentPlayerIndex + 1) % players.size();
    }
    
    lastRaiserIndex = -1;
}

void Game::dealFlop() {
    deck.burn();
    for (int i = 0; i < 3; i++) {
        communityCards.push_back(deck.dealCard());
    }
}

void Game::dealTurn() {
    deck.burn();
    communityCards.push_back(deck.dealCard());
}

void Game::dealRiver() {
    deck.burn();
    communityCards.push_back(deck.dealCard());
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
        }
        pot.reset();
    } else {
        // Normal showdown - distribute pots to winners based on hand evaluation
        pot.distributePots(playerPtrs, communityCards);
    }
    
    stage = Stage::COMPLETE;
}

