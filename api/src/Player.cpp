#include "Player.h"

Player::Player(std::string_view playerId, std::string_view playerName, int startingChips)
    : id(playerId), name(playerName), chips(startingChips), bet(0), totalBet(0),
      state(State::WAITING), lastAction(Action::NONE), position(0),
      isDealer(false), isSmallBlind(false), isBigBlind(false) {}

void Player::dealHoleCards(const std::vector<Card>& cards) {
    holeCards = cards;
}

void Player::clearHoleCards() {
    holeCards.clear();
}

bool Player::placeBet(int amount) {
    if (amount > chips) {
        return false;
    }
    
    chips -= amount;
    bet += amount;
    totalBet += amount;
    
    if (chips == 0) {
        state = State::ALL_IN;
    }
    
    return true;
}

bool Player::postBlind(int amount) {
    int actualAmount = std::min(amount, chips);
    chips -= actualAmount;
    bet += actualAmount;
    totalBet += actualAmount;
    
    if (chips == 0) {
        state = State::ALL_IN;
    }
    
    return true;
}

void Player::fold() {
    state = State::FOLDED;
    lastAction = Action::FOLD;
}

bool Player::check(int currentBet) {
    // Checking is only allowed when player's bet equals the current bet
    if (bet != currentBet) {
        return false;
    }
    lastAction = Action::CHECK;
    return true;
}

bool Player::call(int amountToCall) {
    if (amountToCall > chips) {
        // All-in call
        totalBet += chips;
        bet += chips;
        chips = 0;
        state = State::ALL_IN;
        lastAction = Action::ALL_IN;
        return true;
    }
    
    chips -= amountToCall;
    bet += amountToCall;
    totalBet += amountToCall;
    lastAction = Action::CALL;
    return true;
}

bool Player::makeBet(int amount) {
    if (amount > chips) {
        return false;
    }
    
    chips -= amount;
    bet += amount;
    totalBet += amount;
    
    if (chips == 0) {
        state = State::ALL_IN;
        lastAction = Action::ALL_IN;
    } else {
        lastAction = Action::BET;
    }
    
    return true;
}

bool Player::raise(int totalAmount) {
    if (totalAmount > chips + bet) {
        return false;
    }
    
    int additionalChips = totalAmount - bet;
    chips -= additionalChips;
    bet = totalAmount;
    totalBet += additionalChips;
    
    if (chips == 0) {
        state = State::ALL_IN;
        lastAction = Action::ALL_IN;
    } else {
        lastAction = Action::RAISE;
    }
    
    return true;
}

void Player::goAllIn() {
    bet += chips;
    totalBet += chips;
    chips = 0;
    state = State::ALL_IN;
    lastAction = Action::ALL_IN;
}

void Player::winChips(int amount) {
    chips += amount;
}

void Player::resetBet() {
    bet = 0;
}

void Player::resetLastAction() {
    lastAction = Action::NONE;
}

void Player::resetForNewHand() {
    holeCards.clear();
    bet = 0;
    totalBet = 0;
    lastAction = Action::NONE;
    
    if (chips > 0) {
        state = State::ACTIVE;
    } else {
        state = State::OUT;
    }
    
    isDealer = false;
    isSmallBlind = false;
    isBigBlind = false;
}

bool Player::canAct() const {
    return state == State::ACTIVE && chips > 0;
}

bool Player::isInHand() const {
    return state == State::ACTIVE || state == State::ALL_IN;
}

std::string Player::getActionName() const {
    // Use lookup table for better performance
    static constexpr const char* actionNames[] = {
        "None", "Fold", "Check", "Call", "Bet", "Raise", "All-in"
    };
    
    const int idx = static_cast<int>(lastAction);
    if (idx >= 0 && idx < 7) {
        return actionNames[idx];
    }
    return "Unknown";
}

std::string Player::getStateName() const {
    // Use lookup table for better performance
    static constexpr const char* stateNames[] = {
        "Waiting", "Active", "Folded", "All-in", "Out"
    };
    
    const int idx = static_cast<int>(state);
    if (idx >= 0 && idx < 5) {
        return stateNames[idx];
    }
    return "Unknown";
}

Hand::EvaluatedHand Player::evaluateHand(const std::vector<Card>& communityCards) const {
    return Hand::evaluate(holeCards, communityCards);
}

