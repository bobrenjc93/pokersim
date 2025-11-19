#ifndef PLAYER_H
#define PLAYER_H

#include "Card.h"
#include "Hand.h"
#include <vector>
#include <string>
#include <string_view>

/**
 * Represents a poker player with chips, cards, and game state.
 */
class Player {
public:
    enum class Action {
        NONE,
        FOLD,
        CHECK,
        CALL,
        BET,
        RAISE,
        ALL_IN
    };
    
    enum class State {
        WAITING,      // Waiting for game to start
        ACTIVE,       // Active in current hand
        FOLDED,       // Folded this hand
        ALL_IN,       // All-in
        OUT           // No chips left
    };

private:
    std::string id;
    std::string name;
    int chips;
    int bet;              // Current bet in this round
    int totalBet;         // Total bet in this hand (across all rounds)
    std::vector<Card> holeCards;
    State state;
    Action lastAction;
    int position;         // Seat position (0-indexed)
    bool isDealer;
    bool isSmallBlind;
    bool isBigBlind;

public:
    Player(std::string_view playerId, std::string_view playerName, int startingChips);
    
    // Getters
    const std::string& getId() const noexcept { return id; }
    const std::string& getName() const noexcept { return name; }
    int getChips() const noexcept { return chips; }
    int getBet() const noexcept { return bet; }
    int getTotalBet() const noexcept { return totalBet; }
    const std::vector<Card>& getHoleCards() const noexcept { return holeCards; }
    State getState() const noexcept { return state; }
    Action getLastAction() const noexcept { return lastAction; }
    int getPosition() const noexcept { return position; }
    bool getIsDealer() const noexcept { return isDealer; }
    bool getIsSmallBlind() const noexcept { return isSmallBlind; }
    bool getIsBigBlind() const noexcept { return isBigBlind; }
    
    // Setters
    void setPosition(int pos) { position = pos; }
    void setDealer(bool dealer) { isDealer = dealer; }
    void setSmallBlind(bool sb) { isSmallBlind = sb; }
    void setBigBlind(bool bb) { isBigBlind = bb; }
    void setState(State s) { state = s; }
    
    /**
     * Deals hole cards to the player
     */
    void dealHoleCards(const std::vector<Card>& cards);
    
    /**
     * Clears hole cards at end of hand
     */
    void clearHoleCards();
    
    /**
     * Places a bet (adds to current round bet)
     */
    [[nodiscard]] bool placeBet(int amount);
    
    /**
     * Posts blind (small or big blind)
     */
    [[nodiscard]] bool postBlind(int amount);
    
    /**
     * Performs a fold action
     */
    void fold();
    
    /**
     * Performs a check action
     */
    [[nodiscard]] bool check();
    
    /**
     * Calls the current bet
     */
    [[nodiscard]] bool call(int amountToCall);
    
    /**
     * Makes a bet
     */
    [[nodiscard]] bool makeBet(int amount);
    
    /**
     * Raises to a total bet amount
     */
    [[nodiscard]] bool raise(int totalAmount);
    
    /**
     * Goes all-in
     */
    void goAllIn();
    
    /**
     * Wins chips from pot
     */
    void winChips(int amount);
    
    /**
     * Resets bet at start of new betting round
     */
    void resetBet();
    
    /**
     * Resets player for new hand
     */
    void resetForNewHand();
    
    /**
     * Checks if player can act
     */
    [[nodiscard]] bool canAct() const;
    
    /**
     * Checks if player is in the hand
     */
    [[nodiscard]] bool isInHand() const;
    
    /**
     * Gets action name as string
     */
    [[nodiscard]] std::string getActionName() const;
    
    /**
     * Gets state name as string
     */
    [[nodiscard]] std::string getStateName() const;
    
    /**
     * Evaluates player's best hand given community cards
     */
    [[nodiscard]] Hand::EvaluatedHand evaluateHand(const std::vector<Card>& communityCards) const;
};

#endif // PLAYER_H

