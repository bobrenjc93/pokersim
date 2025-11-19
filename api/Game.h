#ifndef GAME_H
#define GAME_H

#include "Card.h"
#include "Deck.h"
#include "Hand.h"
#include "Player.h"
#include "Pot.h"
#include <vector>
#include <memory>
#include <string>
#include <string_view>
#include <stdexcept>
#include <unordered_map>

/**
 * Main game orchestrator for Texas Hold'em poker.
 * Manages game flow, betting rounds, and player actions.
 */
class Game {
public:
    enum class Stage {
        WAITING,      // Waiting for players
        PREFLOP,      // Before flop
        FLOP,         // 3 community cards
        TURN,         // 4 community cards
        RIVER,        // 5 community cards
        SHOWDOWN,     // Revealing hands
        COMPLETE      // Hand complete
    };
    
    struct GameConfig {
        int smallBlind;
        int bigBlind;
        int startingChips;
        int minPlayers;
        int maxPlayers;
        unsigned int seed; // 0 means random seed
        
        GameConfig() 
            : smallBlind(10), bigBlind(20), startingChips(1000),
              minPlayers(2), maxPlayers(10), seed(0) {}
    };

private:
    std::vector<std::unique_ptr<Player>> players;
    std::unordered_map<std::string, Player*> playerLookup;  // Fast O(1) player lookup by ID
    Deck deck;
    Pot pot;
    std::vector<Card> communityCards;
    Stage stage;
    GameConfig config;
    int dealerPosition;
    int currentPlayerIndex;
    int lastRaiserIndex;
    unsigned int currentSeed;
    int handNumber;

public:
    Game(const GameConfig& cfg = GameConfig());
    
    // Getters
    Stage getStage() const noexcept { return stage; }
    const std::vector<Card>& getCommunityCards() const noexcept { return communityCards; }
    int getPotSize() const noexcept { return pot.getTotalPot(); }
    int getCurrentBet() const noexcept { return pot.getCurrentBet(); }
    int getDealerPosition() const noexcept { return dealerPosition; }
    int getHandNumber() const noexcept { return handNumber; }
    const GameConfig& getConfig() const noexcept { return config; }
    
    /**
     * Adds a player to the game
     */
    [[nodiscard]] bool addPlayer(std::string_view id, std::string_view name);
    
    /**
     * Removes a player from the game
     */
    [[nodiscard]] bool removePlayer(std::string_view id);
    
    /**
     * Gets player by ID
     */
    [[nodiscard]] Player* getPlayer(std::string_view id);
    
    /**
     * Gets all players
     */
    [[nodiscard]] std::vector<Player*> getPlayers();
    
    /**
     * Gets active players (not folded or out)
     */
    [[nodiscard]] std::vector<Player*> getActivePlayers();
    
    /**
     * Starts a new hand
     */
    [[nodiscard]] bool startHand();
    
    /**
     * Processes a player action
     */
    [[nodiscard]] bool processAction(std::string_view playerId, Player::Action action, int amount = 0);
    
    /**
     * Gets current player whose turn it is
     */
    [[nodiscard]] Player* getCurrentPlayer();
    
    /**
     * Gets stage name as string
     */
    std::string getStageName() const;

private:
    /**
     * Advances to next player who can act
     */
    void advanceToNextPlayer();
    
    /**
     * Checks if betting round is complete and advances stage
     */
    void checkBettingRoundComplete();
    
    /**
     * Advances to the next stage of the hand
     */
    void advanceStage();
    
    /**
     * Deals the flop (3 community cards)
     */
    void dealFlop();
    
    /**
     * Deals the turn (4th community card)
     */
    void dealTurn();
    
    /**
     * Deals the river (5th community card)
     */
    void dealRiver();
    
    /**
     * Ends the hand and distributes pots
     */
    void endHand();
};

#endif // GAME_H

