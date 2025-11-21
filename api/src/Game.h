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
#include "json.hpp"

using json = nlohmann::json;

// C++20 transparent hash for string_view compatibility
struct StringHash {
    using is_transparent = void;
    using hash_type = std::hash<std::string_view>;
    
    size_t operator()(std::string_view sv) const { return hash_type{}(sv); }
    size_t operator()(const std::string& s) const { return hash_type{}(s); }
    size_t operator()(const char* s) const { return hash_type{}(s); }
};

struct StringEqual {
    using is_transparent = void;
    
    bool operator()(std::string_view lhs, std::string_view rhs) const {
        return lhs == rhs;
    }
};

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
        std::vector<std::string> exactCards; // If provided, use these exact cards instead of shuffling
        
        GameConfig() 
            : smallBlind(10), bigBlind(20), startingChips(1000),
              minPlayers(2), maxPlayers(10), seed(0) {}
    };

private:
    std::vector<std::unique_ptr<Player>> players;
    // Fast O(1) player lookup by ID with C++20 heterogeneous lookup (no string copies)
    std::unordered_map<std::string, Player*, StringHash, StringEqual> playerLookup;
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
    std::vector<json> history;

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
     * @param id Player ID
     * @param name Player name
     * @param chips Starting chip count (0 means use config.startingChips)
     * @return true if player was added, false if already exists or table is full
     */
    [[nodiscard]] bool addPlayer(std::string_view id, std::string_view name, int chips = 0);
    
    /**
     * Removes a player from the game
     * @return true if player was removed, false if player not found
     */
    [[nodiscard]] bool removePlayer(std::string_view id);
    
    /**
     * Gets player by ID (O(1) lookup)
     * @param id Player ID to look up
     * @return Non-owning pointer to player, or nullptr if not found
     */
    [[nodiscard]] Player* getPlayer(std::string_view id);
    [[nodiscard]] const Player* getPlayer(std::string_view id) const;
    
    /**
     * Gets all players
     */
    [[nodiscard]] std::vector<Player*> getPlayers();
    [[nodiscard]] std::vector<const Player*> getPlayers() const;
    
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
     * @return Non-owning pointer to current player, or nullptr if invalid index
     */
    [[nodiscard]] Player* getCurrentPlayer();
    [[nodiscard]] const Player* getCurrentPlayer() const;
    
    /**
     * Gets stage name as string
     */
    std::string getStageName() const;
    
    /**
     * Advances the game automatically to the next stage
     * Deals community cards and updates game state deterministically
     * Returns true if game was advanced, false if game is complete or betting round not finished
     */
    [[nodiscard]] bool advanceGame();
    
    /**
     * Checks if the current betting round is complete
     * Returns true if all active players have acted and bets are matched
     */
    [[nodiscard]] bool isBettingRoundComplete() const;
    
    /**
     * Gets the full event history
     */
    [[nodiscard]] const std::vector<json>& getHistory() const noexcept { return history; }
    
    /**
     * Adds an event to the history (for event sourcing)
     */
    void recordEvent(const json& event);

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
     * Helper: Performs the stage transition and card dealing
     * Returns true if successfully advanced, false if at terminal state
     */
    bool performStageTransition();
    
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

