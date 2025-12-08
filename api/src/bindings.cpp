#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "PokerEngineAPI.h"
#include "Game.h"
#include "JsonSerializer.h"
#include "json.hpp"
#include "HandStrength.h"
#include "RLStateEncoder.h"
#include "ActionUtils.h"
#include "HeuristicAgents.h"

namespace py = pybind11;

// Helper to convert nlohmann::json to py::object
py::object json_to_py(const nlohmann::json& j) {
    if (j.is_null()) {
        return py::none();
    } else if (j.is_boolean()) {
        return py::bool_(j.get<bool>());
    } else if (j.is_number_integer()) {
        // nlohmann::json::number_integer_t is usually long long
        return py::int_(j.get<nlohmann::json::number_integer_t>());
    } else if (j.is_number_float()) {
        return py::float_(j.get<double>());
    } else if (j.is_string()) {
        return py::str(j.get<std::string>());
    } else if (j.is_array()) {
        py::list l;
        for (const auto& item : j) {
            l.append(json_to_py(item));
        }
        return l;
    } else if (j.is_object()) {
        py::dict d;
        for (auto it = j.begin(); it != j.end(); ++it) {
            d[py::str(it.key())] = json_to_py(it.value());
        }
        return d;
    }
    return py::none();
}

// Helper to expose Game state as JSON string
std::string get_game_state_json(const Game& game) {
    return JsonSerializer::gameToJson(game).dump();
}

// Helper to expose Game state as dict (avoids string parsing in Python)
py::dict get_game_state_dict(const Game& game) {
    auto j = JsonSerializer::gameToJson(game);
    return json_to_py(j).cast<py::dict>();
}

// Helper to process action via string
bool process_action_str(Game& game, const std::string& playerId, const std::string& actionStr, int amount) {
    Player::Action action = JsonSerializer::stringToAction(actionStr);
    return game.processAction(playerId, action, amount);
}

std::string process_request(const std::string& request_str) {
    try {
        auto request_json = nlohmann::json::parse(request_str);
        PokerEngineAPI api;
        auto response_json = api.processRequest(request_json);
        return response_json.dump();
    } catch (const std::exception& e) {
        nlohmann::json error;
        error["success"] = false;
        error["error"] = e.what();
        return error.dump();
    }
}

PYBIND11_MODULE(poker_api_binding, m) {
    m.doc() = "Poker Engine API bindings";
    
    // Existing stateless API
    m.def("process_request", &process_request, "Process a poker engine request (takes JSON string, returns JSON string)");

    // GameConfig binding
    py::class_<Game::GameConfig>(m, "GameConfig")
        .def(py::init<>())
        .def_readwrite("smallBlind", &Game::GameConfig::smallBlind)
        .def_readwrite("bigBlind", &Game::GameConfig::bigBlind)
        .def_readwrite("startingChips", &Game::GameConfig::startingChips)
        .def_readwrite("minPlayers", &Game::GameConfig::minPlayers)
        .def_readwrite("maxPlayers", &Game::GameConfig::maxPlayers)
        .def_readwrite("seed", &Game::GameConfig::seed);

    // Game binding
    py::class_<Game>(m, "Game")
        .def(py::init<const Game::GameConfig&>())
        .def("add_player", [](Game& self, const std::string& id, const std::string& name, int chips) {
            return self.addPlayer(id, name, chips);
        }, py::arg("id"), py::arg("name"), py::arg("chips") = 0)
        .def("remove_player", [](Game& self, const std::string& id) {
            return self.removePlayer(id);
        })
        .def("start_hand", &Game::startHand)
        .def("process_action", &process_action_str, py::arg("player_id"), py::arg("action"), py::arg("amount") = 0)
        .def("advance_game", &Game::advanceGame)
        .def("get_stage_name", &Game::getStageName)
        .def("get_state_json", &get_game_state_json)
        .def("get_state_dict", &get_game_state_dict, "Get game state as a Python dictionary (faster than parsing JSON string)")
        .def("get_current_player_id", [](const Game& self) -> std::optional<std::string> {
            const Player* p = self.getCurrentPlayer();
            if (p) return p->getId();
            return std::nullopt;
        });
    
    // Fast hand strength estimation (for RL training)
    m.def("estimate_hand_strength", 
        &HandStrengthEstimator::estimate,
        "Estimate hand strength (0.0-1.0) from hole cards and community cards.\n\n"
        "This is ~10x faster than the Python equivalent and useful during\n"
        "episode collection for reward shaping.\n\n"
        "Args:\n"
        "    hole_cards: List of 2 card strings (e.g., ['AH', 'KS'])\n"
        "    community_cards: List of 0-5 card strings\n\n"
        "Returns:\n"
        "    float: Hand strength estimate in range [0.0, 1.0]",
        py::arg("hole_cards"),
        py::arg("community_cards")
    );
    
    m.def("estimate_preflop_strength",
        &HandStrengthEstimator::estimatePreflop,
        "Estimate preflop hand strength (0.0-1.0) from hole cards.\n\n"
        "Based on Sklansky-Chubukov rankings.\n\n"
        "Args:\n"
        "    hole_cards: List of 2 card strings (e.g., ['AH', 'KS'])\n\n"
        "Returns:\n"
        "    float: Preflop strength estimate in range [0.0, 1.0]",
        py::arg("hole_cards")
    );
    
    // =========================================================================
    // RLStateEncoder - High-performance state encoding for RL training
    // =========================================================================
    
    py::class_<RLStateEncoder>(m, "RLStateEncoder",
        "High-performance state encoder for RL training.\n\n"
        "Encodes poker game state into a fixed-size feature vector (167 dimensions)\n"
        "for the neural network. This C++ implementation is ~10-20x faster than\n"
        "the Python equivalent.\n\n"
        "Feature breakdown (167 total):\n"
        "  - Hole cards: 34 (2 cards x 17 features)\n"
        "  - Community cards: 85 (5 cards x 17 features)\n"
        "  - Pot/betting info: 5\n"
        "  - Stage: 5 (one-hot)\n"
        "  - Position: 6\n"
        "  - Game theory: 5\n"
        "  - Opponent modeling: 26\n"
        "  - Hand strength: 1")
        .def(py::init<>())
        .def("encode_state", [](RLStateEncoder& self, py::dict state_dict) {
            // Convert Python dict to nlohmann::json
            nlohmann::json state;
            
            // Extract fields from Python dict
            if (state_dict.contains("hole_cards")) {
                py::list cards = state_dict["hole_cards"].cast<py::list>();
                state["hole_cards"] = nlohmann::json::array();
                for (auto card : cards) {
                    if (py::isinstance<py::str>(card)) {
                        state["hole_cards"].push_back(card.cast<std::string>());
                    }
                }
            }
            
            if (state_dict.contains("community_cards")) {
                py::list cards = state_dict["community_cards"].cast<py::list>();
                state["community_cards"] = nlohmann::json::array();
                for (auto card : cards) {
                    if (py::isinstance<py::str>(card)) {
                        state["community_cards"].push_back(card.cast<std::string>());
                    }
                }
            }
            
            // Numeric fields
            auto extract_int = [&](const char* key, const char* alt_key = nullptr) {
                if (state_dict.contains(key)) {
                    state[key] = state_dict[key].cast<int>();
                } else if (alt_key && state_dict.contains(alt_key)) {
                    state[key] = state_dict[alt_key].cast<int>();
                }
            };
            
            auto extract_bool = [&](const char* key) {
                if (state_dict.contains(key)) {
                    state[key] = state_dict[key].cast<bool>();
                }
            };
            
            auto extract_string = [&](const char* key) {
                if (state_dict.contains(key)) {
                    state[key] = state_dict[key].cast<std::string>();
                }
            };
            
            extract_int("pot");
            extract_int("current_bet");
            extract_int("player_chips");
            extract_int("player_bet");
            extract_int("player_total_bet");
            extract_int("num_players");
            extract_int("num_active");
            extract_int("position");
            extract_int("big_blind");
            extract_int("small_blind");
            extract_int("starting_chips");
            extract_int("max_stack");
            extract_int("to_call");
            extract_int("min_bet");
            extract_int("min_raise_total");
            
            extract_bool("is_dealer");
            extract_bool("is_small_blind");
            extract_bool("is_big_blind");
            
            extract_string("stage");
            extract_string("player_id");
            
            return self.encodeState(state);
        },
        "Encode a game state into a feature vector.\n\n"
        "Args:\n"
        "    state: Dict with game state fields (hole_cards, community_cards, pot, etc.)\n\n"
        "Returns:\n"
        "    List of 167 floats representing the encoded state",
        py::arg("state"))
        .def("reset_history", &RLStateEncoder::resetHistory,
            "Reset action history (call at start of new hand)")
        .def("add_action", &RLStateEncoder::addAction,
            "Add an action to history for opponent modeling.\n\n"
            "Args:\n"
            "    player_id: ID of the player who took the action\n"
            "    action_type: Action type string (fold, check, call, bet, raise, all_in)\n"
            "    amount: Bet/raise amount\n"
            "    pot: Pot size at time of action\n"
            "    stage: Game stage (Preflop, Flop, Turn, River)",
            py::arg("player_id"), py::arg("action_type"), py::arg("amount"),
            py::arg("pot"), py::arg("stage"))
        .def_property_readonly_static("FEATURE_DIM", [](py::object) {
            return RLStateEncoder::FEATURE_DIM;
        }, "Feature dimension (167)");
    
    // =========================================================================
    // ActionUtils - Action conversion and legal mask utilities
    // =========================================================================
    
    m.def("convert_action_label", [](const std::string& actionLabel, py::dict state_dict) {
        // Convert Python dict to nlohmann::json
        nlohmann::json state;
        
        auto extract_int = [&](const char* key) {
            if (state_dict.contains(key)) {
                state[key] = state_dict[key].cast<int>();
            }
        };
        
        extract_int("player_chips");
        extract_int("pot");
        extract_int("player_bet");
        extract_int("current_bet");
        extract_int("min_bet");
        extract_int("big_blind");
        extract_int("min_raise_total");
        
        auto result = ActionUtils::convertActionLabel(actionLabel, state);
        return py::make_tuple(result.actionType, result.amount);
    },
    "Convert an action label to (action_type, amount).\n\n"
    "Unified raise_X% actions are converted to either 'bet' or 'raise' based on\n"
    "game context (whether there's already a bet to face).\n\n"
    "Args:\n"
    "    action_label: Action label (e.g., 'raise_50%', 'call', 'fold')\n"
    "    state: Dict with player_chips, pot, player_bet, current_bet, etc.\n\n"
    "Returns:\n"
    "    Tuple of (action_type, amount)",
    py::arg("action_label"), py::arg("state"));
    
    m.def("create_legal_actions_mask",
        &ActionUtils::createLegalActionsMaskFloat,
        "Create a boolean mask for legal actions as floats.\n\n"
        "With the unified action space, 'bet' and 'raise' both enable the same\n"
        "raise_X% actions.\n\n"
        "Args:\n"
        "    legal_actions: List of legal action strings (e.g., ['fold', 'call', 'raise'])\n\n"
        "Returns:\n"
        "    List of 13 floats (1.0 = legal, 0.0 = illegal)",
        py::arg("legal_actions"));
    
    m.def("extract_state", [](py::dict game_state_dict, const std::string& playerId) {
        // Convert Python dict to nlohmann::json for game state
        nlohmann::json gameState;
        
        // Extract key fields
        if (game_state_dict.contains("players")) {
            py::list players = game_state_dict["players"].cast<py::list>();
            gameState["players"] = nlohmann::json::array();
            for (auto p : players) {
                nlohmann::json player_json;
                py::dict pd = p.cast<py::dict>();
                if (pd.contains("id")) player_json["id"] = pd["id"].cast<std::string>();
                if (pd.contains("chips")) player_json["chips"] = pd["chips"].cast<int>();
                if (pd.contains("bet")) player_json["bet"] = pd["bet"].cast<int>();
                if (pd.contains("totalBet")) player_json["totalBet"] = pd["totalBet"].cast<int>();
                if (pd.contains("position")) player_json["position"] = pd["position"].cast<int>();
                if (pd.contains("isDealer")) player_json["isDealer"] = pd["isDealer"].cast<bool>();
                if (pd.contains("isSmallBlind")) player_json["isSmallBlind"] = pd["isSmallBlind"].cast<bool>();
                if (pd.contains("isBigBlind")) player_json["isBigBlind"] = pd["isBigBlind"].cast<bool>();
                if (pd.contains("isInHand")) player_json["isInHand"] = pd["isInHand"].cast<bool>();
                if (pd.contains("holeCards")) {
                    player_json["holeCards"] = nlohmann::json::array();
                    for (auto c : pd["holeCards"].cast<py::list>()) {
                        if (py::isinstance<py::str>(c)) {
                            player_json["holeCards"].push_back(c.cast<std::string>());
                        }
                    }
                }
                gameState["players"].push_back(player_json);
            }
        }
        
        if (game_state_dict.contains("pot")) gameState["pot"] = game_state_dict["pot"].cast<int>();
        if (game_state_dict.contains("currentBet")) gameState["currentBet"] = game_state_dict["currentBet"].cast<int>();
        if (game_state_dict.contains("stage")) gameState["stage"] = game_state_dict["stage"].cast<std::string>();
        
        if (game_state_dict.contains("communityCards")) {
            gameState["communityCards"] = nlohmann::json::array();
            for (auto c : game_state_dict["communityCards"].cast<py::list>()) {
                if (py::isinstance<py::str>(c)) {
                    gameState["communityCards"].push_back(c.cast<std::string>());
                }
            }
        }
        
        if (game_state_dict.contains("config")) {
            py::dict cfg = game_state_dict["config"].cast<py::dict>();
            gameState["config"] = nlohmann::json::object();
            if (cfg.contains("bigBlind")) gameState["config"]["bigBlind"] = cfg["bigBlind"].cast<int>();
            if (cfg.contains("smallBlind")) gameState["config"]["smallBlind"] = cfg["smallBlind"].cast<int>();
            if (cfg.contains("startingChips")) gameState["config"]["startingChips"] = cfg["startingChips"].cast<int>();
        }
        
        if (game_state_dict.contains("actionConstraints")) {
            py::dict ac = game_state_dict["actionConstraints"].cast<py::dict>();
            gameState["actionConstraints"] = nlohmann::json::object();
            if (ac.contains("toCall")) gameState["actionConstraints"]["toCall"] = ac["toCall"].cast<int>();
            if (ac.contains("minBet")) gameState["actionConstraints"]["minBet"] = ac["minBet"].cast<int>();
            if (ac.contains("minRaiseTotal")) gameState["actionConstraints"]["minRaiseTotal"] = ac["minRaiseTotal"].cast<int>();
        }
        
        auto result = ActionUtils::extractState(gameState, playerId);
        return json_to_py(result);
    },
    "Extract state for a specific player from raw API game state.\n\n"
    "Args:\n"
    "    game_state: Raw game state dict from C++ API\n"
    "    player_id: Player ID to extract state for\n\n"
    "Returns:\n"
    "    Dict with extracted state features (hole_cards, pot, player_chips, etc.)",
    py::arg("game_state"), py::arg("player_id"));
    
    m.def("get_action_name", &ActionUtils::getActionName,
        "Get action name from index.\n\n"
        "Args:\n"
        "    action_index: Index in unified action space (0-12)\n\n"
        "Returns:\n"
        "    Action name string",
        py::arg("action_index"));
    
    m.def("get_action_index", &ActionUtils::getActionIndex,
        "Get action index from name.\n\n"
        "Args:\n"
        "    action_name: Action name (e.g., 'fold', 'raise_50%')\n\n"
        "Returns:\n"
        "    Index in unified action space (0-12), or -1 if not found",
        py::arg("action_name"));
    
    // Constants
    m.attr("NUM_ACTIONS") = py::int_(ActionUtils::NUM_ACTIONS);
    m.attr("FEATURE_DIM") = py::int_(RLStateEncoder::FEATURE_DIM);
    
    // Action indices as constants
    m.attr("ACTION_FOLD") = py::int_(static_cast<int>(ActionUtils::FOLD));
    m.attr("ACTION_CHECK") = py::int_(static_cast<int>(ActionUtils::CHECK));
    m.attr("ACTION_CALL") = py::int_(static_cast<int>(ActionUtils::CALL));
    m.attr("ACTION_ALL_IN") = py::int_(static_cast<int>(ActionUtils::ALL_IN));
    
    // =========================================================================
    // HeuristicAgents - Fast C++ opponent agents for RL training
    // =========================================================================
    
    // ActionResult struct
    py::class_<HeuristicAgents::ActionResult>(m, "ActionResult",
        "Result of agent action selection.")
        .def(py::init<>())
        .def_readwrite("action_type", &HeuristicAgents::ActionResult::actionType)
        .def_readwrite("amount", &HeuristicAgents::ActionResult::amount)
        .def_readwrite("action_label", &HeuristicAgents::ActionResult::actionLabel);
    
    // Base Agent class
    py::class_<HeuristicAgents::BaseAgent>(m, "BaseAgent",
        "Base class for C++ heuristic agents.\n\n"
        "These agents are ~10-20x faster than Python equivalents for RL training.")
        .def("reset", &HeuristicAgents::BaseAgent::reset,
            "Reset state for new hand")
        .def("observe_action", &HeuristicAgents::BaseAgent::observeAction,
            "Observe an action (for opponent modeling).\n\n"
            "Args:\n"
            "    player_id: ID of the player who took the action\n"
            "    action_type: Action type string\n"
            "    amount: Bet/raise amount\n"
            "    pot: Pot size at time of action\n"
            "    stage: Game stage",
            py::arg("player_id"), py::arg("action_type"), py::arg("amount"),
            py::arg("pot"), py::arg("stage"))
        .def("select_action", [](HeuristicAgents::BaseAgent& self, py::dict state_dict,
                                 const std::vector<std::string>& legalActions) {
            // Convert Python dict to nlohmann::json
            nlohmann::json state;
            
            // Extract hole_cards and community_cards
            if (state_dict.contains("hole_cards")) {
                state["hole_cards"] = nlohmann::json::array();
                for (auto card : state_dict["hole_cards"].cast<py::list>()) {
                    if (py::isinstance<py::str>(card)) {
                        state["hole_cards"].push_back(card.cast<std::string>());
                    }
                }
            }
            
            if (state_dict.contains("community_cards")) {
                state["community_cards"] = nlohmann::json::array();
                for (auto card : state_dict["community_cards"].cast<py::list>()) {
                    if (py::isinstance<py::str>(card)) {
                        state["community_cards"].push_back(card.cast<std::string>());
                    }
                }
            }
            
            // Extract numeric fields
            auto extract_int = [&](const char* key) {
                if (state_dict.contains(key)) {
                    state[key] = state_dict[key].cast<int>();
                }
            };
            
            extract_int("pot");
            extract_int("current_bet");
            extract_int("player_chips");
            extract_int("player_bet");
            extract_int("player_total_bet");
            extract_int("to_call");
            extract_int("min_bet");
            extract_int("big_blind");
            extract_int("min_raise_total");
            
            auto result = self.selectAction(state, legalActions);
            return py::make_tuple(result.actionType, result.amount, result.actionLabel);
        },
        "Select an action given game state and legal actions.\n\n"
        "Args:\n"
        "    state: Dict with player state (hole_cards, pot, player_chips, etc.)\n"
        "    legal_actions: List of legal action strings\n\n"
        "Returns:\n"
        "    Tuple of (action_type, amount, action_label)",
        py::arg("state"), py::arg("legal_actions"));
    
    // Concrete agent classes
    py::class_<HeuristicAgents::RandomAgent, HeuristicAgents::BaseAgent>(m, "CppRandomAgent",
        "Random agent - selects random valid actions.")
        .def(py::init<>())
        .def(py::init<unsigned int>(), py::arg("seed"));
    
    py::class_<HeuristicAgents::TightAgent, HeuristicAgents::BaseAgent>(m, "CppTightAgent",
        "Tight-aggressive agent that only plays premium hands.")
        .def(py::init<>())
        .def(py::init<unsigned int>(), py::arg("seed"));
    
    py::class_<HeuristicAgents::LoosePassiveAgent, HeuristicAgents::BaseAgent>(m, "CppLoosePassiveAgent",
        "Loose-passive agent that calls too much but rarely raises.")
        .def(py::init<>())
        .def(py::init<unsigned int>(), py::arg("seed"));
    
    py::class_<HeuristicAgents::AggressiveAgent, HeuristicAgents::BaseAgent>(m, "CppAggressiveAgent",
        "Loose-aggressive agent that bets and raises frequently.")
        .def(py::init<>())
        .def(py::init<unsigned int>(), py::arg("seed"));
    
    py::class_<HeuristicAgents::CallingStationAgent, HeuristicAgents::BaseAgent>(m, "CppCallingStationAgent",
        "Calling Station - calls almost everything including all-ins.\n"
        "CRITICAL for teaching model that all-in with weak hands LOSES.")
        .def(py::init<>())
        .def(py::init<unsigned int>(), py::arg("seed"));
    
    py::class_<HeuristicAgents::HeroCallerAgent, HeuristicAgents::BaseAgent>(m, "CppHeroCallerAgent",
        "Hero Caller - calls down suspected bluffs with medium-strength hands.")
        .def(py::init<>())
        .def(py::init<unsigned int>(), py::arg("seed"));
    
    py::class_<HeuristicAgents::HeuristicAgent, HeuristicAgents::BaseAgent>(m, "CppHeuristicAgent",
        "Rule-based heuristic agent for baseline comparison.")
        .def(py::init<>())
        .def(py::init<unsigned int>(), py::arg("seed"));
    
    py::class_<HeuristicAgents::AlwaysRaiseAgent, HeuristicAgents::BaseAgent>(m, "CppAlwaysRaiseAgent",
        "Always raises/bets when possible.")
        .def(py::init<>())
        .def(py::init<unsigned int>(), py::arg("seed"));
    
    py::class_<HeuristicAgents::AlwaysCallAgent, HeuristicAgents::BaseAgent>(m, "CppAlwaysCallAgent",
        "Always checks or calls, never bets/raises.")
        .def(py::init<>())
        .def(py::init<unsigned int>(), py::arg("seed"));
    
    py::class_<HeuristicAgents::AlwaysFoldAgent, HeuristicAgents::BaseAgent>(m, "CppAlwaysFoldAgent",
        "Always folds when facing a bet, checks if free.")
        .def(py::init<>())
        .def(py::init<unsigned int>(), py::arg("seed"));
    
    // Factory function
    m.def("create_cpp_agent", [](const std::string& agentType, unsigned int seed) {
        return HeuristicAgents::createAgent(agentType, seed);
    },
    "Factory function to create a C++ agent by type name.\n\n"
    "Args:\n"
    "    agent_type: Agent type ('random', 'tight', 'loose_passive', 'aggressive',\n"
    "                'calling_station', 'hero_caller', 'heuristic', 'always_raise',\n"
    "                'always_call', 'always_fold')\n"
    "    seed: Random seed (0 for random)\n\n"
    "Returns:\n"
    "    C++ agent instance or None if type unknown",
    py::arg("agent_type"), py::arg("seed") = 0);
    
    m.def("get_cpp_agent_types", &HeuristicAgents::getAgentTypes,
        "Get list of available C++ agent types.");
    
    // =========================================================================
    // Batch operations for vectorized environments
    // =========================================================================
    
    m.def("batch_encode_states", [](const std::vector<py::dict>& states) {
        std::vector<std::vector<float>> results;
        results.reserve(states.size());
        
        RLStateEncoder encoder;
        
        for (const auto& state_dict : states) {
            nlohmann::json state;
            
            // Extract cards
            if (state_dict.contains("hole_cards")) {
                state["hole_cards"] = nlohmann::json::array();
                for (auto card : state_dict["hole_cards"].cast<py::list>()) {
                    if (py::isinstance<py::str>(card)) {
                        state["hole_cards"].push_back(card.cast<std::string>());
                    }
                }
            }
            
            if (state_dict.contains("community_cards")) {
                state["community_cards"] = nlohmann::json::array();
                for (auto card : state_dict["community_cards"].cast<py::list>()) {
                    if (py::isinstance<py::str>(card)) {
                        state["community_cards"].push_back(card.cast<std::string>());
                    }
                }
            }
            
            // Extract numeric fields
            auto extract_int = [&](const char* key) {
                if (state_dict.contains(key)) {
                    state[key] = state_dict[key].cast<int>();
                }
            };
            auto extract_bool = [&](const char* key) {
                if (state_dict.contains(key)) {
                    state[key] = state_dict[key].cast<bool>();
                }
            };
            auto extract_string = [&](const char* key) {
                if (state_dict.contains(key)) {
                    state[key] = state_dict[key].cast<std::string>();
                }
            };
            
            extract_int("pot");
            extract_int("current_bet");
            extract_int("player_chips");
            extract_int("player_bet");
            extract_int("player_total_bet");
            extract_int("num_players");
            extract_int("num_active");
            extract_int("position");
            extract_int("big_blind");
            extract_int("small_blind");
            extract_int("starting_chips");
            extract_int("max_stack");
            extract_int("to_call");
            extract_int("min_bet");
            extract_int("min_raise_total");
            
            extract_bool("is_dealer");
            extract_bool("is_small_blind");
            extract_bool("is_big_blind");
            
            extract_string("stage");
            extract_string("player_id");
            
            results.push_back(encoder.encodeState(state));
            encoder.resetHistory();  // Reset for each state in batch
        }
        
        return results;
    },
    "Batch encode multiple game states into feature vectors.\n\n"
    "This is more efficient than encoding states one by one due to\n"
    "reduced Python/C++ boundary crossing overhead.\n\n"
    "Args:\n"
    "    states: List of state dicts\n\n"
    "Returns:\n"
    "    List of feature vectors (each 167 floats)",
    py::arg("states"));
    
    m.def("batch_create_legal_masks", [](const std::vector<std::vector<std::string>>& legal_actions_list) {
        std::vector<std::vector<float>> results;
        results.reserve(legal_actions_list.size());
        
        for (const auto& legal_actions : legal_actions_list) {
            results.push_back(ActionUtils::createLegalActionsMaskFloat(legal_actions));
        }
        
        return results;
    },
    "Batch create legal action masks for multiple states.\n\n"
    "Args:\n"
    "    legal_actions_list: List of legal action string lists\n\n"
    "Returns:\n"
    "    List of mask vectors (each 13 floats)",
    py::arg("legal_actions_list"));
}
