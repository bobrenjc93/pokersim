#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "PokerEngineAPI.h"
#include "Game.h"
#include "JsonSerializer.h"
#include "json.hpp"
#include "HandStrength.h"

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
}
