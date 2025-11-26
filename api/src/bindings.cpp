#include <pybind11/pybind11.h>
#include "PokerEngineAPI.h"
#include "json.hpp"

namespace py = pybind11;

std::string process_request(const std::string& request_str) {
    try {
        py::gil_scoped_release release;
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
    m.def("process_request", &process_request, "Process a poker engine request (takes JSON string, returns JSON string)");
}

