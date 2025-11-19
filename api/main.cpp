#include <iostream>
#include <string>
#include <sstream>
#include <random>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include "json.hpp"

using json = nlohmann::json;

class GameSimulator {
private:
    std::mt19937 rng;
    
public:
    GameSimulator(unsigned int seed) : rng(seed) {}
    
    /**
     * Simulates the next game state based on current state.
     * This is a simple example - extend with your poker logic.
     */
    json simulateNextState(const json& currentState) {
        json nextState = currentState;
        
        // Example simulation logic
        if (nextState.contains("pot")) {
            int pot = nextState["pot"].get<int>();
            std::uniform_int_distribution<int> betDist(10, 50);
            int newBet = betDist(rng);
            nextState["pot"] = pot + newBet;
            nextState["lastAction"] = "bet";
            nextState["lastBetAmount"] = newBet;
        }
        
        // Add a random card to the board if cards array exists
        if (nextState.contains("cards") && nextState["cards"].is_array()) {
            std::vector<std::string> suits = {"H", "D", "C", "S"};
            std::vector<std::string> ranks = {"2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"};
            
            std::uniform_int_distribution<size_t> suitDist(0, suits.size() - 1);
            std::uniform_int_distribution<size_t> rankDist(0, ranks.size() - 1);
            
            std::string newCard = ranks[rankDist(rng)] + suits[suitDist(rng)];
            nextState["cards"].push_back(newCard);
        }
        
        // Add simulation metadata
        nextState["simulated"] = true;
        nextState["timestamp"] = std::time(nullptr);
        
        return nextState;
    }
};

class HTTPServer {
private:
    int serverSocket;
    int port;
    
    std::string extractBody(const std::string& request) {
        size_t bodyPos = request.find("\r\n\r\n");
        if (bodyPos != std::string::npos) {
            return request.substr(bodyPos + 4);
        }
        return "";
    }
    
    bool isPostSimulate(const std::string& request) {
        return request.find("POST /simulate") == 0;
    }
    
    std::string createResponse(int statusCode, const std::string& body, const std::string& contentType = "application/json") {
        std::ostringstream response;
        
        std::string statusText = (statusCode == 200) ? "OK" : 
                                (statusCode == 400) ? "Bad Request" : 
                                (statusCode == 500) ? "Internal Server Error" : "Error";
        
        response << "HTTP/1.1 " << statusCode << " " << statusText << "\r\n";
        response << "Content-Type: " << contentType << "\r\n";
        response << "Content-Length: " << body.length() << "\r\n";
        response << "Access-Control-Allow-Origin: *\r\n";
        response << "Access-Control-Allow-Methods: POST, OPTIONS\r\n";
        response << "Access-Control-Allow-Headers: Content-Type\r\n";
        response << "Connection: close\r\n";
        response << "\r\n";
        response << body;
        
        return response.str();
    }
    
public:
    HTTPServer(int p) : port(p), serverSocket(-1) {}
    
    bool start() {
        serverSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSocket < 0) {
            std::cerr << "Error creating socket" << std::endl;
            return false;
        }
        
        int opt = 1;
        if (setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
            std::cerr << "Error setting socket options" << std::endl;
            return false;
        }
        
        struct sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(port);
        
        if (bind(serverSocket, (struct sockaddr*)&address, sizeof(address)) < 0) {
            std::cerr << "Error binding socket to port " << port << std::endl;
            close(serverSocket);
            return false;
        }
        
        if (listen(serverSocket, 10) < 0) {
            std::cerr << "Error listening on socket" << std::endl;
            close(serverSocket);
            return false;
        }
        
        std::cout << "🚀 C++ API Server started on http://localhost:" << port << std::endl;
        std::cout << "📡 Listening for POST requests to /simulate" << std::endl;
        
        return true;
    }
    
    void handleRequests() {
        while (true) {
            struct sockaddr_in clientAddress;
            socklen_t clientLen = sizeof(clientAddress);
            
            int clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddress, &clientLen);
            if (clientSocket < 0) {
                std::cerr << "Error accepting connection" << std::endl;
                continue;
            }
            
            char buffer[4096] = {0};
            ssize_t bytesRead = read(clientSocket, buffer, sizeof(buffer) - 1);
            
            if (bytesRead > 0) {
                std::string request(buffer, bytesRead);
                std::string response = processRequest(request);
                write(clientSocket, response.c_str(), response.length());
            }
            
            close(clientSocket);
        }
    }
    
    std::string processRequest(const std::string& request) {
        // Handle OPTIONS for CORS
        if (request.find("OPTIONS") == 0) {
            return createResponse(200, "");
        }
        
        if (!isPostSimulate(request)) {
            json errorResponse = {{"error", "Invalid endpoint. Use POST /simulate"}};
            return createResponse(404, errorResponse.dump());
        }
        
        try {
            std::string body = extractBody(request);
            
            if (body.empty()) {
                json errorResponse = {{"error", "Empty request body"}};
                return createResponse(400, errorResponse.dump());
            }
            
            json requestData = json::parse(body);
            
            if (!requestData.contains("gameState") || !requestData.contains("seed")) {
                json errorResponse = {{"error", "Missing required fields: gameState and seed"}};
                return createResponse(400, errorResponse.dump());
            }
            
            unsigned int seed = requestData["seed"].get<unsigned int>();
            json gameState = requestData["gameState"];
            
            GameSimulator simulator(seed);
            json nextState = simulator.simulateNextState(gameState);
            
            json responseData = {
                {"success", true},
                {"nextGameState", nextState}
            };
            
            std::cout << "✓ Simulated game state with seed " << seed << std::endl;
            
            return createResponse(200, responseData.dump());
            
        } catch (const json::exception& e) {
            json errorResponse = {
                {"error", "Invalid JSON"},
                {"details", e.what()}
            };
            return createResponse(400, errorResponse.dump());
        } catch (const std::exception& e) {
            json errorResponse = {
                {"error", "Internal server error"},
                {"details", e.what()}
            };
            return createResponse(500, errorResponse.dump());
        }
    }
    
    ~HTTPServer() {
        if (serverSocket >= 0) {
            close(serverSocket);
        }
    }
};

int main(int argc, char* argv[]) {
    int port = 8080;
    
    if (argc > 1) {
        port = std::atoi(argv[1]);
    }
    
    HTTPServer server(port);
    
    if (!server.start()) {
        return 1;
    }
    
    server.handleRequests();
    
    return 0;
}

