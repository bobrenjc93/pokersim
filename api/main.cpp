#include <iostream>
#include <string>
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include "json.hpp"

using json = nlohmann::json;

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
                                (statusCode == 404) ? "Not Found" :
                                (statusCode == 500) ? "Internal Server Error" : 
                                (statusCode == 501) ? "Not Implemented" : "Error";
        
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
    HTTPServer(int p) : serverSocket(-1), port(p) {}
    
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
            
            // Poker engine integration point - API contract TBD
            json responseData = {
                {"success", false},
                {"error", "HTTP API integration pending"},
                {"note", "Use poker engine classes directly via Game.h"}
            };
            
            return createResponse(501, responseData.dump());
            
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

