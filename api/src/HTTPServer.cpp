#include "HTTPServer.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include "json.hpp"

using json = nlohmann::json;

HTTPServer::HTTPServer(int p) : serverSocket(-1), port(p) {}

HTTPServer::~HTTPServer() {
    if (serverSocket >= 0) {
        close(serverSocket);
    }
}

bool HTTPServer::start() {
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
    
    std::cout << "🚀 Stateless C++ Poker Engine API started on http://localhost:" << port << std::endl;
    std::cout << "📡 Listening for POST requests to /simulate" << std::endl;
    std::cout << "✨ Server is truly stateless - send config + seed + history + action" << std::endl;
    std::cout << "   No gameIds, no server-side persistence!" << std::endl;
    
    return true;
}

void HTTPServer::handleRequests() {
    while (true) {
        struct sockaddr_in clientAddress;
        socklen_t clientLen = sizeof(clientAddress);
        
        int clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddress, &clientLen);
        if (clientSocket < 0) {
            std::cerr << "Error accepting connection" << std::endl;
            continue;
        }
        
        // Read request with dynamic buffer for large payloads
        std::string request;
        char buffer[16384];  // 16KB buffer for better performance with large histories
        ssize_t bytesRead;
        
        while ((bytesRead = read(clientSocket, buffer, sizeof(buffer))) > 0) {
            request.append(buffer, bytesRead);
            // Check if we've read the full request (look for end of headers + potential body)
            if (request.find("\r\n\r\n") != std::string::npos) {
                // Check Content-Length to see if we need to read more
                size_t clPos = request.find("Content-Length:");
                if (clPos != std::string::npos) {
                    size_t clEnd = request.find("\r\n", clPos);
                    int contentLength = std::stoi(request.substr(clPos + 15, clEnd - clPos - 15));
                    size_t headerEnd = request.find("\r\n\r\n") + 4;
                    if (request.length() >= headerEnd + contentLength) {
                        break;  // Full request received
                    }
                } else {
                    break;  // No body expected
                }
            }
        }
        
        if (!request.empty()) {
            std::string response = processRequest(request);
            write(clientSocket, response.c_str(), response.length());
        }
        
        close(clientSocket);
    }
}

std::string HTTPServer::processRequest(const std::string& request) {
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
        
        // Process request through poker engine API
        json responseData = pokerAPI.processRequest(requestData);
        
        int statusCode = responseData["success"] ? 200 : 400;
        return createResponse(statusCode, responseData.dump());
        
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

std::string HTTPServer::extractBody(const std::string& request) {
    size_t bodyPos = request.find("\r\n\r\n");
    if (bodyPos != std::string::npos) {
        return request.substr(bodyPos + 4);
    }
    return "";
}

bool HTTPServer::isPostSimulate(const std::string& request) {
    return request.find("POST /simulate") == 0;
}

std::string HTTPServer::createResponse(int statusCode, const std::string& body, 
                                      const std::string& contentType) {
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

