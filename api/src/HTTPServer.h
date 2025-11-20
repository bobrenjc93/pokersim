#ifndef HTTP_SERVER_H
#define HTTP_SERVER_H

#include <string>
#include "PokerEngineAPI.h"

/**
 * HTTPServer - Simple HTTP server for the poker engine API
 * 
 * Handles HTTP requests and routes them to the PokerEngineAPI
 * Supports CORS for cross-origin requests
 */
class HTTPServer {
public:
    /**
     * Constructor
     * @param port The port to listen on
     */
    explicit HTTPServer(int port);
    
    /**
     * Destructor - cleans up socket resources
     */
    ~HTTPServer();
    
    /**
     * Start the HTTP server
     * @return true if server started successfully, false otherwise
     */
    bool start();
    
    /**
     * Handle incoming requests (blocking call)
     */
    void handleRequests();
    
    /**
     * Process a single HTTP request
     * @param request The raw HTTP request string
     * @return The HTTP response string
     */
    [[nodiscard]] std::string processRequest(const std::string& request);
    
private:
    int serverSocket;
    int port;
    PokerEngineAPI pokerAPI;
    
    /**
     * Extract the body from an HTTP request
     */
    std::string extractBody(const std::string& request);
    
    /**
     * Check if the request is a POST to /simulate
     */
    bool isPostSimulate(const std::string& request);
    
    /**
     * Create an HTTP response
     */
    std::string createResponse(int statusCode, const std::string& body, 
                              const std::string& contentType = "application/json");
};

#endif // HTTP_SERVER_H

