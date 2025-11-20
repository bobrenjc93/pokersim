#include <iostream>
#include <cstdlib>
#include "HTTPServer.h"

/**
 * Poker Engine API - Entry Point
 * 
 * Starts a stateless HTTP server that processes poker game requests.
 * Usage: ./poker_api [port]
 * Default port: 8080
 */
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
