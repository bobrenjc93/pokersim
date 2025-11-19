cmake_minimum_required(VERSION 3.10)
project(PokerSimAPI)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include json library
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Source directory
include_directories(src)

# Source files
set(POKER_SOURCES
    src/Card.cpp
    src/Deck.cpp
    src/Hand.cpp
    src/Player.cpp
    src/Pot.cpp
    src/Game.cpp
    src/HandStrength.cpp
)

# API sources
set(API_SOURCES
    src/JsonSerializer.cpp
    src/PokerEngineAPI.cpp
    src/HTTPServer.cpp
)

# Add executables
add_executable(poker_api src/main.cpp ${API_SOURCES} ${POKER_SOURCES})

# Link json library to poker_api
target_link_libraries(poker_api PRIVATE nlohmann_json::nlohmann_json)

# Set output directories
set_target_properties(poker_api PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

