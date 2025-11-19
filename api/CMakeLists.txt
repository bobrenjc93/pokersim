cmake_minimum_required(VERSION 3.10)
project(PokerSimAPI)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include json library
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Add executable
add_executable(poker_api main.cpp)

# Link json library
target_link_libraries(poker_api PRIVATE nlohmann_json::nlohmann_json)

# Set output directory
set_target_properties(poker_api PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

