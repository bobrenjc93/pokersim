# Simple Makefile for building without CMake

CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2 -Isrc
BUILD_DIR = build
SRC_DIR = src
TEST_DIR = tests
TARGET = $(BUILD_DIR)/poker_api

# Source files
POKER_SOURCES = $(SRC_DIR)/Card.cpp $(SRC_DIR)/Deck.cpp $(SRC_DIR)/Hand.cpp $(SRC_DIR)/Player.cpp $(SRC_DIR)/Pot.cpp $(SRC_DIR)/Game.cpp $(SRC_DIR)/JsonSerializer.cpp
API_SOURCES = $(SRC_DIR)/PokerEngineAPI.cpp $(SRC_DIR)/HTTPServer.cpp
SRC = $(SRC_DIR)/main.cpp $(API_SOURCES) $(POKER_SOURCES)

# Test targets
TEST_TARGETS = $(BUILD_DIR)/test_card $(BUILD_DIR)/test_deck $(BUILD_DIR)/test_hand $(BUILD_DIR)/test_player $(BUILD_DIR)/test_game

# Check if json.hpp exists, if not provide download instructions
JSON_HEADER = $(SRC_DIR)/json.hpp

all: check_json $(TARGET)

check_json:
	@if [ ! -f $(JSON_HEADER) ]; then \
		echo "Downloading nlohmann/json header..."; \
		curl -L https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp -o $(JSON_HEADER); \
	fi

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(TARGET): $(SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

# Build individual test executables
$(BUILD_DIR)/test_card: $(TEST_DIR)/test_card.cpp $(POKER_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TEST_DIR)/test_card.cpp $(POKER_SOURCES) -o $(BUILD_DIR)/test_card

$(BUILD_DIR)/test_deck: $(TEST_DIR)/test_deck.cpp $(POKER_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TEST_DIR)/test_deck.cpp $(POKER_SOURCES) -o $(BUILD_DIR)/test_deck

$(BUILD_DIR)/test_hand: $(TEST_DIR)/test_hand.cpp $(POKER_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TEST_DIR)/test_hand.cpp $(POKER_SOURCES) -o $(BUILD_DIR)/test_hand

$(BUILD_DIR)/test_player: $(TEST_DIR)/test_player.cpp $(POKER_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TEST_DIR)/test_player.cpp $(POKER_SOURCES) -o $(BUILD_DIR)/test_player

$(BUILD_DIR)/test_game: $(TEST_DIR)/test_game.cpp $(POKER_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TEST_DIR)/test_game.cpp $(POKER_SOURCES) -o $(BUILD_DIR)/test_game

# Build all tests
build_tests: $(TEST_TARGETS)

# Run all tests
test: build_tests
	@echo "\n🎯 Running all test suites...\n"
	@$(BUILD_DIR)/test_card && \
	$(BUILD_DIR)/test_deck && \
	$(BUILD_DIR)/test_hand && \
	$(BUILD_DIR)/test_player && \
	$(BUILD_DIR)/test_game && \
	echo "\n🎉 All test suites passed!\n"

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean check_json test build_tests

