# Simple Makefile for building without CMake

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
BUILD_DIR = build
TARGET = $(BUILD_DIR)/poker_api

# Source files
POKER_SOURCES = Card.cpp Deck.cpp Hand.cpp Player.cpp Pot.cpp Game.cpp
SRC = main.cpp $(POKER_SOURCES)

# Test targets
TEST_TARGETS = $(BUILD_DIR)/test_card $(BUILD_DIR)/test_deck $(BUILD_DIR)/test_hand $(BUILD_DIR)/test_player $(BUILD_DIR)/test_game

# Check if json.hpp exists, if not provide download instructions
JSON_HEADER = json.hpp

all: check_json $(TARGET)

check_json:
	@if [ ! -f $(JSON_HEADER) ]; then \
		echo "Downloading nlohmann/json header..."; \
		curl -L https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp -o json.hpp; \
	fi

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(TARGET): $(SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

# Build individual test executables
$(BUILD_DIR)/test_card: test_card.cpp $(POKER_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) test_card.cpp $(POKER_SOURCES) -o $(BUILD_DIR)/test_card

$(BUILD_DIR)/test_deck: test_deck.cpp $(POKER_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) test_deck.cpp $(POKER_SOURCES) -o $(BUILD_DIR)/test_deck

$(BUILD_DIR)/test_hand: test_hand.cpp $(POKER_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) test_hand.cpp $(POKER_SOURCES) -o $(BUILD_DIR)/test_hand

$(BUILD_DIR)/test_player: test_player.cpp $(POKER_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) test_player.cpp $(POKER_SOURCES) -o $(BUILD_DIR)/test_player

$(BUILD_DIR)/test_game: test_game.cpp $(POKER_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) test_game.cpp $(POKER_SOURCES) -o $(BUILD_DIR)/test_game

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

