# Simple Makefile for building without CMake

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
TARGET = poker_api
SRC = main.cpp

# Check if json.hpp exists, if not provide download instructions
JSON_HEADER = json.hpp

all: check_json $(TARGET)

check_json:
	@if [ ! -f $(JSON_HEADER) ]; then \
		echo "Downloading nlohmann/json header..."; \
		curl -L https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp -o json.hpp; \
	fi

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean check_json

