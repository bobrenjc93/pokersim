#!/usr/bin/env bash
#
# Optimized RL Training Script for Poker AI
#
# This script starts RL training with optimized hyperparameters for better convergence:
# - Heads-up (2 player) for simpler learning
# - Diverse opponent pool: Random (35%), Heuristic (20%), Past checkpoints (30%), Self-play (15%)
# - Increased learning rate (3e-4) for faster learning
# - More exploration via entropy bonus with slower decay
# - Reward normalization for stability
# - Learning rate scheduling aligned with total iterations
# - Periodic evaluation against both Random and Heuristic baselines
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_DIR="$(dirname "$SCRIPT_DIR")/api"

# Configuration
# API_URL="http://localhost:8080/simulate"
# API_PID_FILE="/tmp/pokersim_api.pid"

# Training parameters (optimized for convergence and proper hand selection)
ITERATIONS=5000
EPISODES_PER_ITER=500    # Balanced for variance reduction vs speed
PPO_EPOCHS=20            # Increased to extract more from data
MINI_BATCH_SIZE=512      # Standard batch size
LEARNING_RATE=0.0002     # Slightly increased with larger batch size
ENTROPY_COEF=0.03        # Increased exploration to prevent early convergence to all-in
VALUE_LOSS_COEF=0.5      # Default 0.5
HAND_STRENGTH_LOSS_COEF=0.15  # Auxiliary loss for hand strength prediction (teaches hand evaluation)
GAE_LAMBDA=0.95          # Default 0.95
NUM_PLAYERS=2            # Heads-up (simpler)
SMALL_BLIND=10
BIG_BLIND=20
STARTING_CHIPS=1000
SAVE_INTERVAL=1

# Monte Carlo multi-runout settings
# These enable "run it N times" style training where each hand is simulated
# multiple times with different action sampling to calculate regret
NUM_RUNOUTS=50           # Number of Monte Carlo runouts per decision (0=disabled, 50=recommended)
REGRET_WEIGHT=0.5        # Weight for regret-based reward adjustment (0-1)

# Model version from config
if [ -z "${MODEL_VERSION:-}" ]; then
    # Extract from config.py
    if [ -f "${SCRIPT_DIR}/config.py" ]; then
        MODEL_VERSION=$(grep "^MODEL_VERSION =" "${SCRIPT_DIR}/config.py" | awk '{print $3}' | tr -d ' ')
    fi
fi
MODEL_VERSION=${MODEL_VERSION:-11}

# Output directories
OUTPUT_DIR="/tmp/pokersim/rl_models_v${MODEL_VERSION}"
TENSORBOARD_DIR="/tmp/pokersim/tensorboard_v${MODEL_VERSION}"

# Parse command line arguments
VERBOSE=""
CHECKPOINT=""
CUSTOM_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -c|--checkpoint)
            CHECKPOINT="--checkpoint $2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num-runouts)
            NUM_RUNOUTS="$2"
            shift 2
            ;;
        --regret-weight)
            REGRET_WEIGHT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose              Enable verbose output"
            echo "  -c, --checkpoint PATH      Resume from checkpoint"
            echo "  --iterations N             Number of training iterations (default: $ITERATIONS)"
            echo "  --learning-rate LR         Learning rate (default: $LEARNING_RATE)"
            echo "  --num-runouts N            Monte Carlo runouts per decision (default: $NUM_RUNOUTS, 0=disabled)"
            echo "  --regret-weight W          Weight for regret-based reward (default: $REGRET_WEIGHT)"
            echo "  --help                     Show this help message"
            echo ""
            echo "Optimized settings:"
            echo "  - Heads-up (2 players) for simpler learning"
            echo "  - Diverse opponents: Random, Heuristic, Past checkpoints, Self-play"
            echo "  - Learning rate: 3e-4 (standard for PPO)"
            echo "  - Entropy coefficient: 0.02 with slow decay to 0.005"
            echo "  - Episodes per iteration: 500 (variance reduction)"
            echo "  - Monte Carlo multi-runout: Run each hand $NUM_RUNOUTS times for regret calculation"
            echo "  - Reward normalization enabled"
            echo "  - Learning rate scheduling aligned with total iterations"
            echo "  - Evaluation against both Random and Heuristic"
            exit 0
            ;;
        *)
            CUSTOM_ARGS="$CUSTOM_ARGS $1"
            shift
            ;;
    esac
done

# Print header
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  🃏 Poker AI Reinforcement Learning Training (OPTIMIZED)    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down...${NC}"
    echo -e "${GREEN}✓ Cleanup complete${NC}\n"
}

trap cleanup EXIT INT TERM

# Check if bindings are built
if [ ! -f "$SCRIPT_DIR/poker_api_binding.cpython-312-darwin.so" ] && [ ! -f "$SCRIPT_DIR/poker_api_binding.so" ]; then
    echo -e "${YELLOW}⚠️  Warning: poker_api_binding not found in current directory.${NC}"
    echo -e "${YELLOW}   Please ensure you have built the bindings using 'make module' in the api directory.${NC}"
    # We don't exit here because the file name might vary by python version/platform
    # train.py will fail with a clear error if it can't import it
fi

echo ""
echo -e "${BLUE}Using MODEL_VERSION: v${MODEL_VERSION}${NC}"
echo ""

# Show configuration
if [ -n "$CHECKPOINT" ]; then
    echo -e "${YELLOW}📂 Resuming from checkpoint${NC}"
else
    echo -e "${GREEN}Using optimized training parameters${NC}"
    echo -e "(Run with --help to see all options)"
fi

echo ""
echo -e "${BLUE}📊 Optimizations Enabled:${NC}"
echo "   ✓ Heads-up play (2 players) - simpler to learn"
echo "   ✓ Diverse opponent pool - CallingStation, HeroCaller, Tight, Heuristic, Self-play"
echo "   ✓ Learning rate: ${LEARNING_RATE} - adjusted for stability"
echo "   ✓ Entropy coefficient: ${ENTROPY_COEF} with very slow decay (0.9999) to 0.01"
echo "   ✓ Hand strength auxiliary loss: ${HAND_STRENGTH_LOSS_COEF} - teaches hand evaluation"
echo "   ✓ Model size: Large (1024 dim, 8 layers, 16 heads)"
echo "   ✓ Hand-strength-aware action gating - penalizes all-in with weak hands"
echo "   ✓ Strong reward shaping - punishes bad all-ins even when they win"
echo "   ✓ CallingStation opponents - calls all-ins to show trash hands lose"
echo "   ✓ Learning rate scheduling - aligned with ${ITERATIONS} iterations"
echo "   ✓ ${EPISODES_PER_ITER} episodes/iteration - variance reduction"
echo "   ✓ Evaluation vs both Random and Heuristic baselines"
if [ "$NUM_RUNOUTS" -gt 0 ]; then
echo "   ✓ Monte Carlo multi-runout: ${NUM_RUNOUTS} runouts per decision for regret"
echo "   ✓ Regret weight: ${REGRET_WEIGHT} - blends outcome with regret-based reward"
fi

echo ""
echo -e "${BLUE}📊 TensorBoard Monitoring:${NC}"
echo "   In another terminal, run:"
echo "   tensorboard --logdir=${TENSORBOARD_DIR}"
echo "   Then open: http://localhost:6006"

echo ""
echo -e "${BLUE}📁 Output Locations:${NC}"
echo "   Models: ${OUTPUT_DIR}/"
echo "   Logs: ${TENSORBOARD_DIR}/"

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Starting RL Training...                                       ║${NC}"
echo -e "${BLUE}║  Press Ctrl+C to stop gracefully                              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# Run training with optimized parameters
uv run python train.py \
    --iterations "$ITERATIONS" \
    --episodes-per-iter "$EPISODES_PER_ITER" \
    --ppo-epochs "$PPO_EPOCHS" \
    --mini-batch-size "$MINI_BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --entropy-coef "$ENTROPY_COEF" \
    --value-loss-coef "$VALUE_LOSS_COEF" \
    --hand-strength-loss-coef "$HAND_STRENGTH_LOSS_COEF" \
    --gae-lambda "$GAE_LAMBDA" \
    --hidden-dim 1024 \
    --num-heads 16 \
    --num-layers 8 \
    --num-players "$NUM_PLAYERS" \
    --small-blind "$SMALL_BLIND" \
    --big-blind "$BIG_BLIND" \
    --starting-chips "$STARTING_CHIPS" \
    --save-interval "$SAVE_INTERVAL" \
    --output-dir "$OUTPUT_DIR" \
    --tensorboard-dir "$TENSORBOARD_DIR" \
    --num-runouts "$NUM_RUNOUTS" \
    --regret-weight "$REGRET_WEIGHT" \
    $CHECKPOINT \
    $VERBOSE \
    $CUSTOM_ARGS

