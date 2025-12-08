#!/usr/bin/env bash
#
# RL Training Script for Poker AI
#
# Usage: ./start_rl_training_optimized.sh [OPTIONS]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

# Add project root to PYTHONPATH so we import common/ directly from source,
# not from an installed package. This ensures /elo and /training always share
# the same MODEL_VERSION without manual re-installs.
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Get model version from common/config.py
#
# Prefer explicit env vars so training and ELO evaluation can be pointed at the same run:
# - POKERSIM_MODEL_VERSION: controls /tmp/pokersim/rl_models_vX default
# - POKERSIM_MODELS_DIR: overrides output dir entirely
MODEL_VERSION=${POKERSIM_MODEL_VERSION:-${MODEL_VERSION:-$(uv run python -c "from common import MODEL_VERSION; print(MODEL_VERSION)" 2>/dev/null || echo "16")}}

# Directories
OUTPUT_DIR="${POKERSIM_MODELS_DIR:-/tmp/pokersim/rl_models_v${MODEL_VERSION}}"
TENSORBOARD_DIR="/tmp/pokersim/tensorboard_v${MODEL_VERSION}"

# Auto-detect resources
NUM_CPUS=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 8)
NUM_WORKERS=$((NUM_CPUS > 4 ? NUM_CPUS - 1 : 4))

# Defaults (can override via CLI)
ITERATIONS=5000
EPISODES_PER_ITER=1000
LEARNING_RATE=0.0002
MAX_MEMORY_GB=20
VERBOSE=0
CHECKPOINT_PATH=""
EXTRA_ARGS=()

# Parse arguments
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Poker AI RL Training (v${MODEL_VERSION})

Options:
  -v, --verbose              Enable verbose output
  -c, --checkpoint PATH      Resume from checkpoint
  --iterations N             Training iterations (default: $ITERATIONS)
  --episodes-per-iter N      Episodes per iteration (default: $EPISODES_PER_ITER)
  --learning-rate LR         Learning rate (default: $LEARNING_RATE)
  --num-workers N            Parallel workers (default: $NUM_WORKERS)
  --max-memory-gb N          Max memory in GB (default: $MAX_MEMORY_GB)
  -h, --help                 Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)          VERBOSE=1; shift ;;
        -c|--checkpoint)       CHECKPOINT_PATH="$2"; shift 2 ;;
        --iterations)          ITERATIONS="$2"; shift 2 ;;
        --episodes-per-iter)   EPISODES_PER_ITER="$2"; shift 2 ;;
        --learning-rate)       LEARNING_RATE="$2"; shift 2 ;;
        --num-workers)         NUM_WORKERS="$2"; shift 2 ;;
        --max-memory-gb)       MAX_MEMORY_GB="$2"; shift 2 ;;
        -h|--help)             show_help ;;
        --)                    shift; EXTRA_ARGS+=("$@"); break ;;
        *)                     EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Display config
echo "Poker AI - RL Training (v${MODEL_VERSION})"
echo "Config: ${ITERATIONS} iters × ${EPISODES_PER_ITER} eps | LR=${LEARNING_RATE} | ${NUM_WORKERS} workers"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Cleanup handler
cleanup() { echo -e "\nShutting down..."; }
trap cleanup EXIT INT TERM

# Run training
echo "Starting training..."

ARGS=(
  --iterations "$ITERATIONS"
  --episodes-per-iter "$EPISODES_PER_ITER"
  --learning-rate "$LEARNING_RATE"
  --num-workers "$NUM_WORKERS"
  --max-memory-gb "$MAX_MEMORY_GB"
  --output-dir "$OUTPUT_DIR"
  --tensorboard-dir "$TENSORBOARD_DIR"
)

if [[ -n "$CHECKPOINT_PATH" ]]; then
  ARGS+=(--checkpoint "$CHECKPOINT_PATH")
fi
if [[ "$VERBOSE" -eq 1 ]]; then
  ARGS+=(--verbose)
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  ARGS+=("${EXTRA_ARGS[@]}")
fi

uv run train-poker "${ARGS[@]}"
