#!/bin/bash
#
# Start the continuous training system
#
# This script starts the continuous training system which:
#   - Generates training data in the background
#   - Trains models continuously
#   - Evaluates model performance periodically
#   - Balances system resources automatically
#   - Runs until interrupted (Ctrl+C)
#
# Usage:
#   ./start_continuous_training.sh [options]
#
# Options:
#   --data-dir PATH       Directory for training data (default: /tmp/pokersim/data)
#   --eval-interval N     Run evaluation every N training cycles (default: 5)
#   --num-eval-hands N    Number of hands to play in evaluation (default: 100)
#   --skip-eval           Skip evaluation entirely
#   --no-accumulate       Train on single batches instead of accumulating all data
#   --max-data-files N    Max data files to keep (default: 10, 0 = keep all)
#
# Examples:
#   ./start_continuous_training.sh                          # Default: accumulate all data
#   ./start_continuous_training.sh --data-dir ./my_data     # Custom data directory
#   ./start_continuous_training.sh --hidden-dim 1024        # Larger model
#   ./start_continuous_training.sh --eval-interval 10       # Eval every 10 cycles
#   ./start_continuous_training.sh --skip-eval              # No evaluation
#   ./start_continuous_training.sh --no-accumulate          # Train on single batches (old behavior)

set -e

echo "=================================================="
echo "Continuous Poker Training System with Evaluation"
echo "=================================================="
echo ""
echo "This will run indefinitely until you press Ctrl+C"
echo "The system will automatically:"
echo "  - Generate training data in batches"
echo "  - Accumulate data and train on larger datasets over time"
echo "  - Train models on the generated data"
echo "  - Evaluate model vs random baseline"
echo "  - Balance CPU and memory usage"
echo "  - Update the model continuously"
echo ""
echo "Press Ctrl+C to stop gracefully"
echo ""
echo "=================================================="
echo ""

# Check if dependencies are available
if ! uv run python -c "import psutil" 2>/dev/null; then
    echo "⚠️  Installing required dependencies..."
    uv sync
    echo ""
fi

# Extract data-dir from arguments if specified, otherwise use default
DATA_DIR="/tmp/pokersim/data"
MODEL_DIR="/tmp/pokersim/models"

# Parse command line arguments to find --data-dir (without consuming them)
i=1
while [[ $i -le $# ]]; do
    arg="${!i}"
    case "$arg" in
        --data-dir)
            next=$((i+1))
            DATA_DIR="${!next}"
            ;;
        --data-dir=*)
            DATA_DIR="${arg#*=}"
            ;;
    esac
    i=$((i+1))
done

# Print paths
echo "📁 Storage Paths:"
echo "   Data directory:  $DATA_DIR"
echo "   Model directory: $MODEL_DIR"
echo ""

# Extract and display model configuration
echo "🧠 Model Architecture & Parameters:"
echo ""

# Parse model parameters from command line args with defaults from continuous_trainer.py
HIDDEN_DIM=512
NUM_HEADS=16
NUM_LAYERS=8
DROPOUT=0.2
LEARNING_RATE=0.0001
BATCH_SIZE=32
WEIGHT_DECAY=0.001
LR_SCHEDULER="cosine"
LR_WARMUP=5
EPOCHS_PER_CYCLE=50
GRADIENT_CKPT="enabled"
MIXED_PRECISION="disabled"
ADAPTIVE_SCHEDULE="enabled"

# Parse command line arguments to override defaults
i=1
while [[ $i -le $# ]]; do
    arg="${!i}"
    case "$arg" in
        --hidden-dim)
            next=$((i+1))
            HIDDEN_DIM="${!next}"
            ;;
        --hidden-dim=*)
            HIDDEN_DIM="${arg#*=}"
            ;;
        --num-heads)
            next=$((i+1))
            NUM_HEADS="${!next}"
            ;;
        --num-heads=*)
            NUM_HEADS="${arg#*=}"
            ;;
        --num-layers)
            next=$((i+1))
            NUM_LAYERS="${!next}"
            ;;
        --num-layers=*)
            NUM_LAYERS="${arg#*=}"
            ;;
        --dropout)
            next=$((i+1))
            DROPOUT="${!next}"
            ;;
        --dropout=*)
            DROPOUT="${arg#*=}"
            ;;
        --learning-rate)
            next=$((i+1))
            LEARNING_RATE="${!next}"
            ;;
        --learning-rate=*)
            LEARNING_RATE="${arg#*=}"
            ;;
        --batch-size)
            next=$((i+1))
            BATCH_SIZE="${!next}"
            ;;
        --batch-size=*)
            BATCH_SIZE="${arg#*=}"
            ;;
        --weight-decay)
            next=$((i+1))
            WEIGHT_DECAY="${!next}"
            ;;
        --weight-decay=*)
            WEIGHT_DECAY="${arg#*=}"
            ;;
        --lr-scheduler)
            next=$((i+1))
            LR_SCHEDULER="${!next}"
            ;;
        --lr-scheduler=*)
            LR_SCHEDULER="${arg#*=}"
            ;;
        --lr-warmup-epochs)
            next=$((i+1))
            LR_WARMUP="${!next}"
            ;;
        --lr-warmup-epochs=*)
            LR_WARMUP="${arg#*=}"
            ;;
        --epochs-per-cycle)
            next=$((i+1))
            EPOCHS_PER_CYCLE="${!next}"
            ;;
        --epochs-per-cycle=*)
            EPOCHS_PER_CYCLE="${arg#*=}"
            ;;
        --no-gradient-checkpointing)
            GRADIENT_CKPT="disabled"
            ;;
        --mixed-precision)
            MIXED_PRECISION="enabled"
            ;;
        --no-adaptive-schedule)
            ADAPTIVE_SCHEDULE="disabled"
            ;;
    esac
    i=$((i+1))
done

# Calculate approximate model parameters
# Transformer architecture formula (simplified):
# Total params ≈ (num_layers * (4 * hidden_dim^2 + 2 * hidden_dim * num_heads)) + embeddings
# This is a rough estimate
PARAMS_PER_LAYER=$((4 * HIDDEN_DIM * HIDDEN_DIM + 2 * HIDDEN_DIM * NUM_HEADS))
TOTAL_PARAMS=$((NUM_LAYERS * PARAMS_PER_LAYER))
# Format with commas
TOTAL_PARAMS_FORMATTED=$(printf "%'d" $TOTAL_PARAMS)

echo "   Architecture:         Transformer (PokerTransformer)"
echo "   ─────────────────────────────────────────────────────"
echo "   Model Configuration:"
echo "     • Hidden dimension: $HIDDEN_DIM"
echo "     • Attention heads:  $NUM_HEADS"
echo "     • Transformer layers: $NUM_LAYERS"
echo "     • Dropout rate:     $DROPOUT"
echo "     • Approx. params:   ~$TOTAL_PARAMS_FORMATTED"
echo ""
echo "   Training Configuration:"
echo "     • Learning rate:    $LEARNING_RATE"
echo "     • Batch size:       $BATCH_SIZE"
echo "     • Weight decay:     $WEIGHT_DECAY"
echo "     • LR scheduler:     $LR_SCHEDULER"
echo "     • LR warmup epochs: $LR_WARMUP"
echo "     • Epochs/cycle:     $EPOCHS_PER_CYCLE"
echo ""
echo "   Optimization Features:"
echo "     • Gradient checkpointing: $GRADIENT_CKPT"
echo "     • Mixed precision:        $MIXED_PRECISION"
echo "     • Adaptive schedule:      $ADAPTIVE_SCHEDULE"
echo ""
echo "=================================================="
echo ""

# Run the continuous trainer with evaluation support
# All arguments are passed through to continuous_trainer.py
uv run python continuous_trainer.py "$@"

