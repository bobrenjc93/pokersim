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
#   ./start_continuous_training.sh --batch-size 2000        # Larger batches
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
    uv pip install -r requirements.txt
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

# Run the continuous trainer with evaluation support
# All arguments are passed through to continuous_trainer.py
uv run python continuous_trainer.py "$@"

