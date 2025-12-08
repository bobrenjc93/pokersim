#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

# Add project root to PYTHONPATH so we import common/ directly from source.
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

MODELS_DIR="${1:-}"
if [[ -z "${MODELS_DIR}" ]]; then
  echo "Usage: $0 /path/to/models_dir"
  exit 2
fi

uv run python verify_checkpoints.py --models-dir "${MODELS_DIR}"


