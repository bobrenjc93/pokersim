#!/usr/bin/env bash
set -euo pipefail

# Quick sanity check to confirm:
# - checkpoints load
# - different iterations are actually different weights
# - later iterations don't accidentally evaluate as iter_1 due to misconfiguration
#
# Usage:
#   ./smoke_elo_sanity.sh /tmp/pokersim/rl_models_v17 1 700
#
# If you omit iterations, defaults to (1, last).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

# Add project root to PYTHONPATH so we import common/ directly from source.
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

MODELS_DIR="${1:-}"
if [[ -z "${MODELS_DIR}" ]]; then
  echo "Usage: $0 /path/to/models_dir [iter_a] [iter_b]"
  exit 2
fi

ITER_A="${2:-1}"
ITER_B="${3:-}"

uv run python - "$MODELS_DIR" "$ITER_A" "$ITER_B" <<'PY'
import sys
from pathlib import Path
import torch

from common import parse_checkpoints
from common.model_loading import load_model_from_path
from engine import PokerEloArena

models_dir = Path(sys.argv[1])
it_a = int(sys.argv[2])
it_b_raw = sys.argv[3] if len(sys.argv) > 3 else ""

cps = parse_checkpoints(models_dir)
if not cps:
    raise SystemExit(f"No checkpoints found in {models_dir}")

if it_b_raw:
    it_b = int(it_b_raw)
else:
    it_b = max(i for i,_ in cps if i >= 0)

paths = {it: models_dir / f"iter_{it}.pt" for it in (it_a, it_b)}
for it,p in paths.items():
    if not p.exists():
        raise SystemExit(f"Missing checkpoint: {p}")

def fp(model):
    with torch.no_grad():
        v = torch.cat([p.detach().float().cpu().view(-1) for p in model.parameters()])
    return {"n": int(v.numel()), "l2": float(torch.linalg.vector_norm(v).item()), "mean_abs": float(v.abs().mean().item())}

m_a, _, raw_a = load_model_from_path(str(paths[it_a]), device=torch.device("cpu"), strict=True)
m_b, _, raw_b = load_model_from_path(str(paths[it_b]), device=torch.device("cpu"), strict=True)

print(f"Loaded {paths[it_a].name}: ckpt_epoch={raw_a.get('epoch')} fp={fp(m_a)}")
print(f"Loaded {paths[it_b].name}: ckpt_epoch={raw_b.get('epoch')} fp={fp(m_b)}")

arena = PokerEloArena(device="cpu", base_seed=0, rounds_per_match=20, max_hands_per_round=40)
cfg_a = {"type": "model", "name": f"Iter_{it_a}", "model": m_a, "deterministic": True}
cfg_b = {"type": "model", "name": f"Iter_{it_b}", "model": m_b, "deterministic": True}

res1 = arena.play_match(f"iter_{it_a}", f"iter_{it_b}", cfg_a, cfg_b)
res2 = arena.play_match(f"iter_{it_b}", f"iter_{it_a}", cfg_b, cfg_a)

print("\nMatch results (deterministic, seeded):")
print("A vs B:", res1)
print("B vs A:", res2)
print("\nRatings:", {k: v.rating for k, v in arena.ratings.items()})
PY


