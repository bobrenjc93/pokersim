#!/usr/bin/env python3
"""
Deterministic head-to-head Elo-format evaluator.

Why this exists:
- When "iter_1 looks best" it can be either a real training regression or a
  checkpoint loading/selection bug.
- This script evaluates two specific participants under the same match format
  as the Elo server (freezeout rounds, alternating positions, per-player stacks).

Examples:
  uv run python training/elo_head_to_head.py --a /tmp/pokersim/rl_models_v17/iter_1.pt --b /tmp/pokersim/rl_models_v17/iter_700.pt
  uv run python training/elo_head_to_head.py --a /tmp/.../iter_700.pt --b calling_station
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch

from common import detect_device
# `pokersim-elo` installs `engine.py` as a top-level module. Import it directly so
# `uv run elo-head-to-head ...` works regardless of the current working directory.
from engine import PokerEloArena


def _participant_from_arg(arg: str, name: str) -> Tuple[str, Dict]:
    """
    Return (player_id, config) for PokerEloArena.

    arg may be:
    - path to .pt checkpoint
    - heuristic agent type (e.g. calling_station, tight, random, aggressive, ...)
    """
    p = Path(arg)
    if p.exists() and p.is_file():
        pid = p.stem  # e.g. iter_700
        return pid, {"type": "model", "path": str(p), "name": name, "deterministic": True}
    # Heuristic agent type
    pid = arg
    return pid, {"type": arg, "name": name}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Elo-format head-to-head evaluator")
    ap.add_argument("--a", required=True, help="Checkpoint path or heuristic agent type")
    ap.add_argument("--b", required=True, help="Checkpoint path or heuristic agent type")
    ap.add_argument("--rounds", type=int, default=50, help="Freezeout rounds per match")
    ap.add_argument("--max-hands", type=int, default=200, help="Max hands per freezeout round")
    ap.add_argument("--k", type=float, default=40.0, help="Elo k-factor (only affects ratings, not score)")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed for deterministic decks (set to -1 to disable)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument(
        "--same-decks",
        action="store_true",
        help=(
            "Use stable seed labels so different checkpoints are evaluated on the same deck sequence "
            "(reduces variance when comparing iter_1 vs iter_700, or a model vs a fixed heuristic)."
        ),
    )
    ap.add_argument("--seed-id-a", default="", help="Override seed label for A (advanced)")
    ap.add_argument("--seed-id-b", default="", help="Override seed label for B (advanced)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = detect_device() if args.device == "auto" else torch.device(args.device)

    pid_a, cfg_a = _participant_from_arg(args.a, "A")
    pid_b, cfg_b = _participant_from_arg(args.b, "B")

    # Optional seed-label overrides to make comparisons more apples-to-apples.
    if args.same_decks:
        # If A/B are models, pin their seed label to "model" so iter_1 and iter_700
        # see the same deterministic deck sequence.
        if cfg_a.get("type") == "model":
            cfg_a["seed_id"] = "model"
        else:
            cfg_a["seed_id"] = cfg_a.get("type", pid_a)
        if cfg_b.get("type") == "model":
            cfg_b["seed_id"] = "model"
        else:
            cfg_b["seed_id"] = cfg_b.get("type", pid_b)
    if str(args.seed_id_a).strip():
        cfg_a["seed_id"] = str(args.seed_id_a).strip()
    if str(args.seed_id_b).strip():
        cfg_b["seed_id"] = str(args.seed_id_b).strip()

    arena = PokerEloArena(
        device=str(device),
        k_factor=float(args.k),
        max_cached_models=2,
        base_seed=None if int(args.seed) < 0 else int(args.seed),
        rounds_per_match=int(args.rounds),
        max_hands_per_round=int(args.max_hands),
    )

    res = arena.play_match(pid_a, pid_b, cfg_a, cfg_b)
    if res.get("error"):
        msg = res.get("message") or "match failed"
        print(f"FAIL: {msg}")
        return 2

    print(f"Device: {device}")
    print(f"A: {pid_a} ({cfg_a.get('type')})")
    print(f"B: {pid_b} ({cfg_b.get('type')})")
    print(
        f"Result: score_a={res['score_a']}  "
        f"round_wins={res['round_wins_a']}-{res['round_wins_b']}  "
        f"hands={res['hands_played']}"
    )
    print(
        f"Ratings: {pid_a} {res['old_rating_a']:.1f}->{res['new_rating_a']:.1f}  "
        f"{pid_b} {res['old_rating_b']:.1f}->{res['new_rating_b']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


