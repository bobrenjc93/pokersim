#!/usr/bin/env python3
"""Evaluate trained poker models against opponents."""

import argparse
from dataclasses import dataclass
import sys
import time
from pathlib import Path
from typing import Dict

import torch

from common import ModelAgent, GameConfig, PokerSimulator, detect_device, AGENT_CLASSES, create_agent


@dataclass(frozen=True)
class EvalConfig:
    model: str
    num_hands: int = 100
    opponent: str = "random"
    small_blind: int = 10
    big_blind: int = 20
    starting_chips: int = 1000
    verbose: bool = False
    device: str = "auto"  # auto|cpu|cuda|mps

    @property
    def game_config(self) -> GameConfig:
        return GameConfig(
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            starting_chips=self.starting_chips,
        )

    def resolve_device(self) -> torch.device:
        return detect_device() if self.device == "auto" else torch.device(self.device)


def evaluate(
    model_path: str,
    num_hands: int = 100,
    opponent_type: str = 'random',
    game_config: GameConfig = None,
    device: torch.device = None,
    verbose: bool = False
) -> Dict:
    """Evaluate a trained model against an opponent."""
    device = device or detect_device()
    config = game_config or GameConfig()
    
    try:
        model_agent = ModelAgent('p0', 'Model', model_path=model_path, device=device, deterministic=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return {}
    
    opponent = create_agent(opponent_type, 'p1')
    if not opponent:
        print(f"Unknown opponent: {opponent_type}")
        return {}
    
    print(f"✓ Evaluating vs {opponent_type} for {num_hands} hands on {device}")
    
    sim = PokerSimulator(config)
    agents = {'p0': model_agent, 'p1': opponent}
    stats = {'hands_played': 0, 'hands_won': 0, 'hands_lost': 0, 'hands_tied': 0, 'total_profit': 0}
    
    start = time.time()
    
    for i in range(num_hands):
        result = sim.play_hand(agents)
        
        if not result.get('success'):
            if verbose:
                print(f"Error in hand {i+1}: {result.get('error')}")
            continue
        
        profit = result.get('profits', {}).get('p0', 0)
        stats['total_profit'] += profit
        stats['hands_played'] += 1
        stats['hands_won'] += profit > 0
        stats['hands_lost'] += profit < 0
        stats['hands_tied'] += profit == 0
        
        if (i + 1) % 10 == 0 and stats['hands_played'] > 0:
            n = stats['hands_played']
            elapsed = time.time() - start
            win_rate = stats['hands_won'] / n * 100
            avg_profit = stats['total_profit'] / n
            eta = (num_hands - i - 1) / ((i + 1) / elapsed) if elapsed > 0 else 0
            print(f"  Hand {i+1}/{num_hands} | Win: {win_rate:.1f}% | Avg: {avg_profit:.1f} | ETA: {eta:.0f}s")
    
    return stats


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Evaluate Poker AI Model")
    p.add_argument('--model', type=str, required=True, help="Path to model checkpoint")
    p.add_argument('--num-hands', type=int, default=100)
    p.add_argument('--opponent', type=str, default='random', choices=list(AGENT_CLASSES.keys()))
    p.add_argument('--small-blind', type=int, default=10)
    p.add_argument('--big-blind', type=int, default=20)
    p.add_argument('--starting-chips', type=int, default=1000)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    a = p.parse_args()
    return EvalConfig(
        model=a.model,
        num_hands=a.num_hands,
        opponent=a.opponent,
        small_blind=a.small_blind,
        big_blind=a.big_blind,
        starting_chips=a.starting_chips,
        verbose=a.verbose,
        device=a.device,
    )


def main() -> int:
    cfg = parse_args()

    if not Path(cfg.model).exists():
        print(f"Model not found: {cfg.model}")
        return 1

    device = cfg.resolve_device()
    stats = evaluate(
        model_path=str(cfg.model),
        num_hands=cfg.num_hands,
        opponent_type=cfg.opponent,
        game_config=cfg.game_config,
        device=device,
        verbose=cfg.verbose
    )
    
    if not stats or stats['hands_played'] == 0:
        print("Evaluation failed")
        return 1
    
    n = stats['hands_played']
    print(f"\n{'='*50}\nResults\n{'='*50}")
    print(f"Hands: {n}")
    print(f"Win Rate: {stats['hands_won'] / n * 100:.2f}%")
    print(f"W/L/T: {stats['hands_won']}/{stats['hands_lost']}/{stats['hands_tied']}")
    print(f"Total Profit: {stats['total_profit']}")
    print(f"Avg Profit: {stats['total_profit'] / n:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
