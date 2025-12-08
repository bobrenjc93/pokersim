#!/usr/bin/env python3
"""Reinforcement Learning Training for Poker AI with PPO and self-play."""

import argparse
import random
import resource
import sys
import time
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, get_args, get_origin

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

from common import (
    RLStateEncoder, create_actor_critic, MODEL_VERSION, LOG_LEVEL,
    DEFAULT_MODELS_DIR, GameConfig, parse_checkpoints, select_spread_checkpoints,
    ELO_SMALL_BLIND, ELO_BIG_BLIND, ELO_STARTING_STACK,
)
from ppo import PPOTrainer
from episode_collector import EpisodeCollector
from parallel_rollouts import ParallelRolloutManager
# `pokersim-elo` installs `engine.py` as a top-level module (see elo/pyproject.toml),
# so importing via `elo.engine` only works when the repo root is on PYTHONPATH.
# Keep this import stable for `uv run train-poker` (which runs from `training/`).
from engine import PokerEloArena


@dataclass
class TrainConfig:
    """Training configuration with sensible defaults."""
    # Training loop
    iterations: int = 5000
    episodes_per_iter: int = 1000
    
    # PPO hyperparameters
    ppo_epochs: int = 15
    mini_batch_size: int = 128
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.03  # Higher base entropy to prevent early collapse
    value_loss_coef: float = 0.5
    hand_strength_loss_coef: float = 0.10
    lr_warmup_steps: int = 100
    advantage_clip: float = 10.0
    # PopArt is experimental in this repo. Historically we normalized return targets
    # without consistently normalizing the value predictions/GAE, which can make
    # training *worse* over time. Keep it off by default for correctness.
    use_popart: bool = False
    use_adaptive_entropy: bool = True
    
    # Model architecture
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    gradient_checkpointing: bool = True
    
    # Game settings
    # Default to the same blinds/stacks as the Elo arena so "training improving"
    # lines up with "Elo arena improving".
    small_blind: int = ELO_SMALL_BLIND
    big_blind: int = ELO_BIG_BLIND
    starting_chips: int = ELO_STARTING_STACK
    
    # I/O
    output_dir: str = DEFAULT_MODELS_DIR
    tensorboard_dir: str = f'/tmp/pokersim/tensorboard_v{MODEL_VERSION}'
    use_tensorboard: bool = True
    save_interval: int = 100
    checkpoint: Optional[str] = None
    
    # Evaluation
    elo_eval_interval: int = 50
    elo_eval_rounds: int = 50

    # Eval vs past checkpoints (freezeout format, like the ELO simulation)
    checkpoint_eval_interval: int = 50
    checkpoint_eval_rounds: int = 30
    checkpoint_eval_num_checkpoints: int = 6
    checkpoint_eval_max_hands_per_round: int = 60
    # Deterministic deck seeding for eval (None disables deterministic seeding)
    checkpoint_eval_base_seed: Optional[int] = 0
    
    # Resources
    num_workers: int = 0
    max_memory_gb: float = 20.0
    verbose: bool = False
    seed: Optional[int] = None
    deterministic: bool = False

    @property
    def game_config(self) -> GameConfig:
        return GameConfig(small_blind=self.small_blind, big_blind=self.big_blind, starting_chips=self.starting_chips)

    @property
    def model_config(self) -> Dict:
        return {'hidden_dim': self.hidden_dim, 'num_heads': self.num_heads, 'num_layers': self.num_layers, 'dropout': self.dropout}


def parse_args() -> TrainConfig:
    """Parse CLI arguments into TrainConfig."""
    p = argparse.ArgumentParser(description="Train poker AI with PPO")
    defaults = TrainConfig()
    
    for field in fields(TrainConfig):
        name = field.name.replace('_', '-')
        default = getattr(defaults, field.name)

        # Bool: allow --flag / --no-flag (Python 3.9+)
        if field.type is bool:
            p.add_argument(f"--{name}", default=default, action=argparse.BooleanOptionalAction)
            continue

        origin = get_origin(field.type)
        if origin is None:
            p.add_argument(f"--{name}", type=field.type, default=default)
            continue

        # typing.Optional[X] is typing.Union[X, NoneType]
        if origin is Union and type(None) in get_args(field.type):
            args = [a for a in get_args(field.type) if a is not type(None)]
            if len(args) == 1:
                p.add_argument(f"--{name}", type=args[0], default=default)
            else:
                p.add_argument(f"--{name}", type=str, default=default)
            continue

        # Fallback: treat as string
        p.add_argument(f"--{name}", type=str, default=default)
    
    args = p.parse_args()
    return TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python/NumPy/PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Best-effort: some backends (e.g. MPS) may not support full determinism.
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass


def aggregate_episodes(episodes: List[Dict]) -> Dict[str, torch.Tensor]:
    """Combine episodes into batched tensors for PPO update."""
    rewards, hand_strengths = [], []
    
    for ep in episodes:
        n = len(ep['states'])
        step_r = list(ep.get('step_rewards', [0.0] * n))
        step_r[-1] += ep.get('main_reward', 0.0)
        rewards.extend(step_r)
        hand_strengths.extend(ep.get('hand_strengths', [0.5] * n))
    
    return {
        'states': torch.cat([ep['states'] for ep in episodes]),
        'actions': torch.cat([ep['actions'] for ep in episodes]),
        'log_probs': torch.cat([ep['log_probs'] for ep in episodes]),
        'values': torch.cat([ep['values'] for ep in episodes]),
        'masks': torch.cat([ep['legal_actions_masks'] for ep in episodes]),
        'dones': torch.cat([ep['dones'] for ep in episodes]),
        'rewards': torch.tensor(rewards, dtype=torch.float32),
        'hand_strengths': torch.tensor(hand_strengths, dtype=torch.float32),
    }


class Trainer:
    """PPO Trainer for Poker AI with self-play."""
    
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        # Model setup
        encoder = RLStateEncoder()
        self.input_dim = encoder.get_feature_dim()
        base_model = create_actor_critic(
            input_dim=self.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            gradient_checkpointing=cfg.gradient_checkpointing
        )
        
        self.model_cfg = {**cfg.model_config, 'input_dim': self.input_dim}
        self.ppo = PPOTrainer(
            model=base_model,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_epsilon=cfg.clip_epsilon,
            entropy_coef=cfg.entropy_coef,
            value_loss_coef=cfg.value_loss_coef,
            hand_strength_loss_coef=cfg.hand_strength_loss_coef,
            ppo_epochs=cfg.ppo_epochs,
            mini_batch_size=cfg.mini_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            lr_schedule_steps=cfg.iterations,
            lr_warmup_steps=cfg.lr_warmup_steps,
            use_popart=cfg.use_popart,
            use_adaptive_entropy=cfg.use_adaptive_entropy,
            advantage_clip=cfg.advantage_clip,
            device=self.device,
            accelerator=self.accelerator
        )

        # IMPORTANT: PPOTrainer may wrap the model via Accelerate (e.g. DDP wrappers).
        # Keep a single authoritative reference for rollouts + updates.
        self.model = self.ppo.model
        
        # State
        self.iteration = 0
        self.total_episodes = 0
        self.total_steps = 0
        self.output_dir = Path(cfg.output_dir)
        self.checkpoints: List = []
        self.writer: Optional[SummaryWriter] = None
        self.collector: Optional[EpisodeCollector] = None
        self.rollout_mgr: Optional[ParallelRolloutManager] = None
    
    def _log(self, msg: str):
        if self.cfg.verbose or LOG_LEVEL >= 1:
            print(msg)

    def _unwrapped_state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        """
        Return a CPU state_dict suitable for checkpointing and for sync to
        unwrapped worker models.
        """
        m = self.model
        try:
            if self.accelerator is not None and hasattr(self.accelerator, "unwrap_model"):
                m = self.accelerator.unwrap_model(self.model)
        except Exception:
            m = self.model
        return {k: v.detach().cpu() for k, v in m.state_dict().items()}
    
    def setup(self) -> int:
        """Initialize training. Returns start iteration."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Seeding (do this early so opponent sampling + env seeds are stable)
        if self.cfg.seed is not None:
            seed_everything(self.cfg.seed, deterministic=self.cfg.deterministic)
        
        # Load checkpoint if provided
        start_iter = 0
        if self.cfg.checkpoint and Path(self.cfg.checkpoint).exists():
            ckpt = self.ppo.load_checkpoint(self.cfg.checkpoint)
            start_iter = ckpt.get('epoch', 0)
            self._log(f"✓ Resumed from iter {start_iter}")
        
        # TensorBoard
        if self.cfg.use_tensorboard:
            log_dir = Path(self.cfg.tensorboard_dir) / f"run_{datetime.now():%Y%m%d_%H%M%S}"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
        
        # Load existing checkpoints for opponent pool
        self.checkpoints = parse_checkpoints(self.output_dir)
        if self.checkpoints:
            self._log(f"📚 Loaded {len(self.checkpoints)} checkpoints")
        
        self._setup_collectors()
        
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        workers = f" ({self.cfg.num_workers} workers)" if self.cfg.num_workers > 0 else ""
        self._log(f"✓ Model: {params:,} params on {self.device}")
        self._log(f"🚀 Training: {self.cfg.iterations} iters × {self.cfg.episodes_per_iter} eps{workers}")
        
        return start_iter
    
    def _setup_collectors(self):
        """Setup episode collectors (parallel or single-process)."""
        pool = self._get_opponent_pool()
        game_cfg = self.cfg.game_config.to_dict()
        
        if self.cfg.num_workers > 0:
            try:
                self.rollout_mgr = ParallelRolloutManager(
                    num_workers=self.cfg.num_workers,
                    game_config=game_cfg,
                    model_config=self.model_cfg,
                    model=self.model,
                    device='cpu',
                    opponent_pool=pool,
                    base_seed=self.cfg.seed,
                )
                self.rollout_mgr.start(state_dict=self._unwrapped_state_dict_cpu())
                return
            except (PermissionError, OSError) as e:
                print(f"⚠️  Multiprocessing unavailable ({e}), using single-process")
                self.cfg.num_workers = 0
        
        self.collector = EpisodeCollector(
            model=self.model,
            game_config=game_cfg,
            device=self.device,
            model_config=self.model_cfg,
            opponent_pool=pool,
            seed=self.cfg.seed,
        )
    
    def _get_opponent_pool(self) -> List[str]:
        if not self.checkpoints:
            return []
        return [str(p) for _, p in select_spread_checkpoints(self.checkpoints, max_checkpoints=30)]
    
    def _collect_episodes(self) -> List[Dict]:
        """Collect training episodes."""
        pool = self._get_opponent_pool()
        if self.rollout_mgr:
            self.rollout_mgr.update_opponent_pool(pool)
            return self.rollout_mgr.collect_episodes(self.cfg.episodes_per_iter, verbose=self.cfg.verbose)
        self.collector.opponent_pool = pool
        return self.collector.collect_episodes(self.cfg.episodes_per_iter, verbose=self.cfg.verbose)
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = self.output_dir / f"{name}.pt"
        self.ppo.save_checkpoint(str(path), epoch=self.iteration, game_config=self.cfg.game_config.to_dict(), model_config=self.model_cfg)
        self._log(f"💾 Saved: {path}")
        
        # Track iteration checkpoints for opponent pool
        if name.startswith("iter_"):
            try:
                it = int(name.split("_")[1])
                self.checkpoints.append((it, path))
                self.checkpoints.sort(key=lambda x: x[0])
            except (IndexError, ValueError):
                pass

    def _maybe_elo_eval(self) -> None:
        """
        Periodically run a small ELO-style freezeout evaluation.

        This is a sanity check that training is improving *under the same match
        format used by the ELO server* (unequal stacks supported, alternating
        positions, freezeout rounds).
        """
        if self.cfg.elo_eval_interval <= 0:
            return
        if self.iteration <= 0 or (self.iteration % self.cfg.elo_eval_interval) != 0:
            return

        # Keep this lightweight: evaluate vs a couple of fixed heuristic opponents.
        opponents = ["calling_station", "tight"]
        rounds = max(1, int(self.cfg.elo_eval_rounds))
        max_hands = 60  # cap to keep eval quick during training

        # Use the unwrapped model to avoid DDP/Accelerate wrapper surprises.
        model_for_eval = self.model
        try:
            if self.accelerator is not None and hasattr(self.accelerator, "unwrap_model"):
                model_for_eval = self.accelerator.unwrap_model(self.model)
        except Exception:
            model_for_eval = self.model

        arena = PokerEloArena(
            device=str(self.device),
            k_factor=40.0,
            max_cached_models=2,
            rounds_per_match=rounds,
            max_hands_per_round=max_hands,
        )

        cfg_model = {
            "type": "model",
            "name": f"Iter_{self.iteration}",
            "model": model_for_eval,
            # Keep decks comparable across iterations: the Elo arena's deterministic
            # seeding can use `seed_id` instead of `player_id`.
            "seed_id": "trained_model",
            "deterministic": True,
        }

        for opp in opponents:
            cfg_opp = {"type": opp, "name": opp.replace("_", " ").title(), "seed_id": opp}
            # Unique IDs so ratings don't collide across opponents.
            pid_model = f"iter_{self.iteration}"
            pid_opp = f"opp_{opp}"
            res = arena.play_match(pid_model, pid_opp, cfg_model, cfg_opp)
            if res.get("error"):
                self._log(f"⚠️  ELO eval failed vs {opp}")
                continue

            score = res.get("score_a")
            # score_a is 1/0.5/0 (match win/draw/loss)
            if self.writer:
                self.writer.add_scalar(f"ELO/score_vs_{opp}", float(score), self.iteration)
            self._log(f"🎯 ELO-eval vs {opp}: score={score} (round_wins {res.get('round_wins_a')}-{res.get('round_wins_b')}, hands={res.get('hands_played')})")

    def _maybe_checkpoint_eval(self) -> None:
        """
        Periodically evaluate the current (in-memory) model vs a small set of
        *past saved checkpoints* in the same output directory.
        
        This uses the same freezeout match logic as the ELO simulation and prints
        a comparable summary (round_wins + hands), plus a win-rate.
        """
        if self.cfg.checkpoint_eval_interval <= 0:
            return
        if self.iteration <= 0 or (self.iteration % self.cfg.checkpoint_eval_interval) != 0:
            return

        # Need at least one past checkpoint to compare against.
        if not self.checkpoints:
            return

        # Use the unwrapped model to avoid DDP/Accelerate wrapper surprises.
        model_for_eval = self.model
        try:
            if self.accelerator is not None and hasattr(self.accelerator, "unwrap_model"):
                model_for_eval = self.accelerator.unwrap_model(self.model)
        except Exception:
            model_for_eval = self.model

        # Select a spread of checkpoints strictly before the current iteration.
        # (Avoid evaluating vs ourselves if iter_N was just saved.)
        past = [(it, p) for (it, p) in self.checkpoints if int(it) < int(self.iteration)]
        if not past:
            return

        max_n = max(1, int(self.cfg.checkpoint_eval_num_checkpoints))
        selected = select_spread_checkpoints(sorted(past, key=lambda x: x[0]), max_checkpoints=max_n)
        if not selected:
            return

        rounds = max(1, int(self.cfg.checkpoint_eval_rounds))
        max_hands = max(1, int(self.cfg.checkpoint_eval_max_hands_per_round))

        base_seed = self.cfg.checkpoint_eval_base_seed
        try:
            base_seed = None if base_seed is None else int(base_seed)
        except Exception:
            base_seed = 0

        arena = PokerEloArena(
            device=str(self.device),
            k_factor=40.0,
            max_cached_models=2,
            rounds_per_match=rounds,
            max_hands_per_round=max_hands,
            base_seed=base_seed,
        )

        cfg_model = {
            "type": "model",
            "name": f"Iter_{self.iteration}",
            "model": model_for_eval,
            # Keep decks comparable across training iterations.
            "seed_id": "trained_model",
            "deterministic": True,
        }

        for past_it, past_path in selected:
            # The arena will load the checkpoint via its ModelCache.
            cfg_past = {
                "type": "model",
                "name": "Baseline" if int(past_it) == -1 else f"Iter_{int(past_it)}",
                "path": str(past_path),
                # Stable seed label per checkpoint (don't depend on player_id).
                "seed_id": f"ckpt_{int(past_it)}",
                "deterministic": True,
            }

            pid_model = f"iter_{self.iteration}"
            pid_past = "baseline" if int(past_it) == -1 else f"iter_{int(past_it)}"
            res = arena.play_match(pid_model, pid_past, cfg_model, cfg_past)
            if res.get("error"):
                self._log(f"⚠️  Checkpoint eval failed vs {cfg_past['name']}")
                continue

            rounds_played = int(res.get("rounds_played") or 0)
            w = int(res.get("round_wins_a") or 0)
            l = int(res.get("round_wins_b") or 0)
            d = max(0, rounds_played - w - l)
            win_rate = (w / rounds_played) if rounds_played > 0 else 0.0

            if self.writer:
                tag = f"ELO_vs_checkpoints/win_rate_vs_{pid_past}"
                self.writer.add_scalar(tag, float(win_rate), self.iteration)

            # Print in the same style as the ELO sanity check, but with win-rate.
            self._log(
                f"🎯 ELO-eval vs {pid_past}: win={win_rate:.1%} "
                f"(round_wins {w}-{l}, draws={d}, hands={res.get('hands_played')})"
            )
    
    def train_iteration(self, i: int) -> bool:
        """Run single training iteration. Returns success."""
        t0 = time.time()
        
        # Collect episodes
        episodes = self._collect_episodes()
        valid = [ep for ep in episodes if ep.get('success') and len(ep.get('states', [])) > 0]
        if not valid:
            return False
        
        t_collect = time.time() - t0
        
        # Prepare trajectory for PPO
        trajectory = aggregate_episodes(valid)
        with torch.no_grad():
            _, next_val = self.model(
                trajectory['states'][-1:].to(self.device),
                trajectory['masks'][-1:].to(self.device)
            )
        
        advantages, returns = self.ppo.compute_gae(
            trajectory['rewards'], trajectory['values'],
            trajectory['dones'], next_val.squeeze().cpu()
        )
        
        # PPO update
        t1 = time.time()
        stats = self.ppo.update(
            states=trajectory['states'],
            actions=trajectory['actions'],
            old_log_probs=trajectory['log_probs'],
            old_values=trajectory['values'],
            advantages=advantages,
            returns=returns,
            legal_actions_masks=trajectory['masks'],
            hand_strengths=trajectory['hand_strengths'],
            verbose=self.cfg.verbose
        )
        t_train = time.time() - t1
        
        # Stats
        rewards = [ep.get('main_reward', 0.0) for ep in valid]
        win_rate = sum(1 for r in rewards if r > 0) / len(valid)
        avg_reward = np.mean(rewards)
        
        # Fold-rate monitoring for collapse detection
        fold_rates = [ep.get('fold_rate', 0.0) for ep in valid]
        avg_fold_rate = np.mean(fold_rates) if fold_rates else 0.0
        avg_steps = np.mean([ep.get('num_steps', 0) for ep in valid])
        
        # Detect policy collapse: high fold rate + very short episodes
        if avg_fold_rate > 0.6 and avg_steps < 3:
            print(f"⚠️  POLICY COLLAPSE DETECTED: fold_rate={avg_fold_rate:.1%}, steps={avg_steps:.1f}")
            print(f"   Consider: reset to earlier checkpoint, increase entropy_coef, reduce self-play")
        
        self.iteration = i + 1
        self.total_episodes += len(valid)
        self.total_steps += len(trajectory['states'])
        
        # Sync weights to workers
        if self.rollout_mgr:
            # See _setup_collectors(): workers need an unwrapped state_dict.
            self.rollout_mgr.update_model_weights(state_dict=self._unwrapped_state_dict_cpu())
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar('Training/AvgReward', avg_reward, self.iteration)
            self.writer.add_scalar('Training/WinRate', win_rate, self.iteration)
            for k in ['policy_loss', 'value_loss', 'entropy']:
                if k in stats:
                    self.writer.add_scalar(f'Training/{k}', stats[k], self.iteration)
        
        self._log(f"[{self.iteration}] {len(valid)} eps, {len(trajectory['states'])} steps | "
                  f"reward={avg_reward:.3f} win={win_rate:.1%} | "
                  f"π={stats.get('policy_loss', 0):.3f} | {t_collect:.1f}s+{t_train:.1f}s")
        
        # Save checkpoints
        if self.iteration % self.cfg.save_interval == 0 or i == 0:
            self._save_checkpoint(f"iter_{self.iteration}")
            self._save_checkpoint("latest")

        # Periodic ELO sanity-check
        self._maybe_elo_eval()

        # Periodic evaluation vs past checkpoints (ELO-style freezeout)
        self._maybe_checkpoint_eval()
        
        return True
    
    def cleanup(self):
        """Cleanup resources."""
        if self.rollout_mgr:
            self.rollout_mgr.shutdown()
        if self.writer:
            self.writer.close()


def train(cfg: TrainConfig) -> int:
    """Main training loop."""
    trainer = Trainer(cfg)
    start_iter = trainer.setup()
    
    try:
        for i in range(start_iter, cfg.iterations):
            trainer.train_iteration(i)
        trainer._save_checkpoint("final")
    finally:
        trainer.cleanup()
    
    print(f"✓ Done! {trainer.total_episodes} episodes, {trainer.total_steps} steps")
    return 0


def main() -> int:
    """Entry point."""
    cfg = parse_args()
    
    # Memory limit
    if cfg.max_memory_gb > 0:
        try:
            max_bytes = int(cfg.max_memory_gb * 1024**3)
            resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        except (ValueError, resource.error):
            pass
    
    return train(cfg)


if __name__ == "__main__":
    sys.exit(main())
