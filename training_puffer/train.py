#!/usr/bin/env python3
"""
CleanRL-style Vectorized Training Script for Poker AI

This script provides efficient vectorized environment training with PPO.
Based on CleanRL's PPO implementation for clean, readable code.

Key features:
- Vectorized environments for parallel training using multiprocessing
- Standard PPO implementation with GAE
- Automatic checkpointing and logging
- TensorBoard integration

Usage:
    python train.py --total-timesteps 10000000 --num-envs 64
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Local imports
from config import (
    MODEL_VERSION, DEFAULT_MODELS_DIR, LOG_LEVEL,
    NUM_ACTIONS, ACTION_NAMES
)
from poker_env import PokerEnv
from policy import PokerPolicy, create_policy


# =============================================================================
# Vectorized Environment Wrapper
# =============================================================================

def _create_poker_env(small_blind: int, big_blind: int, starting_chips: int, seed: int) -> PokerEnv:
    """
    Module-level function to create a poker environment.
    Must be at module level for multiprocessing pickling to work with 'spawn' method.
    """
    env = PokerEnv(
        small_blind=small_blind,
        big_blind=big_blind,
        starting_chips=starting_chips,
    )
    env.reset(seed=seed)
    return env


def make_env(
    small_blind: int = 10,
    big_blind: int = 20,
    starting_chips: int = 1000,
    seed: int = 0,
    idx: int = 0,
) -> Callable:
    """Create a callable that creates a poker environment using functools.partial."""
    # Use functools.partial with a module-level function instead of a closure
    # This allows proper pickling for multiprocessing with 'spawn' method
    return partial(
        _create_poker_env,
        small_blind=small_blind,
        big_blind=big_blind,
        starting_chips=starting_chips,
        seed=seed + idx,
    )


class VecEnv:
    """
    Simple vectorized environment using serial execution.
    
    For poker, this is often faster than multiprocessing due to the
    overhead of IPC, since each step is fast with C++ bindings.
    """
    
    def __init__(self, env_fns: List[Callable]):
        """
        Args:
            env_fns: List of functions that create environments
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        
        # Get spaces from first env
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        
        # For vectorized env, we just track the single observation space
        # The actual observations are stacked numpy arrays
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments."""
        observations = []
        infos = []
        
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            observations.append(obs)
            infos.append(info)
        
        return np.stack(observations), infos
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments."""
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(int(action))
            
            # Auto-reset if done
            if terminated or truncated:
                # Store final info
                info["final_observation"] = obs
                info["final_info"] = info.copy()
                # Reset environment
                obs, reset_info = env.reset()
                info.update(reset_info)
            
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        
        return (
            np.stack(observations),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos,
        )
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


class AsyncVecEnv:
    """
    Asynchronous vectorized environment using multiprocessing.
    
    Each environment runs in its own process for true parallelism.
    """
    
    def __init__(self, env_fns: List[Callable]):
        """
        Args:
            env_fns: List of functions that create environments
        """
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False
        
        # Create pipes for communication
        self.parent_pipes = []
        self.child_pipes = []
        self.processes = []
        
        for env_fn in env_fns:
            parent_conn, child_conn = mp.Pipe()
            self.parent_pipes.append(parent_conn)
            self.child_pipes.append(child_conn)
            
            process = mp.Process(target=self._worker, args=(child_conn, env_fn))
            process.daemon = True
            process.start()
            self.processes.append(process)
            child_conn.close()  # Close child end in parent
        
        # Get spaces from first env
        self.parent_pipes[0].send(('get_spaces', None))
        spaces = self.parent_pipes[0].recv()
        self.single_observation_space = spaces['observation_space']
        self.single_action_space = spaces['action_space']
        
        # For vectorized env, we just track the single observation space
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
    
    @staticmethod
    def _worker(conn, env_fn):
        """Worker process that runs an environment."""
        env = env_fn()
        
        while True:
            try:
                cmd, data = conn.recv()
            except EOFError:
                break
            
            if cmd == 'step':
                obs, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    info["final_observation"] = obs
                    obs, reset_info = env.reset()
                    info.update(reset_info)
                conn.send((obs, reward, terminated, truncated, info))
            
            elif cmd == 'reset':
                obs, info = env.reset(seed=data)
                conn.send((obs, info))
            
            elif cmd == 'get_spaces':
                conn.send({
                    'observation_space': env.observation_space,
                    'action_space': env.action_space,
                })
            
            elif cmd == 'close':
                env.close()
                conn.close()
                break
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments."""
        for i, pipe in enumerate(self.parent_pipes):
            env_seed = seed + i if seed is not None else None
            pipe.send(('reset', env_seed))
        
        observations = []
        infos = []
        for pipe in self.parent_pipes:
            obs, info = pipe.recv()
            observations.append(obs)
            infos.append(info)
        
        return np.stack(observations), infos
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments."""
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(('step', int(action)))
        
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for pipe in self.parent_pipes:
            obs, reward, terminated, truncated, info = pipe.recv()
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        
        return (
            np.stack(observations),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos,
        )
    
    def close(self):
        """Close all environments."""
        if self.closed:
            return
        
        for pipe in self.parent_pipes:
            try:
                pipe.send(('close', None))
            except (BrokenPipeError, OSError):
                pass
        
        for process in self.processes:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
        
        self.closed = True


# =============================================================================
# PPO Agent
# =============================================================================

class PPOAgent(nn.Module):
    """
    PPO Agent wrapping the policy for CleanRL-style training.
    """
    
    def __init__(
        self,
        obs_dim: int,
        num_actions: int = NUM_ACTIONS,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        
        # Create underlying policy
        self.policy = create_policy(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            recurrent=False,
        )
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        return self.policy.get_value(obs)
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, and value."""
        # Extract action mask from observation (last NUM_ACTIONS values)
        action_mask = obs[:, -self.num_actions:]
        
        return self.policy.get_action_and_value(
            obs, 
            action=action,
            action_mask=action_mask,
        )


# =============================================================================
# Training Function
# =============================================================================

def train(args):
    """Main training function."""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if LOG_LEVEL >= 1:
        print(f"✓ Using device: {device}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tensorboard_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else None
    if tensorboard_dir:
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup run name
    run_name = f"poker_vec_v{MODEL_VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if LOG_LEVEL >= 1:
        print(f"\n{'='*70}")
        print(f"Vectorized Poker Training - {run_name}")
        print(f"{'='*70}")
        print(f"Total timesteps: {args.total_timesteps:,}")
        print(f"Num envs: {args.num_envs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"{'='*70}\n")
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
    
    # Create vectorized environments
    env_fns = [
        make_env(
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            starting_chips=args.starting_chips,
            seed=args.seed,
            idx=i,
        )
        for i in range(args.num_envs)
    ]
    
    # Use sync or async vectorization based on num_envs
    # For small num_envs, sync is faster due to no IPC overhead
    if args.num_envs <= 8 or args.sync_envs:
        vec_env = VecEnv(env_fns)
        if LOG_LEVEL >= 1:
            print(f"✓ Created {args.num_envs} vectorized environments (sync)")
    else:
        vec_env = AsyncVecEnv(env_fns)
        if LOG_LEVEL >= 1:
            print(f"✓ Created {args.num_envs} vectorized environments (async)")
    
    # Get observation dimension
    obs_dim = vec_env.single_observation_space.shape[0]
    
    # Create agent
    agent = PPOAgent(
        obs_dim=obs_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    
    num_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    if LOG_LEVEL >= 1:
        print(f"✓ Created agent with {num_params:,} parameters")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        agent.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
    )
    
    # Setup TensorBoard writer
    writer = None
    if tensorboard_dir:
        log_dir = tensorboard_dir / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        if LOG_LEVEL >= 1:
            print(f"✓ TensorBoard logging to: {log_dir}")
    
    # Load checkpoint if provided
    start_update = 0
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            if LOG_LEVEL >= 1:
                print(f"\n📂 Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            agent.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_update = checkpoint.get('update', 0)
            if LOG_LEVEL >= 1:
                print(f"✓ Resumed from update {start_update}")
    
    # Training loop
    if LOG_LEVEL >= 1:
        print("\n🚀 Starting training...\n")
    
    # Initialize training state
    batch_size = args.num_envs * args.num_steps
    minibatch_size = args.minibatch_size
    num_updates = args.total_timesteps // batch_size
    
    global_step = start_update * batch_size
    start_time = time.time()
    
    # Initialize rollout buffers
    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # Reset environments
    next_obs, infos = vec_env.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    # Track episode statistics
    episode_rewards = []
    episode_lengths = []
    
    for update in range(start_update, num_updates):
        # Annealing the learning rate
        if args.anneal_lr:
            frac = 1.0 - (update / num_updates)
            lr_now = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now
        
        # Collect rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            # Get action from agent
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value
            
            actions[step] = action
            logprobs[step] = logprob
            
            # Take step in environment
            next_obs_np, reward, terminated, truncated, infos = vec_env.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done).to(device)
            
            # Track episode statistics from infos
            for i, info in enumerate(infos):
                if isinstance(info, dict):
                    ep_reward = info.get("episode_reward", 0)
                    if ep_reward != 0:
                        episode_rewards.append(ep_reward)
                        episode_lengths.append(info.get("episode_length", 0))
        
        # Bootstrap value
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        
        # Flatten the batch
        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # Optimize policy
        b_inds = np.arange(batch_size)
        clipfracs = []
        
        for epoch in range(args.num_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    # Calculate approx_kl
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            
            # Early stopping based on KL divergence
            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        # Compute explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # Logging
        if (update + 1) % 10 == 0 or update == 0:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed) if elapsed > 0 else 0
            
            if LOG_LEVEL >= 1:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
                win_rate = np.mean([1.0 if r > 0 else 0.0 for r in episode_rewards[-100:]]) if episode_rewards else 0.0
                print(f"Update {update+1}/{num_updates} | "
                      f"Step {global_step:,} | "
                      f"SPS: {sps} | "
                      f"Avg Reward: {avg_reward:.3f} | "
                      f"Win Rate: {win_rate:.2%} | "
                      f"Policy Loss: {pg_loss.item():.4f} | "
                      f"Value Loss: {v_loss.item():.4f}")
            
            if writer:
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                writer.add_scalar("losses/explained_variance", explained_var, global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
                if episode_rewards:
                    writer.add_scalar("charts/avg_reward", np.mean(episode_rewards[-100:]), global_step)
                    writer.add_scalar("charts/win_rate", np.mean([1.0 if r > 0 else 0.0 for r in episode_rewards[-100:]]), global_step)
        
        # Save checkpoint
        if (update + 1) % args.save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_{update+1}.pt"
            torch.save({
                'update': update + 1,
                'global_step': global_step,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }, checkpoint_path)
            
            # Also save as latest
            latest_path = output_dir / "latest.pt"
            torch.save({
                'update': update + 1,
                'global_step': global_step,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }, latest_path)
            
            if LOG_LEVEL >= 1:
                print(f"✓ Saved checkpoint: {checkpoint_path}")
    
    # Final save
    final_path = output_dir / "final.pt"
    torch.save({
        'update': num_updates,
        'global_step': global_step,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
    }, final_path)
    
    if LOG_LEVEL >= 1:
        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}")
        print(f"Total timesteps: {global_step:,}")
        print(f"Final model saved to: {final_path}")
        if episode_rewards:
            print(f"Final avg reward: {np.mean(episode_rewards[-100:]):.3f}")
            print(f"Final win rate: {np.mean([1.0 if r > 0 else 0.0 for r in episode_rewards[-100:]]):.2%}")
    
    # Cleanup
    vec_env.close()
    if writer:
        writer.close()
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train poker AI using vectorized PPO"
    )
    
    # Environment settings
    parser.add_argument("--small-blind", type=int, default=10,
                       help="Small blind amount")
    parser.add_argument("--big-blind", type=int, default=20,
                       help="Big blind amount")
    parser.add_argument("--starting-chips", type=int, default=1000,
                       help="Starting chip count")
    
    # Training settings
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
                       help="Total training timesteps")
    parser.add_argument("--num-envs", type=int, default=64,
                       help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=128,
                       help="Steps per environment per update")
    parser.add_argument("--minibatch-size", type=int, default=512,
                       help="Minibatch size for PPO updates")
    parser.add_argument("--num-epochs", type=int, default=4,
                       help="Number of PPO epochs per update")
    parser.add_argument("--sync-envs", action="store_true",
                       help="Force synchronous environment execution")
    
    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                       help="GAE lambda parameter")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                       help="PPO clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                       help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                       help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                       help="Maximum gradient norm")
    parser.add_argument("--target-kl", type=float, default=None,
                       help="Target KL divergence for early stopping")
    parser.add_argument("--anneal-lr", action="store_true",
                       help="Anneal learning rate")
    parser.add_argument("--norm-adv", action="store_true", default=True,
                       help="Normalize advantages")
    parser.add_argument("--clip-vloss", action="store_true", default=True,
                       help="Clip value loss")
    
    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=256,
                       help="Hidden dimension")
    parser.add_argument("--num-heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4,
                       help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    
    # I/O
    parser.add_argument("--output-dir", type=str, default=DEFAULT_MODELS_DIR,
                       help="Output directory for checkpoints")
    parser.add_argument("--tensorboard-dir", type=str, 
                       default=f"/tmp/pokersim/tensorboard_puffer_v{MODEL_VERSION}",
                       help="TensorBoard log directory")
    parser.add_argument("--save-interval", type=int, default=100,
                       help="Save checkpoint every N updates")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Checkpoint to resume from")
    
    # Misc
    parser.add_argument("--cuda", action="store_true", default=True,
                       help="Use CUDA if available")
    parser.add_argument("--seed", type=int, default=1,
                       help="Random seed")
    parser.add_argument("--torch-deterministic", action="store_true", default=True,
                       help="Use deterministic PyTorch operations")
    
    args = parser.parse_args()
    
    # Default batch size
    args.batch_size = args.num_envs * args.num_steps
    
    return train(args)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    sys.exit(main())
