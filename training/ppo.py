#!/usr/bin/env python3
"""
Proximal Policy Optimization (PPO) Algorithm for Poker

This module implements PPO, a state-of-the-art policy gradient RL algorithm.

Key features:
- Clipped surrogate objective (prevents too large policy updates)
- Generalized Advantage Estimation (GAE)
- Value function learning
- Entropy bonus for exploration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
import numpy as np


class PPOTrainer:
    """
    PPO training algorithm for poker RL.
    
    PPO maintains a balance between:
    - Improving the policy (learn better actions)
    - Staying close to old policy (stability)
    - Estimating state values accurately
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,  # Discount factor
        gae_lambda: float = 0.95,  # GAE parameter
        clip_epsilon: float = 0.2,  # PPO clip parameter
        value_loss_coef: float = 0.5,  # Weight for value loss
        entropy_coef: float = 0.01,  # Weight for entropy bonus
        max_grad_norm: float = 1.0,  # Gradient clipping (relaxed from 0.5)
        ppo_epochs: int = 4,  # Number of PPO epochs per update
        mini_batch_size: int = 64,  # Mini-batch size for PPO
        gradient_accumulation_steps: int = 4,  # Accumulate gradients for larger effective batch
        target_kl: Optional[float] = 0.02,  # Early stopping KL threshold (relaxed for poker)
        lr_schedule_steps: int = 5000,  # Total steps for LR scheduling (should match training iterations)
        device: torch.device = torch.device('cpu'),
        accelerator = None
    ):
        """
        Args:
            model: Actor-critic model
            learning_rate: Learning rate
            gamma: Discount factor for rewards
            gae_lambda: Lambda parameter for GAE
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Max gradient norm for clipping
            ppo_epochs: Number of epochs to train on each batch
            mini_batch_size: Size of mini-batches
            target_kl: Target KL divergence for early stopping
            device: Device to train on
            accelerator: Hugging Face Accelerator instance
        """
        self.model = model
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.target_kl = target_kl
        self.device = device
        self.accelerator = accelerator
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
        
        # Prepare with accelerator if provided
        if self.accelerator is not None:
            self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        
        # Learning rate scheduler (cosine annealing for smooth decay)
        # T_max should match total training iterations for proper scheduling
        self.initial_lr = learning_rate
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=lr_schedule_steps, eta_min=learning_rate * 0.1
        )
        
        # Training statistics
        self.stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': [],
            'explained_variance': [],
            'learning_rate': [],
            'grad_norm': [],
            'value_mean': [],
            'return_mean': []
        }
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        GAE provides a good bias-variance tradeoff for advantage estimation.
        
        Args:
            rewards: Rewards tensor of shape (num_steps,)
            values: Value predictions of shape (num_steps,)
            dones: Done flags of shape (num_steps,)
            next_value: Value of next state of shape (1,)
        
        Returns:
            Tuple of:
            - advantages: Advantage estimates of shape (num_steps,)
            - returns: Target returns of shape (num_steps,)
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        # Append next_value for easier computation
        values_extended = torch.cat([values, next_value.unsqueeze(0)])
        
        # Compute advantages using GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values_extended[t + 1]
            
            # TD error: r + gamma * V(s') - V(s)
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            
            # GAE: advantage = delta + gamma * lambda * advantage_next
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            last_gae = advantages[t]
        
        # Returns = advantages + values
        returns = advantages + values
        
        return advantages, returns
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        legal_actions_masks: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Update policy using PPO algorithm.
        
        Args:
            states: State tensor of shape (num_samples, state_dim)
            actions: Action tensor of shape (num_samples,)
            old_log_probs: Old log probabilities of shape (num_samples,)
            old_values: Old value predictions of shape (num_samples,)
            advantages: Advantage estimates of shape (num_samples,)
            returns: Target returns of shape (num_samples,)
            legal_actions_masks: Legal action masks of shape (num_samples, num_actions)
            verbose: Whether to print detailed progress
        
        Returns:
            Dictionary with training statistics
        """
        # Normalize advantages (improves stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        num_samples = states.size(0)
        
        # Track statistics
        epoch_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': []
        }
        
        # PPO epochs with gradient accumulation
        for epoch in range(self.ppo_epochs):
            # Generate random indices (on CPU to match input tensors)
            indices = torch.randperm(num_samples)
            
            # Track accumulation steps
            accumulation_step = 0
            self.optimizer.zero_grad()
            
            for start in range(0, num_samples, self.mini_batch_size):
                end = start + self.mini_batch_size
                if end > num_samples:
                    end = num_samples
                
                mb_indices = indices[start:end]
                
                # Mini-batch data - Move to device (GPU) here for offloading
                # Input tensors are expected to be on CPU
                mb_states = states[mb_indices].to(self.device)
                mb_actions = actions[mb_indices].to(self.device)
                mb_old_log_probs = old_log_probs[mb_indices].to(self.device)
                mb_old_values = old_values[mb_indices].to(self.device)
                mb_advantages = advantages[mb_indices].to(self.device)
                mb_returns = returns[mb_indices].to(self.device)
                
                mb_legal_masks = None
                if legal_actions_masks is not None:
                    mb_legal_masks = legal_actions_masks[mb_indices].to(self.device)
                
                # Evaluate actions with current policy
                new_log_probs, values, entropy = self.model.evaluate_actions(
                    mb_states, mb_actions, mb_legal_masks
                )
                
                # PPO clipped surrogate loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with clipping (improves stability)
                values = values.squeeze() # Ensure shape match
                
                # Unclipped loss - use Huber loss (SmoothL1) for robustness against outliers
                v_loss_unclipped = F.smooth_l1_loss(values, mb_returns, reduction='none')
                
                # Clipped loss
                v_clipped = mb_old_values + torch.clamp(
                    values - mb_old_values,
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                v_loss_clipped = F.smooth_l1_loss(v_clipped, mb_returns, reduction='none')
                
                # Max of clipped and unclipped (conservative update)
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                value_loss = 0.5 * v_loss_max.mean()
                
                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss - scale by accumulation steps for proper gradient averaging
                loss = (
                    policy_loss +
                    self.value_loss_coef * value_loss +
                    self.entropy_coef * entropy_loss
                ) / self.gradient_accumulation_steps
                
                # Backward pass (accumulate gradients)
                if self.accelerator is not None:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                
                accumulation_step += 1
                
                # Step optimizer after accumulating enough gradients
                if accumulation_step >= self.gradient_accumulation_steps:
                    # Gradient clipping and tracking
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accumulation_step = 0
                else:
                    grad_norm = 0.0  # Will be updated on actual step
                
                # Track statistics (use unscaled loss values for accurate reporting)
                with torch.no_grad():
                    # KL divergence (approximate)
                    kl_div = (mb_old_log_probs - new_log_probs).mean()
                    
                    # Clip fraction (how often we clip the ratio)
                    clip_fraction = ((ratio < 1.0 - self.clip_epsilon) | 
                                   (ratio > 1.0 + self.clip_epsilon)).float().mean()
                    
                    # Store unscaled loss values (loss was scaled for gradient accumulation)
                    unscaled_loss = loss.item() * self.gradient_accumulation_steps
                    
                    epoch_stats['policy_loss'].append(policy_loss.item())
                    epoch_stats['value_loss'].append(value_loss.item())
                    epoch_stats['entropy'].append(-entropy_loss.item())
                    epoch_stats['total_loss'].append(unscaled_loss)
                    epoch_stats['kl_divergence'].append(kl_div.item())
                    epoch_stats['clip_fraction'].append(clip_fraction.item())
                    if 'grad_norm' not in epoch_stats: epoch_stats['grad_norm'] = []
                    epoch_stats['grad_norm'].append(grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm)
            
            # Handle any remaining accumulated gradients at end of epoch
            if accumulation_step > 0:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Early stopping based on KL divergence
            if self.target_kl is not None:
                mean_kl = np.mean(epoch_stats['kl_divergence'][-num_samples//self.mini_batch_size:])
                if mean_kl > self.target_kl * 1.5:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1} due to KL divergence: {mean_kl:.4f}")
                    break
        
        # Compute explained variance (how well value function predicts returns)
        # Compute in batches to save memory
        with torch.no_grad():
            all_values_list = []
            for start in range(0, num_samples, self.mini_batch_size):
                end = min(start + self.mini_batch_size, num_samples)
                mb_states = states[start:end].to(self.device)
                mb_actions = actions[start:end].to(self.device)
                mb_legal_masks = None
                if legal_actions_masks is not None:
                    mb_legal_masks = legal_actions_masks[start:end].to(self.device)
                
                _, values, _ = self.model.evaluate_actions(mb_states, mb_actions, mb_legal_masks)
                all_values_list.append(values.cpu())
            
            all_values = torch.cat(all_values_list).squeeze()
            # Move returns to CPU for calculation
            returns_cpu = returns.cpu()
            var_returns = returns_cpu.var()
            explained_var = 1 - (returns_cpu - all_values).var() / (var_returns + 1e-8)
            epoch_stats['explained_variance'] = [explained_var.item()]
            epoch_stats['value_mean'] = [all_values.mean().item()]
            epoch_stats['return_mean'] = [returns_cpu.mean().item()]
        
        # Average statistics
        avg_stats = {
            key: np.mean(values) for key, values in epoch_stats.items()
        }
        
        # Update global statistics
        for key, value in avg_stats.items():
            self.stats[key].append(value)
        
        # Track learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.stats['learning_rate'].append(current_lr)
        avg_stats['learning_rate'] = current_lr
        
        # Step learning rate scheduler
        self.scheduler.step()
        
        return avg_stats
    
    def get_stats(self) -> Dict[str, List[float]]:
        """Get training statistics"""
        return self.stats
    
    def save_checkpoint(self, path: str, epoch: int, **kwargs):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': self.stats,
            'hyperparameters': {
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'value_loss_coef': self.value_loss_coef,
                'entropy_coef': self.entropy_coef,
            },
            **kwargs
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'stats' in checkpoint:
            self.stats = checkpoint['stats']
        return checkpoint

