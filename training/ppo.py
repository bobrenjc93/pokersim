#!/usr/bin/env python3
"""
Proximal Policy Optimization (PPO) Algorithm for Poker

This module implements PPO with modern enhancements:
- PopArt value normalization for stable learning
- Adaptive entropy scheduling
- Warmup + cosine annealing LR schedule
- Improved advantage estimation
- Mixed precision support

Key features:
- Clipped surrogate objective (prevents too large policy updates)
- Generalized Advantage Estimation (GAE)
- Value function learning with normalization
- Entropy bonus for exploration
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
import numpy as np


class PopArtValueNormalizer:
    """
    PopArt (Preserving Outputs Precisely, while Adaptively Rescaling Targets)
    
    Normalizes value function targets adaptively during training.
    This helps stabilize training when reward scales vary significantly.
    
    Reference: "Learning values across many orders of magnitude" (DeepMind, 2016)
    """
    
    def __init__(self, beta: float = 0.0003, epsilon: float = 1e-5):
        """
        Args:
            beta: Update rate for running statistics (lower = more stable)
            epsilon: Small constant for numerical stability
        """
        self.beta = beta
        self.epsilon = epsilon
        
        # Running statistics
        self.mu = 0.0  # Running mean
        self.nu = 1.0  # Running second moment (E[x^2])
        self.count = 0
    
    @property
    def std(self) -> float:
        """Compute standard deviation from running statistics."""
        return max(self.epsilon, math.sqrt(self.nu - self.mu ** 2))
    
    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize values using current statistics."""
        return (values - self.mu) / self.std
    
    def denormalize(self, normalized_values: torch.Tensor) -> torch.Tensor:
        """Convert normalized values back to original scale."""
        return normalized_values * self.std + self.mu
    
    def update(self, values: torch.Tensor) -> Tuple[float, float]:
        """
        Update running statistics with new batch of values.
        
        Args:
            values: Tensor of target values
            
        Returns:
            Tuple of (old_mu, old_std) for potential weight rescaling
        """
        old_mu = self.mu
        old_std = self.std
        
        # Compute batch statistics
        batch_mean = values.mean().item()
        batch_var = values.var().item() if values.numel() > 1 else 0.0
        batch_count = values.numel()
        
        # Update running statistics with exponential moving average
        self.count += batch_count
        
        # Welford's online algorithm with momentum
        delta = batch_mean - self.mu
        self.mu += self.beta * delta
        
        # Update second moment
        batch_nu = batch_var + batch_mean ** 2
        self.nu += self.beta * (batch_nu - self.nu)
        
        return old_mu, old_std


class AdaptiveEntropyScheduler:
    """
    Adaptively adjusts entropy coefficient based on policy entropy.
    
    - If entropy is too low (policy too deterministic), increase coefficient
    - If entropy is too high (policy too random), decrease coefficient
    """
    
    def __init__(
        self,
        initial_coef: float = 0.01,
        min_coef: float = 0.001,
        max_coef: float = 0.1,
        target_entropy_ratio: float = 0.5,  # Target as ratio of max entropy
        adaptation_rate: float = 0.001
    ):
        """
        Args:
            initial_coef: Starting entropy coefficient
            min_coef: Minimum allowed coefficient
            max_coef: Maximum allowed coefficient  
            target_entropy_ratio: Target entropy as ratio of maximum possible
            adaptation_rate: How quickly to adapt the coefficient
        """
        self.coef = initial_coef
        self.min_coef = min_coef
        self.max_coef = max_coef
        self.target_ratio = target_entropy_ratio
        self.adaptation_rate = adaptation_rate
        
        # Track entropy history for smoothing
        self.entropy_history: List[float] = []
        self.history_size = 100
    
    def update(self, current_entropy: float, max_entropy: float) -> float:
        """
        Update entropy coefficient based on current vs target entropy.
        
        Args:
            current_entropy: Current policy entropy
            max_entropy: Maximum possible entropy (log(num_actions))
            
        Returns:
            Updated entropy coefficient
        """
        # Add to history
        self.entropy_history.append(current_entropy)
        if len(self.entropy_history) > self.history_size:
            self.entropy_history.pop(0)
        
        # Use smoothed entropy
        smoothed_entropy = sum(self.entropy_history) / len(self.entropy_history)
        
        # Compute target entropy
        target_entropy = self.target_ratio * max_entropy
        
        # Adjust coefficient
        # If entropy < target, increase coefficient to encourage exploration
        # If entropy > target, decrease coefficient
        entropy_ratio = smoothed_entropy / max(target_entropy, 1e-8)
        
        if entropy_ratio < 0.8:
            # Too deterministic, increase entropy bonus
            self.coef *= (1 + self.adaptation_rate * 2)
        elif entropy_ratio > 1.2:
            # Too random, decrease entropy bonus
            self.coef *= (1 - self.adaptation_rate)
        
        # Clamp to valid range
        self.coef = max(self.min_coef, min(self.max_coef, self.coef))
        
        return self.coef


def get_warmup_cosine_schedule(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1
):
    """
    Create a learning rate schedule with linear warmup followed by cosine annealing.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of initial LR
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class PPOTrainer:
    """
    PPO training algorithm for poker RL with modern enhancements.
    
    PPO maintains a balance between:
    - Improving the policy (learn better actions)
    - Staying close to old policy (stability)
    - Estimating state values accurately
    - Predicting hand strength (auxiliary task to improve hand evaluation)
    
    Enhancements over standard PPO:
    - PopArt value normalization
    - Adaptive entropy scheduling
    - Warmup + cosine annealing LR
    - Improved advantage normalization
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
        hand_strength_loss_coef: float = 0.1,  # Weight for hand strength prediction loss
        max_grad_norm: float = 1.0,  # Gradient clipping
        ppo_epochs: int = 4,  # Number of PPO epochs per update
        mini_batch_size: int = 64,  # Mini-batch size for PPO
        gradient_accumulation_steps: int = 4,  # Accumulate gradients
        target_kl: Optional[float] = 0.02,  # Early stopping KL threshold
        lr_schedule_steps: int = 5000,  # Total steps for LR scheduling
        lr_warmup_steps: int = 100,  # Warmup steps for LR
        use_popart: bool = True,  # Use PopArt value normalization
        use_adaptive_entropy: bool = True,  # Use adaptive entropy scheduling
        advantage_clip: float = 10.0,  # Clip extreme advantages
        device: torch.device = torch.device('cpu'),
        accelerator=None
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
            hand_strength_loss_coef: Coefficient for auxiliary hand strength loss
            max_grad_norm: Max gradient norm for clipping
            ppo_epochs: Number of epochs to train on each batch
            mini_batch_size: Size of mini-batches
            gradient_accumulation_steps: Steps to accumulate gradients
            target_kl: Target KL divergence for early stopping
            lr_schedule_steps: Total training steps for LR schedule
            lr_warmup_steps: Number of warmup steps
            use_popart: Whether to use PopArt normalization
            use_adaptive_entropy: Whether to use adaptive entropy scheduling
            advantage_clip: Maximum absolute value for advantages
            device: Device to train on
            accelerator: Hugging Face Accelerator instance
        """
        self.model = model
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.hand_strength_loss_coef = hand_strength_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.target_kl = target_kl
        self.advantage_clip = advantage_clip
        self.device = device
        self.accelerator = accelerator
        
        # Create optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            eps=1e-5,
            weight_decay=0.01
        )
        
        # Prepare with accelerator if provided
        if self.accelerator is not None:
            self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        
        # Learning rate scheduler with warmup
        self.initial_lr = learning_rate
        self.scheduler = get_warmup_cosine_schedule(
            self.optimizer,
            warmup_steps=lr_warmup_steps,
            total_steps=lr_schedule_steps,
            min_lr_ratio=0.1
        )
        
        # PopArt value normalizer
        self.use_popart = use_popart
        self.value_normalizer = PopArtValueNormalizer(beta=0.0003) if use_popart else None
        
        # Adaptive entropy scheduler
        self.use_adaptive_entropy = use_adaptive_entropy
        self.entropy_scheduler = AdaptiveEntropyScheduler(
            initial_coef=entropy_coef,
            min_coef=0.001,
            max_coef=0.1,
            target_entropy_ratio=0.5
        ) if use_adaptive_entropy else None
        
        # Maximum entropy for action space (log of num_actions)
        self.max_entropy = math.log(22)  # 22 actions in the action space
        
        # Training step counter
        self.training_step = 0
        
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
            'return_mean': [],
            'hand_strength_loss': [],
            'advantage_mean': [],
            'advantage_std': [],
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
    
    def _normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        """
        Normalize and clip advantages for stable training.
        
        Args:
            advantages: Raw advantage estimates
            
        Returns:
            Normalized and clipped advantages
        """
        # Normalize
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        normalized = (advantages - adv_mean) / (adv_std + 1e-8)
        
        # Clip extreme values
        clipped = torch.clamp(normalized, -self.advantage_clip, self.advantage_clip)
        
        return clipped
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        legal_actions_masks: Optional[torch.Tensor] = None,
        hand_strengths: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Update policy using PPO algorithm with modern enhancements.
        
        Args:
            states: State tensor of shape (num_samples, state_dim)
            actions: Action tensor of shape (num_samples,)
            old_log_probs: Old log probabilities of shape (num_samples,)
            old_values: Old value predictions of shape (num_samples,)
            advantages: Advantage estimates of shape (num_samples,)
            returns: Target returns of shape (num_samples,)
            legal_actions_masks: Legal action masks of shape (num_samples, num_actions)
            hand_strengths: Target hand strengths of shape (num_samples,) for auxiliary loss
            verbose: Whether to print detailed progress
        
        Returns:
            Dictionary with training statistics
        """
        # Increment training step
        self.training_step += 1
        
        # Update PopArt statistics and normalize returns
        if self.use_popart and self.value_normalizer is not None:
            self.value_normalizer.update(returns)
            normalized_returns = self.value_normalizer.normalize(returns)
        else:
            normalized_returns = returns
        
        # Normalize and clip advantages
        advantages = self._normalize_advantages(advantages)
        
        num_samples = states.size(0)
        
        # Check if we have hand strength targets for auxiliary loss
        use_hand_strength_loss = hand_strengths is not None and self.hand_strength_loss_coef > 0
        
        # Track statistics
        epoch_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': [],
            'hand_strength_loss': [],
            'advantage_mean': [],
            'advantage_std': [],
        }
        
        # Record advantage statistics
        epoch_stats['advantage_mean'].append(advantages.mean().item())
        epoch_stats['advantage_std'].append(advantages.std().item())
        
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
                mb_states = states[mb_indices].to(self.device)
                mb_actions = actions[mb_indices].to(self.device)
                mb_old_log_probs = old_log_probs[mb_indices].to(self.device)
                mb_old_values = old_values[mb_indices].to(self.device)
                mb_advantages = advantages[mb_indices].to(self.device)
                mb_returns = normalized_returns[mb_indices].to(self.device)
                
                mb_legal_masks = None
                if legal_actions_masks is not None:
                    mb_legal_masks = legal_actions_masks[mb_indices].to(self.device)
                
                mb_hand_strengths = None
                if use_hand_strength_loss:
                    mb_hand_strengths = hand_strengths[mb_indices].to(self.device)
                
                # Evaluate actions with current policy
                if use_hand_strength_loss:
                    new_log_probs, values, entropy, hand_strength_pred = self.model.evaluate_actions(
                        mb_states, mb_actions, mb_legal_masks, return_hand_strength=True
                    )
                else:
                    new_log_probs, values, entropy = self.model.evaluate_actions(
                        mb_states, mb_actions, mb_legal_masks, return_hand_strength=False
                    )
                
                # PPO clipped surrogate loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with clipping - use Huber loss for robustness
                values = values.squeeze()
                
                # Unclipped value loss
                v_loss_unclipped = F.smooth_l1_loss(values, mb_returns, reduction='none')
                
                # Clipped value loss
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
                entropy_mean = entropy.mean()
                entropy_loss = -entropy_mean
                
                # Hand strength prediction loss (auxiliary task)
                hand_strength_loss = torch.tensor(0.0, device=self.device)
                if use_hand_strength_loss and mb_hand_strengths is not None:
                    hand_strength_loss = F.mse_loss(hand_strength_pred, mb_hand_strengths)
                
                # Get current entropy coefficient
                current_entropy_coef = self.entropy_coef
                if self.use_adaptive_entropy and self.entropy_scheduler is not None:
                    current_entropy_coef = self.entropy_scheduler.update(
                        entropy_mean.item(), self.max_entropy
                    )
                
                # Total loss - scale by accumulation steps for proper gradient averaging
                loss = (
                    policy_loss +
                    self.value_loss_coef * value_loss +
                    current_entropy_coef * entropy_loss +
                    self.hand_strength_loss_coef * hand_strength_loss
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
                    grad_norm = 0.0
                
                # Track statistics (use unscaled loss values)
                with torch.no_grad():
                    # KL divergence (approximate)
                    kl_div = (mb_old_log_probs - new_log_probs).mean()
                    
                    # Clip fraction
                    clip_fraction = ((ratio < 1.0 - self.clip_epsilon) | 
                                   (ratio > 1.0 + self.clip_epsilon)).float().mean()
                    
                    unscaled_loss = loss.item() * self.gradient_accumulation_steps
                    
                    epoch_stats['policy_loss'].append(policy_loss.item())
                    epoch_stats['value_loss'].append(value_loss.item())
                    epoch_stats['entropy'].append(entropy_mean.item())
                    epoch_stats['total_loss'].append(unscaled_loss)
                    epoch_stats['kl_divergence'].append(kl_div.item())
                    epoch_stats['clip_fraction'].append(clip_fraction.item())
                    epoch_stats['hand_strength_loss'].append(
                        hand_strength_loss.item() if hasattr(hand_strength_loss, 'item') else hand_strength_loss
                    )
                    if 'grad_norm' not in epoch_stats: 
                        epoch_stats['grad_norm'] = []
                    epoch_stats['grad_norm'].append(grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm)
            
            # Handle remaining accumulated gradients at end of epoch
            if accumulation_step > 0:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Early stopping based on KL divergence
            if self.target_kl is not None:
                recent_kls = epoch_stats['kl_divergence'][-max(1, num_samples//self.mini_batch_size):]
                mean_kl = np.mean(recent_kls) if recent_kls else 0
                if mean_kl > self.target_kl * 1.5:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1} due to KL divergence: {mean_kl:.4f}")
                    break
        
        # Step LR scheduler
        self.scheduler.step()
        
        # Compute explained variance
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
            if key not in self.stats:
                self.stats[key] = []
            self.stats[key].append(value)
        
        # Track learning rate and entropy coefficient
        current_lr = self.optimizer.param_groups[0]['lr']
        self.stats['learning_rate'].append(current_lr)
        avg_stats['learning_rate'] = current_lr
        
        # Track adaptive entropy coefficient
        if self.use_adaptive_entropy and self.entropy_scheduler:
            avg_stats['entropy_coef'] = self.entropy_scheduler.coef
        
        return avg_stats
    
    def get_stats(self) -> Dict[str, List[float]]:
        """Get training statistics"""
        return self.stats
    
    def save_checkpoint(self, path: str, epoch: int, **kwargs):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'training_step': self.training_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
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
        
        # Save PopArt state if used
        if self.use_popart and self.value_normalizer:
            checkpoint['popart'] = {
                'mu': self.value_normalizer.mu,
                'nu': self.value_normalizer.nu,
                'count': self.value_normalizer.count,
            }
        
        # Save entropy scheduler state if used
        if self.use_adaptive_entropy and self.entropy_scheduler:
            checkpoint['entropy_scheduler'] = {
                'coef': self.entropy_scheduler.coef,
                'entropy_history': self.entropy_scheduler.entropy_history,
            }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'training_step' in checkpoint:
            self.training_step = checkpoint['training_step']
        
        if 'stats' in checkpoint:
            self.stats = checkpoint['stats']
        
        # Load PopArt state
        if self.use_popart and 'popart' in checkpoint and self.value_normalizer:
            self.value_normalizer.mu = checkpoint['popart']['mu']
            self.value_normalizer.nu = checkpoint['popart']['nu']
            self.value_normalizer.count = checkpoint['popart']['count']
        
        # Load entropy scheduler state
        if self.use_adaptive_entropy and 'entropy_scheduler' in checkpoint and self.entropy_scheduler:
            self.entropy_scheduler.coef = checkpoint['entropy_scheduler']['coef']
            self.entropy_scheduler.entropy_history = checkpoint['entropy_scheduler'].get('entropy_history', [])
        
        return checkpoint
