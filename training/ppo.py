#!/usr/bin/env python3
"""PPO Trainer with PopArt normalization and adaptive entropy."""

import math
from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPOTrainer:
    """PPO trainer with PopArt normalization and adaptive entropy."""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        hand_strength_loss_coef: float = 0.1,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        gradient_accumulation_steps: int = 4,
        target_kl: Optional[float] = 0.02,
        lr_schedule_steps: int = 5000,
        lr_warmup_steps: int = 100,
        use_popart: bool = True,
        use_adaptive_entropy: bool = True,
        advantage_clip: float = 10.0,
        device: torch.device = torch.device('cpu'),
        accelerator=None,
        use_amp: bool = True,
    ):
        self.model = model
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.hs_loss_coef = hand_strength_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.grad_accum_steps = gradient_accumulation_steps
        self.target_kl = target_kl
        self.advantage_clip = advantage_clip
        self.device = device
        self.accelerator = accelerator
        self.step = 0
        
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-5, weight_decay=0.01)
        if accelerator:
            self.model, self.optimizer = accelerator.prepare(model, self.optimizer)
        
        # LR schedule: warmup + cosine annealing
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda s: (
            s / max(1, lr_warmup_steps) if s < lr_warmup_steps else
            max(0.1, 0.5 * (1 + math.cos(math.pi * (s - lr_warmup_steps) / max(1, lr_schedule_steps - lr_warmup_steps))))
        ))
        
        # PopArt normalization state
        self.use_popart = use_popart
        self._pop_mu, self._pop_nu = 0.0, 1.0
        
        # Adaptive entropy state
        self.use_adaptive_ent = use_adaptive_entropy
        self._ent_coef = entropy_coef
        self._ent_avg = 0.0
        self._max_ent = math.log(13)  # log(num_actions)
        
        # AMP (CUDA only)
        self.use_amp = use_amp and torch.cuda.is_available() and device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
    
    def _pop_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._pop_mu) / max(1e-5, math.sqrt(self._pop_nu - self._pop_mu ** 2))
    
    def _pop_update(self, x: torch.Tensor, beta: float = 0.0003):
        mean = x.mean().item()
        self._pop_mu += beta * (mean - self._pop_mu)
        self._pop_nu += beta * ((x.var().item() if x.numel() > 1 else 0.0) + mean ** 2 - self._pop_nu)
    
    def _update_entropy_coef(self, entropy: float) -> float:
        """Adaptive entropy coefficient with anti-collapse safeguards.
        
        The key insight: policy collapse to folding happens when entropy drops
        too low (model becomes deterministic). We need to:
        1. Keep a higher minimum entropy floor (0.02 instead of 0.001)
        2. Slow down the decay rate
        3. Aggressively boost entropy when it gets critically low
        """
        self._ent_avg = 0.99 * self._ent_avg + 0.01 * entropy
        target_ratio = self._ent_avg / max(0.4 * self._max_ent, 1e-8)
        
        # More conservative adjustment: slower decay, faster recovery
        if target_ratio < 0.5:
            # Entropy critically low - aggressively increase coefficient
            self._ent_coef *= 1.01
        elif target_ratio < 0.7:
            # Entropy low - gently increase
            self._ent_coef *= 1.003
        elif target_ratio > 1.5:
            # Entropy high - gentle decrease (but not too fast)
            self._ent_coef *= 0.998
        # else: keep stable
        
        # Higher floor (0.02) prevents collapse; cap at 0.15 for stability
        self._ent_coef = max(0.02, min(0.15, self._ent_coef))
        return self._ent_coef
    
    def compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor,
        dones: torch.Tensor, next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        # Shape hygiene: callers sometimes provide (T, 1) tensors. If we don't
        # flatten to 1D here, broadcasting can silently produce wrong shapes
        # (e.g. (T,) + (T,1) -> (T,T)).
        rewards = rewards.view(-1).to(dtype=torch.float32)
        values = values.view(-1).to(dtype=torch.float32)
        dones = dones.view(-1).to(dtype=torch.float32)
        next_value = next_value.view(-1).to(dtype=torch.float32)
        next_value = next_value.squeeze()

        n = rewards.numel()
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)
        
        for t in reversed(range(n)):
            next_v = next_value if t == n - 1 else values[t + 1]
            mask = 1.0 - dones[t]
            delta = rewards[t] + (self.gamma * next_v * mask) - values[t]
            gae = delta + (self.gamma * self.gae_lambda * mask * gae)
            advantages[t] = gae
        
        return advantages, advantages + values
    
    def _compute_losses(self, batch: Dict, use_hs: bool) -> Tuple[torch.Tensor, Dict]:
        """Compute PPO losses for a mini-batch."""
        states = batch['states']
        # Ensure all 1D tensors stay 1D even for batch size 1. Using .squeeze()
        # here is dangerous because it can turn (1,) into a scalar, triggering
        # silent broadcasting and incorrect losses.
        actions = batch['actions'].view(-1).long()
        old_lp = batch['old_log_probs'].view(-1)
        old_v = batch['old_values'].view(-1)
        adv = batch['advantages'].view(-1)
        ret = batch['returns'].view(-1)
        mask, hs = batch.get('mask'), batch.get('hand_strengths')
        
        # Forward pass
        if use_hs:
            new_lp, vals, ent, hs_pred = self.model.evaluate_actions(states, actions, mask, return_hand_strength=True)
        else:
            new_lp, vals, ent = self.model.evaluate_actions(states, actions, mask)
            hs_pred = None

        new_lp = new_lp.view(-1)
        vals = vals.view(-1)
        ent = ent.view(-1)
        
        # Policy loss (clipped surrogate)
        ratio = torch.exp(new_lp - old_lp)
        pi_loss = -torch.min(
            ratio * adv,
            torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
        ).mean()
        
        # Value loss (clipped)
        v_pred = vals
        v_pred_clipped = old_v + torch.clamp(v_pred - old_v, -self.clip_epsilon, self.clip_epsilon)
        v_loss_unclipped = F.smooth_l1_loss(v_pred, ret, reduction='none')
        v_loss_clipped = F.smooth_l1_loss(v_pred_clipped, ret, reduction='none')
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        
        ent_mean = ent.mean()
        if use_hs and hs_pred is not None and hs is not None:
            hs_loss = F.mse_loss(hs_pred.view(-1), hs.view(-1))
        else:
            hs_loss = torch.tensor(0.0, device=self.device)
        ent_coef = self._update_entropy_coef(ent_mean.item()) if self.use_adaptive_ent else self.entropy_coef
        
        total = pi_loss + self.value_loss_coef * v_loss - ent_coef * ent_mean + self.hs_loss_coef * hs_loss
        
        with torch.no_grad():
            clip_frac = ((ratio < 1 - self.clip_epsilon) | (ratio > 1 + self.clip_epsilon)).float().mean().item()
        
        return total, {
            'policy_loss': pi_loss.item(),
            'value_loss': v_loss.item(),
            'entropy': ent_mean.item(),
            'kl_divergence': (old_lp - new_lp).mean().item(),
            'clip_fraction': clip_frac,
            'hand_strength_loss': hs_loss.item() if hasattr(hs_loss, 'item') else 0,
        }
    
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
        """Run PPO update on collected trajectories."""
        # Make sure we're in train mode during the optimization step.
        # Rollout collection sets eval() for inference; leaving the model in eval()
        # can silently change training dynamics (dropout off, checkpointing off).
        was_training = bool(getattr(self.model, "training", False))
        self.model.train()
        self.step += 1
        n = states.size(0)
        use_hs = hand_strengths is not None and self.hs_loss_coef > 0
        
        # PopArt (IMPORTANT)
        # ------------------
        # Proper PopArt requires keeping the critic's outputs and the advantage/GAE
        # computation in a *consistent* scale, typically by adjusting the value head
        # parameters when the running (mu, sigma) changes.
        #
        # A previous implementation normalized the return targets for the value loss,
        # while leaving rollouts/GAE in raw reward units. That silently corrupts
        # advantages over time and can make later iterations perform worse.
        #
        # For now, keep training targets in the raw reward scale for correctness.
        # We still track PopArt stats (mu/nu) for potential future use / debugging.
        ret_for_value = returns
        if self.use_popart:
            self._pop_update(returns)
        
        # Normalize and clip advantages
        adv = torch.clamp(
            (advantages - advantages.mean()) / (advantages.std() + 1e-8),
            -self.advantage_clip, self.advantage_clip
        )
        
        # Collect per-minibatch stats so we never end up with empty aggregates
        # (which would otherwise yield NaNs via np.mean([])).
        all_stats = []
        last_grad_norm: Optional[float] = None
        ctx = torch.amp.autocast(device_type='cuda') if self.use_amp else nullcontext()
        
        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(n)
            accum, epoch_kl = 0, []
            self.optimizer.zero_grad()
            
            for start in range(0, n, self.mini_batch_size):
                idx = indices[start:min(start + self.mini_batch_size, n)]
                
                batch = {
                    'states': states[idx].to(self.device),
                    'actions': actions[idx].to(self.device),
                    'old_log_probs': old_log_probs[idx].to(self.device),
                    'old_values': old_values[idx].to(self.device),
                    'advantages': adv[idx].to(self.device),
                    'returns': ret_for_value[idx].to(self.device),
                    'mask': legal_actions_masks[idx].to(self.device) if legal_actions_masks is not None else None,
                    'hand_strengths': hand_strengths[idx].to(self.device) if use_hs else None,
                }
                
                with ctx:
                    loss, stats = self._compute_losses(batch, use_hs)
                    loss = loss / self.grad_accum_steps
                
                if self.accelerator:
                    self.accelerator.backward(loss)
                elif self.use_amp and self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accum += 1
                epoch_kl.append(stats['kl_divergence'])
                all_stats.append(stats)
                
                if accum >= self.grad_accum_steps:
                    last_grad_norm = self._optimizer_step()
                    accum = 0
            
            if accum > 0:
                last_grad_norm = self._optimizer_step()
            
            # Early stopping on high KL
            if self.target_kl and epoch_kl and np.mean(epoch_kl) > self.target_kl * 1.5:
                if verbose:
                    print(f"  Early stop epoch {epoch+1}: KL={np.mean(epoch_kl):.4f}")
                break
        
        self.scheduler.step()
        
        # Aggregate stats
        keys = ['policy_loss', 'value_loss', 'entropy', 'kl_divergence', 'clip_fraction', 'hand_strength_loss', 'grad_norm']
        avg = {k: float(np.mean([s.get(k, 0) for s in all_stats])) for k in keys}
        if last_grad_norm is not None:
            avg['grad_norm'] = float(last_grad_norm)
        avg['learning_rate'] = self.optimizer.param_groups[0]['lr']
        avg['total_loss'] = avg['policy_loss'] + self.value_loss_coef * avg['value_loss']
        if self.use_adaptive_ent:
            avg['entropy_coef'] = self._ent_coef
        # Restore previous mode (avoid surprising callers).
        if not was_training:
            self.model.eval()
        return avg
    
    def _optimizer_step(self) -> float:
        """Optimizer step with gradient clipping."""
        if self.use_amp and self.scaler:
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        self.optimizer.zero_grad()
        return grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
    
    def save_checkpoint(self, path: str, epoch: int, **kwargs):
        """Save training checkpoint."""
        # If we're under Accelerate, unwrap the model so checkpoints are loadable
        # outside the training process (e.g. ELO server, eval scripts).
        model_for_save = self.model
        try:
            if self.accelerator is not None and hasattr(self.accelerator, "unwrap_model"):
                model_for_save = self.accelerator.unwrap_model(self.model)
        except Exception:
            model_for_save = self.model

        ckpt = {
            'epoch': epoch,
            'training_step': self.step,
            'model_state_dict': model_for_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'hyperparameters': {
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'value_loss_coef': self.value_loss_coef,
                'entropy_coef': self.entropy_coef,
            },
            **kwargs
        }
        if self.use_popart:
            ckpt['popart'] = {'mu': self._pop_mu, 'nu': self._pop_nu}
        if self.use_adaptive_ent:
            ckpt['adaptive_entropy'] = {'coef': self._ent_coef, 'avg': self._ent_avg}
        torch.save(ckpt, path)
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load training checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'training_step' in ckpt:
            self.step = ckpt['training_step']
        if self.use_popart and 'popart' in ckpt:
            self._pop_mu, self._pop_nu = ckpt['popart']['mu'], ckpt['popart']['nu']
        if self.use_adaptive_ent and 'adaptive_entropy' in ckpt:
            self._ent_coef = ckpt['adaptive_entropy']['coef']
            self._ent_avg = ckpt['adaptive_entropy'].get('avg', 0.0)
        return ckpt
