#!/usr/bin/env python3
"""
Actor-Critic Neural Network Architecture for Poker RL

This module implements a transformer-based actor-critic architecture:
- Shared encoder (transformer)
- Policy head (actor): outputs action probabilities
- Value head (critic): outputs expected value (chip EV)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PokerActorCritic(nn.Module):
    """
    Actor-Critic architecture for poker RL.
    
    Components:
    1. Shared Transformer Encoder - processes state features
    2. Policy Head (Actor) - outputs action probabilities
    3. Value Head (Critic) - outputs expected chip value
    
    This architecture allows the model to:
    - Learn what actions to take (policy)
    - Estimate how good the current state is (value)
    - Share representations between policy and value (more efficient)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        num_actions: int = 13,  # fold, check, call, 9 raise sizes, all-in (unified action space)
        dropout: float = 0.1,
        gradient_checkpointing: bool = False
    ):
        """
        Args:
            input_dim: Size of input feature vector
            hidden_dim: Dimension of transformer embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
            num_actions: Number of possible actions
            dropout: Dropout rate
            gradient_checkpointing: Enable gradient checkpointing
        """
        super().__init__()
        
        # Ensure hidden_dim is divisible by num_heads
        if hidden_dim % num_heads != 0:
            hidden_dim = ((hidden_dim // num_heads) + 1) * num_heads
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_actions = num_actions
        self.gradient_checkpointing = gradient_checkpointing
        
        # Token organization (same as PokerTransformer)
        self.hole_card_dim = 17
        self.community_card_dim = 17
        self.pot_dim = 5
        self.stage_dim = 5
        self.position_dim = 6
        self.gametheory_dim = 5
        self.opponent_dim = 26
        self.handstrength_dim = 1
        
        # Input projections for different token types
        self.hole_card_embed = nn.Linear(self.hole_card_dim, hidden_dim)
        self.community_card_embed = nn.Linear(self.community_card_dim, hidden_dim)
        self.pot_embed = nn.Linear(self.pot_dim, hidden_dim)
        self.stage_embed = nn.Linear(self.stage_dim, hidden_dim)
        self.position_embed = nn.Linear(self.position_dim, hidden_dim)
        
        # New embeddings for RL features
        self.gametheory_embed = nn.Linear(self.gametheory_dim, hidden_dim)
        self.opponent_embed = nn.Linear(self.opponent_dim, hidden_dim)
        self.handstrength_embed = nn.Linear(self.handstrength_dim, hidden_dim)
        
        # Total tokens: 1 (CLS) + 2 (hole) + 5 (community) + 1 (pot) + 1 (stage) + 
        #               1 (position) + 1 (game theory) + 1 (opponent) + 1 (hand strength)
        self.num_tokens = 14
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_tokens, hidden_dim))
        
        # CLS token for global state aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Shared transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Policy head (actor) - outputs action logits
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # Action space prior adjustment to balance probability mass
        # Unified action space: fold(1), check(1), call(1), raise_X%(9), all_in(1)
        # Single-slot actions need boost relative to multi-slot actions
        action_prior = torch.zeros(num_actions)
        
        log_9 = 2.197  # log(9) 
        
        # Boost check and call, but REDUCE fold boost to prevent over-folding
        # Fold should be less attractive than check/call to encourage playing
        fold_idx = 0
        check_idx = 1
        call_idx = 2
        
        if fold_idx < num_actions:
            action_prior[fold_idx] = log_9 * 0.5  # Reduced fold prior (was log_9)
        if check_idx < num_actions:
            action_prior[check_idx] = log_9 * 1.2  # Boost check to prefer over fold
        if call_idx < num_actions:
            action_prior[call_idx] = log_9 * 1.1  # Boost call slightly
        
        # VERY STRONG NEGATIVE prior for all_in to discourage overuse
        # A value of -3.5 makes all-in roughly 33x less likely than without the prior
        # With unified action space, all_in is now at index 12
        all_in_idx = 12
        if all_in_idx < num_actions:
            action_prior[all_in_idx] = -3.5  # Very strong penalty
        
        self.register_buffer('action_prior', action_prior)
        
        # Hand-strength-aware action gating
        # This network learns to penalize aggressive actions when hand strength is low
        # It outputs a scaling factor for each action based on hand strength
        self.hand_strength_gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 4),  # +1 for hand strength
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_actions),
        )
        
        # All-in penalty weights based on hand strength
        # This creates a direct penalty term for all-in when hand strength is low
        # The model learns these weights, but they're initialized to penalize weak-hand all-ins
        # Threshold lowered from 0.50 to 0.35 - only truly trash hands get penalized
        self.hand_strength_threshold = 0.35  # Below this, all-in is strongly discouraged
        
        # Value head (critic) - outputs expected value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Hand strength prediction head (auxiliary task)
        # This forces the model to explicitly learn hand evaluation
        # which helps with action selection (don't all-in with weak hands)
        self.hand_strength_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate scaling"""
        # Policy head - smaller initialization for more stable early training
        for module in self.policy_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Value head - normal initialization
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input features into transformer tokens and return pooled representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            pooled: Pooled representation of shape (batch_size, hidden_dim)
        """
        batch_size = x.size(0)
        
        # Parse input features into tokens
        idx = 0
        
        # Hole cards (2 tokens)
        hole_cards = []
        for i in range(2):
            hole_card = x[:, idx:idx+self.hole_card_dim]
            hole_cards.append(self.hole_card_embed(hole_card))
            idx += self.hole_card_dim
        
        # Community cards (5 tokens)
        community_cards = []
        for i in range(5):
            comm_card = x[:, idx:idx+self.community_card_dim]
            community_cards.append(self.community_card_embed(comm_card))
            idx += self.community_card_dim
        
        # Pot/betting info (1 token)
        pot_info = x[:, idx:idx+self.pot_dim]
        pot_token = self.pot_embed(pot_info)
        idx += self.pot_dim
        
        # Stage (1 token)
        stage_info = x[:, idx:idx+self.stage_dim]
        stage_token = self.stage_embed(stage_info)
        idx += self.stage_dim
        
        # Position (1 token)
        position_info = x[:, idx:idx+self.position_dim]
        position_token = self.position_embed(position_info)
        idx += self.position_dim
        
        # Game theory features (1 token)
        gametheory_info = x[:, idx:idx+self.gametheory_dim]
        gametheory_token = self.gametheory_embed(gametheory_info)
        idx += self.gametheory_dim
        
        # Opponent modeling features (1 token)
        opponent_info = x[:, idx:idx+self.opponent_dim]
        opponent_token = self.opponent_embed(opponent_info)
        idx += self.opponent_dim
        
        # Hand strength (1 token)
        handstrength_info = x[:, idx:idx+self.handstrength_dim]
        handstrength_token = self.handstrength_embed(handstrength_info)
        
        # Stack all tokens
        tokens = torch.stack(
            hole_cards + community_cards + 
            [pot_token, stage_token, position_token, gametheory_token, 
             opponent_token, handstrength_token],
            dim=1
        )
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        
        # Add positional encodings
        tokens = tokens + self.pos_encoding
        
        # Apply transformer
        if self.gradient_checkpointing and self.training:
            transformed = torch.utils.checkpoint.checkpoint(
                self.transformer, tokens, use_reentrant=False
            )
        else:
            transformed = self.transformer(tokens)
        
        # Apply layer norm
        transformed = self.norm(transformed)
        
        # Aggregate tokens using CLS token
        pooled = transformed[:, 0, :]  # (batch_size, hidden_dim)
        
        return pooled
    
    def forward(
        self, 
        x: torch.Tensor, 
        legal_actions_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor-critic network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            legal_actions_mask: Boolean mask of shape (batch_size, num_actions)
                                True for legal actions, False for illegal
        
        Returns:
            Tuple of:
            - action_logits: Raw logits of shape (batch_size, num_actions)
            - value: State value of shape (batch_size, 1)
        """
        # Get pooled representation
        pooled = self._encode_tokens(x)
        
        # Extract hand strength from input (last feature in the state encoding)
        # Hand strength is at index 166 (0-indexed) = position 167 total
        hand_strength = x[:, -1:].detach()  # (batch_size, 1)
        
        # Policy head - compute action logits
        action_logits = self.policy_head(pooled)  # (batch_size, num_actions)
        
        # Apply action prior to balance probability mass across action types
        action_logits = action_logits + self.action_prior
        
        # Apply hand-strength-aware action gating
        # This penalizes aggressive actions (especially all-in) when hand strength is low
        gate_input = torch.cat([pooled, hand_strength], dim=-1)
        action_gate = self.hand_strength_gate(gate_input)  # (batch_size, num_actions)
        
        # The gate outputs adjustments that are larger (more negative) for all-in with weak hands
        # We scale the gate based on inverse hand strength for aggressive actions
        # When hand_strength < threshold, all-in gets extra penalty
        weak_hand_mask = (hand_strength < self.hand_strength_threshold).float()  # (batch_size, 1)
        
        # Create penalty tensor - mostly zeros, but penalize all-in when hand is weak
        # Penalty reduced from -5.0 to -3.0 to allow some bluffing/semi-bluffs
        all_in_penalty = torch.zeros_like(action_logits)
        all_in_penalty[:, 12] = -3.0 * weak_hand_mask.squeeze()  # All-in index is 12 in unified action space
        
        # Also penalize large raises (indices 17-20: raise_100%, raise_150%, raise_200%, raise_300%)
        # Penalty reduced from -2.0 to -1.0
        for raise_idx in [17, 18, 19, 20]:
            if raise_idx < action_logits.size(1):
                all_in_penalty[:, raise_idx] = -1.0 * weak_hand_mask.squeeze()
        
        # Apply gate (learned adjustment) + fixed penalty
        action_logits = action_logits + action_gate + all_in_penalty
        
        # Apply legal action mask if provided
        if legal_actions_mask is not None:
            action_logits = action_logits.masked_fill(~legal_actions_mask, -1e9)
        
        # Value head - compute state value
        value = self.value_head(pooled)  # (batch_size, 1)
        
        return action_logits, value
    
    def forward_with_hand_strength(
        self, 
        x: torch.Tensor, 
        legal_actions_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass including hand strength prediction (for training with auxiliary loss).
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            legal_actions_mask: Boolean mask of shape (batch_size, num_actions)
        
        Returns:
            Tuple of:
            - action_logits: Raw logits of shape (batch_size, num_actions)
            - value: State value of shape (batch_size, 1)
            - hand_strength_pred: Predicted hand strength of shape (batch_size, 1)
        """
        # Get pooled representation
        pooled = self._encode_tokens(x)
        
        # Extract hand strength from input
        hand_strength = x[:, -1:].detach()
        
        # Policy head
        action_logits = self.policy_head(pooled)
        action_logits = action_logits + self.action_prior
        
        # Apply hand-strength-aware action gating
        gate_input = torch.cat([pooled, hand_strength], dim=-1)
        action_gate = self.hand_strength_gate(gate_input)
        
        weak_hand_mask = (hand_strength < self.hand_strength_threshold).float()
        all_in_penalty = torch.zeros_like(action_logits)
        all_in_penalty[:, 12] = -3.0 * weak_hand_mask.squeeze()  # All-in index is 12
        for raise_idx in [17, 18, 19, 20]:
            if raise_idx < action_logits.size(1):
                all_in_penalty[:, raise_idx] = -1.0 * weak_hand_mask.squeeze()
        
        action_logits = action_logits + action_gate + all_in_penalty
        
        if legal_actions_mask is not None:
            action_logits = action_logits.masked_fill(~legal_actions_mask, -1e9)
        
        # Value head
        value = self.value_head(pooled)
        
        # Hand strength prediction head
        hand_strength_pred = self.hand_strength_head(pooled)
        
        return action_logits, value, hand_strength_pred
    
    def get_action_probs(
        self, 
        x: torch.Tensor, 
        legal_actions_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action probabilities (for inference).
        
        Args:
            x: Input state
            legal_actions_mask: Boolean mask for legal actions
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            Tuple of:
            - action_probs: Probabilities of shape (batch_size, num_actions)
            - value: State value of shape (batch_size, 1)
        """
        action_logits, value = self.forward(x, legal_actions_mask)
        
        # Apply temperature
        action_logits = action_logits / temperature
        
        # Convert to probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs, value
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        legal_actions_mask: Optional[torch.Tensor] = None,
        return_hand_strength: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, ...]:
        """
        Evaluate actions for PPO training.
        
        Args:
            states: State tensor of shape (batch_size, input_dim)
            actions: Action indices of shape (batch_size,)
            legal_actions_mask: Boolean mask for legal actions
            return_hand_strength: If True, also return hand strength prediction
        
        Returns:
            Tuple of:
            - log_probs: Log probabilities of taken actions (batch_size,)
            - values: State values (batch_size,)
            - entropy: Policy entropy (batch_size,)
            - (optional) hand_strength_pred: Predicted hand strength (batch_size,)
        """
        if return_hand_strength:
            action_logits, values, hand_strength_pred = self.forward_with_hand_strength(
                states, legal_actions_mask
            )
        else:
            action_logits, values = self.forward(states, legal_actions_mask)
        
        # Compute action probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Log probabilities of taken actions
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Entropy for exploration bonus
        entropy = -(action_probs * log_probs).sum(dim=-1)
        
        if return_hand_strength:
            return action_log_probs, values.squeeze(1), entropy, hand_strength_pred.squeeze(1)
        return action_log_probs, values.squeeze(1), entropy


def create_actor_critic(
    input_dim: int,
    hidden_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 4,
    dropout: float = 0.1,
    gradient_checkpointing: bool = False
) -> PokerActorCritic:
    """
    Factory function to create an actor-critic model.
    
    Args:
        input_dim: Size of input feature vector
        hidden_dim: Transformer hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        gradient_checkpointing: Enable gradient checkpointing
    
    Returns:
        PokerActorCritic model
    """
    return PokerActorCritic(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        gradient_checkpointing=gradient_checkpointing
    )

