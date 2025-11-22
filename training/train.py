#!/usr/bin/env python3
"""
Training Example

This example shows how to:
1. Load generated rollout data
2. Convert poker states into tensors
3. Train a neural network to predict actions
4. Save the trained model

The model learns to map game states to action probabilities, which can be
used for decision-making in poker.

Prerequisites:
- Generated rollout data (from generate_rollouts.py)
- PyTorch (pip install torch)

Usage:
    python train.py --data data/quickstart_rollouts.json --epochs 100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# Model Version Management
# =============================================================================

# Model architecture versions
MODEL_VERSION_LEGACY = 1  # Feed-forward network (PokerNet)
MODEL_VERSION_TRANSFORMER = 2  # Transformer-based network (PokerTransformer)
MODEL_VERSION_CURRENT = MODEL_VERSION_TRANSFORMER  # Default for new models

# =============================================================================
# Feature Encoding
# =============================================================================

# Card mapping: rank to index
RANK_MAP = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 
            'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
# Suit mapping
SUIT_MAP = {'C': 0, 'D': 1, 'H': 2, 'S': 3}

# Stage mapping
STAGE_MAP = {'Preflop': 0, 'Flop': 1, 'Turn': 2, 'River': 3, 'Complete': 4}

# Action mapping with granular bet sizing
# Bet/Raise sizes as percentages of pot: 10%, 25%, 33%, 50%, 75%, 100%, 150%, 200%, 300%
ACTION_MAP = {
    'fold': 0, 'check': 1, 'call': 2,
    'bet_10%': 3, 'bet_25%': 4, 'bet_33%': 5, 'bet_50%': 6, 'bet_75%': 7,
    'bet_100%': 8, 'bet_150%': 9, 'bet_200%': 10, 'bet_300%': 11,
    'raise_10%': 12, 'raise_25%': 13, 'raise_33%': 14, 'raise_50%': 15, 'raise_75%': 16,
    'raise_100%': 17, 'raise_150%': 18, 'raise_200%': 19, 'raise_300%': 20,
    'all_in': 21
}
ACTION_NAMES = [
    'fold', 'check', 'call',
    'bet_10%', 'bet_25%', 'bet_33%', 'bet_50%', 'bet_75%', 'bet_100%', 'bet_150%', 'bet_200%', 'bet_300%',
    'raise_10%', 'raise_25%', 'raise_33%', 'raise_50%', 'raise_75%', 'raise_100%', 'raise_150%', 'raise_200%', 'raise_300%',
    'all_in'
]

# Bet size percentages for each action (as fraction of pot)
BET_SIZE_MAP = {
    'bet_10%': 0.10, 'bet_25%': 0.25, 'bet_33%': 0.33, 'bet_50%': 0.50, 'bet_75%': 0.75,
    'bet_100%': 1.0, 'bet_150%': 1.5, 'bet_200%': 2.0, 'bet_300%': 3.0,
    'raise_10%': 0.10, 'raise_25%': 0.25, 'raise_33%': 0.33, 'raise_50%': 0.50, 'raise_75%': 0.75,
    'raise_100%': 1.0, 'raise_150%': 1.5, 'raise_200%': 2.0, 'raise_300%': 3.0,
}


def encode_card(card: str) -> tuple[int, int]:
    """
    Encode a card string (e.g., 'AS', 'KC') into rank and suit indices.
    
    Args:
        card: Card string (rank + suit)
    
    Returns:
        tuple: (rank_index, suit_index)
    """
    if len(card) != 2:
        return (0, 0)
    rank = RANK_MAP.get(card[0], 0)
    suit = SUIT_MAP.get(card[1], 0)
    return (rank, suit)


def encode_state(state: dict[str, Any]) -> torch.Tensor:
    """
    Convert a poker state dictionary into a feature vector.
    
    Features include:
    - Hole cards (2 cards, one-hot encoded)
    - Community cards (5 cards, one-hot encoded)
    - Pot size (normalized)
    - Current bet (normalized)
    - Player chips (normalized)
    - Player bet (normalized)
    - Stage (one-hot)
    - Position features
    
    Args:
        state: State dictionary from rollout
    
    Returns:
        Tensor of shape (feature_dim,)
    """
    features = []
    
    # Encode hole cards (2 cards × (13 ranks + 4 suits) = 34 features)
    hole_cards = state.get('hole_cards', [])
    for i in range(2):
        if i < len(hole_cards):
            rank, suit = encode_card(hole_cards[i])
            rank_onehot = [0] * 13
            rank_onehot[rank] = 1
            suit_onehot = [0] * 4
            suit_onehot[suit] = 1
            features.extend(rank_onehot + suit_onehot)
        else:
            features.extend([0] * 17)  # No card
    
    # Encode community cards (5 cards × 17 features = 85 features)
    community_cards = state.get('community_cards', [])
    for i in range(5):
        if i < len(community_cards):
            rank, suit = encode_card(community_cards[i])
            rank_onehot = [0] * 13
            rank_onehot[rank] = 1
            suit_onehot = [0] * 4
            suit_onehot[suit] = 1
            features.extend(rank_onehot + suit_onehot)
        else:
            features.extend([0] * 17)  # No card
    
    # Normalize chip values (divide by starting stack, typically 1000)
    starting_chips = 1000.0
    pot = state.get('pot', 0) / starting_chips
    current_bet = state.get('current_bet', 0) / starting_chips
    player_chips = state.get('player_chips', 0) / starting_chips
    player_bet = state.get('player_bet', 0) / starting_chips
    player_total_bet = state.get('player_total_bet', 0) / starting_chips
    
    features.extend([pot, current_bet, player_chips, player_bet, player_total_bet])
    
    # Stage (one-hot, 5 features)
    stage = state.get('stage', 'Preflop')
    stage_idx = STAGE_MAP.get(stage, 0)
    stage_onehot = [0] * 5
    stage_onehot[stage_idx] = 1
    features.extend(stage_onehot)
    
    # Position features
    num_players = state.get('num_players', 2)
    num_active = state.get('num_active', 2)
    position = state.get('position', 0)
    is_dealer = float(state.get('is_dealer', False))
    is_small_blind = float(state.get('is_small_blind', False))
    is_big_blind = float(state.get('is_big_blind', False))
    
    features.extend([
        num_players / 10.0,  # Normalize by max players
        num_active / 10.0,
        position / 10.0,
        is_dealer,
        is_small_blind,
        is_big_blind
    ])
    
    return torch.tensor(features, dtype=torch.float32)


def get_bet_size_bucket(amount: int, pot: int, min_amount: int, max_amount: int, is_bet: bool) -> str:
    """
    Determine which bet size bucket an amount falls into.
    
    Args:
        amount: The bet/raise amount
        pot: Current pot size
        min_amount: Minimum legal bet/raise amount
        max_amount: Maximum legal bet/raise amount (player chips)
        is_bet: True if this is a bet, False if it's a raise
    
    Returns:
        Action label (e.g., 'bet_50%', 'raise_100%')
    """
    # All-in case (at or very close to max)
    if amount >= max_amount * 0.98:  # Within 2% of max is considered all-in
        return 'all_in'
    
    # Avoid division by zero
    if pot <= 0:
        pot = min_amount  # Use minimum as fallback
    
    # Calculate as percentage of pot
    pot_fraction = amount / pot
    
    # Define bet size buckets (sorted)
    buckets = [(0.10, '10%'), (0.25, '25%'), (0.33, '33%'), (0.50, '50%'), (0.75, '75%'),
               (1.0, '100%'), (1.5, '150%'), (2.0, '200%'), (3.0, '300%')]
    
    # Find closest bucket
    best_bucket = buckets[-1][1]  # Default to largest
    min_diff = float('inf')
    
    for threshold, bucket_name in buckets:
        diff = abs(pot_fraction - threshold)
        if diff < min_diff:
            min_diff = diff
            best_bucket = bucket_name
    
    prefix = 'bet_' if is_bet else 'raise_'
    return prefix + best_bucket


def encode_action(action: dict[str, Any], state: dict[str, Any] = None) -> int:
    """
    Convert action dictionary to action index.
    
    Args:
        action: Action dictionary with 'action' key
        state: Optional state dictionary for bet sizing context
    
    Returns:
        Action index (0-21)
    """
    # Check if we have a granular action label (e.g., 'bet_50%', 'raise_100%')
    action_label = action.get('action_label')
    if action_label and action_label in ACTION_MAP:
        return ACTION_MAP[action_label]
    
    # Get basic action type
    action_type = action.get('action', 'fold')
    
    # For bet/raise actions, determine the size bucket
    if action_type in ['bet', 'raise'] and state is not None:
        amount = action.get('amount', 0)
        pot = state.get('pot', 0)
        player_chips = state.get('player_chips', 0)
        
        if action_type == 'bet':
            min_bet = state.get('min_bet', state.get('big_blind', 20))
            action_label = get_bet_size_bucket(amount, pot, min_bet, player_chips, is_bet=True)
        else:  # raise
            min_raise_total = state.get('min_raise_total', state.get('big_blind', 20))
            action_label = get_bet_size_bucket(amount, pot, min_raise_total, player_chips, is_bet=False)
        
        if action_label in ACTION_MAP:
            return ACTION_MAP[action_label]
    
    # Fall back to direct mapping (fold, check, call, all_in)
    return ACTION_MAP.get(action_type, 0)


# =============================================================================
# Adaptive Training Scheduler
# =============================================================================

class AdaptiveDataScheduler:
    """
    Manages progressive dataset growth during training.
    Starts with a small subset of data and increases when validation loss plateaus.
    """
    
    def __init__(self, total_samples: int, initial_fraction: float = 0.1, 
                 growth_factor: float = 1.5, plateau_patience: int = 5,
                 plateau_threshold: float = 0.001):
        """
        Args:
            total_samples: Total number of available samples
            initial_fraction: Initial fraction of data to use (0.0-1.0)
            growth_factor: Multiply current size by this when growing
            plateau_patience: Epochs without improvement before growing dataset
            plateau_threshold: Minimum improvement to not be considered plateaued
        """
        self.total_samples = total_samples
        self.initial_fraction = max(0.01, min(1.0, initial_fraction))
        self.growth_factor = max(1.1, growth_factor)
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold
        
        # Start with initial fraction, but at least 100 samples or 1% of data
        self.current_size = max(
            100,
            int(total_samples * self.initial_fraction)
        )
        self.current_size = min(self.current_size, total_samples)
        
        # Track validation loss history
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.growth_history = [self.current_size]
        
    def get_current_size(self) -> int:
        """Get current dataset size to use"""
        return self.current_size
    
    def get_progress(self) -> float:
        """Get progress through curriculum (0.0 to 1.0)"""
        return self.current_size / self.total_samples
    
    def update(self, val_loss: float) -> bool:
        """
        Update scheduler with validation loss.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if dataset size was increased, False otherwise
        """
        # Check if we improved
        if val_loss < self.best_val_loss - self.plateau_threshold:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            return False
        
        # No improvement
        self.epochs_without_improvement += 1
        
        # Check if we should grow dataset
        if self.epochs_without_improvement >= self.plateau_patience:
            # Only grow if we haven't reached full dataset yet
            if self.current_size < self.total_samples:
                new_size = int(self.current_size * self.growth_factor)
                new_size = min(new_size, self.total_samples)
                
                if new_size > self.current_size:
                    print(f"\n📈 Validation loss plateaued. Growing dataset:")
                    print(f"   {self.current_size:,} → {new_size:,} samples ({new_size/self.total_samples*100:.1f}% of total)")
                    
                    self.current_size = new_size
                    self.growth_history.append(new_size)
                    self.epochs_without_improvement = 0
                    return True
        
        return False
    
    def is_at_full_size(self) -> bool:
        """Check if using full dataset"""
        return self.current_size >= self.total_samples


# =============================================================================
# Dataset
# =============================================================================

class PokerDataset(Dataset):
    """
    PyTorch dataset for poker state-action pairs.
    
    Each sample is a (state_features, action_label) pair where:
    - state_features: Encoded state vector
    - action_label: Action taken (0-5)
    """
    
    def __init__(self, rollouts: list[dict[str, Any]]):
        """
        Args:
            rollouts: List of rollout dictionaries
        """
        self.samples = []
        
        # Extract state-action pairs from all rollouts
        for rollout in rollouts:
            states = rollout.get('states', [])
            actions = rollout.get('actions', [])
            
            # Match states with actions (should be 1:1)
            for state, action in zip(states, actions):
                # Only keep if the action is for the same player as the state
                if state.get('player_id') == action.get('playerId'):
                    state_tensor = encode_state(state)
                    action_idx = encode_action(action, state)  # Pass state for bet sizing context
                    self.samples.append((state_tensor, action_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def get_feature_dim(self):
        """Return the dimension of state features"""
        if len(self.samples) > 0:
            return self.samples[0][0].shape[0]
        return 0


# =============================================================================
# Neural Network Model
# =============================================================================

class PokerTransformer(nn.Module):
    """
    Transformer-based neural network for poker action prediction.
    
    Architecture:
    - Input: State features organized into tokens (cards, pot info, position)
    - Embedding: Project each token to embedding dimension
    - Positional Encoding: Add positional information
    - Transformer Encoder: Multi-head self-attention layers
    - Aggregation: Pool transformer outputs
    - Output: Action probabilities (6 classes)
    
    This modern architecture can better capture:
    - Relationships between cards (e.g., suited connectors)
    - Complex betting patterns
    - Position-dependent strategies
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_heads: int = 8, 
                 num_layers: int = 4, dropout: float = 0.1):
        """
        Args:
            input_dim: Size of input feature vector
            hidden_dim: Dimension of transformer embeddings (must be divisible by num_heads)
            num_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Ensure hidden_dim is divisible by num_heads
        if hidden_dim % num_heads != 0:
            hidden_dim = ((hidden_dim // num_heads) + 1) * num_heads
            print(f"  Adjusted hidden_dim to {hidden_dim} to be divisible by {num_heads} heads")
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Token organization:
        # - 2 hole card tokens (17 features each)
        # - 5 community card tokens (17 features each)
        # - 1 pot/betting token (5 features: pot, current_bet, player_chips, player_bet, player_total_bet)
        # - 1 stage token (5 features: one-hot stage)
        # - 1 position token (6 features: position info)
        
        self.hole_card_dim = 17
        self.community_card_dim = 17
        self.pot_dim = 5
        self.stage_dim = 5
        self.position_dim = 6
        
        # Input projections for different token types
        self.hole_card_embed = nn.Linear(self.hole_card_dim, hidden_dim)
        self.community_card_embed = nn.Linear(self.community_card_dim, hidden_dim)
        self.pot_embed = nn.Linear(self.pot_dim, hidden_dim)
        self.stage_embed = nn.Linear(self.stage_dim, hidden_dim)
        self.position_embed = nn.Linear(self.position_dim, hidden_dim)
        
        # Learnable positional encodings for each token type
        # Total: 2 (hole) + 5 (community) + 1 (pot) + 1 (stage) + 1 (position) = 10 tokens
        self.num_tokens = 10
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_tokens, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',  # GELU is more modern than ReLU
            batch_first=True,
            norm_first=True  # Pre-LayerNorm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output head with residual projection
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 22)  # 22 possible actions (including granular bet sizes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, 6) with action logits
        """
        batch_size = x.size(0)
        
        # Parse input features into tokens
        # Feature layout: 2*17 (hole) + 5*17 (community) + 5 (pot) + 5 (stage) + 6 (position) = 135
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
        
        # Stack all tokens: (batch_size, num_tokens, hidden_dim)
        tokens = torch.stack(
            hole_cards + community_cards + [pot_token, stage_token, position_token],
            dim=1
        )
        
        # Add positional encodings
        tokens = tokens + self.pos_encoding
        
        # Apply transformer
        transformed = self.transformer(tokens)  # (batch_size, num_tokens, hidden_dim)
        
        # Apply layer norm
        transformed = self.norm(transformed)
        
        # Aggregate tokens using mean pooling
        pooled = transformed.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Output head
        logits = self.output_head(pooled)  # (batch_size, 6)
        
        return logits


# Legacy model for backward compatibility
class PokerNet(nn.Module):
    """
    Legacy feed-forward neural network for poker action prediction.
    Kept for backward compatibility with existing checkpoints.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """
        Args:
            input_dim: Size of input feature vector
            hidden_dim: Size of hidden layers
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 22)  # 22 possible actions (including granular bet sizes)
        )
    
    def forward(self, x):
        """Forward pass"""
        return self.network(x)


# =============================================================================
# Training
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        total_epochs: Total number of epochs for training
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (states, actions) in enumerate(train_loader):
        states = states.to(device)
        actions = actions.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(states)
        loss = criterion(outputs, actions)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += actions.size(0)
        correct += predicted.eq(actions).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """
    Validate the model on validation data.
    
    Args:
        model: Neural network
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Tuple of (average loss, accuracy) for validation set
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for states, actions in val_loader:
            states = states.to(device)
            actions = actions.to(device)
            
            # Forward pass
            outputs = model(states)
            loss = criterion(outputs, actions)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += actions.size(0)
            correct += predicted.eq(actions).sum().item()
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    accuracy = 100. * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def main() -> int:
    """Main training function"""
    parser = argparse.ArgumentParser(
        description="Train a poker action prediction model from rollout data"
    )
    
    parser.add_argument('--data', type=str, default='data/quickstart_rollouts.json',
                       help='Path to rollout data JSON file or directory of JSON files')
    parser.add_argument('--output', type=str, default='/tmp/pokersim/models/poker_model.pt',
                       help='Output path for trained model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden layer dimension (for transformer: embedding dimension)')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads in transformer (default: 8)')
    parser.add_argument('--num-layers', type=int, default=4,
                       help='Number of transformer encoder layers (default: 4)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--model-version', type=int, default=MODEL_VERSION_CURRENT,
                       choices=[MODEL_VERSION_LEGACY, MODEL_VERSION_TRANSFORMER],
                       help=f'Model architecture version (1=legacy feed-forward, 2=transformer, default={MODEL_VERSION_CURRENT})')
    
    # Adaptive training schedule
    parser.add_argument('--adaptive-schedule', action='store_true',
                       help='Enable adaptive training schedule (start small, grow as loss plateaus)')
    parser.add_argument('--initial-data-fraction', type=float, default=0.1,
                       help='Initial fraction of data to train on (default: 0.1)')
    parser.add_argument('--data-growth-factor', type=float, default=1.5,
                       help='Factor to grow dataset by when plateauing (default: 1.5)')
    parser.add_argument('--plateau-patience', type=int, default=5,
                       help='Epochs without improvement before increasing data size (default: 5)')
    parser.add_argument('--plateau-threshold', type=float, default=0.001,
                       help='Minimum improvement to not be considered plateaued (default: 0.001)')
    
    args = parser.parse_args()
    
    # Load data - support both single file and directory
    data_path = Path(args.data) if Path(args.data).is_absolute() else Path(__file__).parent / args.data
    
    if not data_path.exists():
        print(f"✗ Error: Data path not found: {data_path}")
        print(f"  Please generate rollouts first:")
        print(f"  python generate_rollouts.py --num-rollouts 1000")
        return 1
    
    rollouts = []
    if data_path.is_dir():
        # Load all JSON files from directory
        json_files = sorted(data_path.glob("*.json"))
        if not json_files:
            print(f"✗ Error: No JSON files found in directory: {data_path}")
            return 1
        
        print(f"📂 Loading data from {len(json_files)} files in {data_path}...")
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    file_rollouts = json.load(f)
                    rollouts.extend(file_rollouts)
                    print(f"  ✓ Loaded {len(file_rollouts)} rollouts from {json_file.name}")
            except Exception as e:
                print(f"  ⚠️  Error loading {json_file.name}: {e}")
    else:
        # Load single file
        with open(data_path, 'r') as f:
            rollouts = json.load(f)
    
    # Create full dataset
    full_dataset = PokerDataset(rollouts)
    
    if len(full_dataset) == 0:
        print("✗ Error: No training samples found")
        return 1
    
    feature_dim = full_dataset.get_feature_dim()
    
    print(f"✓ Loaded {len(rollouts)} rollouts ({len(full_dataset)} state-action pairs)")
    
    # Setup adaptive scheduler if enabled
    adaptive_scheduler = None
    if args.adaptive_schedule:
        adaptive_scheduler = AdaptiveDataScheduler(
            total_samples=len(full_dataset),
            initial_fraction=args.initial_data_fraction,
            growth_factor=args.data_growth_factor,
            plateau_patience=args.plateau_patience,
            plateau_threshold=args.plateau_threshold
        )
        print(f"\n🎯 Adaptive training schedule enabled:")
        print(f"   Starting with {adaptive_scheduler.get_current_size():,} samples ({adaptive_scheduler.get_progress()*100:.1f}%)")
        print(f"   Will grow by {args.data_growth_factor}x when loss plateaus")
    
    # Helper function to create dataloaders with current curriculum size
    def create_dataloaders(dataset, current_size=None):
        """Create train/val dataloaders, optionally with a subset of data"""
        if current_size is None or current_size >= len(dataset):
            # Use full dataset
            working_dataset = dataset
        else:
            # Use subset
            indices = list(range(len(dataset)))
            import random
            random.shuffle(indices)
            subset_indices = indices[:current_size]
            working_dataset = torch.utils.data.Subset(dataset, subset_indices)
        
        # Split into train/val (80/20)
        train_size = int(0.8 * len(working_dataset))
        val_size = len(working_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            working_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader, train_size, val_size
    
    # Create initial dataloaders
    if adaptive_scheduler:
        train_loader, val_loader, train_size, val_size = create_dataloaders(
            full_dataset, adaptive_scheduler.get_current_size()
        )
    else:
        train_loader, val_loader, train_size, val_size = create_dataloaders(full_dataset)
    
    print(f"✓ Train samples: {train_size}, Validation samples: {val_size}")
    
    # Setup device (prioritize CUDA, then MPS, then CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using device: CUDA (GPU)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✓ Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print(f"✓ Using device: CPU")
    print(f"  Device: {device}")
    
    # Create model based on version
    if args.model_version == MODEL_VERSION_LEGACY:
        print(f"Using model version {MODEL_VERSION_LEGACY}: Legacy feed-forward network")
        model = PokerNet(input_dim=feature_dim, hidden_dim=args.hidden_dim)
    elif args.model_version == MODEL_VERSION_TRANSFORMER:
        print(f"Using model version {MODEL_VERSION_TRANSFORMER}: Transformer-based network")
        model = PokerTransformer(
            input_dim=feature_dim, 
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model version: {args.model_version}")
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model parameters: {num_params:,}")
    
    # Loss and optimizer with weight decay (L2 regularization)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    start_epoch = 1
    
    # Load from checkpoint if provided
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint) if Path(args.checkpoint).is_absolute() else Path(__file__).parent / args.checkpoint
        if checkpoint_path.exists():
            print(f"\n📂 Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Check if checkpoint model version matches current version
            # Handle legacy checkpoints that used 'model_type' string
            if 'model_version' in checkpoint:
                checkpoint_version = checkpoint['model_version']
            else:
                # Legacy checkpoint compatibility
                checkpoint_model_type = checkpoint.get('model_type', 'legacy')
                checkpoint_version = MODEL_VERSION_LEGACY if checkpoint_model_type == 'legacy' else MODEL_VERSION_TRANSFORMER
            
            if checkpoint_version != args.model_version:
                print(f"⚠️  Warning: Checkpoint is model version {checkpoint_version} but current model is version {args.model_version}")
                print(f"   Starting training from scratch with new architecture")
            else:
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint.get('epoch', 0) + 1
                    best_val_loss = checkpoint.get('val_loss', float('inf'))
                    print(f"✓ Resumed from epoch {checkpoint.get('epoch', 0)}, best val loss: {best_val_loss:.4f}")
                except Exception as e:
                    print(f"⚠️  Error loading checkpoint: {e}")
                    print(f"   Starting training from scratch")
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}, starting from scratch")
    
    print(f"\nStarting training for {args.epochs} epochs...")
    if start_epoch > 1:
        print(f"  Continuing from epoch {start_epoch}")

    current_epoch = start_epoch
    total_epochs_run = 0
    max_total_epochs = args.epochs * 5  # Safety limit to prevent infinite training
    
    while total_epochs_run < max_total_epochs:
        # Determine how many epochs to run in this phase
        if adaptive_scheduler and not adaptive_scheduler.is_at_full_size():
            # When using adaptive schedule, run fewer epochs per phase
            epochs_this_phase = min(args.epochs, max_total_epochs - total_epochs_run)
        else:
            # Regular training or at full dataset size
            epochs_this_phase = args.epochs
        
        phase_end_epoch = current_epoch + epochs_this_phase
        
        for epoch in range(current_epoch, phase_end_epoch):
            train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device, epoch, phase_end_epoch - 1)
            val_loss, val_acc = validate_model(model, val_loader, criterion, device)
            
            # Progress indicator for adaptive schedule
            if adaptive_scheduler:
                progress_str = f" [Data: {adaptive_scheduler.get_progress()*100:.0f}%]"
            else:
                progress_str = ""
            
            print(f"Epoch {epoch}/{phase_end_epoch - 1}{progress_str} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%", flush=True)
            
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                output_path = Path(args.output) if Path(args.output).is_absolute() else Path(__file__).parent / args.output
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save model state and metadata
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'feature_dim': feature_dim,
                    'hidden_dim': args.hidden_dim,
                    'action_names': ACTION_NAMES,
                    'model_version': args.model_version,
                }
                
                # Add version-specific parameters
                if args.model_version == MODEL_VERSION_TRANSFORMER:
                    checkpoint_data.update({
                        'num_heads': args.num_heads,
                        'num_layers': args.num_layers,
                        'dropout': args.dropout,
                    })
                
                torch.save(checkpoint_data, output_path)
                
                print(f"  → Best model saved (val loss: {best_val_loss:.4f})", flush=True)
            else:
                epochs_without_improvement += 1
                # Only trigger early stopping if not using adaptive schedule or already at full size
                if (not adaptive_scheduler or adaptive_scheduler.is_at_full_size()) and \
                   epochs_without_improvement >= args.early_stopping_patience:
                    print(f"\n  ⚠️  Early stopping triggered after {epoch} epochs (no improvement for {args.early_stopping_patience} epochs)")
                    total_epochs_run = max_total_epochs  # Exit outer loop too
                    break
            
            # Update adaptive scheduler if enabled
            if adaptive_scheduler:
                dataset_grew = adaptive_scheduler.update(val_loss)
                
                if dataset_grew:
                    # Recreate dataloaders with new dataset size
                    train_loader, val_loader, train_size, val_size = create_dataloaders(
                        full_dataset, adaptive_scheduler.get_current_size()
                    )
                    print(f"   New split - Train: {train_size:,}, Val: {val_size:,}")
                    
                    # Reset improvement counter when we grow dataset
                    epochs_without_improvement = 0
                    
                    # Break to start new phase with larger dataset
                    current_epoch = epoch + 1
                    total_epochs_run += (epoch - current_epoch + epochs_this_phase)
                    break
        else:
            # Completed this phase normally (no break)
            total_epochs_run += epochs_this_phase
            current_epoch = phase_end_epoch
            
            # If not using adaptive schedule or at full size, we're done
            if not adaptive_scheduler or adaptive_scheduler.is_at_full_size():
                break
        
        # Safety check
        if total_epochs_run >= max_total_epochs:
            print(f"\n  ⚠️  Reached maximum epochs ({max_total_epochs})")
            break
    
    print(f"\n✓ Training complete! Best validation loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

