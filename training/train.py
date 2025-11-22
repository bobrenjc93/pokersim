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
# Feature Encoding
# =============================================================================

# Card mapping: rank to index
RANK_MAP = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 
            'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
# Suit mapping
SUIT_MAP = {'C': 0, 'D': 1, 'H': 2, 'S': 3}

# Stage mapping
STAGE_MAP = {'Preflop': 0, 'Flop': 1, 'Turn': 2, 'River': 3, 'Complete': 4}

# Action mapping
ACTION_MAP = {'fold': 0, 'check': 1, 'call': 2, 'bet': 3, 'raise': 4, 'all_in': 5}
ACTION_NAMES = ['fold', 'check', 'call', 'bet', 'raise', 'all_in']


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


def encode_action(action: dict[str, Any]) -> int:
    """
    Convert action dictionary to action index.
    
    Args:
        action: Action dictionary with 'action' key
    
    Returns:
        Action index (0-5)
    """
    action_type = action.get('action', 'fold')
    return ACTION_MAP.get(action_type, 0)


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
                    action_idx = encode_action(action)
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
            nn.Linear(hidden_dim // 4, 6)  # 6 possible actions
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
            nn.Linear(hidden_dim // 4, 6)  # 6 possible actions
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


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description="Train a poker action prediction model from rollout data"
    )
    
    parser.add_argument('--data', type=str, default='data/quickstart_rollouts.json',
                       help='Path to rollout data JSON file or directory of JSON files')
    parser.add_argument('--output', type=str, default='/tmp/models/poker_model.pt',
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
    parser.add_argument('--use-legacy-model', action='store_true',
                       help='Use legacy feed-forward model instead of transformer')
    
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
    
    # Create dataset
    dataset = PokerDataset(rollouts)
    
    if len(dataset) == 0:
        print("✗ Error: No training samples found")
        return 1
    
    feature_dim = dataset.get_feature_dim()
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"✓ Loaded {len(rollouts)} rollouts ({len(dataset)} state-action pairs)")
    print(f"✓ Train samples: {train_size}, Validation samples: {val_size}")
    
    # Create data loaders
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
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    # Create model
    if args.use_legacy_model:
        print("Using legacy feed-forward model")
        model = PokerNet(input_dim=feature_dim, hidden_dim=args.hidden_dim)
    else:
        print("Using transformer-based model")
        model = PokerTransformer(
            input_dim=feature_dim, 
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
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
            
            # Check if checkpoint matches model type
            checkpoint_model_type = checkpoint.get('model_type', 'legacy')
            current_model_type = 'legacy' if args.use_legacy_model else 'transformer'
            
            if checkpoint_model_type != current_model_type:
                print(f"⚠️  Warning: Checkpoint is {checkpoint_model_type} but current model is {current_model_type}")
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

    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device, epoch, start_epoch + args.epochs - 1)
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}/{start_epoch + args.epochs - 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%", flush=True)
        
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
                'model_type': 'legacy' if args.use_legacy_model else 'transformer',
            }
            
            # Add transformer-specific parameters if using transformer
            if not args.use_legacy_model:
                checkpoint_data.update({
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout,
                })
            
            torch.save(checkpoint_data, output_path)
            
            print(f"  → Best model saved (val loss: {best_val_loss:.4f})", flush=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"\n  ⚠️  Early stopping triggered after {epoch} epochs (no improvement for {args.early_stopping_patience} epochs)")
                break
    
    print(f"\n✓ Training complete! Best validation loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

