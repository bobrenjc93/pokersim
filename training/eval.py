#!/usr/bin/env python3
"""
Evaluation Script - Play Against Random Agent

This script evaluates a trained model by:
1. Loading the trained model
2. Playing 100 hands against a random agent
3. Computing performance metrics (win rate, profit)
4. Comparing performance to baseline

Prerequisites:
- Trained model (from train.py)
- Running API server

Usage:
    python eval.py --model /tmp/models/poker_model.pt
    python eval.py --model /tmp/models/poker_model.pt --num-hands 100 --num-players 2
"""

import argparse
import json
import sys
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import from train.py
from train import PokerNet, PokerTransformer, PokerDataset, encode_state, encode_action, ACTION_NAMES

# Import agent classes from generate_rollouts
try:
    from generate_rollouts import RandomAgent, RolloutGenerator
    HAS_ROLLOUT_GENERATOR = True
except ImportError:
    HAS_ROLLOUT_GENERATOR = False
    print("⚠️  Warning: Could not import from generate_rollouts.py")

# Try to import requests
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# =============================================================================
# Model Agent
# =============================================================================

class ModelAgent:
    """Agent that uses a trained neural network to make decisions"""
    
    def __init__(self, model: nn.Module, device: torch.device, player_id: str, name: str):
        """
        Initialize model agent.
        
        Args:
            model: Trained PyTorch model
            device: Device to run inference on
            player_id: Player ID
            name: Player name
        """
        self.model = model
        self.device = device
        self.player_id = player_id
        self.name = name
        self.model.eval()
    
    def select_action(self, state: dict[str, Any], legal_actions: list[str]) -> tuple[str, int]:
        """
        Select an action using the trained model.
        
        Args:
            state: Game state dictionary
            legal_actions: List of legal action strings
        
        Returns:
            tuple: (action_type, amount)
        """
        if not legal_actions:
            return ("fold", 0)
        
        # Encode the state
        state_tensor = encode_state(state).unsqueeze(0).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(state_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            # Get action probabilities for legal actions only
            legal_action_probs = []
            legal_action_indices = []
            
            for action_str in legal_actions:
                if action_str in ACTION_NAMES:
                    action_idx = ACTION_NAMES.index(action_str)
                    legal_action_probs.append(probs[0, action_idx].item())
                    legal_action_indices.append(action_idx)
            
            # If no legal actions match our action space, pick randomly
            if not legal_action_probs:
                action_type = random.choice(legal_actions)
            else:
                # Choose the legal action with highest probability
                best_idx = legal_action_probs.index(max(legal_action_probs))
                action_idx = legal_action_indices[best_idx]
                action_type = ACTION_NAMES[action_idx]
        
        # Calculate amount for bet/raise
        amount = 0
        if action_type == "raise":
            min_raise_total = state.get('min_raise_total', state.get('big_blind', 20))
            max_raise = state['player_chips']
            if max_raise >= min_raise_total:
                # Raise to about 2x minimum
                amount = min(max_raise, min_raise_total * 2)
            else:
                amount = max_raise
        
        elif action_type == "bet":
            min_bet = state.get('min_bet', state.get('big_blind', 20))
            max_bet = state['player_chips']
            pot = state.get('pot', 0)
            if max_bet >= min_bet:
                # Bet about half pot
                amount = max(min_bet, min(max_bet, pot // 2))
            else:
                amount = max_bet
        
        return (action_type, amount)


# =============================================================================
# Evaluation Metrics
# =============================================================================

class EvaluationMetrics:
    """Track and compute evaluation metrics"""
    
    def __init__(self, num_actions: int = 6):
        self.num_actions = num_actions
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total = 0
        self.correct = 0
        self.total_loss = 0.0
        
        # Per-action metrics
        self.action_counts = defaultdict(int)
        self.action_correct = defaultdict(int)
        self.action_predicted = defaultdict(int)
        
        # Confusion matrix
        self.confusion = [[0] * self.num_actions for _ in range(self.num_actions)]
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float):
        """
        Update metrics with batch results.
        
        Args:
            predictions: Predicted action indices
            targets: True action indices
            loss: Loss value for this batch
        """
        self.total += targets.size(0)
        self.total_loss += loss * targets.size(0)
        
        for pred, target in zip(predictions, targets):
            pred_idx = pred.item()
            target_idx = target.item()
            
            # Overall accuracy
            if pred_idx == target_idx:
                self.correct += 1
                self.action_correct[target_idx] += 1
            
            # Per-action counts
            self.action_counts[target_idx] += 1
            self.action_predicted[pred_idx] += 1
            
            # Confusion matrix
            self.confusion[target_idx][pred_idx] += 1
    
    def get_accuracy(self) -> float:
        """Get overall accuracy"""
        if self.total == 0:
            return 0.0
        return 100.0 * self.correct / self.total
    
    def get_average_loss(self) -> float:
        """Get average loss"""
        if self.total == 0:
            return 0.0
        return self.total_loss / self.total
    
    def get_per_action_accuracy(self) -> dict[int, float]:
        """Get accuracy for each action"""
        accuracies = {}
        for action_idx in range(self.num_actions):
            if self.action_counts[action_idx] > 0:
                accuracies[action_idx] = (
                    100.0 * self.action_correct[action_idx] / self.action_counts[action_idx]
                )
            else:
                accuracies[action_idx] = 0.0
        return accuracies
    
    def print_summary(self, action_names: list[str] = None):
        """Print evaluation summary"""
        if action_names is None:
            action_names = [f"Action {i}" for i in range(self.num_actions)]
        
        print("=" * 60)
        print("Evaluation Summary")
        print("=" * 60)
        print()
        
        # Overall metrics
        print(f"Total samples: {self.total}")
        print(f"Overall accuracy: {self.get_accuracy():.2f}%")
        print(f"Average loss: {self.get_average_loss():.4f}")
        print()
        
        # Per-action statistics
        print("Per-Action Performance:")
        print("-" * 60)
        print(f"{'Action':<15} {'Count':<10} {'Predicted':<12} {'Accuracy':<10}")
        print("-" * 60)
        
        per_action_acc = self.get_per_action_accuracy()
        for action_idx in range(self.num_actions):
            action_name = action_names[action_idx] if action_idx < len(action_names) else f"Action {action_idx}"
            count = self.action_counts[action_idx]
            predicted = self.action_predicted[action_idx]
            accuracy = per_action_acc[action_idx]
            print(f"{action_name:<15} {count:<10} {predicted:<12} {accuracy:>6.2f}%")
        print()
        
        # Confusion matrix
        print("Confusion Matrix (rows=true, cols=predicted):")
        print("-" * 60)
        
        # Header
        print(f"{'True \\ Pred':<15}", end="")
        for action_idx in range(self.num_actions):
            action_name = action_names[action_idx][:6] if action_idx < len(action_names) else f"A{action_idx}"
            print(f"{action_name:<8}", end="")
        print()
        
        # Matrix
        for true_idx in range(self.num_actions):
            true_name = action_names[true_idx][:12] if true_idx < len(action_names) else f"Action {true_idx}"
            print(f"{true_name:<15}", end="")
            for pred_idx in range(self.num_actions):
                count = self.confusion[true_idx][pred_idx]
                print(f"{count:<8}", end="")
            print()
        print()


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> EvaluationMetrics:
    """
    Evaluate the model on test data.
    
    Args:
        model: Neural network
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
    
    Returns:
        EvaluationMetrics object with results
    """
    model.eval()
    metrics = EvaluationMetrics(num_actions=6)
    
    with torch.no_grad():
        for batch_idx, (states, actions) in enumerate(test_loader):
            states = states.to(device)
            actions = actions.to(device)
            
            # Forward pass
            outputs = model(states)
            loss = criterion(outputs, actions)
            
            # Predictions
            _, predicted = outputs.max(1)
            
            # Update metrics
            metrics.update(predicted, actions, loss.item())
            
            # Progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(test_loader)}: "
                      f"acc={metrics.get_accuracy():.2f}%")
    
    return metrics


# =============================================================================
# Play vs Random Agent
# =============================================================================

def play_vs_random(
    model: nn.Module,
    device: torch.device,
    num_hands: int = 100,
    num_players: int = 2,
    api_url: str = "http://localhost:8080/simulate",
    small_blind: int = 10,
    big_blind: int = 20,
    starting_chips: int = 1000,
    verbose: bool = False
) -> dict:
    """
    Play hands against random agent(s) to evaluate model performance.
    
    Args:
        model: Trained model
        device: Device for inference
        num_hands: Number of hands to play
        num_players: Total number of players (including model agent)
        api_url: API server URL
        small_blind: Small blind amount
        big_blind: Big blind amount
        starting_chips: Starting chips per hand
        verbose: Print detailed progress
    
    Returns:
        Dictionary with evaluation metrics
    """
    if not HAS_ROLLOUT_GENERATOR:
        print("✗ Error: Cannot import RolloutGenerator from generate_rollouts.py")
        return {}
    
    generator = RolloutGenerator(api_url=api_url)
    
    # Check server
    if not generator.check_server():
        print(f"✗ Error: Cannot connect to API server at {api_url}")
        print("  Make sure the API server is running:")
        print("  cd api && ./build/poker_api 8080")
        return {}
    
    print(f"✓ Connected to API server")
    print(f"  Playing {num_hands} hands with {num_players} players")
    print()
    
    # Track statistics
    model_id = "p0"  # Model agent is always player 0
    stats = {
        'hands_played': 0,
        'hands_won': 0,
        'hands_lost': 0,
        'hands_tied': 0,
        'total_profit': 0,
        'profits': []
    }
    
    start_time = time.time()
    
    for hand_num in range(num_hands):
        # Create agents
        agents = []
        
        # Model agent is player 0
        model_agent = ModelAgent(model, device, model_id, "ModelAgent")
        agents.append(model_agent)
        
        # Other players are random
        for i in range(1, num_players):
            random_agent = RandomAgent(f"p{i}", f"RandomAgent{i}")
            agents.append(random_agent)
        
        # Game configuration
        config = {
            'smallBlind': small_blind,
            'bigBlind': big_blind,
            'startingChips': starting_chips,
            'minPlayers': num_players,
            'maxPlayers': num_players,
            'seed': random.randint(0, 1000000)
        }
        
        # Generate rollout
        rollout = generator.generate_rollout(
            agents=agents,
            config=config,
            max_steps=1000,
            verbose=verbose
        )
        
        # Calculate model's profit
        model_profit = rollout['rewards'].get(model_id, 0)
        stats['total_profit'] += model_profit
        stats['profits'].append(model_profit)
        stats['hands_played'] += 1
        
        if model_profit > 0:
            stats['hands_won'] += 1
        elif model_profit < 0:
            stats['hands_lost'] += 1
        else:
            stats['hands_tied'] += 1
        
        # Progress
        if (hand_num + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (hand_num + 1) / elapsed
            eta = (num_hands - hand_num - 1) / rate if rate > 0 else 0
            print(f"  Progress: {hand_num+1}/{num_hands} hands "
                  f"(win rate: {stats['hands_won']}/{stats['hands_played']}, "
                  f"avg profit: {stats['total_profit']/stats['hands_played']:.1f}, "
                  f"ETA: {eta:.0f}s)")
    
    return stats


# =============================================================================
# Main
# =============================================================================

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained poker model by playing against random agents"
    )
    
    parser.add_argument('--model', type=str, default='/tmp/models/poker_model.pt',
                       help='Path to trained model')
    parser.add_argument('--num-hands', type=int, default=100,
                       help='Number of hands to play (default: 100)')
    parser.add_argument('--num-players', type=int, default=2,
                       help='Number of players including model (default: 2)')
    parser.add_argument('--api-url', type=str, default='http://localhost:8080/simulate',
                       help='API server URL (default: http://localhost:8080/simulate)')
    parser.add_argument('--small-blind', type=int, default=10,
                       help='Small blind amount (default: 10)')
    parser.add_argument('--big-blind', type=int, default=20,
                       help='Big blind amount (default: 20)')
    parser.add_argument('--starting-chips', type=int, default=1000,
                       help='Starting chips per hand (default: 1000)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Poker Model Evaluation - Play vs Random Agent")
    print("=" * 70)
    print()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model_path = Path(args.model) if Path(args.model).is_absolute() else Path(__file__).parent / args.model
    
    if not model_path.exists():
        print(f"✗ Error: Model file not found: {model_path}")
        print(f"  Please train a model first:")
        print(f"  python train.py --data data/quickstart_rollouts.json")
        return 1
    
    checkpoint = torch.load(model_path, map_location='cpu')
    feature_dim = checkpoint['feature_dim']
    hidden_dim = checkpoint['hidden_dim']
    model_type = checkpoint.get('model_type', 'legacy')
    
    print(f"✓ Model metadata:")
    print(f"  - Model type: {model_type}")
    print(f"  - Feature dim: {feature_dim}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Trained epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"  - Validation loss: {checkpoint['val_loss']:.4f}")
    
    # Print transformer-specific parameters
    if model_type == 'transformer':
        print(f"  - Transformer layers: {checkpoint.get('num_layers', 4)}")
        print(f"  - Attention heads: {checkpoint.get('num_heads', 8)}")
        print(f"  - Dropout: {checkpoint.get('dropout', 0.1)}")
    print()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    print()
    
    # Create and load model
    print("Loading model weights...")
    if model_type == 'transformer':
        model = PokerTransformer(
            input_dim=feature_dim, 
            hidden_dim=hidden_dim,
            num_heads=checkpoint.get('num_heads', 8),
            num_layers=checkpoint.get('num_layers', 4),
            dropout=checkpoint.get('dropout', 0.1)
        )
    else:
        model = PokerNet(input_dim=feature_dim, hidden_dim=hidden_dim)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully")
    print()
    
    # Play against random agents
    print("=" * 70)
    print("Playing against Random Agents")
    print("=" * 70)
    print()
    
    stats = play_vs_random(
        model=model,
        device=device,
        num_hands=args.num_hands,
        num_players=args.num_players,
        api_url=args.api_url,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        starting_chips=args.starting_chips,
        verbose=args.verbose
    )
    
    if not stats:
        print("✗ Evaluation failed")
        return 1
    
    # Print results
    print()
    print("=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print()
    
    hands_played = stats['hands_played']
    hands_won = stats['hands_won']
    hands_lost = stats['hands_lost']
    hands_tied = stats['hands_tied']
    total_profit = stats['total_profit']
    avg_profit = total_profit / hands_played if hands_played > 0 else 0
    
    print(f"Hands played:     {hands_played}")
    print(f"Hands won:        {hands_won} ({100*hands_won/hands_played:.1f}%)")
    print(f"Hands lost:       {hands_lost} ({100*hands_lost/hands_played:.1f}%)")
    print(f"Hands tied:       {hands_tied} ({100*hands_tied/hands_played:.1f}%)")
    print()
    print(f"Total profit:     {total_profit:+.0f} chips")
    print(f"Average profit:   {avg_profit:+.2f} chips/hand")
    print()
    
    # Calculate expected random baseline
    # Against (n-1) random opponents, expected profit is 0 (fair game)
    # But there's variance, so we estimate the random baseline
    random_baseline = 0.0
    improvement = avg_profit - random_baseline
    
    print("=" * 70)
    print("Performance vs Random Baseline")
    print("=" * 70)
    print()
    print(f"Random baseline:  {random_baseline:+.2f} chips/hand (expected)")
    print(f"Model performance: {avg_profit:+.2f} chips/hand")
    print(f"Improvement:      {improvement:+.2f} chips/hand")
    print()
    
    if avg_profit > 10:
        print("✓ EXCELLENT: Model is significantly outperforming random play!")
    elif avg_profit > 5:
        print("✓ GOOD: Model is outperforming random play")
    elif avg_profit > 0:
        print("⚠ FAIR: Model is slightly better than random")
    elif avg_profit > -5:
        print("⚠ POOR: Model is performing close to random baseline")
    else:
        print("✗ NEEDS IMPROVEMENT: Model is underperforming random play")
    
    print()
    print("Tips for improvement:")
    if avg_profit < 5:
        print("- Train for more epochs")
        print("- Generate more diverse training data")
        print("- Try larger hidden dimensions")
        print("- Add more features to state encoding")
    else:
        print("- Model is performing well!")
        print("- Consider training against stronger opponents")
        print("- Try fine-tuning with more data")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

