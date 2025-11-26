
import torch
import sys

path = "/tmp/pokersim/rl_models_v16/poker_rl_iter_1.pt"
try:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    print(f"Keys in checkpoint: {list(checkpoint.keys())}")
    if 'hyperparameters' in checkpoint:
        print(f"Hyperparameters: {checkpoint['hyperparameters']}")
    
except Exception as e:
    print(f"Error loading checkpoint: {e}")
