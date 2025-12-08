"""
Checkpoint utilities for managing model checkpoints.

Provides functions for:
- Parsing checkpoint files from a directory
- Selecting checkpoints with temporal spread (for diverse opponent pools)
"""

import math
import re
from pathlib import Path
from typing import List, Tuple


def parse_checkpoints(directory: Path) -> List[Tuple[int, Path]]:
    """Find and sort checkpoints by iteration number.
    
    Args:
        directory: Directory to scan for checkpoint files
        
    Returns:
        List of (iteration_number, path) tuples sorted by iteration.
        Baseline model uses iteration -1 to distinguish from actual iterations.
    """
    checkpoints = []
    pattern = re.compile(r"(?:poker_rl_)?iter_(\d+)\.pt")
    
    if not directory.exists():
        return []
        
    for f in directory.glob("*.pt"):
        if f.name == "poker_rl_baseline.pt":
            # Use -1 for baseline to distinguish from actual iteration 0
            checkpoints.append((-1, f))
            continue
        
        # Skip non-iteration checkpoints like "latest.pt", "final.pt"
        if f.name in ("latest.pt", "final.pt", "checkpoint.pt"):
            continue
            
        match = pattern.match(f.name)
        if match:
            iteration = int(match.group(1))
            checkpoints.append((iteration, f))
            
    return sorted(checkpoints, key=lambda x: x[0])


def select_spread_checkpoints(
    checkpoints: List[Tuple[int, Path]], 
    max_checkpoints: int
) -> List[Tuple[int, Path]]:
    """
    Select checkpoints with maximum coverage across all iterations.
    
    Uses farthest-first (maximin distance) greedy selection with logarithmic
    scaling of iteration numbers. This gives more weight to early iterations
    where learning changes are typically most dramatic, while still maintaining
    good coverage of later training.
    
    Strategy:
    - Always keep first checkpoint (baseline/-1 or earliest iteration)
    - Always keep last checkpoint (most recent)
    - Use log-scaled iteration numbers for distance calculation
    - Greedily select remaining checkpoints to maximize minimum distance
      to already-selected checkpoints (farthest-first traversal)
    
    Args:
        checkpoints: List of (iteration_number, path) tuples, sorted by iteration
        max_checkpoints: Maximum number of checkpoints to select
        
    Returns:
        List of (iteration_number, path) tuples with maximized spread
    """
    n = len(checkpoints)
    if n <= max_checkpoints:
        return checkpoints
    
    # Extract iteration numbers and apply log scaling for distance calculation
    # Add offset to handle baseline (-1) and iter_0 cases
    iterations = [cp[0] for cp in checkpoints]
    min_iter = min(iterations)
    offset = 2 - min_iter  # Ensure all values are >= 2 for log scaling
    
    def log_scale(iter_num: int) -> float:
        """Apply log scaling to iteration number."""
        return math.log(iter_num + offset)
    
    # Pre-compute log-scaled values for all checkpoints
    log_iterations = [log_scale(it) for it in iterations]
    
    # Start with first and last checkpoints (always included)
    selected_indices = [0, n - 1]
    selected_log_values = {log_iterations[0], log_iterations[n - 1]}
    
    # Greedily select remaining checkpoints using farthest-first on log scale
    remaining_slots = max_checkpoints - 2
    available_indices = set(range(1, n - 1))
    
    for _ in range(remaining_slots):
        if not available_indices:
            break
        
        best_idx = None
        best_min_dist = -1.0
        
        # Find the checkpoint that maximizes minimum log-distance to selected set
        for idx in available_indices:
            log_val = log_iterations[idx]
            # Calculate minimum distance to any already-selected checkpoint (in log space)
            min_dist = min(abs(log_val - sel_log) for sel_log in selected_log_values)
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            selected_log_values.add(log_iterations[best_idx])
            available_indices.remove(best_idx)
    
    # Sort and return selected checkpoints
    sorted_indices = sorted(selected_indices)
    return [checkpoints[i] for i in sorted_indices]


def get_spread_checkpoint_paths(
    directory: Path,
    max_checkpoints: int = 30
) -> List[Path]:
    """
    Convenience function to get a spread of checkpoint paths from a directory.
    
    Args:
        directory: Directory containing checkpoint files
        max_checkpoints: Maximum number of checkpoints to select
        
    Returns:
        List of Path objects with good temporal spread
    """
    checkpoints = parse_checkpoints(directory)
    if not checkpoints:
        return []
    
    selected = select_spread_checkpoints(checkpoints, max_checkpoints)
    return [cp[1] for cp in selected]




