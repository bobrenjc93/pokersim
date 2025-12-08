"""
Model caching utilities for efficient checkpoint loading.

Provides LRU caching for model checkpoints, shared between training and ELO systems.
"""

import gc
import os
import traceback
from collections import OrderedDict
from typing import Any, Dict, Optional, TYPE_CHECKING, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .rl_model import PokerActorCritic


class ModelCache:
    """
    LRU cache for model checkpoints.
    
    Handles efficient loading and caching of model checkpoints to avoid
    repeated disk I/O and model initialization overhead.
    """
    
    def __init__(self, device: torch.device = None, max_size: int = 10):
        self.device = device or torch.device('cpu')
        self.max_size = max_size
        self._cache: OrderedDict[str, nn.Module] = OrderedDict()
        # Track on-disk file identity so we don't accidentally reuse stale weights
        # when a path is overwritten (e.g. "latest.pt" during training).
        self._cache_fingerprint: Dict[str, Tuple[int, int]] = {}
        self._last_error: Dict[str, str] = {}
    
    def _stat_fingerprint(self, path: str) -> Optional[Tuple[int, int]]:
        """Return (mtime_ns, size) for a file path, or None if missing/unstatable."""
        try:
            st = os.stat(path)
            return int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))), int(st.st_size)
        except Exception:
            return None

    def get(self, path: str, config: Optional[Dict[str, Any]] = None, *, strict: bool = False) -> Optional[nn.Module]:
        """
        Load and cache a model checkpoint.
        
        Args:
            path: Path to checkpoint file
            config: Optional model config (input_dim, hidden_dim, etc.)
                   If not provided, will try to infer from checkpoint.
        
        Returns:
            Loaded model or None if loading failed
        """
        fp = self._stat_fingerprint(path)
        if path in self._cache:
            cached_fp = self._cache_fingerprint.get(path)
            # Cache hit only if the on-disk file looks unchanged.
            if fp is not None and cached_fp == fp:
                self._cache.move_to_end(path)
                return self._cache[path]
            # Otherwise, treat as a cache miss and reload.
            try:
                self._cache.pop(path, None)
                self._cache_fingerprint.pop(path, None)
            except Exception:
                pass
        
        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        try:
            model = self._load_model(path, config)
            if model:
                self._cache[path] = model
                if fp is not None:
                    self._cache_fingerprint[path] = fp
            return model
        except Exception as e:
            tb = traceback.format_exc()
            self._last_error[path] = f"{type(e).__name__}: {e}\n{tb}"
            if strict:
                raise
            return None

    def get_with_error(self, path: str, config: Optional[Dict[str, Any]] = None) -> Tuple[Optional[nn.Module], Optional[str]]:
        """
        Load a checkpoint and return (model, error_string).

        This never raises; callers can surface the error upstream.
        """
        model = self.get(path, config=config, strict=False)
        return model, self._last_error.get(path)

    def last_error(self, path: str) -> Optional[str]:
        """Return the most recent load error for a path, if any."""
        return self._last_error.get(path)
    
    def _load_model(self, path: str, config: Optional[Dict[str, Any]]) -> Optional[nn.Module]:
        """Load model from checkpoint."""
        from .model_loading import load_model_from_path

        model, _cfg, _raw = load_model_from_path(
            path,
            device=self.device,
            override_config=config,
            strict=True,
            map_location="cpu",
        )
        return model
    
    def clear(self):
        """Clear all cached models."""
        self._cache.clear()
        self._cache_fingerprint.clear()
        self._last_error.clear()
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, path: str) -> bool:
        return path in self._cache


