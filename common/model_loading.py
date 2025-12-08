"""
Shared checkpoint/model loading utilities.

This repo historically had multiple slightly-different implementations of:
- "read checkpoint"
- "infer model config"
- "instantiate actor-critic"
- "load state_dict"

That divergence can lead to subtle issues where training, Elo evaluation, and
API usage load the *same* checkpoint into *different* architectures.

This module is the single canonical implementation used by:
- `common.model_cache.ModelCache`
- `common.model_agent.ModelAgent`
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


def load_checkpoint(path: str, *, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    # weights_only=False is required for checkpoints that include numpy scalars.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint at {path} is not a dict (got {type(ckpt).__name__})")
    return ckpt


def infer_model_config(ckpt: Dict[str, Any], *, override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Infer the model architecture config needed to instantiate the network.

    Priority:
    1) Explicit override dict
    2) ckpt['model_config'] (preferred, produced by PPOTrainer.save_checkpoint)
    3) Legacy flat keys (input_dim, hidden_dim, ...)
    4) Best-effort inference from state_dict shapes
    """
    override = dict(override or {})
    model_cfg = dict(ckpt.get("model_config", {}) or {})
    state_dict = ckpt.get("model_state_dict", {}) or {}

    def _get(key: str, default: Any) -> Any:
        if key in override and override[key] is not None:
            return override[key]
        if key in model_cfg and model_cfg[key] is not None:
            return model_cfg[key]
        if key in ckpt and ckpt[key] is not None:
            return ckpt[key]
        return default

    input_dim = int(_get("input_dim", 167))
    dropout = float(_get("dropout", 0.1))

    hidden_dim = _get("hidden_dim", None)
    num_heads = _get("num_heads", None)
    num_layers = _get("num_layers", None)

    # Infer hidden_dim from positional encoding if available.
    if hidden_dim is None and isinstance(state_dict, dict) and "pos_encoding" in state_dict:
        try:
            hidden_dim = int(state_dict["pos_encoding"].shape[2])
        except Exception:
            hidden_dim = None

    # Infer layer count from transformer weight keys if missing.
    if num_layers is None and isinstance(state_dict, dict):
        max_layer = -1
        for k in state_dict.keys():
            if k.startswith("transformer.layers."):
                parts = k.split(".")
                if len(parts) >= 3:
                    try:
                        max_layer = max(max_layer, int(parts[2]))
                    except Exception:
                        pass
        if max_layer >= 0:
            num_layers = max_layer + 1

    hidden_dim = int(hidden_dim) if hidden_dim is not None else 256
    num_layers = int(num_layers) if num_layers is not None else 4

    # Heuristic: default to 8 heads, but if hidden_dim isn't divisible by 8 and is divisible by 4, use 4.
    if num_heads is None:
        num_heads = 8 if hidden_dim % 8 == 0 else (4 if hidden_dim % 4 == 0 else 8)
    num_heads = int(num_heads)

    return {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
    }


def build_model_from_checkpoint(
    ckpt: Dict[str, Any],
    *,
    device: torch.device,
    override_config: Optional[Dict[str, Any]] = None,
    strict: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Instantiate the actor-critic model and load weights from a checkpoint.

    Returns (model, resolved_model_config).
    """
    from .rl_model import create_actor_critic

    cfg = infer_model_config(ckpt, override=override_config)
    model = create_actor_critic(
        input_dim=int(cfg["input_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_heads=int(cfg["num_heads"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
        gradient_checkpointing=False,
    )

    state_dict = ckpt.get("model_state_dict", None)
    if not isinstance(state_dict, dict):
        raise KeyError("Checkpoint missing 'model_state_dict' dict")
    model.load_state_dict(state_dict, strict=strict)
    model.to(device).eval()
    return model, cfg


def load_model_from_path(
    path: str,
    *,
    device: torch.device,
    override_config: Optional[Dict[str, Any]] = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    Convenience wrapper: load checkpoint from disk and return (model, model_cfg, raw_ckpt).
    """
    ckpt = load_checkpoint(path, map_location=map_location)
    model, cfg = build_model_from_checkpoint(ckpt, device=device, override_config=override_config, strict=strict)
    return model, cfg, ckpt


