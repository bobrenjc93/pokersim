#!/usr/bin/env python3
"""
Verify that checkpoints in a directory are loadable and represent distinct weights.

This is meant as a fast sanity check when you see suspicious evaluation behavior
(e.g. iter_1 always looks best): it will confirm that later checkpoints load and
that parameter tensors differ across iterations.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch

from common import ModelCache, parse_checkpoints, detect_device


def _fingerprint(model: torch.nn.Module) -> Tuple[int, float, float]:
    """
    Return a small fingerprint: (num_params, l2_norm, mean_abs)
    """
    params = [p.detach().float().cpu().view(-1) for p in model.parameters()]
    if not params:
        return 0, 0.0, 0.0
    v = torch.cat(params)
    return int(v.numel()), float(torch.linalg.vector_norm(v).item()), float(v.abs().mean().item())

def _iter_from_name(p: Path) -> int:
    """
    Best-effort parse iteration from filename.
    Returns -999 if unknown/non-iter checkpoint.
    """
    name = p.name
    if "iter_" not in name:
        return -999
    try:
        return int(name.split("iter_")[1].split(".")[0])
    except Exception:
        return -999


def main() -> int:
    p = argparse.ArgumentParser(description="Verify poker RL checkpoints")
    p.add_argument("--models-dir", type=str, required=True)
    p.add_argument("--max", type=int, default=10, help="Max checkpoints to inspect (spread-selected)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--check-epoch", action="store_true", help="Verify ckpt['epoch'] matches iter_XXX in filename")
    args = p.parse_args()

    models_dir = Path(args.models_dir)
    cps = parse_checkpoints(models_dir)
    if not cps:
        print(f"No checkpoints found in {models_dir}")
        return 2

    # spread selection is handled by parse_checkpoints caller elsewhere; keep it simple here:
    # just take the first, last, and some middle ones.
    if len(cps) > args.max:
        # naive spread: uniform indices
        idxs = sorted(set([0, len(cps) - 1] + [int(i * (len(cps) - 1) / (args.max - 1)) for i in range(args.max)]))
        cps = [cps[i] for i in idxs]

    device = detect_device() if args.device == "auto" else torch.device(args.device)
    cache = ModelCache(device=device, max_size=max(2, len(cps)))

    print(f"Device: {device}")
    print(f"Found {len(cps)} checkpoints to verify")

    fingerprints: List[Tuple[int, str, Tuple[int, float, float]]] = []
    for it, path in cps:
        if args.check_epoch:
            try:
                raw = torch.load(str(path), map_location="cpu", weights_only=False)
                epoch = int(raw.get("epoch", -1))
                name_it = _iter_from_name(path)
                # Only enforce on iter checkpoints.
                if name_it >= 0 and epoch not in (-1, name_it):
                    print(f"WARNING: epoch mismatch for {path.name}: ckpt['epoch']={epoch} but filename iter={name_it}")
            except Exception as e:
                print(f"WARNING: failed to read epoch from {path.name}: {e}")

        model, err = cache.get_with_error(str(path))
        if model is None:
            print(f"FAIL load iter={it}: {path}\n{err or ''}".strip())
            return 1
        fp = _fingerprint(model)
        fingerprints.append((it, str(path), fp))
        print(f"OK   iter={it:<6} fp=(n={fp[0]}, l2={fp[1]:.4e}, mean_abs={fp[2]:.4e})  {path.name}")

    # Check for identical fingerprints (weak but useful smoke test)
    seen = {}
    dup = []
    for it, path, fp in fingerprints:
        if fp in seen:
            dup.append((seen[fp], (it, path)))
        else:
            seen[fp] = (it, path)

    if dup:
        print("\nWARNING: Some checkpoints have identical fingerprints (possible accidental overwrite):")
        for (it_a, p_a), (it_b, p_b) in dup:
            print(f"  - iter {it_a} ({Path(p_a).name}) == iter {it_b} ({Path(p_b).name})")
        return 3

    print("\n✓ Checkpoints loaded and appear distinct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


