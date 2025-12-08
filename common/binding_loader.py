"""
Robust loader for the native `poker_api_binding` extension.

Why this exists
---------------
Many entrypoints in this repo are installed as console scripts (e.g. `train-poker`).
When Python runs a console script, `sys.path[0]` points at the venv's `bin/` dir,
not the current working directory. That means a locally-built extension like:

  training/poker_api_binding.cpython-312-darwin.so

is *not* importable unless it's installed into site-packages.

This helper makes imports robust by:
- Trying a normal `import poker_api_binding`
- If it fails, searching common locations relative to the current working dir
  (and its parents) and adding the directory containing the `.so` to `sys.path`
  before retrying.
"""

from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Optional


def _candidate_dirs() -> list[Path]:
    """Return directories to search for `poker_api_binding*.so`."""
    dirs: list[Path] = []

    # Explicit override for power users / CI.
    override = os.environ.get("POKER_API_BINDING_DIR")
    if override:
        dirs.append(Path(override))

    # Current working directory and parents (covers `cd training && uv run train-poker`).
    cwd = Path.cwd().resolve()
    dirs.append(cwd)
    dirs.extend(list(cwd.parents)[:6])

    # Common project subdirs (relative to each candidate root).
    roots: list[Path] = []
    for d in list(dirs):
        # Treat any dir that contains a known project marker as a "root"
        # (this avoids scanning random parents like /Users).
        if (d / "pyproject.toml").exists() or (d / "README.md").exists():
            roots.append(d)

    for r in roots:
        for sub in ("training", "elo", "playground", "api", "api/build", "hand-viewer"):
            dirs.append(r / sub)

    # De-dupe, keep existing dirs only.
    seen: set[Path] = set()
    out: list[Path] = []
    for d in dirs:
        d = d.resolve()
        if d in seen or not d.exists() or not d.is_dir():
            continue
        seen.add(d)
        out.append(d)
    return out


def load_poker_api_binding() -> Optional[object]:
    """
    Attempt to import and return the `poker_api_binding` module.

    Returns:
        The imported module, or None if it can't be loaded.
    """
    try:
        return import_module("poker_api_binding")
    except Exception:
        pass

    # Search for a local `.so` and add its directory to sys.path.
    for d in _candidate_dirs():
        try:
            hit = next(d.glob("poker_api_binding*.so"), None)
        except Exception:
            hit = None
        if not hit:
            continue

        sys.path.insert(0, str(d))
        try:
            return import_module("poker_api_binding")
        except Exception:
            # Keep searching; some hits might be wrong-arch / wrong-python, etc.
            try:
                sys.path.remove(str(d))
            except Exception:
                pass

    return None


