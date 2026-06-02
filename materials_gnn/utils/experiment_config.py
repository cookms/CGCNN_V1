"""Experiment configuration serialization helpers.

The example training scripts intentionally remain lightweight instead of depending on a
full experiment tracker. These helpers provide a small JSON-safe config artifact that can
be saved beside every run and embedded in checkpoints for reproducible inference.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def make_json_safe(value: Any) -> Any:
    """Convert common scientific Python objects into JSON-serializable values."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [make_json_safe(item) for item in value]

    # NumPy scalars/arrays expose item()/tolist() without requiring a hard dependency here.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # pragma: no cover - defensive fallback for unusual objects
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:  # pragma: no cover - defensive fallback for unusual objects
            pass
    return str(value)


def write_experiment_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Write an experiment config as deterministic, human-readable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(make_json_safe(dict(config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
