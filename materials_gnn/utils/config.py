"""Lightweight dataclass configs.

The package can later switch to Hydra, Pydantic, or OmegaConf, but dataclasses keep the
first prototype explicit and dependency-light.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json


@dataclass
class GraphConfig:
    cutoff: float = 5.0
    num_rbf: int = 64
    num_angle_rbf: int = 32
    include_line_graph: bool = False


@dataclass
class ModelConfig:
    name: str = "alignn_like"
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.0
    pooling: str = "mean"


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42
    device: str = "cpu"


def save_config(config: Any, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
    return output
