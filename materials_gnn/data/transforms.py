"""Data transforms used by datasets and training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


@dataclass
class TargetNormalizer:
    """Standardize scalar targets with train-set mean and standard deviation."""

    mean: float
    std: float
    eps: float = 1e-12

    @classmethod
    def from_tensor(cls, targets: Tensor | Iterable[float]) -> "TargetNormalizer":
        y = torch.as_tensor(targets, dtype=torch.float32)
        if y.numel() == 0:
            raise ValueError("Cannot fit TargetNormalizer on an empty target array")
        std = float(y.std(unbiased=False).item())
        return cls(mean=float(y.mean().item()), std=max(std, cls(mean=0.0, std=1.0).eps))

    def transform(self, values: Tensor | Iterable[float] | float) -> Tensor:
        y = torch.as_tensor(values, dtype=torch.float32)
        return (y - self.mean) / max(self.std, self.eps)

    def inverse_transform(self, values: Tensor | Iterable[float] | float) -> Tensor:
        y = torch.as_tensor(values, dtype=torch.float32)
        return y * max(self.std, self.eps) + self.mean

    def state_dict(self) -> dict[str, float]:
        return {"mean": float(self.mean), "std": float(self.std), "eps": float(self.eps)}

    @classmethod
    def from_state_dict(cls, state: dict[str, float]) -> "TargetNormalizer":
        return cls(mean=float(state["mean"]), std=float(state["std"]), eps=float(state.get("eps", 1e-12)))
