"""Loss factories."""

from __future__ import annotations

from torch import nn


def get_loss(name: str = "mse") -> nn.Module:
    """Return a PyTorch loss module by name."""

    normalized = name.lower()
    if normalized in {"mse", "l2"}:
        return nn.MSELoss()
    if normalized in {"mae", "l1"}:
        return nn.L1Loss()
    if normalized in {"huber", "smooth_l1"}:
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss: {name!r}")
