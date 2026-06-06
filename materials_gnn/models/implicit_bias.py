"""Implicit-bias readout components.

This module implements a small, unrolled fixed-point implicit-bias activation for the
post-pooling readout head. It intentionally does not change graph construction or message
passing.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _activation_module(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported implicit-bias activation: {name!r}")


def _build_coupling(dim: int, coupling: str) -> Tensor:
    if dim <= 0:
        raise ValueError("dim must be positive")

    weights = torch.zeros((dim, dim), dtype=torch.float32)
    if coupling == "ring":
        if dim == 1:
            return weights
        for idx in range(dim):
            weights[idx, (idx - 1) % dim] = 1.0
            weights[idx, (idx + 1) % dim] = 1.0
    elif coupling == "dense":
        if dim == 1:
            return weights
        weights.fill_(1.0)
        weights.fill_diagonal_(0.0)
    else:
        raise ValueError(f"Unsupported implicit-bias coupling: {coupling!r}")

    row_sums = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return weights / row_sums


class ImplicitBiasActivation(nn.Module):
    """Apply an unrolled implicit-bias correction followed by a pointwise activation."""

    def __init__(
        self,
        dim: int,
        ib_lambda: float = 0.01,
        sigma_slope: float = 1.0,
        fixed_point_iters: int = 8,
        coupling: str = "ring",
        activation: str = "silu",
        trainable_lambda: bool = False,
    ) -> None:
        super().__init__()
        if fixed_point_iters < 0:
            raise ValueError("fixed_point_iters must be non-negative")

        self.dim = dim
        self.sigma_slope = float(sigma_slope)
        self.fixed_point_iters = fixed_point_iters
        self.activation = _activation_module(activation)
        self.register_buffer("coupling", _build_coupling(dim, coupling))

        lambda_tensor = torch.tensor(float(ib_lambda), dtype=torch.float32)
        if trainable_lambda:
            self.ib_lambda = nn.Parameter(lambda_tensor)
        else:
            self.register_buffer("ib_lambda", lambda_tensor)

    def forward(self, y: Tensor) -> Tensor:
        if y.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {y.shape[-1]}")

        ib_lambda = self.ib_lambda.to(dtype=y.dtype, device=y.device)
        if self.fixed_point_iters == 0 or ib_lambda.detach().item() == 0.0:
            return self.activation(y)

        original_shape = y.shape
        y_flat = y.reshape(-1, self.dim)
        z = y_flat
        coupling = self.coupling.to(dtype=y.dtype, device=y.device)

        for _ in range(self.fixed_point_iters):
            diff = z.unsqueeze(1) - z.unsqueeze(2)
            bias = (coupling.unsqueeze(0) * torch.sigmoid(self.sigma_slope * diff)).sum(dim=-1)
            z = y_flat - ib_lambda * bias

        return self.activation(z.reshape(original_shape))


class ImplicitBiasMLPReadout(nn.Module):
    """One-hidden-layer readout using an implicit-bias hidden activation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        ib_lambda: float = 0.01,
        sigma_slope: float = 1.0,
        fixed_point_iters: int = 8,
        coupling: str = "ring",
        activation: str = "silu",
        trainable_lambda: bool = False,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or input_dim
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden),
            ImplicitBiasActivation(
                hidden,
                ib_lambda=ib_lambda,
                sigma_slope=sigma_slope,
                fixed_point_iters=fixed_point_iters,
                coupling=coupling,
                activation=activation,
                trainable_lambda=trainable_lambda,
            ),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, crystal_embedding: Tensor) -> Tensor:
        return self.net(crystal_embedding)
