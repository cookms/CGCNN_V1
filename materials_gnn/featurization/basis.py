"""Configurable basis expansions for scalar geometric features.

Distances and angles are continuous scalars. Crystal GNNs usually expand these scalars
into a richer vector before message passing: a short bond near 2 A, a medium contact near
3 A, and a nearly linear bond angle should be easy for a small neural network to
separate. This module keeps those expansions configurable so the same graph builder can
compare Gaussian RBFs, Bessel/sine bases, Fourier features, and trainable basis functions.

The non-learnable functions are useful during preprocessing. The ``nn.Module`` classes are
useful when a basis should be trained end-to-end from raw ``distance`` or ``angle`` fields.
Future equivariant layers can reuse the same scalar bases while additionally consuming
``edge_vec`` or normalized directions.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Iterable, Literal

import torch
from torch import Tensor, nn

BasisKind = Literal["gaussian", "bessel", "fourier", "learnable_gaussian"]


def _as_tensor(values: Tensor | Iterable[float] | float, *, dtype: torch.dtype | None = None) -> Tensor:
    if isinstance(values, Tensor):
        return values if dtype is None else values.to(dtype=dtype)
    return torch.as_tensor(values, dtype=dtype or torch.get_default_dtype())


def _scaled_values(values: Tensor, start: float, stop: float) -> Tensor:
    if stop <= start:
        raise ValueError("stop/cutoff must be greater than start")
    return (values - start) / (stop - start)


def gaussian_rbf(values: Tensor, centers: Tensor, gamma: float | Tensor) -> Tensor:
    """Expand scalar values with Gaussian radial basis functions.

    Args:
        values: Tensor of shape ``[...]``.
        centers: Tensor of shape ``[num_basis]`` containing basis centers.
        gamma: Width parameter. Larger values create narrower Gaussians.

    Returns:
        Tensor of shape ``[..., num_basis]``.
    """

    values = values.to(device=centers.device, dtype=centers.dtype)
    gamma_t = torch.as_tensor(gamma, device=centers.device, dtype=centers.dtype)
    shape = [1] * values.ndim + [-1]
    return torch.exp(-gamma_t * (values.unsqueeze(-1) - centers.view(*shape)).pow(2))


def bessel_basis(
    values: Tensor | Iterable[float],
    *,
    num_basis: int,
    start: float = 0.0,
    cutoff: float = 5.0,
    envelope_exponent: int = 2,
    eps: float = 1e-7,
) -> Tensor:
    """Sine/Bessel-style basis for distances or other bounded scalars.

    For distances this resembles the radial bases used in several geometric GNNs: sine
    waves over a cutoff domain, multiplied by a smooth polynomial envelope so features go
    toward zero near the cutoff. The implementation is intentionally simple and stable,
    not a specialized reproduction of any one equivariant architecture.
    """

    if num_basis <= 0:
        raise ValueError("num_basis must be positive")
    v = _as_tensor(values)
    scaled = _scaled_values(v, start, cutoff)
    shifted = (v - start).clamp_min(eps)
    frequencies = torch.arange(1, num_basis + 1, device=v.device, dtype=v.dtype) * math.pi
    raw = torch.sin(frequencies * scaled.unsqueeze(-1)) / shifted.unsqueeze(-1)

    # Keep the basis local to the configured interval while preserving gradients inside it.
    inside = ((scaled >= 0.0) & (scaled <= 1.0)).to(dtype=v.dtype).unsqueeze(-1)
    envelope = (1.0 - scaled.clamp(0.0, 1.0)).pow(envelope_exponent).unsqueeze(-1)
    return raw * envelope * inside


def fourier_basis(
    values: Tensor | Iterable[float],
    *,
    num_basis: int,
    start: float = 0.0,
    cutoff: float = 5.0,
    include_constant: bool = True,
) -> Tensor:
    """Fourier features over a bounded scalar interval.

    Fourier features are useful for testing periodic or oscillatory scalar encodings. For
    angles they can naturally represent angular periodicity; for distances they are a
    configurable alternative to localized RBFs. The returned tensor always has exactly
    ``num_basis`` columns.
    """

    if num_basis <= 0:
        raise ValueError("num_basis must be positive")
    v = _as_tensor(values)
    scaled = _scaled_values(v, start, cutoff)
    features: list[Tensor] = []
    if include_constant:
        features.append(torch.ones_like(scaled))
    frequency = 1
    while len(features) < num_basis:
        phase = 2.0 * math.pi * frequency * scaled
        features.append(torch.sin(phase))
        if len(features) < num_basis:
            features.append(torch.cos(phase))
        frequency += 1
    return torch.stack(features[:num_basis], dim=-1)


def scalar_basis_expansion(
    values: Tensor | Iterable[float],
    *,
    num_basis: int,
    start: float,
    cutoff: float,
    basis_type: str = "gaussian",
    gamma: float | None = None,
    **kwargs: object,
) -> Tensor:
    """Expand scalar values with a named non-learnable basis.

    ``learnable_gaussian`` is intentionally not supported here because preprocessing has no
    trainable parameters. Use ``make_basis_expansion('learnable_gaussian', ...)`` inside a
    model when the centers and widths should be optimized.
    """

    kind = basis_type.lower()
    if kind == "gaussian":
        v = _as_tensor(values)
        centers = torch.linspace(start, cutoff, num_basis, device=v.device, dtype=v.dtype)
        spacing = (cutoff - start) / max(num_basis - 1, 1)
        gamma_value = gamma if gamma is not None else 1.0 / (spacing**2 + 1e-12)
        return gaussian_rbf(v, centers, gamma_value)
    if kind == "bessel":
        return bessel_basis(values, num_basis=num_basis, start=start, cutoff=cutoff, **kwargs)
    if kind == "fourier":
        return fourier_basis(values, num_basis=num_basis, start=start, cutoff=cutoff, **kwargs)
    if kind == "learnable_gaussian":
        raise ValueError("learnable_gaussian is a model layer, not a preprocessing basis")
    raise ValueError(f"Unknown basis_type {basis_type!r}")


def radial_basis_expansion(
    distances: Tensor | Iterable[float],
    *,
    num_basis: int = 64,
    start: float = 0.0,
    cutoff: float = 5.0,
    gamma: float | None = None,
    basis_type: str = "gaussian",
    **kwargs: object,
) -> Tensor:
    """Encode bond distances with a configurable scalar basis.

    Distances in a crystal graph are bond lengths between atoms in the periodic neighbor
    graph. Expanding distances lets a message-passing layer distinguish short covalent
    bonds from longer coordination-shell contacts without hard bins.
    """

    return scalar_basis_expansion(
        distances,
        num_basis=num_basis,
        start=start,
        cutoff=cutoff,
        basis_type=basis_type,
        gamma=gamma,
        **kwargs,
    )


def angle_basis_expansion(
    angles: Tensor | Iterable[float],
    *,
    num_basis: int = 32,
    use_cosine: bool = False,
    gamma: float | None = None,
    basis_type: str = "gaussian",
    **kwargs: object,
) -> Tensor:
    """Encode bond angles or angle cosines with a configurable scalar basis.

    Args:
        angles: Either angles in radians in ``[0, pi]`` or cosines in ``[-1, 1]``.
        num_basis: Number of basis functions.
        use_cosine: If true, interpret input as cosine values and place centers on
            ``[-1, 1]``. If false, interpret input as radians and place centers on
            ``[0, pi]``.
        gamma: Optional Gaussian width. If omitted, a spacing-based default is used.
        basis_type: ``gaussian``, ``bessel``, or ``fourier`` for preprocessing.
    """

    lo, hi = (-1.0, 1.0) if use_cosine else (0.0, math.pi)
    return scalar_basis_expansion(
        angles,
        num_basis=num_basis,
        start=lo,
        cutoff=hi,
        basis_type=basis_type,
        gamma=gamma,
        **kwargs,
    )


class ScalarBasisExpansion(ABC, nn.Module):
    """Base module for trainable or fixed scalar basis expansions."""

    output_dim: int

    @abstractmethod
    def forward(self, values: Tensor) -> Tensor:
        """Return expanded values with shape ``[..., output_dim]``."""


class GaussianBasisExpansion(ScalarBasisExpansion):
    """Gaussian basis module with optionally trainable centers and widths."""

    def __init__(
        self,
        start: float,
        stop: float,
        num_basis: int,
        gamma: float | None = None,
        *,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if num_basis <= 0:
            raise ValueError("num_basis must be positive")
        if stop <= start:
            raise ValueError("stop must be greater than start")
        centers = torch.linspace(start, stop, num_basis)
        spacing = (stop - start) / max(num_basis - 1, 1)
        gamma_value = float(gamma if gamma is not None else 1.0 / (spacing**2 + 1e-12))
        self.output_dim = num_basis
        if trainable:
            self.centers = nn.Parameter(centers)
            self.log_gamma = nn.Parameter(torch.tensor(math.log(gamma_value)))
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("log_gamma", torch.tensor(math.log(gamma_value)))

    def forward(self, values: Tensor) -> Tensor:
        return gaussian_rbf(values, self.centers, torch.exp(self.log_gamma))


class BesselBasisExpansion(ScalarBasisExpansion):
    """Fixed Bessel/sine basis module."""

    def __init__(self, start: float, stop: float, num_basis: int, envelope_exponent: int = 2) -> None:
        super().__init__()
        self.start = float(start)
        self.stop = float(stop)
        self.output_dim = int(num_basis)
        self.envelope_exponent = int(envelope_exponent)
        if self.output_dim <= 0:
            raise ValueError("num_basis must be positive")
        if self.stop <= self.start:
            raise ValueError("stop must be greater than start")

    def forward(self, values: Tensor) -> Tensor:
        return bessel_basis(
            values,
            num_basis=self.output_dim,
            start=self.start,
            cutoff=self.stop,
            envelope_exponent=self.envelope_exponent,
        )


class FourierBasisExpansion(ScalarBasisExpansion):
    """Fixed Fourier feature module."""

    def __init__(self, start: float, stop: float, num_basis: int, include_constant: bool = True) -> None:
        super().__init__()
        self.start = float(start)
        self.stop = float(stop)
        self.output_dim = int(num_basis)
        self.include_constant = bool(include_constant)
        if self.output_dim <= 0:
            raise ValueError("num_basis must be positive")
        if self.stop <= self.start:
            raise ValueError("stop must be greater than start")

    def forward(self, values: Tensor) -> Tensor:
        return fourier_basis(
            values,
            num_basis=self.output_dim,
            start=self.start,
            cutoff=self.stop,
            include_constant=self.include_constant,
        )


# Backward-compatible alias used by the first prototype.
RBFExpansion = GaussianBasisExpansion


def make_basis_expansion(
    basis_type: str,
    *,
    start: float,
    cutoff: float,
    num_basis: int,
    gamma: float | None = None,
    trainable: bool | None = None,
    **kwargs: object,
) -> ScalarBasisExpansion:
    """Factory for scalar basis modules.

    Args:
        basis_type: ``gaussian``, ``bessel``, ``fourier``, or ``learnable_gaussian``.
        start: Lower bound of the scalar domain.
        cutoff: Upper bound of the scalar domain.
        num_basis: Output feature dimension.
        trainable: Optional override for Gaussian trainability.
    """

    kind = basis_type.lower()
    if kind in {"gaussian", "learnable_gaussian"}:
        return GaussianBasisExpansion(
            start,
            cutoff,
            num_basis,
            gamma=gamma,
            trainable=(kind == "learnable_gaussian") if trainable is None else trainable,
        )
    if kind == "bessel":
        return BesselBasisExpansion(start, cutoff, num_basis, **kwargs)
    if kind == "fourier":
        return FourierBasisExpansion(start, cutoff, num_basis, **kwargs)
    raise ValueError(f"Unknown basis_type {basis_type!r}")
