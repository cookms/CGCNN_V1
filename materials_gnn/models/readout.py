"""Graph pooling and readout heads.

Message passing creates atom-level features. Scalar materials properties such as formation
energy, band gap, or elastic modulus are crystal-level labels, so atom states must be
pooled into one crystal representation before a final prediction head is applied.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from materials_gnn.models.layers import build_mlp


def global_mean_pool(x: Tensor, batch: Tensor | None = None) -> Tensor:
    """Mean-pool node features into graph features.

    Args:
        x: Node features ``[num_nodes, hidden_dim]``.
        batch: Optional graph id for each node. If omitted, all nodes are treated as one
            crystal graph.
    """

    if batch is None:
        return x.mean(dim=0, keepdim=True)

    batch = batch.long()
    if batch.numel() != x.shape[0]:
        raise ValueError("batch length must equal number of nodes")
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    pooled = x.new_zeros((num_graphs, x.shape[-1]))
    counts = x.new_zeros((num_graphs, 1))
    pooled.index_add_(0, batch, x)
    counts.index_add_(0, batch, torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype))
    return pooled / counts.clamp_min(1.0)


def global_add_pool(x: Tensor, batch: Tensor | None = None) -> Tensor:
    """Sum-pool node features into graph features."""

    if batch is None:
        return x.sum(dim=0, keepdim=True)
    batch = batch.long()
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    pooled = x.new_zeros((num_graphs, x.shape[-1]))
    pooled.index_add_(0, batch, x)
    return pooled


def pool_nodes(x: Tensor, batch: Tensor | None = None, *, mode: str = "mean") -> Tensor:
    """Pool node features using a named strategy."""

    if mode == "mean":
        return global_mean_pool(x, batch)
    if mode == "sum":
        return global_add_pool(x, batch)
    raise ValueError(f"Unsupported pooling mode: {mode!r}")


class MLPReadout(nn.Module):
    """Small MLP mapping crystal embeddings to scalar or vector targets."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        *,
        hidden_dim: int | None = None,
        num_hidden_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or input_dim
        hidden_dims = [hidden] * num_hidden_layers
        self.net = build_mlp(input_dim, hidden_dims, output_dim, dropout=dropout)

    def forward(self, crystal_embedding: Tensor) -> Tensor:
        return self.net(crystal_embedding)
