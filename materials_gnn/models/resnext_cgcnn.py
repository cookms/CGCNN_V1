"""Aggregated residual message passing on the existing CGCNN crystal graph."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from materials_gnn.models.cgcnn import CGCNNModel
from materials_gnn.models.layers import GatedGraphConv


class AggregatedResidualGraphBlock(nn.Module):
    """Aggregate independent CGCNN updates over one shared graph.

    Each branch sees the full node and edge state, but ``branch_dim`` controls the
    internal MLP bottleneck. Branches return normalized residual states, so their
    transformations are measured relative to the common input before aggregation.
    At C=1 and full branch width this is exactly one baseline GatedGraphConv.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        cardinality: int = 4,
        branch_dim: int | None = None,
        aggregation: str = "mean",
        branch_weighting: str = "uniform",
        **conv_kwargs: Any,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or cardinality <= 0:
            raise ValueError("hidden_dim and cardinality must be positive")
        if branch_dim is not None and branch_dim <= 0:
            raise ValueError("branch_dim must be positive")
        if aggregation not in {"sum", "mean"}:
            raise ValueError("aggregation must be 'sum' or 'mean'")
        if branch_weighting not in {"uniform", "learned"}:
            raise ValueError("branch_weighting must be 'uniform' or 'learned'")
        self.cardinality = cardinality
        self.branch_dim = branch_dim or hidden_dim
        self.aggregation = aggregation
        self.branch_weighting = branch_weighting
        self.branches = nn.ModuleList(
            GatedGraphConv(hidden_dim, hidden_dim, hidden_dim=self.branch_dim, **conv_kwargs)
            for _ in range(cardinality)
        )
        self.branch_logits = (
            nn.Parameter(torch.zeros(cardinality)) if branch_weighting == "learned" else None
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_weight: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.cardinality == 1:
            return self.branches[0](x, edge_index, edge_attr, edge_weight=edge_weight)
        weights = (
            torch.softmax(self.branch_logits, dim=0)
            if self.branch_logits is not None else None
        )
        node_delta: Tensor | None = None
        edge_delta: Tensor | None = None
        # Sequential branches share edge_index, edge_attr and edge_weight. Accumulating
        # deltas avoids stacking C full node/edge output tensors.
        for index, branch in enumerate(self.branches):
            branch_x, branch_edge = branch(x, edge_index, edge_attr, edge_weight=edge_weight)
            scale = weights[index] * self.cardinality if weights is not None else 1.0
            this_node_delta = (branch_x - x) * scale
            this_edge_delta = (branch_edge - edge_attr) * scale
            node_delta = this_node_delta if node_delta is None else node_delta + this_node_delta
            edge_delta = this_edge_delta if edge_delta is None else edge_delta + this_edge_delta
        assert node_delta is not None and edge_delta is not None
        if self.aggregation == "mean":
            node_delta = node_delta / self.cardinality
            edge_delta = edge_delta / self.cardinality
        return x + node_delta, edge_attr + edge_delta


class ResNeXtCGCNNModel(CGCNNModel):
    """Opt-in CGCNN with independent aggregated residual graph branches."""

    def __init__(
        self,
        *,
        cardinality: int = 4,
        branch_dim: int | None = None,
        aggregation: str = "mean",
        branch_weighting: str = "uniform",
        **cgcnn_kwargs: Any,
    ) -> None:
        super().__init__(**cgcnn_kwargs)
        hidden_dim = cgcnn_kwargs.get("hidden_dim", 128)
        num_layers = cgcnn_kwargs.get("num_layers", 3)
        self.cardinality = cardinality
        self.branch_dim = branch_dim or hidden_dim
        self.aggregation = aggregation
        self.branch_weighting = branch_weighting
        self.hidden_dim = hidden_dim
        self.convs = nn.ModuleList(
            AggregatedResidualGraphBlock(
                hidden_dim,
                cardinality=cardinality,
                branch_dim=branch_dim,
                aggregation=aggregation,
                branch_weighting=branch_weighting,
                dropout=cgcnn_kwargs.get("dropout", 0.0),
                conv_activation_type=cgcnn_kwargs.get("conv_activation_type", "silu"),
                conv_ib_lambda=cgcnn_kwargs.get("conv_ib_lambda", 0.01),
                conv_ib_sigma_slope=cgcnn_kwargs.get("conv_ib_sigma_slope", 1.0),
                conv_ib_fixed_point_iters=cgcnn_kwargs.get("conv_ib_fixed_point_iters", 8),
                conv_ib_coupling=cgcnn_kwargs.get("conv_ib_coupling", "ring"),
                conv_ib_trainable_lambda=cgcnn_kwargs.get("conv_ib_trainable_lambda", False),
                conv_ib_targets=cgcnn_kwargs.get("conv_ib_targets", ()),
            )
            for _ in range(num_layers)
        )


def model_parameter_summary(model: CGCNNModel) -> dict[str, Any]:
    """Report parameter counts for the embedding, graph blocks and readout."""

    def count(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    embedding = count(model.atom_embedding) + count(model.bond_embedding)
    if model.distance_basis is not None:
        embedding += count(model.distance_basis)
    result: dict[str, Any] = {
        "model_type": "resnext_cgcnn" if isinstance(model, ResNeXtCGCNNModel) else "cgcnn",
        "hidden_dim": model.atom_embedding.output_dim,
        "num_layers": len(model.convs),
        "embedding": embedding,
        "graph_blocks": count(model.convs),
        "readout": count(model.readout),
        "total": count(model),
    }
    if isinstance(model, ResNeXtCGCNNModel):
        result.update(
            cardinality=model.cardinality,
            branch_dim=model.branch_dim,
            aggregation=model.aggregation,
            branch_weighting=model.branch_weighting,
            parameters_per_branch=[
                count(branch) for block in model.convs for branch in block.branches
            ],
        )
    return result
