"""ALIGNN-inspired atom/bond/angle model.

The model is not an exact reimplementation of ALIGNN. It keeps the central idea: build a
line graph where directed bonds become nodes, update bond states using bond-angle features,
then update atom states using the updated bond states. Alternating these two updates gives
the model access to two-body distances and three-body angles.

As in ``CGCNNModel``, distance and angle bases can be precomputed in preprocessing or
computed inside the model. Model-level learnable Gaussian bases make scalar geometry
featurization part of the trainable architecture.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from materials_gnn.featurization.basis import ScalarBasisExpansion, make_basis_expansion
from materials_gnn.featurization.elemental_features import AtomFeatureEncoder
from materials_gnn.models.layers import GatedGraphConv
from materials_gnn.models.readout import make_readout, pool_nodes


class ALIGNNLikeModel(nn.Module):
    """ALIGNN-inspired model using both bond graph and line graph."""

    def __init__(
        self,
        *,
        edge_input_dim: int = 64,
        angle_input_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_dim: int = 1,
        max_atomic_number: int = 118,
        pooling: str = "mean",
        dropout: float = 0.0,
        atom_feature_names: Sequence[str] | str | None = None,
        atom_feature_kwargs: Mapping[str, Any] | None = None,
        atom_input_dim: int | None = None,
        atom_feature_combine: str = "concat_project",
        distance_basis_type: str | None = None,
        distance_basis_start: float = 0.0,
        distance_basis_cutoff: float = 5.0,
        distance_basis_kwargs: Mapping[str, Any] | None = None,
        angle_basis_type: str | None = None,
        angle_basis_use_cosine: bool = False,
        angle_basis_kwargs: Mapping[str, Any] | None = None,
        readout_type: str = "mlp",
        ib_lambda: float = 0.01,
        ib_sigma_slope: float = 1.0,
        ib_fixed_point_iters: int = 8,
        ib_coupling: str = "ring",
        ib_trainable_lambda: bool = False,
        use_edge_weight: bool = False,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.use_edge_weight = use_edge_weight
        self.angle_basis_use_cosine = angle_basis_use_cosine
        self.atom_embedding = AtomFeatureEncoder(
            hidden_dim,
            max_atomic_number=max_atomic_number,
            descriptor_names=atom_feature_names,
            descriptor_kwargs=atom_feature_kwargs,
            external_feature_dim=atom_input_dim,
            combine=atom_feature_combine,
        )

        self.distance_basis: ScalarBasisExpansion | None = None
        if distance_basis_type is not None:
            self.distance_basis = make_basis_expansion(
                distance_basis_type,
                start=distance_basis_start,
                cutoff=distance_basis_cutoff,
                num_basis=edge_input_dim,
                **dict(distance_basis_kwargs or {}),
            )

        self.angle_basis: ScalarBasisExpansion | None = None
        if angle_basis_type is not None:
            angle_start, angle_stop = (-1.0, 1.0) if angle_basis_use_cosine else (0.0, math.pi)
            self.angle_basis = make_basis_expansion(
                angle_basis_type,
                start=angle_start,
                cutoff=angle_stop,
                num_basis=angle_input_dim,
                **dict(angle_basis_kwargs or {}),
            )

        self.bond_embedding = nn.Sequential(nn.Linear(edge_input_dim, hidden_dim), nn.SiLU())
        self.angle_embedding = nn.Sequential(nn.Linear(angle_input_dim, hidden_dim), nn.SiLU())

        # In the line graph, bond representations e are nodes and angle representations t
        # are edges. The same edge-gated convolution can update (bond, angle) states.
        self.line_convs = nn.ModuleList(
            [GatedGraphConv(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.bond_convs = nn.ModuleList(
            [GatedGraphConv(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.readout = make_readout(
            readout_type,
            hidden_dim,
            output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            ib_kwargs={
                "ib_lambda": ib_lambda,
                "sigma_slope": ib_sigma_slope,
                "fixed_point_iters": ib_fixed_point_iters,
                "coupling": ib_coupling,
                "trainable_lambda": ib_trainable_lambda,
            },
        )

    def _edge_features(self, graph: Mapping[str, Tensor | int], *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.distance_basis is None:
            edge_attr = graph["edge_attr"]
            if not isinstance(edge_attr, Tensor):
                raise TypeError("graph['edge_attr'] must be a tensor")
            return edge_attr.to(device=device, dtype=dtype)
        distance = graph["distance"]
        if not isinstance(distance, Tensor):
            raise TypeError("graph['distance'] must be a tensor for model-level distance bases")
        return self.distance_basis(distance.to(device=device, dtype=dtype))

    def _angle_features(self, graph: Mapping[str, Tensor | int], *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.angle_basis is None:
            line_edge_attr = graph["line_edge_attr"]
            if not isinstance(line_edge_attr, Tensor):
                raise TypeError("graph['line_edge_attr'] must be a tensor")
            return line_edge_attr.to(device=device, dtype=dtype)
        key = "cosine" if self.angle_basis_use_cosine else "angle"
        values = graph.get(key)  # type: ignore[attr-defined]
        if not isinstance(values, Tensor):
            raise KeyError(f"graph must contain {key!r} for model-level angle bases")
        return self.angle_basis(values.to(device=device, dtype=dtype))

    def forward(self, graph: Mapping[str, Tensor | int]) -> Tensor:
        required = ["z", "edge_index", "edge_attr", "line_edge_index", "line_edge_attr"]
        missing = [key for key in required if key not in graph]
        if missing:
            raise KeyError(f"ALIGNNLikeModel requires line-graph fields; missing {missing}")

        z = graph["z"]
        edge_index = graph["edge_index"]
        line_edge_index = graph["line_edge_index"]
        batch = graph.get("batch")  # type: ignore[attr-defined]
        atom_attr = graph.get("atom_attr")  # type: ignore[attr-defined]

        if not all(isinstance(v, Tensor) for v in [z, edge_index, line_edge_index]):
            raise TypeError("graph fields z, edge_index, and line_edge_index must be tensors")

        h = self.atom_embedding(z.long(), atom_attr=atom_attr if isinstance(atom_attr, Tensor) else None)  # type: ignore[union-attr]
        device = h.device
        dtype = h.dtype
        edge_index = edge_index.to(device=device)  # type: ignore[union-attr]
        line_edge_index = line_edge_index.to(device=device)  # type: ignore[union-attr]
        e = self.bond_embedding(self._edge_features(graph, device=device, dtype=dtype))
        t = self.angle_embedding(self._angle_features(graph, device=device, dtype=dtype))
        batch_tensor = batch.to(device=device) if isinstance(batch, Tensor) else None
        edge_weight = graph.get("edge_weight") if self.use_edge_weight else None

        for line_conv, bond_conv in zip(self.line_convs, self.bond_convs, strict=True):
            # Bonds are nodes in the line graph; angles are line-graph edges.
            e, t = line_conv(e, line_edge_index, t)
            # Updated bonds then mediate atom-graph message passing.
            h, e = bond_conv(h, edge_index, e, edge_weight=edge_weight)  # type: ignore[arg-type]

        crystal_embedding = pool_nodes(h, batch_tensor, mode=self.pooling)
        prediction = self.readout(crystal_embedding)
        return prediction.squeeze(-1)
