"""A compact CGCNN-style model.

This model uses an atom/bond graph only: atoms are nodes, periodic neighbor contacts are
edges, distances are expanded into basis edge features, and graph pooling converts atom
states into a crystal-level representation for scalar property prediction.

The model can either consume precomputed ``edge_attr`` from preprocessing or recompute a
trainable/fixed basis from raw ``distance`` values inside the network. The latter is useful
for learnable basis-function experiments.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from materials_gnn.featurization.basis import ScalarBasisExpansion, make_basis_expansion
from materials_gnn.featurization.elemental_features import AtomFeatureEncoder
from materials_gnn.models.layers import GatedGraphConv
from materials_gnn.models.readout import MLPReadout, pool_nodes


class CGCNNModel(nn.Module):
    """Distance-only atom/bond graph model for scalar property prediction.

    Args:
        edge_input_dim: Dimension of edge features after basis expansion.
        hidden_dim: Width of atom, bond, and readout hidden states.
        atom_feature_names: Optional elemental descriptor list or ``"default"``. These
            descriptors are merged with learned atomic-number embeddings.
        atom_input_dim: Dimension of external ``graph['atom_attr']`` if descriptors are
            precomputed outside the model.
        distance_basis_type: Optional model-level basis. If supplied, the model ignores
            precomputed ``edge_attr`` and expands raw ``graph['distance']`` instead. Use
            ``"learnable_gaussian"`` for trainable centers and widths.
        distance_basis_cutoff: Upper bound for model-level distance basis.
    """

    def __init__(
        self,
        *,
        edge_input_dim: int = 64,
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
    ) -> None:
        super().__init__()
        self.pooling = pooling
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
        self.bond_embedding = nn.Sequential(nn.Linear(edge_input_dim, hidden_dim), nn.SiLU())
        self.convs = nn.ModuleList(
            [GatedGraphConv(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.readout = MLPReadout(hidden_dim, output_dim, hidden_dim=hidden_dim, dropout=dropout)

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

    def forward(self, graph: Mapping[str, Tensor | int]) -> Tensor:
        z = graph["z"]
        edge_index = graph["edge_index"]
        batch = graph.get("batch")  # type: ignore[attr-defined]
        atom_attr = graph.get("atom_attr")  # type: ignore[attr-defined]

        if not isinstance(z, Tensor) or not isinstance(edge_index, Tensor):
            raise TypeError("graph must contain tensor fields z and edge_index")

        # z has already been moved to the model device by the trainer; keep the code robust
        # for direct calls by moving optional atom_attr to the same device inside the encoder.
        h = self.atom_embedding(z.long(), atom_attr=atom_attr if isinstance(atom_attr, Tensor) else None)
        e = self.bond_embedding(self._edge_features(graph, device=h.device, dtype=h.dtype))
        edge_index = edge_index.to(device=h.device)
        batch_tensor = batch.to(device=h.device) if isinstance(batch, Tensor) else None

        for conv in self.convs:
            h, e = conv(h, edge_index, e)

        crystal_embedding = pool_nodes(h, batch_tensor, mode=self.pooling)
        prediction = self.readout(crystal_embedding)
        return prediction.squeeze(-1)
