from __future__ import annotations

import numpy as np
import torch

from materials_gnn.featurization import (
    AdaptiveShellNeighborStrategy,
    CutoffNeighborStrategy,
    KNearestNeighborStrategy,
    NeighborList,
    StrainJitterConsensusNeighborStrategy,
    make_neighbor_strategy,
    structure_to_bond_graph,
)


class _FakeSpecie:
    Z = 14


class _FakeSite:
    specie = _FakeSpecie()


class _FakeLattice:
    matrix = np.eye(3)


class _FakeStructure:
    lattice = _FakeLattice()
    cart_coords = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def __len__(self) -> int:
        return 2

    def __iter__(self):
        return iter([_FakeSite(), _FakeSite()])


class _OneEdgeStrategy:
    name = "one_edge"

    def build(self, structure: object) -> NeighborList:
        return NeighborList(
            center_indices=np.asarray([0]),
            neighbor_indices=np.asarray([1]),
            image_vectors=np.asarray([[0, 0, 0]]),
            distances=np.asarray([1.0]),
            weights=np.asarray([0.25]),
        )


def test_make_neighbor_strategy_factory() -> None:
    assert isinstance(make_neighbor_strategy(None, cutoff=3.0), CutoffNeighborStrategy)
    assert isinstance(make_neighbor_strategy("cutoff", cutoff=3.0), CutoffNeighborStrategy)
    assert isinstance(make_neighbor_strategy("knn", cutoff=3.0, k=8), KNearestNeighborStrategy)
    assert isinstance(make_neighbor_strategy("adaptive_shell", cutoff=3.0), AdaptiveShellNeighborStrategy)
    assert isinstance(make_neighbor_strategy("strain_consensus", cutoff=3.0), StrainJitterConsensusNeighborStrategy)


def test_custom_neighbor_strategy_integrates_with_graph_builder(monkeypatch) -> None:
    import materials_gnn.featurization.crystal_graph as crystal_graph

    # Avoid requiring pymatgen for this unit test; the fake structure implements only the
    # attributes needed after a strategy has already produced neighbor arrays.
    monkeypatch.setattr(crystal_graph, "_require_pymatgen", lambda: None)

    graph = structure_to_bond_graph(
        _FakeStructure(),
        cutoff=2.0,
        num_rbf=4,
        neighbor_strategy=_OneEdgeStrategy(),
    )

    assert graph["z"].tolist() == [14, 14]
    assert torch.equal(graph["edge_index"], torch.tensor([[0], [1]]))
    assert torch.allclose(graph["edge_vec"], torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(graph["distance"], torch.tensor([1.0]))
    assert graph["edge_attr"].shape == (1, 4)
    assert torch.allclose(graph["edge_weight"], torch.tensor([0.25]))
