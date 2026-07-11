from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from materials_gnn.featurization import (
    AdaptiveShellNeighborStrategy,
    CutoffNeighborStrategy,
    KNearestNeighborStrategy,
    NeighborList,
    StrainJitterConsensusNeighborStrategy,
    VoronoiNeighborError,
    VoronoiNeighborStrategy,
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


class _FakeVoronoiStructure(_FakeStructure):
    frac_coords = _FakeStructure.cart_coords.copy()

    def __init__(self) -> None:
        self.neighbor_list_calls: list[tuple[float, bool]] = []

    def get_neighbor_list(self, r: float, *, exclude_self: bool = True):
        self.neighbor_list_calls.append((r, exclude_self))
        return (
            np.asarray([0, 1]),
            np.asarray([1, 0]),
            np.asarray([[0, 0, 0], [0, 0, 0]]),
            np.asarray([1.0, 1.0]),
        )


def _install_fake_voronoi(monkeypatch, responses: dict[int, object]) -> None:
    class _FakeVoronoiNN:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def get_nn_info(self, structure: object, center: int):
            response = responses[center]
            if isinstance(response, BaseException):
                raise response
            return response

    local_env = ModuleType("pymatgen.analysis.local_env")
    local_env.VoronoiNN = _FakeVoronoiNN  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pymatgen.analysis.local_env", local_env)


def _two_center_voronoi_info() -> dict[int, object]:
    return {
        0: [{"site_index": 1, "image": [1, 0, 0], "weight": 0.25}],
        1: [{"site_index": 0, "image": [0, 0, 0], "weight": 0.75}],
    }


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
    voronoi = make_neighbor_strategy(
        "voronoi",
        cutoff=3.0,
        strategy_kwargs={"cutoff": 12.0},
        failure_policy="empty",
    )
    assert isinstance(voronoi, VoronoiNeighborStrategy)
    assert voronoi.cutoff == 12.0
    assert voronoi.failure_policy == "empty"


def test_voronoi_success_preserves_directed_periodic_edges(monkeypatch) -> None:
    _install_fake_voronoi(monkeypatch, _two_center_voronoi_info())

    neighbors = VoronoiNeighborStrategy(cutoff=8.0).build(_FakeVoronoiStructure())

    assert neighbors.center_indices.tolist() == [0, 1]
    assert neighbors.neighbor_indices.tolist() == [1, 0]
    assert neighbors.image_vectors.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert neighbors.distances.tolist() == [2.0, 1.0]
    assert neighbors.weights is not None
    assert neighbors.weights.tolist() == [0.25, 0.75]


def test_voronoi_default_raises_when_one_center_has_no_neighbors(monkeypatch) -> None:
    responses = _two_center_voronoi_info()
    responses[1] = []
    _install_fake_voronoi(monkeypatch, responses)

    with pytest.raises(VoronoiNeighborError) as exc_info:
        VoronoiNeighborStrategy(
            cutoff=7.5,
            tol=0.1,
            allow_pathological=False,
        ).build(_FakeVoronoiStructure())

    message = str(exc_info.value)
    assert "atom index 1" in message
    assert "structure size 2" in message
    assert "cutoff=7.5" in message
    assert "tolerance=0.1" in message
    assert "allow_pathological=False" in message
    assert "No partial Voronoi graph was returned" in message


def test_voronoi_empty_policy_handles_every_center_without_neighbors(monkeypatch) -> None:
    _install_fake_voronoi(monkeypatch, {0: [], 1: []})

    neighbors = VoronoiNeighborStrategy(failure_policy="empty").build(
        _FakeVoronoiStructure()
    )

    assert neighbors.center_indices.shape == (0,)
    assert neighbors.neighbor_indices.shape == (0,)
    assert neighbors.image_vectors.shape == (0, 3)
    assert neighbors.distances.shape == (0,)
    assert neighbors.weights is None


def test_voronoi_default_reports_when_every_center_has_no_neighbors(monkeypatch) -> None:
    _install_fake_voronoi(monkeypatch, {0: [], 1: []})

    with pytest.raises(VoronoiNeighborError, match="no neighbors for every atom"):
        VoronoiNeighborStrategy().build(_FakeVoronoiStructure())


def test_voronoi_wraps_expected_get_nn_info_exception(monkeypatch) -> None:
    responses = _two_center_voronoi_info()
    responses[1] = RuntimeError("Qhull failed")
    _install_fake_voronoi(monkeypatch, responses)

    with pytest.raises(VoronoiNeighborError) as exc_info:
        VoronoiNeighborStrategy().build(_FakeVoronoiStructure())

    assert "atom index 1" in str(exc_info.value)
    assert "RuntimeError: Qhull failed" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_voronoi_does_not_catch_unexpected_exceptions(monkeypatch) -> None:
    _install_fake_voronoi(monkeypatch, {0: KeyError("unexpected"), 1: []})

    with pytest.raises(KeyError, match="unexpected"):
        VoronoiNeighborStrategy().build(_FakeVoronoiStructure())


def test_voronoi_cutoff_policy_rebuilds_the_whole_graph(monkeypatch) -> None:
    responses = _two_center_voronoi_info()
    responses[1] = []
    _install_fake_voronoi(monkeypatch, responses)
    structure = _FakeVoronoiStructure()

    neighbors = VoronoiNeighborStrategy(cutoff=7.5, failure_policy="cutoff").build(
        structure
    )

    assert structure.neighbor_list_calls == [(7.5, True)]
    assert neighbors.center_indices.tolist() == [0, 1]
    assert neighbors.neighbor_indices.tolist() == [1, 0]
    assert neighbors.image_vectors.shape == (2, 3)
    assert neighbors.distances.tolist() == [1.0, 1.0]
    assert neighbors.weights is None


def test_graph_builder_accepts_voronoi_cutoff_in_neighbor_kwargs(monkeypatch) -> None:
    import materials_gnn.featurization.crystal_graph as crystal_graph

    _install_fake_voronoi(monkeypatch, _two_center_voronoi_info())
    monkeypatch.setattr(crystal_graph, "_require_pymatgen", lambda: None)

    graph = structure_to_bond_graph(
        _FakeVoronoiStructure(),
        cutoff=5.0,
        rbf_cutoff=12.0,
        num_rbf=4,
        neighbor_strategy="voronoi",
        neighbor_kwargs={"cutoff": 12.0, "failure_policy": "raise"},
    )

    assert torch.equal(graph["edge_index"], torch.tensor([[0, 1], [1, 0]]))
    assert torch.allclose(graph["distance"], torch.tensor([2.0, 1.0]))
    assert torch.allclose(graph["edge_weight"], torch.tensor([0.25, 0.75]))


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
