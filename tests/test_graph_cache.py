from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

import materials_gnn.data.datasets as dataset_module
from materials_gnn.data.datasets import CrystalGraphDataset, GraphConstructionError
from materials_gnn.data.graph_cache import GraphCache, stable_hash


def _fake_graph() -> dict[str, torch.Tensor | int]:
    edge_vec = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    return {
        "z": torch.tensor([14, 14], dtype=torch.long),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "edge_vec": edge_vec,
        "distance": torch.tensor([1.0], dtype=torch.float32),
        "edge_attr": torch.ones((1, 4), dtype=torch.float32),
        "num_nodes": 2,
    }


def test_stable_hash_is_order_independent() -> None:
    assert stable_hash({"a": 1, "b": {"x": 2, "y": 3}}) == stable_hash({"b": {"y": 3, "x": 2}, "a": 1})


def test_graph_cache_round_trip(tmp_path: Path) -> None:
    cache = GraphCache(tmp_path)
    key = cache.key({"material_id": "mp-test", "cutoff": 5.0})
    cache.save(key, _fake_graph(), metadata={"source": "unit-test"})

    loaded = cache.load(key)

    assert loaded is not None
    assert torch.equal(loaded["z"], torch.tensor([14, 14]))
    assert loaded["num_nodes"] == 2


def test_dataset_reuses_persistent_graph_cache(tmp_path: Path, monkeypatch) -> None:
    cif_path = tmp_path / "toy.cif"
    cif_path.write_text("not a real cif; structure loading is monkeypatched\n")
    csv_path = tmp_path / "id_prop.csv"
    pd.DataFrame({"material_id": ["toy"], "cif_path": [cif_path.name], "target": [1.0]}).to_csv(csv_path, index=False)

    calls = {"count": 0}

    def fake_structure_to_bond_graph(structure, **kwargs):
        calls["count"] += 1
        return _fake_graph()

    monkeypatch.setattr(dataset_module, "structure_to_bond_graph", fake_structure_to_bond_graph)
    monkeypatch.setattr(CrystalGraphDataset, "_load_structure", lambda self, path: object())

    first = CrystalGraphDataset(
        csv_path,
        graph_cache_dir=tmp_path / "cache",
        graph_kwargs={"num_rbf": 4},
    )
    first_item = first[0]
    assert calls["count"] == 1
    assert first_item["graph"]["edge_attr"].shape == (1, 4)

    second = CrystalGraphDataset(
        csv_path,
        graph_cache_dir=tmp_path / "cache",
        graph_kwargs={"num_rbf": 4},
    )
    second_item = second[0]
    assert calls["count"] == 1
    assert torch.equal(second_item["graph"]["edge_index"], first_item["graph"]["edge_index"])

    changed = CrystalGraphDataset(
        csv_path,
        graph_cache_dir=tmp_path / "cache",
        graph_kwargs={"num_rbf": 8},
    )
    _ = changed[0]
    assert calls["count"] == 2


def test_graph_cache_ignores_target_column_and_material_id(tmp_path: Path, monkeypatch) -> None:
    cif_path = tmp_path / "toy.cif"
    cif_path.write_text("not a real cif; structure loading is monkeypatched\n")
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    pd.DataFrame(
        {"material_id": ["toy-a"], "cif_path": [cif_path.name], "target_a": [1.0], "target_b": [2.0]}
    ).to_csv(csv_a, index=False)
    pd.DataFrame(
        {"material_id": ["toy-b"], "cif_path": [cif_path.name], "target_a": [3.0], "target_b": [4.0]}
    ).to_csv(csv_b, index=False)

    calls = {"count": 0}

    def fake_structure_to_bond_graph(structure, **kwargs):
        calls["count"] += 1
        return _fake_graph()

    monkeypatch.setattr(dataset_module, "structure_to_bond_graph", fake_structure_to_bond_graph)
    monkeypatch.setattr(CrystalGraphDataset, "_load_structure", lambda self, path: object())

    first = CrystalGraphDataset(csv_a, target_column="target_a", graph_cache_dir=tmp_path / "cache")
    _ = first[0]
    assert calls["count"] == 1

    second = CrystalGraphDataset(csv_b, target_column="target_b", graph_cache_dir=tmp_path / "cache")
    _ = second[0]
    assert calls["count"] == 1

    assert first.graph_cache_key(0) == second.graph_cache_key(0)


def test_voronoi_failure_policy_is_explicit_in_cache_identity(tmp_path: Path) -> None:
    cif_path = tmp_path / "toy.cif"
    cif_path.write_text("cache identity only\n")
    csv_path = tmp_path / "id_prop.csv"
    pd.DataFrame(
        {"material_id": ["toy"], "cif_path": [cif_path.name], "target": [1.0]}
    ).to_csv(csv_path, index=False)

    default = CrystalGraphDataset(
        csv_path,
        neighbor_strategy="voronoi",
        graph_cache_dir=tmp_path / "cache",
    )
    empty = CrystalGraphDataset(
        csv_path,
        neighbor_strategy="voronoi",
        neighbor_kwargs={"failure_policy": "empty"},
        graph_cache_dir=tmp_path / "cache",
    )

    assert default.neighbor_kwargs["failure_policy"] == "raise"
    assert default._cache_identity(0, cif_path)["neighbor_kwargs"]["failure_policy"] == "raise"
    assert default.graph_cache_key(0) != empty.graph_cache_key(0)


def test_dataset_graph_failure_identifies_problem_structure(tmp_path: Path, monkeypatch) -> None:
    cif_path = tmp_path / "broken.cif"
    cif_path.write_text("not a valid structure\n")
    csv_path = tmp_path / "id_prop.csv"
    pd.DataFrame(
        {"material_id": ["problem-42"], "cif_path": [cif_path.name], "target": [1.0]}
    ).to_csv(csv_path, index=False)

    def fail_graph_construction(structure, **kwargs):
        raise ValueError("synthetic Voronoi failure")

    monkeypatch.setattr(dataset_module, "structure_to_bond_graph", fail_graph_construction)
    monkeypatch.setattr(CrystalGraphDataset, "_load_structure", lambda self, path: object())
    dataset = CrystalGraphDataset(csv_path, neighbor_strategy="voronoi")

    with pytest.raises(GraphConstructionError) as exc_info:
        _ = dataset[0]

    message = str(exc_info.value)
    assert "dataset row 0" in message
    assert "material_id='problem-42'" in message
    assert f"cif_path={cif_path}" in message
    assert "neighbor_strategy='voronoi'" in message
    assert "ValueError: synthetic Voronoi failure" in message
    assert isinstance(exc_info.value.__cause__, ValueError)
