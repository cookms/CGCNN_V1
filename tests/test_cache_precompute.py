from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

from examples import precompute_graph_cache as precompute_graph_cache_script
import materials_gnn.data.datasets as dataset_module
from materials_gnn.data import CrystalGraphDataset, precompute_graph_cache


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


def test_precompute_graph_cache_counts_built_and_hits(tmp_path: Path, monkeypatch) -> None:
    cif_path = tmp_path / "toy.cif"
    cif_path.write_text("not a real cif; structure loading is monkeypatched\n")
    csv_path = tmp_path / "id_prop.csv"
    pd.DataFrame({"material_id": ["toy"], "cif_path": [cif_path.name], "target": [1.0]}).to_csv(
        csv_path, index=False
    )

    calls = {"count": 0}

    def fake_structure_to_bond_graph(structure, **kwargs):
        calls["count"] += 1
        return _fake_graph()

    monkeypatch.setattr(dataset_module, "structure_to_bond_graph", fake_structure_to_bond_graph)
    monkeypatch.setattr(CrystalGraphDataset, "_load_structure", lambda self, path: object())

    dataset = CrystalGraphDataset(csv_path, graph_cache_dir=tmp_path / "cache")
    first = precompute_graph_cache(dataset)
    assert first.requested == 1
    assert first.graphs_built == 1
    assert first.cache_hits == 0
    assert calls["count"] == 1

    warm_dataset = CrystalGraphDataset(csv_path, graph_cache_dir=tmp_path / "cache")
    second = precompute_graph_cache(warm_dataset)
    assert second.requested == 1
    assert second.graphs_built == 0
    assert second.cache_hits == 1
    assert calls["count"] == 1


def test_precompute_graph_cache_requires_persistent_cache(tmp_path: Path) -> None:
    csv_path = tmp_path / "id_prop.csv"
    pd.DataFrame({"material_id": ["toy"], "cif_path": ["toy.cif"], "target": [1.0]}).to_csv(
        csv_path, index=False
    )
    dataset = CrystalGraphDataset(csv_path)

    try:
        precompute_graph_cache(dataset)
    except ValueError as exc:
        assert "graph_cache_dir" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError")


class _PickleablePrecomputeDataset:
    graph_cache = object()
    overwrite_graph_cache = False

    def __init__(self, n: int = 4) -> None:
        self.n = n
        self.table = pd.DataFrame({"material_id": [f"m{i}" for i in range(n)]})
        self.id_column = "material_id"

    def __len__(self) -> int:
        return self.n

    def is_graph_cached(self, index: int) -> bool:
        return index % 2 == 0

    def __getitem__(self, index: int) -> dict:
        return {"graph": _fake_graph(), "material_id": f"m{index}"}


def test_precompute_graph_cache_supports_multiprocessing() -> None:
    dataset = _PickleablePrecomputeDataset(4)
    result = precompute_graph_cache(dataset, num_workers=2)

    assert result.requested == 4
    assert result.cache_hits == 2
    assert result.graphs_built == 2
    assert result.failed == 0


def test_precompute_graph_cache_cli_writes_cache_and_summary(tmp_path: Path, monkeypatch) -> None:
    cif_path = tmp_path / "toy.cif"
    cif_path.write_text("structure loading is monkeypatched\n")
    csv_path = tmp_path / "id_prop.csv"
    pd.DataFrame({"material_id": ["toy"], "cif_path": [cif_path.name], "target": [1.0]}).to_csv(
        csv_path,
        index=False,
    )
    cache_dir = tmp_path / "cache"
    summary_path = tmp_path / "precompute_summary.json"

    calls = {"count": 0}

    def fake_structure_to_bond_graph(structure, **kwargs):
        calls["count"] += 1
        return _fake_graph()

    monkeypatch.setattr(dataset_module, "structure_to_bond_graph", fake_structure_to_bond_graph)
    monkeypatch.setattr(CrystalGraphDataset, "_load_structure", lambda self, path: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "precompute_graph_cache.py",
            "--csv",
            str(csv_path),
            "--model",
            "cgcnn",
            "--num-rbf",
            "4",
            "--graph-cache-dir",
            str(cache_dir),
            "--max-samples",
            "1",
            "--num-workers",
            "0",
            "--progress-every",
            "0",
            "--output-json",
            str(summary_path),
        ],
    )

    precompute_graph_cache_script.main()

    assert summary_path.exists()
    assert list((cache_dir / "graphs").glob("*.pt"))
    payload = json.loads(summary_path.read_text())
    assert payload["include_line_graph"] is False
    assert payload["result"]["requested"] == 1
    assert payload["result"]["graphs_built"] == 1
    assert payload["result"]["cache_hits"] == 0

    warm_dataset = CrystalGraphDataset(
        csv_path,
        neighbor_strategy="cutoff",
        graph_cache_dir=cache_dir,
        graph_kwargs={"num_rbf": 4, "distance_basis_type": "gaussian", "atom_feature_names": None},
        include_line_graph=False,
    )
    warm_result = precompute_graph_cache(warm_dataset)
    assert warm_result.cache_hits == 1
    assert calls["count"] == 1
