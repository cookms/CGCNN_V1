from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

from examples import analyze_graph_dataset
import materials_gnn.data.datasets as dataset_module
from materials_gnn.data.datasets import CrystalGraphDataset
from materials_gnn.data.graph_stats import (
    analyze_dataset_graphs,
    compute_graph_stats,
    graph_stats_to_rows,
    summarize_graph_stats,
)
from materials_gnn.featurization.line_graph import build_line_graph


def _graph() -> dict:
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_vec = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, -1.5, 0.0],
        ],
        dtype=torch.float32,
    )
    graph = {
        "z": torch.tensor([14, 14, 14]),
        "edge_index": edge_index,
        "edge_vec": edge_vec,
        "distance": torch.linalg.norm(edge_vec, dim=1),
        "edge_attr": torch.ones((4, 8)),
        "num_nodes": 3,
    }
    graph.update(build_line_graph(edge_index, edge_vec, num_nodes=3, num_angle_rbf=4, skip_backtracking=False))
    return graph


def test_compute_graph_stats_reports_counts_and_risks() -> None:
    stats = compute_graph_stats(
        _graph(),
        material_id="toy",
        hidden_dim=16,
        num_layers=2,
        batch_size=1,
        thresholds={"max_line_edges": 1, "max_estimated_batch_mb": 0.0},
    )

    assert stats.material_id == "toy"
    assert stats.num_atoms == 3
    assert stats.num_edges == 4
    assert stats.num_line_edges > 1
    assert stats.distance_min == 1.0
    assert stats.distance_max == 1.5
    assert any(flag.startswith("line_edges>") for flag in stats.risk_flags)
    assert any(flag.startswith("estimated_batch_mb>") for flag in stats.risk_flags)


def test_graph_stats_summary_and_rows_are_serializable() -> None:
    stats = [compute_graph_stats(_graph(), material_id="toy")]
    rows = graph_stats_to_rows(stats)
    summary = summarize_graph_stats(stats)

    assert rows[0]["material_id"] == "toy"
    assert isinstance(rows[0]["risk_flags"], str)
    assert summary["count"] == 1
    assert summary["num_edges"]["max"] == 4
    assert summary["riskiest"][0]["material_id"] == "toy"


class _PickleableGraphStatsDataset:
    cache_graphs = False
    _graph_cache = {}

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict:
        return {"graph": _graph(), "material_id": f"toy-{index}"}


def test_analyze_dataset_graphs_supports_multiprocessing() -> None:
    stats = analyze_dataset_graphs(_PickleableGraphStatsDataset(), num_workers=2)

    assert [item.material_id for item in stats] == ["toy-0", "toy-1", "toy-2"]
    assert all(item.num_edges == 4 for item in stats)


def test_analyze_graph_dataset_cli_writes_csv_and_json(tmp_path: Path, monkeypatch) -> None:
    cif_path = tmp_path / "toy.cif"
    cif_path.write_text("structure loading is monkeypatched\n")
    csv_path = tmp_path / "id_prop.csv"
    pd.DataFrame({"material_id": ["toy"], "cif_path": [cif_path.name], "target": [1.0]}).to_csv(
        csv_path,
        index=False,
    )
    output_csv = tmp_path / "graph_stats.csv"
    output_json = tmp_path / "graph_stats.json"

    def fake_structure_to_bond_graph(structure, **kwargs):
        return _graph()

    monkeypatch.setattr(dataset_module, "structure_to_bond_graph", fake_structure_to_bond_graph)
    monkeypatch.setattr(CrystalGraphDataset, "_load_structure", lambda self, path: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_graph_dataset.py",
            "--csv",
            str(csv_path),
            "--num-rbf",
            "8",
            "--num-angle-rbf",
            "4",
            "--max-samples",
            "1",
            "--num-workers",
            "0",
            "--progress-every",
            "0",
            "--output-csv",
            str(output_csv),
            "--output-json",
            str(output_json),
        ],
    )

    analyze_graph_dataset.main()

    assert output_csv.exists()
    assert output_json.exists()
    rows = pd.read_csv(output_csv)
    assert list(rows["material_id"]) == ["toy"]
    assert {"num_atoms", "num_edges", "num_line_edges", "risk_flags"}.issubset(rows.columns)
    payload = json.loads(output_json.read_text())
    assert payload["analysis_config"]["csv"] == str(csv_path)
    assert payload["summary"]["count"] == 1
    assert payload["summary"]["num_edges"]["max"] == 4
