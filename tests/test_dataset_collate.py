from __future__ import annotations

import torch

from materials_gnn.data.datasets import collate_graphs
from materials_gnn.featurization.line_graph import build_line_graph


def _sample(material_id: str, shift: float) -> dict:
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_vec = torch.tensor([[1.0 + shift, 0.0, 0.0], [-(1.0 + shift), 0.0, 0.0]])
    graph = {
        "z": torch.tensor([14, 14]),
        "edge_index": edge_index,
        "edge_vec": edge_vec,
        "distance": torch.linalg.norm(edge_vec, dim=1),
        "edge_attr": torch.randn(2, 4),
        "edge_unit_vec": edge_vec / torch.linalg.norm(edge_vec, dim=1).clamp_min(1e-12).unsqueeze(-1),
        "atom_attr": torch.randn(2, 3),
        "num_nodes": 2,
    }
    graph.update(build_line_graph(edge_index, edge_vec, num_nodes=2, num_angle_rbf=3, skip_backtracking=False))
    return {"graph": graph, "y": torch.tensor(shift), "material_id": material_id}


def test_collate_offsets_atom_and_line_graph_indices() -> None:
    batch = collate_graphs([_sample("a", 0.0), _sample("b", 0.2)])

    assert batch["z"].shape == (4,)
    assert batch["edge_index"].shape == (2, 4)
    assert batch["line_edge_index"].shape[0] == 2
    assert batch["line_edge_index"].max().item() >= 2
    assert batch["batch"].tolist() == [0, 0, 1, 1]
    assert batch["atom_attr"].shape == (4, 3)
    assert batch["edge_unit_vec"].shape == (4, 3)
    assert "angle" in batch and "cosine" in batch
    assert batch["y"].shape == (2,)
    assert batch["material_id"] == ["a", "b"]


def test_collate_preserves_optional_edge_weights() -> None:
    first = _sample("weighted", 0.0)
    first["graph"]["edge_weight"] = torch.tensor([0.2, 0.8])
    second = _sample("plain", 0.1)

    batch = collate_graphs([first, second])

    assert "edge_weight" in batch
    assert torch.allclose(batch["edge_weight"], torch.tensor([0.2, 0.8, 1.0, 1.0]))
