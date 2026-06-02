from __future__ import annotations

import math

import torch

from materials_gnn.featurization.line_graph import build_line_graph


def test_line_graph_computes_bond_angle() -> None:
    # Directed bonds: 0 -> 1 and 1 -> 2. The angle at atom 1 is between 1 -> 0
    # and 1 -> 2. With the vectors below, that angle is 90 degrees.
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_vec = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    line_graph = build_line_graph(edge_index, edge_vec, num_nodes=3, num_angle_rbf=8)

    assert line_graph["line_edge_index"].shape == (2, 1)
    assert torch.equal(line_graph["line_edge_index"], torch.tensor([[0], [1]]))
    assert line_graph["line_edge_attr"].shape == (1, 8)
    assert torch.allclose(line_graph["angle"], torch.tensor([math.pi / 2]), atol=1e-6)


def test_line_graph_skips_backtracking() -> None:
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_vec = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    skipped = build_line_graph(edge_index, edge_vec, num_nodes=2, skip_backtracking=True)
    kept = build_line_graph(edge_index, edge_vec, num_nodes=2, skip_backtracking=False)

    assert skipped["line_edge_index"].shape[1] == 0
    assert kept["line_edge_index"].shape[1] == 2


def test_line_graph_caps_outgoing_neighbors() -> None:
    # One incoming bond into center atom 1, with three possible outgoing bonds.
    edge_index = torch.tensor(
        [
            [0, 1, 1, 1],
            [1, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    edge_vec = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=torch.float32,
    )

    full = build_line_graph(edge_index, edge_vec, num_nodes=5, num_angle_rbf=4)
    capped = build_line_graph(
        edge_index,
        edge_vec,
        num_nodes=5,
        num_angle_rbf=4,
        max_outgoing_neighbors=1,
    )

    assert full["line_edge_index"].shape[1] == 3
    assert capped["line_edge_index"].shape[1] == 1
    assert capped["num_line_edges"] == 1


def test_line_graph_hard_cap() -> None:
    edge_index = torch.tensor(
        [
            [0, 1, 1, 1, 2, 3, 4],
            [1, 2, 3, 4, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    edge_vec = torch.randn(edge_index.shape[1], 3)

    capped = build_line_graph(
        edge_index,
        edge_vec,
        num_nodes=5,
        num_angle_rbf=4,
        max_line_edges=2,
    )

    assert capped["line_edge_index"].shape[1] <= 2
    assert capped["line_edge_attr"].shape[0] <= 2
