from __future__ import annotations

import torch

from materials_gnn.featurization.line_graph import build_line_graph
from materials_gnn.models import ALIGNNLikeModel, CGCNNModel, GatedGraphConv


def toy_graph(num_rbf: int = 16, num_angle_rbf: int = 8) -> dict[str, torch.Tensor | int]:
    z = torch.tensor([14, 14, 8], dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 0, 2],
            [1, 0, 2, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    edge_vec = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    distance = torch.linalg.norm(edge_vec, dim=1)
    edge_attr = torch.randn(edge_index.shape[1], num_rbf)
    graph: dict[str, torch.Tensor | int] = {
        "z": z,
        "edge_index": edge_index,
        "edge_vec": edge_vec,
        "distance": distance,
        "edge_attr": edge_attr,
        "num_nodes": 3,
        "batch": torch.zeros(3, dtype=torch.long),
    }
    graph.update(build_line_graph(edge_index, edge_vec, num_nodes=3, num_angle_rbf=num_angle_rbf))
    return graph


def test_gated_graph_conv_shapes() -> None:
    conv = GatedGraphConv(node_dim=12, edge_dim=6)
    x = torch.randn(4, 12)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, 6)

    x_out, e_out = conv(x, edge_index, edge_attr)

    assert x_out.shape == x.shape
    assert e_out.shape == edge_attr.shape


def test_cgcnn_forward_shape() -> None:
    graph = toy_graph()
    model = CGCNNModel(edge_input_dim=16, hidden_dim=32, num_layers=2)

    out = model(graph)

    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_alignn_like_forward_shape() -> None:
    graph = toy_graph()
    model = ALIGNNLikeModel(edge_input_dim=16, angle_input_dim=8, hidden_dim=32, num_layers=2)

    out = model(graph)

    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_cgcnn_forward_with_external_atom_attr_and_learnable_basis() -> None:
    graph = toy_graph()
    graph["atom_attr"] = torch.randn(3, 5)
    model = CGCNNModel(
        edge_input_dim=16,
        hidden_dim=32,
        num_layers=2,
        atom_input_dim=5,
        distance_basis_type="learnable_gaussian",
        distance_basis_cutoff=3.0,
    )

    out = model(graph)

    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_alignn_forward_with_model_level_distance_and_angle_bases() -> None:
    graph = toy_graph()
    model = ALIGNNLikeModel(
        edge_input_dim=16,
        angle_input_dim=8,
        hidden_dim=32,
        num_layers=2,
        distance_basis_type="learnable_gaussian",
        distance_basis_cutoff=3.0,
        angle_basis_type="learnable_gaussian",
    )

    out = model(graph)

    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_gated_graph_conv_cpu_autocast_dtype_safe() -> None:
    """Aggregation buffers must match autocast message dtype.

    CUDA AMP can produce half-precision MLP outputs even when the input node
    tensor is float32. The layer should not fail at ``index_add_`` when that
    happens. CPU autocast with bfloat16 exercises the same dtype path without
    requiring a GPU in CI.
    """

    conv = GatedGraphConv(node_dim=12, edge_dim=6)
    x = torch.randn(4, 12)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, 6)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        x_out, e_out = conv(x, edge_index, edge_attr)

    assert x_out.shape == x.shape
    assert e_out.shape == edge_attr.shape
    assert torch.isfinite(x_out).all()
    assert torch.isfinite(e_out).all()
