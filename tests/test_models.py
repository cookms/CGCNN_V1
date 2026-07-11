from __future__ import annotations

import pytest
import torch

from materials_gnn.featurization.line_graph import build_line_graph
from materials_gnn.models import (
    ALIGNNLikeModel,
    CGCNNModel,
    GatedGraphConv,
    ImplicitBiasActivation,
    ImplicitBiasMLPReadout,
)


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


def test_gated_graph_conv_accepts_1d_edge_weight() -> None:
    conv = GatedGraphConv(node_dim=12, edge_dim=6)
    x = torch.randn(4, 12)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, 6)
    edge_weight = torch.tensor([0.5, 1.0, 1.5])

    x_out, e_out = conv(x, edge_index, edge_attr, edge_weight=edge_weight)

    assert x_out.shape == x.shape
    assert e_out.shape == edge_attr.shape
    assert torch.isfinite(x_out).all()
    assert torch.isfinite(e_out).all()
    assert torch.isfinite(x_out).all()
    assert torch.isfinite(e_out).all()


def _has_implicit_bias_activation(module: torch.nn.Module) -> bool:
    return any(isinstance(child, ImplicitBiasActivation) for child in module.modules())


@pytest.mark.parametrize(
    ("conv_ib_targets", "expected_targets"),
    [
        (("edge",), {"edge"}),
        (("message",), {"message"}),
        (("node",), {"node"}),
        (("all",), {"edge", "message", "node"}),
    ],
    ids=["edge", "message", "node", "all"],
)
def test_gated_graph_conv_with_implicit_bias_activation_targets(
    conv_ib_targets: tuple[str, ...],
    expected_targets: set[str],
) -> None:
    conv = GatedGraphConv(
        node_dim=12,
        edge_dim=6,
        conv_activation_type="implicit_bias",
        conv_ib_targets=conv_ib_targets,
        conv_ib_fixed_point_iters=2,
    )
    x = torch.randn(4, 12)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, 6)

    x_out, e_out = conv(x, edge_index, edge_attr)

    assert x_out.shape == x.shape
    assert e_out.shape == edge_attr.shape
    assert torch.isfinite(x_out).all()
    assert torch.isfinite(e_out).all()
    assert _has_implicit_bias_activation(conv.edge_mlp) is ("edge" in expected_targets)
    assert _has_implicit_bias_activation(conv.message_mlp) is ("message" in expected_targets)
    assert _has_implicit_bias_activation(conv.node_mlp) is ("node" in expected_targets)
    assert not _has_implicit_bias_activation(conv.gate_mlp)


def test_gated_graph_conv_implicit_bias_node_activation_handles_edgeless_graph() -> None:
    conv = GatedGraphConv(
        node_dim=12,
        edge_dim=6,
        conv_activation_type="implicit_bias",
        conv_ib_targets=("node",),
        conv_ib_fixed_point_iters=2,
    )
    x = torch.randn(4, 12)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 6))

    x_out, e_out = conv(x, edge_index, edge_attr)

    assert x_out.shape == x.shape
    assert e_out.shape == edge_attr.shape
    assert torch.isfinite(x_out).all()
    assert torch.isfinite(e_out).all()


def test_gated_graph_conv_implicit_bias_gradients_flow() -> None:
    conv = GatedGraphConv(
        node_dim=12,
        edge_dim=6,
        conv_activation_type="implicit_bias",
        conv_ib_targets=("all",),
        conv_ib_fixed_point_iters=2,
    )
    x = torch.randn(4, 12, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, 6, requires_grad=True)

    x_out, e_out = conv(x, edge_index, edge_attr)
    loss = x_out.pow(2).mean() + e_out.pow(2).mean()
    loss.backward()

    finite_nonzero_grads = [
        param.grad
        for param in conv.parameters()
        if param.grad is not None
        and torch.isfinite(param.grad).all()
        and param.grad.detach().abs().sum() > 0
    ]
    assert finite_nonzero_grads


def test_gated_graph_conv_accepts_column_edge_weight() -> None:
    conv = GatedGraphConv(node_dim=12, edge_dim=6)
    x = torch.randn(4, 12)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, 6)
    edge_weight = torch.tensor([[0.5], [1.0], [1.5]])

    x_out, e_out = conv(x, edge_index, edge_attr, edge_weight=edge_weight)

    assert x_out.shape == x.shape
    assert e_out.shape == edge_attr.shape


def test_gated_graph_conv_rejects_mismatched_edge_weight_length() -> None:
    conv = GatedGraphConv(node_dim=12, edge_dim=6)
    x = torch.randn(4, 12)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, 6)
    edge_weight = torch.ones(2)

    with pytest.raises(ValueError, match="edge_weight length must match num_edges"):
        conv(x, edge_index, edge_attr, edge_weight=edge_weight)


def test_gated_graph_conv_ones_edge_weight_matches_unweighted() -> None:
    torch.manual_seed(7)
    conv = GatedGraphConv(node_dim=12, edge_dim=6)
    conv.eval()
    x = torch.randn(4, 12)
    edge_index = torch.tensor([[0, 1, 2, 3], [2, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.randn(4, 6)

    x_unweighted, e_unweighted = conv(x, edge_index, edge_attr)
    x_weighted, e_weighted = conv(x, edge_index, edge_attr, edge_weight=torch.ones(4))

    assert torch.allclose(x_weighted, x_unweighted, atol=1e-6, rtol=1e-6)
    assert torch.allclose(e_weighted, e_unweighted, atol=1e-6, rtol=1e-6)


def test_gated_graph_conv_nonuniform_edge_weight_changes_output() -> None:
    torch.manual_seed(8)
    conv = GatedGraphConv(node_dim=12, edge_dim=6)
    conv.eval()
    x = torch.randn(4, 12)
    edge_index = torch.tensor([[0, 1, 2, 3], [2, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.randn(4, 6)

    x_unweighted, _ = conv(x, edge_index, edge_attr)
    x_weighted, _ = conv(
        x,
        edge_index,
        edge_attr,
        edge_weight=torch.tensor([0.1, 2.0, 1.0, 1.0]),
    )

    assert not torch.allclose(x_weighted, x_unweighted, atol=1e-6, rtol=1e-6)


def test_cgcnn_forward_shape() -> None:
    graph = toy_graph()
    model = CGCNNModel(edge_input_dim=16, hidden_dim=32, num_layers=2)

    out = model(graph)

    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_cgcnn_forward_with_convolution_implicit_bias() -> None:
    graph = toy_graph()
    model = CGCNNModel(
        edge_input_dim=16,
        hidden_dim=32,
        num_layers=2,
        conv_activation_type="implicit_bias",
        conv_ib_targets=("all",),
        conv_ib_fixed_point_iters=2,
    )

    out = model(graph)

    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_cgcnn_ignores_edge_weight_when_disabled() -> None:
    torch.manual_seed(10)
    graph = toy_graph()
    graph["edge_weight"] = torch.ones(6)
    model = CGCNNModel(edge_input_dim=16, hidden_dim=32, num_layers=2, use_edge_weight=False)
    model.eval()

    out_ones = model(graph)
    graph["edge_weight"] = torch.tensor([0.1, 2.0, 0.5, 3.0, 1.0, 0.25])
    out_nonuniform = model(graph)

    assert torch.allclose(out_nonuniform, out_ones, atol=1e-6, rtol=1e-6)


def test_cgcnn_uses_edge_weight_when_enabled() -> None:
    torch.manual_seed(11)
    graph = toy_graph()
    graph["edge_weight"] = torch.ones(6)
    model = CGCNNModel(edge_input_dim=16, hidden_dim=32, num_layers=2, use_edge_weight=True)
    model.eval()

    out_ones = model(graph)
    graph["edge_weight"] = torch.tensor([0.1, 2.0, 0.5, 3.0, 1.0, 0.25])
    out_nonuniform = model(graph)

    assert not torch.allclose(out_nonuniform, out_ones, atol=1e-6, rtol=1e-6)


def test_alignn_like_forward_shape() -> None:
    graph = toy_graph()
    model = ALIGNNLikeModel(edge_input_dim=16, angle_input_dim=8, hidden_dim=32, num_layers=2)

    out = model(graph)

    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_alignn_like_forward_with_convolution_implicit_bias() -> None:
    graph = toy_graph()
    model = ALIGNNLikeModel(
        edge_input_dim=16,
        angle_input_dim=8,
        hidden_dim=32,
        num_layers=2,
        conv_activation_type="implicit_bias",
        conv_ib_targets=("all",),
        conv_ib_fixed_point_iters=2,
    )

    out = model(graph)

    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_alignn_like_runs_with_atom_edge_weight() -> None:
    graph = toy_graph()
    graph["edge_weight"] = torch.tensor([0.1, 2.0, 0.5, 3.0, 1.0, 0.25])
    model = ALIGNNLikeModel(
        edge_input_dim=16,
        angle_input_dim=8,
        hidden_dim=32,
        num_layers=2,
        use_edge_weight=True,
    )

    out = model(graph)

    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_alignn_like_edge_weight_does_not_require_line_edge_weight() -> None:
    graph = toy_graph()
    graph["edge_weight"] = torch.ones(6)
    graph.pop("line_edge_weight", None)
    model = ALIGNNLikeModel(
        edge_input_dim=16,
        angle_input_dim=8,
        hidden_dim=32,
        num_layers=2,
        use_edge_weight=True,
    )

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


def test_implicit_bias_activation_preserves_shape_and_is_finite() -> None:
    activation = ImplicitBiasActivation(dim=5, ib_lambda=0.01, fixed_point_iters=3)
    y = torch.randn(2, 3, 5)

    out = activation(y)

    assert out.shape == y.shape
    assert torch.isfinite(out).all()


def test_implicit_bias_activation_gradients_flow() -> None:
    activation = ImplicitBiasActivation(
        dim=4,
        ib_lambda=0.01,
        fixed_point_iters=3,
        coupling="dense",
        trainable_lambda=True,
    )
    y = torch.randn(2, 4, requires_grad=True)

    loss = activation(y).sum()
    loss.backward()

    assert y.grad is not None
    assert torch.isfinite(y.grad).all()
    assert activation.ib_lambda.grad is not None
    assert torch.isfinite(activation.ib_lambda.grad).all()


def test_implicit_bias_activation_zero_lambda_matches_silu() -> None:
    activation = ImplicitBiasActivation(dim=6, ib_lambda=0.0, fixed_point_iters=3)
    y = torch.randn(4, 6)

    out = activation(y)

    assert torch.allclose(out, torch.nn.functional.silu(y), atol=1e-7, rtol=1e-7)


def test_cgcnn_forward_with_implicit_bias_readout() -> None:
    torch.manual_seed(0)
    graph = toy_graph()
    config = {
        "edge_input_dim": 16,
        "hidden_dim": 32,
        "num_layers": 2,
        "readout_type": "implicit_bias",
        "ib_lambda": 0.02,
        "ib_sigma_slope": 1.25,
        "ib_fixed_point_iters": 3,
        "ib_coupling": "dense",
        "ib_trainable_lambda": True,
    }
    model = CGCNNModel(**config)

    out = model(graph)

    assert isinstance(model.readout, ImplicitBiasMLPReadout)
    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_alignn_like_forward_with_implicit_bias_readout() -> None:
    torch.manual_seed(0)
    graph = toy_graph()
    config = {
        "edge_input_dim": 16,
        "angle_input_dim": 8,
        "hidden_dim": 32,
        "num_layers": 2,
        "readout_type": "implicit_bias",
        "ib_lambda": 0.02,
        "ib_sigma_slope": 1.25,
        "ib_fixed_point_iters": 3,
        "ib_coupling": "dense",
        "ib_trainable_lambda": True,
    }
    model = ALIGNNLikeModel(**config)

    out = model(graph)

    assert isinstance(model.readout, ImplicitBiasMLPReadout)
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
