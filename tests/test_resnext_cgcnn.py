from __future__ import annotations

import io
import sys

import pytest
import torch
from torch.utils.data import DataLoader

from examples import train_cgcnn
from examples.predict_from_cif import _build_model
from materials_gnn.models import (
    AggregatedResidualGraphBlock,
    CGCNNModel,
    ResNeXtCGCNNModel,
    model_parameter_summary,
)
from materials_gnn.training import train_model


def _graph() -> dict[str, torch.Tensor]:
    return {
        "z": torch.tensor([14, 8, 14], dtype=torch.long),
        "edge_index": torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]], dtype=torch.long),
        "edge_attr": torch.randn(4, 8),
        "batch": torch.zeros(3, dtype=torch.long),
    }


def test_single_branch_matches_baseline_after_weight_copy() -> None:
    baseline = CGCNNModel(edge_input_dim=8, hidden_dim=16, num_layers=2)
    variant = ResNeXtCGCNNModel(edge_input_dim=8, hidden_dim=16, num_layers=2, cardinality=1)
    variant.atom_embedding.load_state_dict(baseline.atom_embedding.state_dict())
    variant.bond_embedding.load_state_dict(baseline.bond_embedding.state_dict())
    variant.readout.load_state_dict(baseline.readout.state_dict())
    for old, new in zip(baseline.convs, variant.convs, strict=True):
        new.branches[0].load_state_dict(old.state_dict())
    graph = _graph()
    baseline.eval()
    variant.eval()
    torch.testing.assert_close(variant(graph), baseline(graph), rtol=0, atol=0)
    assert model_parameter_summary(variant)["total"] == model_parameter_summary(baseline)["total"]


@pytest.mark.parametrize("cardinality", [1, 2, 4, 8])
@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_forward_backward_all_branches(cardinality: int, aggregation: str) -> None:
    model = ResNeXtCGCNNModel(
        edge_input_dim=8,
        hidden_dim=16,
        num_layers=1,
        output_dim=3,
        cardinality=cardinality,
        branch_dim=4,
        aggregation=aggregation,
    ).to("cpu")
    output = model(_graph())
    assert output.shape == (1, 3)
    assert torch.isfinite(output).all()
    output.square().sum().backward()
    branches = model.convs[0].branches
    assert len({id(branch.edge_mlp[0].weight) for branch in branches}) == cardinality
    assert all(branch.edge_mlp[0].weight.grad is not None for branch in branches)
    assert all(torch.isfinite(branch.edge_mlp[0].weight.grad).all() for branch in branches)


def test_learned_weighting_and_checkpoint_roundtrip() -> None:
    config = dict(
        edge_input_dim=8, hidden_dim=16, num_layers=1, cardinality=4,
        branch_dim=4, branch_weighting="learned",
    )
    model = ResNeXtCGCNNModel(**config)
    logits = model.convs[0].branch_logits
    assert logits is not None
    torch.testing.assert_close(torch.softmax(logits, dim=0).sum(), torch.tensor(1.0))
    graph = _graph()
    model(graph).square().sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    buffer = io.BytesIO()
    torch.save({"metadata": {"model_name": "resnext_cgcnn", "model_config": config},
                "model_state_dict": model.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=False)
    restored = _build_model(
        checkpoint["metadata"]["model_name"], checkpoint["metadata"]["model_config"]
    )
    restored.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    restored.eval()
    torch.testing.assert_close(restored(graph), model(graph))


def test_block_validation() -> None:
    with pytest.raises(ValueError, match="cardinality"):
        AggregatedResidualGraphBlock(16, cardinality=0)
    with pytest.raises(ValueError, match="branch_dim"):
        AggregatedResidualGraphBlock(16, branch_dim=0)


def test_training_cli_records_resnext_configuration(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "train_cgcnn.py", "--csv", "id_prop.csv", "--model", "resnext_cgcnn",
        "--cardinality", "4", "--branch-dim", "32", "--aggregation", "sum",
        "--branch-weighting", "learned",
    ])
    args = train_cgcnn.parse_args()
    assert args.output_dir == "runs/resnext_cgcnn_C4_B32_sum_learned"
    config = train_cgcnn._model_config_from_args(args)
    assert (config["cardinality"], config["branch_dim"]) == (4, 32)
    assert (config["aggregation"], config["branch_weighting"]) == ("sum", "learned")
    summary = model_parameter_summary(ResNeXtCGCNNModel(**config))
    experiment = train_cgcnn._experiment_config_from_args(
        args, device=torch.device("cpu"), split_indices=([0], [1], [2]),
        normalizer=None, parameter_summary=summary,
    )
    metadata = train_cgcnn._checkpoint_metadata_from_experiment_config(experiment)
    assert metadata["model_name"] == "resnext_cgcnn"
    assert metadata["model_config"] == config
    assert metadata["experiment_config"]["model"]["parameter_summary"] == summary


def test_model_moves_branches_and_logits_to_device() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNeXtCGCNNModel(
        edge_input_dim=8, hidden_dim=16, num_layers=1,
        cardinality=2, branch_weighting="learned",
    ).to(device)
    graph = {key: value.to(device) for key, value in _graph().items()}
    assert model.convs[0].branch_logits.device == device
    assert all(next(branch.parameters()).device == device for branch in model.convs[0].branches)
    assert model(graph).device == device


@pytest.mark.parametrize("cardinality", [0, 1, 4])
def test_training_checkpoint_and_inference_smoke(tmp_path, cardinality: int) -> None:
    config = dict(edge_input_dim=8, hidden_dim=16, num_layers=1)
    name = "cgcnn" if cardinality == 0 else "resnext_cgcnn"
    model = (
        CGCNNModel(**config) if cardinality == 0 else
        ResNeXtCGCNNModel(**config, cardinality=cardinality, branch_dim=16 // cardinality)
    )
    graph = _graph()
    graph["y"] = torch.tensor([0.5])
    loader = DataLoader([graph], batch_size=1, collate_fn=lambda rows: rows[0])
    path = tmp_path / "final.pt"
    model_config = dict(config)
    if cardinality:
        model_config.update(cardinality=cardinality, branch_dim=16 // cardinality)
    history = train_model(
        model, loader, epochs=1, device="cpu", verbose=False,
        final_checkpoint_path=path,
        checkpoint_metadata={"model_name": name, "model_config": model_config},
    )
    assert torch.isfinite(torch.tensor(history[0]["train_loss"]))
    checkpoint = torch.load(path, weights_only=False)
    restored = _build_model(
        checkpoint["metadata"]["model_name"], checkpoint["metadata"]["model_config"]
    )
    restored.load_state_dict(checkpoint["model_state_dict"])
    assert torch.isfinite(restored(graph)).all()
