from __future__ import annotations

import json
from pathlib import Path

import torch

from examples import train_cgcnn
from materials_gnn.data.transforms import TargetNormalizer
from materials_gnn.models import CGCNNModel
from materials_gnn.training.trainer import _make_checkpoint_payload
from materials_gnn.utils import write_experiment_config
from tests.test_models import toy_graph


def test_write_experiment_config_handles_paths_and_tensors(tmp_path: Path) -> None:
    path = write_experiment_config(
        {
            "path": tmp_path / "data.csv",
            "scalar": torch.tensor(1.5),
            "vector": torch.tensor([1, 2, 3]),
        },
        tmp_path / "experiment_config.json",
    )

    payload = json.loads(path.read_text())
    assert payload["path"].endswith("data.csv")
    assert payload["scalar"] == 1.5
    assert payload["vector"] == [1, 2, 3]


def test_cgcnn_checkpoint_metadata_round_trips_model_for_inference(tmp_path: Path) -> None:
    torch.manual_seed(0)
    graph = toy_graph()
    model_config = {
        "edge_input_dim": 16,
        "hidden_dim": 32,
        "num_layers": 2,
        "readout_type": "implicit_bias",
        "ib_lambda": 0.02,
        "ib_sigma_slope": 1.5,
        "ib_fixed_point_iters": 3,
        "ib_coupling": "dense",
        "ib_trainable_lambda": True,
        "use_edge_weight": True,
    }
    graph["edge_weight"] = torch.tensor([0.2, 0.8, 1.0, 1.5, 0.5, 1.2])
    model = CGCNNModel(**model_config)
    model.eval()

    with torch.no_grad():
        pred_before = model(graph)

    experiment_config = {
        "package": {"version": "test"},
        "model": {"name": "cgcnn", "architecture": "CGCNNModel", "config": model_config},
        "training": {"cli_args": {"target": "formation_energy_per_atom"}},
        "inference_config": {
            "cutoff": 5.0,
            "neighbor_strategy": "cutoff",
            "neighbor_kwargs": {},
            "rbf_cutoff": None,
            "include_line_graph": False,
            "graph_kwargs": {"num_rbf": 16},
            "line_graph_kwargs": None,
        },
    }
    metadata = train_cgcnn._checkpoint_metadata_from_experiment_config(experiment_config)
    checkpoint_path = tmp_path / "cgcnn_checkpoint.pt"
    torch.save(
        _make_checkpoint_payload(
            model,
            epoch=7,
            target_normalizer=TargetNormalizer(mean=1.5, std=0.25),
            checkpoint_metadata=metadata,
            history=[{"epoch": 7, "train_loss": 0.1}],
        ),
        checkpoint_path,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    loaded_metadata = checkpoint["metadata"]
    reconstructed = CGCNNModel(**loaded_metadata["model_config"])
    reconstructed.load_state_dict(checkpoint["model_state_dict"])
    reconstructed.eval()

    with torch.no_grad():
        pred_after = reconstructed(graph)

    assert checkpoint["checkpoint_schema_version"] == 2
    assert loaded_metadata["model_name"] == "cgcnn"
    assert loaded_metadata["model_config"]["use_edge_weight"] is True
    assert loaded_metadata["model_config"]["readout_type"] == "implicit_bias"
    assert loaded_metadata["model_config"]["ib_coupling"] == "dense"
    assert loaded_metadata["inference_config"]["graph_kwargs"]["num_rbf"] == 16
    assert loaded_metadata["experiment_config"]["model"]["config"] == model_config
    assert checkpoint["target_normalizer"] == {"mean": 1.5, "std": 0.25, "eps": 1e-12}
    assert pred_after.shape == (1,)
    assert torch.isfinite(pred_after).all()
    assert torch.allclose(pred_after, pred_before, atol=1e-6, rtol=1e-6)
