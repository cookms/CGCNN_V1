from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from examples import predict_from_cif, train_cgcnn
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


def test_predict_from_cif_loads_implicit_bias_checkpoint_from_metadata_and_fallback_args(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    torch.manual_seed(0)
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
    model = CGCNNModel(**model_config)
    model.eval()
    inference_config = {
        "cutoff": 5.0,
        "neighbor_strategy": "cutoff",
        "neighbor_kwargs": {},
        "rbf_cutoff": None,
        "include_line_graph": False,
        "graph_kwargs": {"num_rbf": 16},
        "line_graph_kwargs": None,
    }
    metadata = {
        "model_name": "cgcnn",
        "model_config": model_config,
        "inference_config": inference_config,
    }
    metadata_checkpoint_path = tmp_path / "implicit_bias_with_metadata.pt"
    legacy_checkpoint_path = tmp_path / "implicit_bias_legacy.pt"
    metadata_payload = _make_checkpoint_payload(
        model,
        epoch=1,
        checkpoint_metadata=metadata,
    )
    torch.save(metadata_payload, metadata_checkpoint_path)
    legacy_payload = dict(metadata_payload)
    legacy_payload.pop("metadata")
    torch.save(legacy_payload, legacy_checkpoint_path)
    graph = toy_graph(num_rbf=16)
    graph["edge_weight"] = torch.ones(6)

    def fake_predict_cif(model, cif_path, **kwargs):
        assert str(cif_path) == "toy.cif"
        assert kwargs["device"] == "cpu"
        assert kwargs["graph_kwargs"]["num_rbf"] == 16
        with torch.no_grad():
            return model(graph)

    monkeypatch.setattr(predict_from_cif, "predict_cif", fake_predict_cif)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict_from_cif.py",
            "--cif",
            "toy.cif",
            "--checkpoint",
            str(metadata_checkpoint_path),
            "--device",
            "cpu",
        ],
    )
    predict_from_cif.main()
    metadata_stdout = capsys.readouterr().out.strip()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict_from_cif.py",
            "--cif",
            "toy.cif",
            "--checkpoint",
            str(legacy_checkpoint_path),
            "--model",
            "cgcnn",
            "--num-rbf",
            "16",
            "--hidden-dim",
            "32",
            "--num-layers",
            "2",
            "--use-edge-weight",
            "--readout-type",
            "implicit_bias",
            "--ib-lambda",
            "0.02",
            "--ib-sigma-slope",
            "1.5",
            "--ib-fixed-point-iters",
            "3",
            "--ib-coupling",
            "dense",
            "--ib-trainable-lambda",
            "--device",
            "cpu",
        ],
    )
    predict_from_cif.main()
    fallback_stdout = capsys.readouterr().out.strip()

    assert torch.isfinite(torch.tensor(float(metadata_stdout)))
    assert fallback_stdout == metadata_stdout
