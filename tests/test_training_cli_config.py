from __future__ import annotations

import sys

import torch

from examples import train_alignn_like, train_cgcnn


def test_train_cgcnn_model_config_captures_architecture_and_research_toggles(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_cgcnn.py",
            "--csv",
            "data.csv",
            "--num-rbf",
            "12",
            "--hidden-dim",
            "24",
            "--num-layers",
            "4",
            "--learnable-distance-basis",
            "--use-edge-weight",
            "--readout-type",
            "implicit_bias",
            "--ib-lambda",
            "0.03",
            "--ib-sigma-slope",
            "1.7",
            "--ib-fixed-point-iters",
            "5",
            "--ib-coupling",
            "dense",
            "--ib-trainable-lambda",
        ],
    )

    args = train_cgcnn.parse_args()
    config = train_cgcnn._model_config_from_args(args)

    assert config["edge_input_dim"] == 12
    assert config["hidden_dim"] == 24
    assert config["num_layers"] == 4
    assert config["distance_basis_type"] == "learnable_gaussian"
    assert config["use_edge_weight"] is True
    assert config["readout_type"] == "implicit_bias"
    assert config["ib_lambda"] == 0.03
    assert config["ib_sigma_slope"] == 1.7
    assert config["ib_fixed_point_iters"] == 5
    assert config["ib_coupling"] == "dense"
    assert config["ib_trainable_lambda"] is True

    inference_config = train_cgcnn._inference_config_from_args(args)
    assert inference_config["include_line_graph"] is False
    assert inference_config["graph_kwargs"]["num_rbf"] == 12


def test_train_alignn_like_model_config_captures_architecture_and_research_toggles(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_alignn_like.py",
            "--csv",
            "data.csv",
            "--num-rbf",
            "12",
            "--num-angle-rbf",
            "6",
            "--hidden-dim",
            "24",
            "--num-layers",
            "4",
            "--learnable-distance-basis",
            "--learnable-angle-basis",
            "--angle-basis-use-cosine",
            "--use-edge-weight",
            "--readout-type",
            "implicit_bias",
            "--ib-lambda",
            "0.03",
            "--ib-sigma-slope",
            "1.7",
            "--ib-fixed-point-iters",
            "5",
            "--ib-coupling",
            "dense",
            "--ib-trainable-lambda",
        ],
    )

    args = train_alignn_like.parse_args()
    config = train_alignn_like._model_config_from_args(args)

    assert config["edge_input_dim"] == 12
    assert config["angle_input_dim"] == 6
    assert config["hidden_dim"] == 24
    assert config["num_layers"] == 4
    assert config["distance_basis_type"] == "learnable_gaussian"
    assert config["angle_basis_type"] == "learnable_gaussian"
    assert config["angle_basis_use_cosine"] is True
    assert config["use_edge_weight"] is True
    assert config["readout_type"] == "implicit_bias"
    assert config["ib_lambda"] == 0.03
    assert config["ib_sigma_slope"] == 1.7
    assert config["ib_fixed_point_iters"] == 5
    assert config["ib_coupling"] == "dense"
    assert config["ib_trainable_lambda"] is True

    inference_config = train_alignn_like._inference_config_from_args(args)
    assert inference_config["include_line_graph"] is True
    assert inference_config["graph_kwargs"]["num_rbf"] == 12
    assert inference_config["line_graph_kwargs"]["num_angle_rbf"] == 6
    assert inference_config["line_graph_kwargs"]["use_cosine_basis"] is True


def test_checkpoint_metadata_preserves_full_experiment_configs(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_cgcnn.py",
            "--csv",
            "data.csv",
            "--target",
            "band_gap",
            "--use-edge-weight",
            "--readout-type",
            "implicit_bias",
        ],
    )
    args = train_cgcnn.parse_args()

    experiment_config = train_cgcnn._experiment_config_from_args(
        args,
        device=torch.device("cpu"),
        split_indices=([0, 1], [2], [3]),
        normalizer=None,
    )
    metadata = train_cgcnn._checkpoint_metadata_from_experiment_config(experiment_config)

    assert metadata["model_config"] == experiment_config["model"]["config"]
    assert metadata["inference_config"] == experiment_config["inference_config"]
    assert metadata["training_config"]["target"] == "band_gap"
    assert metadata["experiment_config"] == experiment_config
    assert metadata["model_config"]["use_edge_weight"] is True
    assert metadata["model_config"]["readout_type"] == "implicit_bias"
