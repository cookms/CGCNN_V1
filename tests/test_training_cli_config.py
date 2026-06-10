from __future__ import annotations

import sys

from examples import train_alignn_like, train_cgcnn


def test_train_cgcnn_use_edge_weight_reaches_model_config(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_cgcnn.py", "--csv", "data.csv", "--use-edge-weight"],
    )

    args = train_cgcnn.parse_args()
    config = train_cgcnn._model_config_from_args(args)

    assert config["use_edge_weight"] is True


def test_train_alignn_like_use_edge_weight_reaches_model_config(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_alignn_like.py", "--csv", "data.csv", "--use-edge-weight"],
    )

    args = train_alignn_like.parse_args()
    config = train_alignn_like._model_config_from_args(args)

    assert config["use_edge_weight"] is True
