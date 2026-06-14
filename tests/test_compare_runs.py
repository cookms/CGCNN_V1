from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from examples import compare_runs


def _write_run(
    path: Path,
    *,
    readout_type: str,
    ib_lambda: float = 0.01,
    val_mae: float = 0.2,
    residuals: list[float],
) -> None:
    path.mkdir(parents=True)
    config = {
        "model": {
            "name": "cgcnn",
            "architecture": "CGCNNModel",
            "config": {
                "edge_input_dim": 16,
                "hidden_dim": 32,
                "num_layers": 2,
                "readout_type": readout_type,
                "ib_lambda": ib_lambda,
                "ib_sigma_slope": 1.5,
                "ib_fixed_point_iters": 3,
                "ib_coupling": "dense",
                "ib_trainable_lambda": True,
                "use_edge_weight": False,
            },
        },
        "data": {"target_column": "target"},
        "featurization": {"num_rbf": 16, "distance_basis": "gaussian"},
        "graph": {"cutoff": 5.0, "neighbor_strategy": "cutoff"},
        "split": {"seed": 42},
        "training": {"batch_size": 4, "epochs": 2, "lr": 0.001, "weight_decay": 1e-5},
    }
    (path / "experiment_config.json").write_text(json.dumps(config))
    history = {
        "history": [
            {"epoch": 1, "train_loss": 0.5, "val_mae": val_mae + 0.1},
            {
                "epoch": 2,
                "train_loss": 0.4,
                "val_mae": val_mae,
                "val_rmse": val_mae * 2,
                "train_samples_per_second": 12.5,
                "elapsed_seconds": 9.0,
            },
        ]
    }
    (path / "training_history.json").write_text(json.dumps(history))
    with (path / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["material_id", "y_true", "y_pred", "residual"])
        writer.writeheader()
        for idx, residual in enumerate(residuals):
            writer.writerow(
                {
                    "material_id": f"m{idx}",
                    "y_true": idx,
                    "y_pred": idx + residual,
                    "residual": residual,
                }
            )


def test_run_row_flattens_readout_config_and_test_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "ib_run"
    _write_run(run_dir, readout_type="implicit_bias", ib_lambda=0.02, val_mae=0.12, residuals=[0.1, -0.3])

    row = compare_runs._run_row(run_dir)

    assert row["run_name"] == "ib_run"
    assert row["readout_type"] == "implicit_bias"
    assert row["ib_lambda"] == 0.02
    assert row["best_epoch"] == 2
    assert row["best_val_mae"] == 0.12
    assert row["test_n"] == 2
    assert row["test_mae"] == 0.2
    assert row["test_rmse"] == (0.05) ** 0.5


def test_mlp_rows_hide_unused_implicit_bias_defaults(tmp_path: Path) -> None:
    run_dir = tmp_path / "mlp_run"
    _write_run(run_dir, readout_type="mlp", residuals=[0.1])

    row = compare_runs._run_row(run_dir)

    assert row["readout_type"] == "mlp"
    assert row["ib_lambda"] is None
    assert row["ib_coupling"] is None


def test_compare_runs_cli_writes_sorted_csv(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "runs"
    _write_run(root / "worse", readout_type="mlp", val_mae=0.3, residuals=[0.3])
    _write_run(root / "better", readout_type="implicit_bias", val_mae=0.1, residuals=[0.1])
    output_csv = tmp_path / "comparison.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_runs.py",
            "--root",
            str(root),
            "--output-csv",
            str(output_csv),
            "--top",
            "0",
        ],
    )

    compare_runs.main()

    with output_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["run_name"] for row in rows] == ["better", "worse"]
    assert rows[0]["readout_type"] == "implicit_bias"
