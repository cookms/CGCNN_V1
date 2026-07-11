#!/usr/bin/env python
"""Compare completed training run directories.

The training examples write each run as a small artifact directory containing
``experiment_config.json``, ``training_history.json``, and ``test_predictions.csv``.
This script flattens those artifacts into one comparison table, with enough model,
readout, graph, validation, test, and timing fields to compare sweeps such as MLP vs
implicit-bias readouts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_COLUMNS = [
    "rank",
    "run_name",
    "model_name",
    "readout_type",
    "ib_lambda",
    "ib_sigma_slope",
    "ib_fixed_point_iters",
    "ib_coupling",
    "ib_trainable_lambda",
    "use_edge_weight",
    "hidden_dim",
    "num_layers",
    "edge_input_dim",
    "angle_input_dim",
    "num_rbf",
    "num_angle_rbf",
    "cutoff",
    "neighbor_strategy",
    "distance_basis",
    "learnable_distance_basis",
    "angle_basis",
    "learnable_angle_basis",
    "target",
    "seed",
    "batch_size",
    "epochs_requested",
    "lr",
    "weight_decay",
    "best_epoch",
    "best_val_mae",
    "best_val_rmse",
    "best_val_r2",
    "best_val_loss",
    "final_epoch",
    "final_train_loss",
    "final_val_mae",
    "train_samples_per_second_best",
    "train_samples_per_second_final",
    "total_elapsed_seconds",
    "test_n",
    "test_mae",
    "test_rmse",
    "test_r2",
    "status",
    "notes",
    "run_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare materials-gnn training runs")
    parser.add_argument(
        "runs",
        nargs="*",
        type=Path,
        help="Run directories to compare. If omitted, discover runs under --root.",
    )
    parser.add_argument("--root", type=Path, default=Path("runs"), help="Root directory for run discovery")
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Glob pattern under --root for discovered runs. Can be repeated. Default: *",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Use recursive globbing when discovering runs under --root",
    )
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional comparison CSV output path")
    parser.add_argument(
        "--sort-by",
        default="auto",
        help="Column to sort by, or 'auto' for test_mae then best_val_mae",
    )
    parser.add_argument("--descending", action="store_true", help="Sort largest values first")
    parser.add_argument("--top", type=int, default=20, help="Number of rows to print in the console table; use 0 for all")
    parser.add_argument(
        "--group-by",
        default=None,
        help="Optional column for a compact group summary, for example readout_type or model_name",
    )
    parser.add_argument("--summary-csv", type=Path, default=None, help="Optional grouped summary CSV output path")
    return parser.parse_args()


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _get(mapping: Any, path: str, default: Any = None) -> Any:
    value = mapping
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _history_records(run_dir: Path) -> list[dict[str, Any]]:
    payload = _load_json(run_dir / "training_history.json")
    if isinstance(payload, dict):
        records = payload.get("history", [])
    else:
        records = payload
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def _best_history_record(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [record for record in history if _as_float(record.get("val_mae")) is not None]
    if candidates:
        return min(candidates, key=lambda record: float(record["val_mae"]))

    candidates = [record for record in history if _as_float(record.get("val_loss")) is not None]
    if candidates:
        return min(candidates, key=lambda record: float(record["val_loss"]))

    return history[-1] if history else None


def _prediction_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"test_n": 0}

    residuals: list[float] = []
    y_true: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            residual = _as_float(row.get("residual"))
            true_value = _as_float(row.get("y_true"))
            pred_value = _as_float(row.get("y_pred"))
            if residual is None and true_value is not None and pred_value is not None:
                residual = pred_value - true_value
            if residual is None:
                continue
            residuals.append(residual)
            if true_value is not None:
                y_true.append(true_value)

    if not residuals:
        return {"test_n": 0}

    n = len(residuals)
    mae = sum(abs(value) for value in residuals) / n
    rmse = math.sqrt(sum(value * value for value in residuals) / n)
    result: dict[str, Any] = {"test_n": n, "test_mae": mae, "test_rmse": rmse}

    if len(y_true) == n:
        mean_true = sum(y_true) / n
        sst = sum((value - mean_true) ** 2 for value in y_true)
        if sst > 0:
            sse = sum(value * value for value in residuals)
            result["test_r2"] = 1.0 - sse / sst
    return result


def _discover_runs(args: argparse.Namespace) -> list[Path]:
    if args.runs:
        return sorted({path.resolve() for path in args.runs})

    patterns = args.pattern or ["*"]
    discovered: set[Path] = set()
    for pattern in patterns:
        iterator = args.root.rglob(pattern) if args.recursive else args.root.glob(pattern)
        for path in iterator:
            if path.is_dir() and _looks_like_run_dir(path):
                discovered.add(path.resolve())
    return sorted(discovered)


def _looks_like_run_dir(path: Path) -> bool:
    return any(
        (path / artifact).exists()
        for artifact in ["experiment_config.json", "training_history.json", "test_predictions.csv"]
    )


def _run_row(run_dir: Path) -> dict[str, Any]:
    config = _load_json(run_dir / "experiment_config.json") or {}
    history = _history_records(run_dir)
    best = _best_history_record(history) or {}
    final = history[-1] if history else {}
    model_config = _get(config, "model.config", {}) or {}
    training_config = _get(config, "training", {}) or {}
    featurization = _get(config, "featurization", {}) or {}
    graph = _get(config, "graph", {}) or {}

    row: dict[str, Any] = {
        "run_name": run_dir.name,
        "run_path": str(run_dir),
        "model_name": _get(config, "model.name"),
        "readout_type": model_config.get("readout_type", "mlp"),
        "ib_lambda": model_config.get("ib_lambda"),
        "ib_sigma_slope": model_config.get("ib_sigma_slope"),
        "ib_fixed_point_iters": model_config.get("ib_fixed_point_iters"),
        "ib_coupling": model_config.get("ib_coupling"),
        "ib_trainable_lambda": model_config.get("ib_trainable_lambda"),
        "use_edge_weight": model_config.get("use_edge_weight"),
        "hidden_dim": model_config.get("hidden_dim"),
        "num_layers": model_config.get("num_layers"),
        "edge_input_dim": model_config.get("edge_input_dim"),
        "angle_input_dim": model_config.get("angle_input_dim"),
        "num_rbf": featurization.get("num_rbf") or _get(config, "inference_config.graph_kwargs.num_rbf"),
        "num_angle_rbf": featurization.get("num_angle_rbf")
        or _get(config, "inference_config.line_graph_kwargs.num_angle_rbf"),
        "cutoff": graph.get("cutoff") or _get(config, "inference_config.cutoff"),
        "neighbor_strategy": graph.get("neighbor_strategy") or _get(config, "inference_config.neighbor_strategy"),
        "distance_basis": featurization.get("distance_basis"),
        "learnable_distance_basis": featurization.get("learnable_distance_basis"),
        "angle_basis": featurization.get("angle_basis"),
        "learnable_angle_basis": featurization.get("learnable_angle_basis"),
        "target": _get(config, "data.target_column") or _get(training_config, "cli_args.target"),
        "seed": _get(config, "split.seed") or _get(training_config, "cli_args.seed"),
        "batch_size": training_config.get("batch_size") or _get(training_config, "cli_args.batch_size"),
        "epochs_requested": training_config.get("epochs") or _get(training_config, "cli_args.epochs"),
        "lr": training_config.get("lr") or _get(training_config, "cli_args.lr"),
        "weight_decay": training_config.get("weight_decay") or _get(training_config, "cli_args.weight_decay"),
        "best_epoch": best.get("epoch"),
        "best_val_mae": best.get("val_mae"),
        "best_val_rmse": best.get("val_rmse"),
        "best_val_r2": best.get("val_r2"),
        "best_val_loss": best.get("val_loss"),
        "final_epoch": final.get("epoch"),
        "final_train_loss": final.get("train_loss"),
        "final_val_mae": final.get("val_mae"),
        "train_samples_per_second_best": best.get("train_samples_per_second"),
        "train_samples_per_second_final": final.get("train_samples_per_second"),
        "total_elapsed_seconds": final.get("elapsed_seconds"),
        "status": "ok",
        "notes": "",
    }
    row.update(_prediction_metrics(run_dir / "test_predictions.csv"))

    notes = []
    if not config:
        notes.append("missing experiment_config.json")
    if not history:
        notes.append("missing training_history.json")
    if not (run_dir / "test_predictions.csv").exists():
        notes.append("missing test_predictions.csv")
    if notes:
        row["status"] = "partial"
        row["notes"] = "; ".join(notes)
    if row["readout_type"] != "implicit_bias":
        for key in [
            "ib_lambda",
            "ib_sigma_slope",
            "ib_fixed_point_iters",
            "ib_coupling",
            "ib_trainable_lambda",
        ]:
            row[key] = None
    return row


def _sort_key(row: dict[str, Any], column: str) -> tuple[int, float | str]:
    value = row.get(column)
    numeric = _as_float(value)
    if numeric is not None:
        return (0, numeric)
    if value is None or value == "":
        return (1, "")
    return (0, str(value))


def _sort_rows(rows: list[dict[str, Any]], sort_by: str, *, descending: bool) -> list[dict[str, Any]]:
    if sort_by == "auto":
        sort_by = "test_mae" if any(_as_float(row.get("test_mae")) is not None for row in rows) else "best_val_mae"
    return sorted(rows, key=lambda row: _sort_key(row, sort_by), reverse=descending)


def _format_value(value: Any) -> str:
    number = _as_float(value)
    if number is not None:
        return f"{number:.6g}"
    if value is None:
        return ""
    return str(value)


def _print_table(rows: list[dict[str, Any]], *, top: int, columns: list[str] | None = None) -> None:
    if not rows:
        print("No run directories found.")
        return

    columns = columns or [
        "rank",
        "run_name",
        "model_name",
        "readout_type",
        "ib_lambda",
        "ib_sigma_slope",
        "best_val_mae",
        "test_mae",
        "test_rmse",
        "test_r2",
        "train_samples_per_second_final",
    ]
    visible = rows[: max(top, 0)] if top else rows
    widths = {
        column: max(len(column), *(len(_format_value(row.get(column))) for row in visible))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in visible:
        print("  ".join(_format_value(row.get(column)).ljust(widths[column]) for column in columns))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _group_summary(rows: list[dict[str, Any]], column: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_format_value(row.get(column)) or "<missing>"].append(row)

    summary: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        test_maes = [_as_float(row.get("test_mae")) for row in group_rows]
        test_maes = [value for value in test_maes if value is not None]
        val_maes = [_as_float(row.get("best_val_mae")) for row in group_rows]
        val_maes = [value for value in val_maes if value is not None]
        summary.append(
            {
                column: key,
                "num_runs": len(group_rows),
                "best_test_mae": min(test_maes) if test_maes else None,
                "mean_test_mae": sum(test_maes) / len(test_maes) if test_maes else None,
                "best_val_mae": min(val_maes) if val_maes else None,
                "mean_val_mae": sum(val_maes) / len(val_maes) if val_maes else None,
                "best_run": group_rows[0].get("run_name") if group_rows else None,
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    run_dirs = _discover_runs(args)
    rows = [_run_row(run_dir) for run_dir in run_dirs]
    rows = _sort_rows(rows, args.sort_by, descending=args.descending)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    _print_table(rows, top=args.top)

    if args.output_csv is not None:
        _write_csv(args.output_csv, rows, DEFAULT_COLUMNS)
        print(f"\nwrote comparison CSV: {args.output_csv}")

    if args.group_by is not None:
        print(f"\nGrouped by {args.group_by}:")
        summary = _group_summary(rows, args.group_by)
        summary_columns = [
            args.group_by,
            "num_runs",
            "best_test_mae",
            "mean_test_mae",
            "best_val_mae",
            "mean_val_mae",
            "best_run",
        ]
        _print_table(summary, top=0, columns=summary_columns)
        if args.summary_csv is not None:
            _write_csv(args.summary_csv, summary, summary_columns)
            print(f"\nwrote summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
