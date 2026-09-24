"""Matbench benchmarking utilities for repository-native crystal GNNs.

The benchmark runner preserves the official Matbench outer folds. Any validation
needed for early stopping or checkpoint selection is carved only from the outer
training portion. Held-out targets are requested only after predictions are frozen.
"""

from __future__ import annotations

import json
import platform
import random
import time
from dataclasses import asdict, dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from materials_gnn.data.datasets import collate_graphs
from materials_gnn.data.matbench import MatbenchStructureDataset, graph_diagnostics
from materials_gnn.data.transforms import TargetNormalizer
from materials_gnn.models import ALIGNNLikeModel, CGCNNModel
from materials_gnn.models.resnext_cgcnn import ResNeXtCGCNNModel
from materials_gnn.training.device import dataloader_device_kwargs, resolve_device
from materials_gnn.training.early_stopping import EarlyStopping
from materials_gnn.training.metrics import mae, r2_score, rmse
from materials_gnn.training.trainer import move_batch_to_device, train_model


@dataclass(frozen=True)
class BenchmarkConfig:
    task_name: str = "matbench_mp_e_form"
    experiment_name: str = "cgcnn_baseline"
    model_name: str = "cgcnn"

    neighbor_strategy: str = "cutoff"
    cutoff: float = 5.0
    neighbor_k: int = 12
    neighbor_max_radius: float | None = 8.0
    rbf_cutoff: float | None = None
    strain_epsilon: float = 0.02
    min_survival_fraction: float = 0.5
    voronoi_failure_policy: str = "raise"
    num_rbf: int = 64
    distance_basis: str = "gaussian"
    learnable_distance_basis: bool = False
    atom_features: str | tuple[str, ...] | None = None
    use_edge_weight: bool = False

    num_angle_rbf: int = 32
    angle_basis: str = "gaussian"
    learnable_angle_basis: bool = False
    angle_basis_use_cosine: bool = False
    max_line_neighbors: int | None = 8
    max_line_edges: int | None = None
    line_neighbor_selection: str = "nearest"

    hidden_dim: int = 128
    num_layers: int = 3
    pooling: str = "mean"
    dropout: float = 0.0
    readout_type: str = "mlp"
    ib_lambda: float = 0.01
    ib_sigma_slope: float = 1.0
    ib_fixed_point_iters: int = 8
    ib_coupling: str = "ring"
    ib_trainable_lambda: bool = False
    conv_activation_type: str = "silu"
    conv_ib_lambda: float = 0.01
    conv_ib_sigma_slope: float = 1.0
    conv_ib_fixed_point_iters: int = 8
    conv_ib_coupling: str = "ring"
    conv_ib_trainable_lambda: bool = False
    conv_ib_targets: tuple[str, ...] = ()

    cardinality: int = 4
    branch_dim: int | None = None
    aggregation: str = "mean"
    branch_weighting: str = "uniform"

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float | None = 5.0

    val_fraction: float = 0.1
    patience: int = 20
    early_stopping_min_delta: float = 1e-4
    early_stopping_warmup: int = 0
    early_stopping_adaptive: bool = False
    early_stopping_factor: float = 0.25
    early_stopping_max_patience: int = 75
    early_stopping_smoothing: int = 1

    seed: int = 42
    device: str = "auto"
    amp: bool = True
    num_workers: int = 0
    matmul_precision: str | None = None

    folds: tuple[int, ...] | None = None
    quick_test: bool = False
    quick_train_samples: int = 256
    quick_test_samples: int = 128
    quick_epochs: int = 3
    overwrite: bool = False

    graph_cache_dir: str = ".cache/matbench_graphs"
    output_dir: str = "runs/matbench"

    def clone(self, **changes: Any) -> "BenchmarkConfig":
        return replace(self, **changes)


def require_matbench():
    try:
        from matbench.bench import MatbenchBenchmark
    except ImportError as exc:
        raise ImportError(
            "Matbench benchmarking is optional. Install it with pip install -e .[matbench]."
        ) from exc
    return MatbenchBenchmark


def discover_structure_regression_tasks() -> pd.DataFrame:
    """Discover compatible structure-regression tasks from Matbench metadata."""
    MatbenchBenchmark = require_matbench()
    mb = MatbenchBenchmark(autoload=False)
    rows = []
    for task in mb.tasks:
        md = dict(task.metadata)
        if md.get("input_type") == "structure" and md.get("task_type") == "regression":
            rows.append(
                {
                    "task_name": task.dataset_name,
                    "target": md.get("target"),
                    "n_samples": md.get("n_samples"),
                    "task_type": md.get("task_type"),
                    "input_type": md.get("input_type"),
                    "unit": md.get("unit"),
                }
            )
    return pd.DataFrame(rows).sort_values("task_name").reset_index(drop=True)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def internal_train_val_indices(
    n: int, *, val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Deterministically split only an outer-fold training set."""
    if n < 2:
        raise ValueError("Need at least two outer-training samples for internal validation")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_val = max(1, min(n - 1, int(round(n * val_fraction))))
    return indices[n_val:].tolist(), indices[:n_val].tolist()


def _neighbor_kwargs(config: BenchmarkConfig) -> dict[str, Any]:
    if config.neighbor_strategy == "knn":
        out: dict[str, Any] = {"k": config.neighbor_k}
        if config.neighbor_max_radius is not None:
            out["max_radius"] = config.neighbor_max_radius
        return out
    if config.neighbor_strategy == "voronoi":
        out = {"failure_policy": config.voronoi_failure_policy}
        if config.neighbor_max_radius is not None:
            out["cutoff"] = config.neighbor_max_radius
        return out
    if config.neighbor_strategy == "adaptive_shell":
        return (
            {"max_radius": config.neighbor_max_radius}
            if config.neighbor_max_radius is not None else {}
        )
    if config.neighbor_strategy == "strain_consensus":
        out = {
            "strain_epsilon": config.strain_epsilon,
            "min_survival_fraction": config.min_survival_fraction,
        }
        if config.neighbor_max_radius is not None:
            out["cutoff"] = config.neighbor_max_radius
        return out
    return {}


def _resolved_rbf_cutoff(config: BenchmarkConfig) -> float | None:
    if config.rbf_cutoff is not None:
        return config.rbf_cutoff
    if config.neighbor_strategy in {"knn", "voronoi", "adaptive_shell", "strain_consensus"}:
        return config.neighbor_max_radius
    return None


def graph_config(config: BenchmarkConfig) -> dict[str, Any]:
    alignn = config.model_name == "alignn_like"
    return {
        "cutoff": config.cutoff,
        "neighbor_strategy": config.neighbor_strategy,
        "neighbor_kwargs": _neighbor_kwargs(config),
        "rbf_cutoff": _resolved_rbf_cutoff(config),
        "graph_kwargs": {
            "num_rbf": config.num_rbf,
            "distance_basis_type": config.distance_basis,
            "atom_feature_names": config.atom_features,
        },
        "include_line_graph": alignn,
        "line_graph_kwargs": {
            "num_angle_rbf": config.num_angle_rbf,
            "angle_basis_type": config.angle_basis,
            "use_cosine_basis": config.angle_basis_use_cosine,
            "max_outgoing_neighbors": config.max_line_neighbors,
            "max_line_edges": config.max_line_edges,
            "line_neighbor_selection": config.line_neighbor_selection,
        } if alignn else {},
    }


def model_config(config: BenchmarkConfig) -> dict[str, Any]:
    common = {
        "edge_input_dim": config.num_rbf,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "pooling": config.pooling,
        "dropout": config.dropout,
        "atom_feature_names": config.atom_features,
        "distance_basis_type": "learnable_gaussian" if config.learnable_distance_basis else None,
        "distance_basis_cutoff": _resolved_rbf_cutoff(config) or config.cutoff,
        "readout_type": config.readout_type,
        "ib_lambda": config.ib_lambda,
        "ib_sigma_slope": config.ib_sigma_slope,
        "ib_fixed_point_iters": config.ib_fixed_point_iters,
        "ib_coupling": config.ib_coupling,
        "ib_trainable_lambda": config.ib_trainable_lambda,
        "use_edge_weight": config.use_edge_weight,
        "conv_activation_type": config.conv_activation_type,
        "conv_ib_lambda": config.conv_ib_lambda,
        "conv_ib_sigma_slope": config.conv_ib_sigma_slope,
        "conv_ib_fixed_point_iters": config.conv_ib_fixed_point_iters,
        "conv_ib_coupling": config.conv_ib_coupling,
        "conv_ib_trainable_lambda": config.conv_ib_trainable_lambda,
        "conv_ib_targets": config.conv_ib_targets,
    }
    if config.model_name == "alignn_like":
        common.update(
            angle_input_dim=config.num_angle_rbf,
            angle_basis_type="learnable_gaussian" if config.learnable_angle_basis else None,
            angle_basis_use_cosine=config.angle_basis_use_cosine,
        )
    return common


def create_model(config: BenchmarkConfig) -> torch.nn.Module:
    kwargs = model_config(config)
    if config.model_name == "cgcnn":
        return CGCNNModel(**kwargs)
    if config.model_name == "alignn_like":
        return ALIGNNLikeModel(**kwargs)
    if config.model_name == "resnext_cgcnn":
        return ResNeXtCGCNNModel(
            cardinality=config.cardinality,
            branch_dim=config.branch_dim,
            aggregation=config.aggregation,
            branch_weighting=config.branch_weighting,
            **kwargs,
        )
    raise ValueError(f"Unknown model_name {config.model_name!r}")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _series_values(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        return list(values.tolist())
    return list(values)


def _sample_ids(values: Any) -> list[str]:
    index = getattr(values, "index", None)
    if index is None:
        return [str(i) for i in range(len(values))]
    return [str(value) for value in index]


def make_dataset(
    structures: Any,
    targets: Any | None,
    *,
    config: BenchmarkConfig,
    sample_ids: Sequence[str] | None = None,
    normalizer: TargetNormalizer | None = None,
) -> MatbenchStructureDataset:
    gc = graph_config(config)
    return MatbenchStructureDataset(
        _series_values(structures),
        None if targets is None else _series_values(targets),
        sample_ids=sample_ids or _sample_ids(structures),
        task_name=config.task_name,
        target_normalizer=normalizer,
        cache_graphs=False,
        graph_cache_dir=config.graph_cache_dir,
        overwrite_graph_cache=False,
        **gc,
    )


def _loader(dataset, indices, *, config: BenchmarkConfig, shuffle: bool):
    device = resolve_device(config.device)
    kwargs = dataloader_device_kwargs(device, num_workers=config.num_workers)
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_graphs,
        **kwargs,
    )


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: str | torch.device,
    normalizer: TargetNormalizer | None,
    mixed_precision: bool,
) -> tuple[list[str], np.ndarray]:
    resolved = resolve_device(device)
    model.eval().to(resolved)
    amp = bool(mixed_precision and resolved.type == "cuda")
    ids: list[str] = []
    predictions: list[torch.Tensor] = []
    for batch in loader:
        ids.extend(batch.get("material_id", []))
        batch = move_batch_to_device(batch, resolved)
        with torch.autocast(device_type=resolved.type, dtype=torch.float16, enabled=amp):
            pred = model(batch).view(-1).detach().cpu()
        if normalizer is not None:
            pred = normalizer.inverse_transform(pred)
        predictions.append(pred)
    if not predictions:
        return ids, np.empty(0, dtype=float)
    return ids, torch.cat(predictions).numpy()


def package_versions() -> dict[str, str]:
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for name in ["materials-gnn", "pymatgen", "matbench", "scikit-learn"]:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return result


def _json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _completion_path(fold_dir: Path) -> Path:
    return fold_dir / "completed.json"


def fold_is_complete(fold_dir: str | Path, config: BenchmarkConfig) -> bool:
    path = _completion_path(Path(fold_dir))
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("complete") and data.get("config") == asdict(config))


def _extract_test_targets(task: Any, fold: int) -> tuple[Any, Any]:
    data = task.get_test_data(fold, include_target=True)
    if isinstance(data, tuple) and len(data) == 2:
        return data
    raise RuntimeError(
        "This Matbench version did not return (inputs, targets) for include_target=True"
    )


def _quick_subset(values: Any, count: int) -> Any:
    if hasattr(values, "iloc"):
        return values.iloc[:count]
    return values[:count]


def run_matbench_fold(task: Any, fold: int, config: BenchmarkConfig) -> dict[str, Any]:
    """Train/evaluate one official outer fold with train-only internal validation."""
    set_reproducibility(config.seed + int(fold))
    fold_dir = Path(config.output_dir) / config.experiment_name / config.task_name / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    if not config.overwrite and fold_is_complete(fold_dir, config):
        return json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8"))

    train_inputs, train_targets = task.get_train_and_val_data(fold)
    if config.quick_test:
        n = min(config.quick_train_samples, len(train_inputs))
        train_inputs = _quick_subset(train_inputs, n)
        train_targets = _quick_subset(train_targets, n)

    train_idx, val_idx = internal_train_val_indices(
        len(train_inputs), val_fraction=config.val_fraction, seed=config.seed + int(fold)
    )
    normalizer = TargetNormalizer.from_tensor(
        torch.as_tensor(np.asarray(train_targets)[train_idx], dtype=torch.float32)
    )
    train_dataset = make_dataset(
        train_inputs, train_targets, config=config, normalizer=normalizer
    )
    train_loader = _loader(train_dataset, train_idx, config=config, shuffle=True)
    val_loader = _loader(train_dataset, val_idx, config=config, shuffle=False)

    model = create_model(config)
    params = count_parameters(model)
    checkpoint_path = fold_dir / "best_model.pt"
    early = EarlyStopping(
        monitor="val_mae",
        mode="min",
        patience=config.patience,
        min_delta=config.early_stopping_min_delta,
        warmup=config.early_stopping_warmup,
        adaptive=config.early_stopping_adaptive,
        adaptive_factor=config.early_stopping_factor,
        max_patience=max(config.patience, config.early_stopping_max_patience),
        smoothing=config.early_stopping_smoothing,
    )

    if resolve_device(config.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=config.quick_epochs if config.quick_test else config.epochs,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        target_normalizer=normalizer,
        checkpoint_path=checkpoint_path,
        final_checkpoint_path=fold_dir / "final_model.pt",
        checkpoint_metadata={
            "benchmark": "matbench",
            "task": config.task_name,
            "fold": int(fold),
            "config": asdict(config),
        },
        early_stopping=early,
        grad_clip_norm=config.grad_clip_norm,
        mixed_precision=config.amp,
        matmul_precision=config.matmul_precision,
        verbose=True,
    )
    train_time = time.perf_counter() - start

    test_inputs = task.get_test_data(fold, include_target=False)
    official_test_count = len(test_inputs)
    if config.quick_test:
        test_inputs = _quick_subset(
            test_inputs, min(config.quick_test_samples, len(test_inputs))
        )

    test_dataset = make_dataset(test_inputs, None, config=config)
    test_loader = _loader(
        test_dataset, range(len(test_dataset)), config=config, shuffle=False
    )
    pred_start = time.perf_counter()
    sample_ids, predictions = predict_loader(
        model,
        test_loader,
        device=config.device,
        normalizer=normalizer,
        mixed_precision=config.amp,
    )
    prediction_time = time.perf_counter() - pred_start

    official = bool(not config.quick_test and len(predictions) == official_test_count)
    true_targets: np.ndarray | None = None
    if official:
        task.record(fold, predictions)
        _, test_targets = _extract_test_targets(task, fold)
        true_targets = np.asarray(test_targets, dtype=float).reshape(-1)

    best_record = min(
        (row for row in history if "val_mae" in row),
        key=lambda row: row["val_mae"],
        default={},
    )
    metrics: dict[str, Any] = {
        "experiment": config.experiment_name,
        "task": config.task_name,
        "fold": int(fold),
        "model": config.model_name,
        "official_matbench": official,
        "quick_test": config.quick_test,
        "params": params,
        "epochs_ran": len(history),
        "best_epoch": best_record.get("epoch"),
        "val_mae": best_record.get("val_mae"),
        "train_time": train_time,
        "prediction_time": prediction_time,
        "total_runtime": train_time + prediction_time,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / 1024**2
            if resolve_device(config.device).type == "cuda" else None
        ),
    }
    if true_targets is not None:
        yt = torch.as_tensor(true_targets, dtype=torch.float32)
        yp = torch.as_tensor(predictions, dtype=torch.float32)
        metrics.update(
            test_mae=mae(yt, yp),
            test_rmse=rmse(yt, yp),
            test_r2=r2_score(yt, yp),
        )

    pred_frame = pd.DataFrame(
        {"sample_index": sample_ids, "prediction": predictions, "fold": int(fold)}
    )
    if true_targets is not None:
        pred_frame["true_target"] = true_targets
        pred_frame["residual"] = pred_frame["prediction"] - pred_frame["true_target"]
        diagnostics = [
            graph_diagnostics(test_dataset._build_graph(i))
            for i in range(len(test_dataset))
        ]
        pred_frame = pd.concat(
            [pred_frame.reset_index(drop=True), pd.DataFrame(diagnostics)], axis=1
        )
    pred_frame["task"] = config.task_name
    pred_frame["experiment"] = config.experiment_name
    pred_frame.to_csv(fold_dir / "predictions.csv", index=False)
    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)
    _json_dump(asdict(config), fold_dir / "config.json")
    _json_dump(package_versions(), fold_dir / "versions.json")
    _json_dump(metrics, fold_dir / "metrics.json")
    _json_dump({"complete": True, "config": asdict(config)}, _completion_path(fold_dir))
    return metrics


def run_matbench_benchmark(
    config: BenchmarkConfig,
    folds: Sequence[int] | None = None,
) -> tuple[Any, pd.DataFrame]:
    """Run selected folds and return the Matbench object plus tidy fold results."""
    MatbenchBenchmark = require_matbench()
    mb = MatbenchBenchmark(autoload=False, subset=[config.task_name])
    task = mb.tasks[0]
    if (
        task.metadata.get("input_type") != "structure"
        or task.metadata.get("task_type") != "regression"
    ):
        raise ValueError(
            "This notebook currently supports structure-based regression tasks only"
        )
    task.load()

    requested = tuple(folds) if folds is not None else config.folds
    selected = tuple(task.folds) if requested is None else tuple(requested)
    if config.quick_test:
        selected = selected[:1]

    rows = [run_matbench_fold(task, int(fold), config) for fold in selected]
    results = pd.DataFrame(rows)
    root = Path(config.output_dir) / config.experiment_name / config.task_name
    results.to_csv(root / "summary.csv", index=False)

    if not results.empty and "test_mae" in results:
        numeric = [
            c
            for c in [
                "val_mae",
                "test_mae",
                "test_rmse",
                "test_r2",
                "train_time",
                "prediction_time",
                "total_runtime",
                "peak_gpu_memory_mb",
            ]
            if c in results
        ]
        aggregate = (
            results[numeric].agg(["mean", "std", "min", "max"]).T.reset_index()
        )
        aggregate = aggregate.rename(columns={"index": "metric"})
        aggregate.to_csv(root / "aggregate.csv", index=False)

    return mb, results


def collect_oof_predictions(config: BenchmarkConfig) -> pd.DataFrame:
    """Collect saved held-out predictions from completed official folds."""
    root = Path(config.output_dir) / config.experiment_name / config.task_name
    frames = []
    for path in sorted(root.glob("fold_*/predictions.csv")):
        frame = pd.read_csv(path)
        if {"true_target", "prediction", "residual"}.issubset(frame.columns):
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not result.empty:
        result.to_csv(root / "oof_predictions.csv", index=False)
    return result


def completed_experiment_summary(
    configs: Iterable[BenchmarkConfig],
) -> pd.DataFrame:
    """Combine summary rows across architecture configurations."""
    frames = []
    for config in configs:
        path = (
            Path(config.output_dir)
            / config.experiment_name
            / config.task_name
            / "summary.csv"
        )
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
