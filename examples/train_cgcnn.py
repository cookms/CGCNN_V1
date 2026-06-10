#!/usr/bin/env python
"""Train the CGCNN-style distance-only model from a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

import materials_gnn
from materials_gnn.data.datasets import CrystalGraphDataset, collate_graphs
from materials_gnn.data.splits import split_dataset
from materials_gnn.evaluation import export_predictions_csv
from materials_gnn.models import CGCNNModel
from materials_gnn.training import dataloader_device_kwargs, evaluate_model, resolve_device, train_model
from materials_gnn.utils import write_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CGCNN-style crystal GNN")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", default="target")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument(
        "--neighbor-strategy",
        choices=["cutoff", "knn", "voronoi", "adaptive_shell", "strain_consensus"],
        default="cutoff",
    )
    parser.add_argument("--neighbor-k", type=int, default=12)
    parser.add_argument("--neighbor-max-radius", type=float, default=None)
    parser.add_argument("--rbf-cutoff", type=float, default=None)
    parser.add_argument("--strain-epsilon", type=float, default=0.02)
    parser.add_argument("--min-survival-fraction", type=float, default=0.5)
    parser.add_argument("--num-rbf", type=int, default=64)
    parser.add_argument("--distance-basis", choices=["gaussian", "bessel", "fourier"], default="gaussian", help="Static preprocessing basis for bond distances")
    parser.add_argument("--learnable-distance-basis", action="store_true", help="Use a trainable Gaussian distance basis inside the model")
    parser.add_argument(
        "--use-edge-weight",
        action="store_true",
        help="Use optional graph['edge_weight'] scalars to weight atom-graph message aggregation.",
    )
    parser.add_argument("--atom-features", default="", help="Comma-separated elemental descriptors or 'default'")
    parser.add_argument("--readout-type", choices=["mlp", "implicit_bias"], default="mlp")
    parser.add_argument("--ib-lambda", type=float, default=0.01)
    parser.add_argument("--ib-sigma-slope", type=float, default=1.0)
    parser.add_argument("--ib-fixed-point-iters", type=int, default=8)
    parser.add_argument("--ib-coupling", choices=["ring", "dense"], default="ring")
    parser.add_argument("--ib-trainable-lambda", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0, help="Max gradient norm; set <=0 to disable clipping")
    parser.add_argument("--no-finite-checks", action="store_true", help="Disable NaN/Inf checks in batches, predictions, losses, gradients, and parameters")
    parser.add_argument("--detect-anomaly", action="store_true", help="Enable torch autograd anomaly detection for debugging NaNs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--cache-graphs", action="store_true", help="Cache graph tensors in RAM after first load")
    parser.add_argument("--graph-cache-dir", default=None, help="Directory for persistent precomputed graph cache")
    parser.add_argument("--overwrite-graph-cache", action="store_true", help="Rebuild graphs even if cached entries exist")
    parser.add_argument("--matmul-precision", default=None, choices=["highest", "high", "medium"], help="Optional torch float32 matmul precision")
    parser.add_argument("--output-dir", default="runs/cgcnn")
    return parser.parse_args()


def _neighbor_kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Translate CLI flags into strategy-specific kwargs."""

    if args.neighbor_strategy == "knn":
        kwargs: dict[str, object] = {"k": args.neighbor_k}
        if args.neighbor_max_radius is not None:
            kwargs["max_radius"] = args.neighbor_max_radius
        return kwargs
    if args.neighbor_strategy == "voronoi":
        kwargs = {}
        if args.neighbor_max_radius is not None:
            kwargs["cutoff"] = args.neighbor_max_radius
        return kwargs
    if args.neighbor_strategy == "adaptive_shell":
        kwargs = {}
        if args.neighbor_max_radius is not None:
            kwargs["max_radius"] = args.neighbor_max_radius
        return kwargs
    if args.neighbor_strategy == "strain_consensus":
        kwargs = {
            "strain_epsilon": args.strain_epsilon,
            "min_survival_fraction": args.min_survival_fraction,
        }
        if args.neighbor_max_radius is not None:
            kwargs["cutoff"] = args.neighbor_max_radius
        return kwargs
    return {}


def _rbf_cutoff_from_args(args: argparse.Namespace) -> float | None:
    if args.rbf_cutoff is not None:
        return args.rbf_cutoff
    if args.neighbor_strategy in {"knn", "voronoi", "adaptive_shell", "strain_consensus"}:
        return args.neighbor_max_radius
    return None


def _graph_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_rbf": args.num_rbf,
        "distance_basis_type": args.distance_basis,
        "atom_feature_names": args.atom_features or None,
    }


def _model_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "edge_input_dim": args.num_rbf,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "atom_feature_names": args.atom_features or None,
        "distance_basis_type": "learnable_gaussian" if args.learnable_distance_basis else None,
        "distance_basis_cutoff": _rbf_cutoff_from_args(args) or args.cutoff,
        "use_edge_weight": args.use_edge_weight,
        "readout_type": args.readout_type,
        "ib_lambda": args.ib_lambda,
        "ib_sigma_slope": args.ib_sigma_slope,
        "ib_fixed_point_iters": args.ib_fixed_point_iters,
        "ib_coupling": args.ib_coupling,
        "ib_trainable_lambda": args.ib_trainable_lambda,
    }


def _inference_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cutoff": args.cutoff,
        "neighbor_strategy": args.neighbor_strategy,
        "neighbor_kwargs": _neighbor_kwargs_from_args(args),
        "rbf_cutoff": _rbf_cutoff_from_args(args),
        "include_line_graph": False,
        "graph_kwargs": _graph_kwargs_from_args(args),
        "line_graph_kwargs": None,
    }


def _experiment_config_from_args(
    args: argparse.Namespace,
    *,
    device: torch.device,
    split_indices: tuple[list[int], list[int], list[int]],
    normalizer: Any,
) -> dict[str, Any]:
    train_idx, val_idx, test_idx = split_indices
    return {
        "experiment_config_schema_version": 1,
        "package": {
            "name": "materials-gnn",
            "version": materials_gnn.__version__,
        },
        "script": "examples/train_cgcnn.py",
        "model": {
            "name": "cgcnn",
            "architecture": "CGCNNModel",
            "config": _model_config_from_args(args),
        },
        "data": {
            "csv": args.csv,
            "target_column": args.target,
            "id_column": "material_id",
            "cif_path_column": "cif_path",
        },
        "graph": {
            "cutoff": args.cutoff,
            "neighbor_strategy": args.neighbor_strategy,
            "neighbor_kwargs": _neighbor_kwargs_from_args(args),
            "rbf_cutoff": _rbf_cutoff_from_args(args),
            "include_line_graph": False,
            "line_graph_kwargs": None,
        },
        "featurization": {
            "graph_kwargs": _graph_kwargs_from_args(args),
            "distance_basis": args.distance_basis,
            "learnable_distance_basis": args.learnable_distance_basis,
            "atom_features": args.atom_features or None,
            "num_rbf": args.num_rbf,
        },
        "split": {
            "seed": args.seed,
            "train_size": 0.8,
            "val_size": 0.1,
            "test_size": 0.1,
            "shuffle": True,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx,
        },
        "training": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm if args.grad_clip_norm > 0 else None,
            "amp": args.amp,
            "matmul_precision": args.matmul_precision,
            "num_workers": args.num_workers,
            "device_requested": args.device,
            "device_resolved": str(device),
            "check_finite": not args.no_finite_checks,
            "detect_anomaly": args.detect_anomaly,
            "cli_args": vars(args).copy(),
        },
        "target_normalizer": normalizer.state_dict() if normalizer is not None else None,
        "inference_config": _inference_config_from_args(args),
    }


def _checkpoint_metadata_from_experiment_config(experiment_config: dict[str, Any]) -> dict[str, Any]:
    """Keep legacy checkpoint keys while embedding the complete run config."""

    return {
        "materials_gnn_version": experiment_config["package"]["version"],
        "model_name": experiment_config["model"]["name"],
        "model_config": experiment_config["model"]["config"],
        "inference_config": experiment_config["inference_config"],
        "training_config": experiment_config["training"]["cli_args"],
        "experiment_config": experiment_config,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CrystalGraphDataset(
        args.csv,
        target_column=args.target,
        cutoff=args.cutoff,
        neighbor_strategy=args.neighbor_strategy,
        neighbor_kwargs=_neighbor_kwargs_from_args(args),
        rbf_cutoff=_rbf_cutoff_from_args(args),
        graph_kwargs=_graph_kwargs_from_args(args),
        include_line_graph=False,
        cache_graphs=args.cache_graphs,
        graph_cache_dir=args.graph_cache_dir,
        overwrite_graph_cache=args.overwrite_graph_cache,
    )
    train_set, val_set, test_set, split_indices = split_dataset(dataset, seed=args.seed)
    train_idx, _val_idx, _test_idx = split_indices
    normalizer = dataset.fit_target_normalizer(train_idx)

    loader_kwargs = dataloader_device_kwargs(device, num_workers=args.num_workers)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_graphs,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_graphs,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_graphs,
        **loader_kwargs,
    )

    model = CGCNNModel(**_model_config_from_args(args))
    experiment_config = _experiment_config_from_args(
        args,
        device=device,
        split_indices=split_indices,
        normalizer=normalizer,
    )
    write_experiment_config(experiment_config, output_dir / "experiment_config.json")
    checkpoint_metadata = _checkpoint_metadata_from_experiment_config(experiment_config)
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        target_normalizer=normalizer,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        mixed_precision=args.amp,
        matmul_precision=args.matmul_precision,
        checkpoint_path=output_dir / "best_model.pt",
        final_checkpoint_path=output_dir / "final_model.pt",
        checkpoint_metadata=checkpoint_metadata,
        check_finite=not args.no_finite_checks,
        detect_anomaly=args.detect_anomaly,
    )
    write_experiment_config({"history": history}, output_dir / "training_history.json")

    test_metrics = evaluate_model(model, test_loader, device=device, target_normalizer=normalizer, mixed_precision=args.amp, check_finite=not args.no_finite_checks)
    print(
        "test "
        f"mae={test_metrics['mae']:.5f} "
        f"rmse={test_metrics['rmse']:.5f} "
        f"r2={test_metrics['r2']:.4f}"
    )
    export_predictions_csv(
        test_metrics["material_id"],
        test_metrics["y_true"],
        test_metrics["y_pred"],
        output_dir / "test_predictions.csv",
    )


if __name__ == "__main__":
    main()
