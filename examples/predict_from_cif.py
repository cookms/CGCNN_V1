#!/usr/bin/env python
"""Predict from a CIF using a saved checkpoint.

New checkpoints written by the example training scripts include enough metadata to
reconstruct the model architecture and CIF-to-graph settings automatically. CLI graph and
model arguments are retained as a fallback for older checkpoints, or when
``--ignore-checkpoint-config`` is supplied.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch

from materials_gnn.data.transforms import TargetNormalizer
from materials_gnn.inference import predict_cif
from materials_gnn.models import ALIGNNLikeModel, CGCNNModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a scalar property from a CIF")
    parser.add_argument("--cif", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--ignore-checkpoint-config",
        action="store_true",
        help="Use CLI model/graph arguments instead of metadata saved in the checkpoint",
    )
    parser.add_argument("--model", choices=["cgcnn", "alignn_like"], default="alignn_like")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument(
        "--neighbor-strategy",
        choices=["cutoff", "knn", "voronoi", "adaptive_shell", "strain_consensus"],
        default="cutoff",
    )
    parser.add_argument("--neighbor-k", type=int, default=12)
    parser.add_argument("--neighbor-max-radius", type=float, default=None)
    parser.add_argument(
        "--voronoi-failure-policy",
        choices=["raise", "empty", "cutoff"],
        default="raise",
        help="Whole-graph policy when Voronoi construction fails or any atom has no neighbors",
    )
    parser.add_argument("--rbf-cutoff", type=float, default=None)
    parser.add_argument("--strain-epsilon", type=float, default=0.02)
    parser.add_argument("--min-survival-fraction", type=float, default=0.5)
    parser.add_argument("--num-rbf", type=int, default=64)
    parser.add_argument("--distance-basis", choices=["gaussian", "bessel", "fourier"], default="gaussian", help="Static preprocessing basis for bond distances")
    parser.add_argument("--learnable-distance-basis", action="store_true", help="Use a trainable Gaussian distance basis inside the model")
    parser.add_argument(
        "--use-edge-weight",
        action="store_true",
        help="Fallback option to use optional graph['edge_weight'] scalars in message aggregation",
    )
    parser.add_argument("--atom-features", default="", help="Comma-separated elemental descriptors or 'default'")
    parser.add_argument("--num-angle-rbf", type=int, default=32)
    parser.add_argument("--angle-basis", choices=["gaussian", "bessel", "fourier"], default="gaussian", help="Static preprocessing basis for bond angles")
    parser.add_argument("--angle-basis-use-cosine", action="store_true", help="Encode cos(theta) instead of theta for static and learnable angle bases")
    parser.add_argument("--learnable-angle-basis", action="store_true", help="Use a trainable Gaussian angle basis inside the model")
    parser.add_argument(
        "--max-line-neighbors",
        type=int,
        default=None,
        help="Fallback cap on outgoing j->k bonds per incoming i->j bond for older ALIGNN checkpoints",
    )
    parser.add_argument(
        "--max-line-edges",
        type=int,
        default=None,
        help="Fallback hard cap on line-graph angle edges per crystal for older ALIGNN checkpoints",
    )
    parser.add_argument(
        "--line-neighbor-selection",
        choices=["nearest", "first"],
        default="nearest",
        help="Fallback line-graph neighbor selection rule for older ALIGNN checkpoints",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--readout-type", choices=["mlp", "implicit_bias"], default="mlp")
    parser.add_argument("--ib-lambda", type=float, default=0.01)
    parser.add_argument("--ib-sigma-slope", type=float, default=1.0)
    parser.add_argument("--ib-fixed-point-iters", type=int, default=8)
    parser.add_argument("--ib-coupling", choices=["ring", "dense"], default="ring")
    parser.add_argument("--ib-trainable-lambda", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _neighbor_kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Translate CLI flags into strategy-specific kwargs."""

    if args.neighbor_strategy == "knn":
        kwargs: dict[str, object] = {"k": args.neighbor_k}
        if args.neighbor_max_radius is not None:
            kwargs["max_radius"] = args.neighbor_max_radius
        return kwargs
    if args.neighbor_strategy == "voronoi":
        kwargs = {"failure_policy": args.voronoi_failure_policy}
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


def _legacy_settings_from_args(args: argparse.Namespace) -> tuple[str, dict[str, Any], dict[str, Any]]:
    distance_basis_cutoff = _rbf_cutoff_from_args(args) or args.cutoff
    graph_kwargs = {
        "num_rbf": args.num_rbf,
        "distance_basis_type": args.distance_basis,
        "atom_feature_names": args.atom_features or None,
    }
    inference_config: dict[str, Any] = {
        "cutoff": args.cutoff,
        "neighbor_strategy": args.neighbor_strategy,
        "neighbor_kwargs": _neighbor_kwargs_from_args(args),
        "rbf_cutoff": _rbf_cutoff_from_args(args),
        "include_line_graph": args.model == "alignn_like",
        "graph_kwargs": graph_kwargs,
        "line_graph_kwargs": None,
    }

    if args.model == "cgcnn":
        model_config = {
            "edge_input_dim": args.num_rbf,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "atom_feature_names": args.atom_features or None,
            "distance_basis_type": "learnable_gaussian" if args.learnable_distance_basis else None,
            "distance_basis_cutoff": distance_basis_cutoff,
            "use_edge_weight": args.use_edge_weight,
            "readout_type": args.readout_type,
            "ib_lambda": args.ib_lambda,
            "ib_sigma_slope": args.ib_sigma_slope,
            "ib_fixed_point_iters": args.ib_fixed_point_iters,
            "ib_coupling": args.ib_coupling,
            "ib_trainable_lambda": args.ib_trainable_lambda,
        }
    else:
        inference_config["line_graph_kwargs"] = {
            "num_angle_rbf": args.num_angle_rbf,
            "angle_basis_type": args.angle_basis,
            "use_cosine_basis": args.angle_basis_use_cosine,
            "max_outgoing_neighbors": args.max_line_neighbors,
            "max_line_edges": args.max_line_edges,
            "line_neighbor_selection": args.line_neighbor_selection,
        }
        model_config = {
            "edge_input_dim": args.num_rbf,
            "angle_input_dim": args.num_angle_rbf,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "atom_feature_names": args.atom_features or None,
            "distance_basis_type": "learnable_gaussian" if args.learnable_distance_basis else None,
            "distance_basis_cutoff": distance_basis_cutoff,
            "angle_basis_type": "learnable_gaussian" if args.learnable_angle_basis else None,
            "angle_basis_use_cosine": args.angle_basis_use_cosine,
            "use_edge_weight": args.use_edge_weight,
            "readout_type": args.readout_type,
            "ib_lambda": args.ib_lambda,
            "ib_sigma_slope": args.ib_sigma_slope,
            "ib_fixed_point_iters": args.ib_fixed_point_iters,
            "ib_coupling": args.ib_coupling,
            "ib_trainable_lambda": args.ib_trainable_lambda,
        }
    return args.model, model_config, inference_config


def _settings_from_checkpoint_or_args(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if not args.ignore_checkpoint_config:
        metadata = checkpoint.get("metadata")
        if isinstance(metadata, dict):
            model_name = metadata.get("model_name")
            model_config = metadata.get("model_config")
            inference_config = metadata.get("inference_config")
            if isinstance(model_name, str) and isinstance(model_config, dict) and isinstance(inference_config, dict):
                return model_name, dict(model_config), dict(inference_config)
    return _legacy_settings_from_args(args)


def _build_model(model_name: str, model_config: dict[str, Any]) -> torch.nn.Module:
    if model_name == "cgcnn":
        return CGCNNModel(**model_config)
    if model_name == "alignn_like":
        return ALIGNNLikeModel(**model_config)
    raise ValueError(f"Unsupported checkpoint model_name {model_name!r}")


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name, model_config, inference_config = _settings_from_checkpoint_or_args(checkpoint, args)
    model = _build_model(model_name, model_config)

    model.load_state_dict(checkpoint["model_state_dict"])
    normalizer_state = checkpoint.get("target_normalizer")
    normalizer = TargetNormalizer.from_state_dict(normalizer_state) if normalizer_state else None
    pred = predict_cif(
        model,
        args.cif,
        target_normalizer=normalizer,
        device=args.device,
        **inference_config,
    )
    print(float(pred.view(-1)[0]))


if __name__ == "__main__":
    main()
