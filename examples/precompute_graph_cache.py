#!/usr/bin/env python
"""Precompute persistent graph-cache entries before training.

This script builds the same graph dictionaries used by the training examples and saves
those CPU tensors into ``--graph-cache-dir``. Run it with the same graph, feature, and
line-graph flags you plan to use for training; subsequent training runs can reuse the
warmed cache instead of constructing graphs during the first epoch.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from materials_gnn.data import CrystalGraphDataset, precompute_graph_cache
from materials_gnn.utils import write_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute persistent crystal graph cache entries")
    parser.add_argument("--csv", required=True, help="CSV with material_id,cif_path,target columns")
    parser.add_argument("--target", default="target", help="Target column name; not included in the graph-cache key")
    parser.add_argument(
        "--model",
        choices=["cgcnn", "alignn_like"],
        default="alignn_like",
        help="Graph family to precompute. alignn_like includes line graphs; cgcnn does not.",
    )
    parser.add_argument("--cutoff", type=float, default=5.0, help="Periodic neighbor cutoff / default length scale in Angstrom")
    parser.add_argument(
        "--neighbor-strategy",
        choices=["cutoff", "knn", "voronoi", "adaptive_shell", "strain_consensus"],
        default="cutoff",
        help="Periodic graph construction rule",
    )
    parser.add_argument("--neighbor-k", type=int, default=12, help="K for --neighbor-strategy knn")
    parser.add_argument("--neighbor-max-radius", type=float, default=None, help="Search radius for knn/voronoi/adaptive_shell")
    parser.add_argument(
        "--voronoi-failure-policy",
        choices=["raise", "empty", "cutoff"],
        default="raise",
        help="Whole-graph policy when Voronoi construction fails or any atom has no neighbors",
    )
    parser.add_argument("--rbf-cutoff", type=float, default=None, help="Final distance RBF center; fixed across the dataset")
    parser.add_argument("--strain-epsilon", type=float, default=0.02, help="Virtual strain size for strain_consensus")
    parser.add_argument("--min-survival-fraction", type=float, default=0.5, help="Minimum edge survival for strain_consensus")
    parser.add_argument("--num-rbf", type=int, default=64, help="Distance basis size")
    parser.add_argument(
        "--distance-basis",
        choices=["gaussian", "bessel", "fourier"],
        default="gaussian",
        help="Static preprocessing basis for bond distances",
    )
    parser.add_argument("--atom-features", default="", help="Comma-separated elemental descriptors or 'default'")
    parser.add_argument("--num-angle-rbf", type=int, default=32, help="Angle basis size for alignn_like graphs")
    parser.add_argument(
        "--angle-basis",
        choices=["gaussian", "bessel", "fourier"],
        default="gaussian",
        help="Static preprocessing basis for bond angles",
    )
    parser.add_argument("--angle-basis-use-cosine", action="store_true", help="Encode cos(theta) instead of theta")
    parser.add_argument(
        "--max-line-neighbors",
        type=int,
        default=None,
        help="Cap outgoing j->k bonds per incoming i->j bond in the line graph",
    )
    parser.add_argument(
        "--max-line-edges",
        type=int,
        default=None,
        help="Hard cap on line-graph angle edges per crystal",
    )
    parser.add_argument(
        "--line-neighbor-selection",
        choices=["nearest", "first"],
        default="nearest",
        help="How to choose line-graph outgoing bonds when --max-line-neighbors is set",
    )
    parser.add_argument("--graph-cache-dir", required=True, help="Directory for persistent precomputed graph cache")
    parser.add_argument("--graph-cache-namespace", default="graphs", help="Subdirectory namespace under --graph-cache-dir")
    parser.add_argument("--overwrite-graph-cache", action="store_true", help="Rebuild graphs even if cached entries exist")
    parser.add_argument(
        "--no-cache-file-hash",
        action="store_true",
        help="Use path/size/mtime invalidation instead of hashing CIF contents",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Precompute only the first N rows")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="CPU worker processes for graph construction; 0 or 1 runs in the main process",
    )
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N rows; set <=0 to disable")
    parser.add_argument("--continue-on-error", action="store_true", help="Record failures and continue instead of failing fast")
    parser.add_argument("--output-json", default=None, help="Optional JSON summary path")
    return parser.parse_args()


def _neighbor_kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
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


def _graph_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_rbf": args.num_rbf,
        "distance_basis_type": args.distance_basis,
        "atom_feature_names": args.atom_features or None,
    }


def _line_graph_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_angle_rbf": args.num_angle_rbf,
        "angle_basis_type": args.angle_basis,
        "use_cosine_basis": args.angle_basis_use_cosine,
        "max_outgoing_neighbors": args.max_line_neighbors,
        "max_line_edges": args.max_line_edges,
        "line_neighbor_selection": args.line_neighbor_selection,
    }


def main() -> None:
    args = parse_args()
    include_line_graph = args.model == "alignn_like"
    graph_cache_dir = Path(args.graph_cache_dir)
    dataset = CrystalGraphDataset(
        args.csv,
        target_column=args.target,
        cutoff=args.cutoff,
        neighbor_strategy=args.neighbor_strategy,
        neighbor_kwargs=_neighbor_kwargs_from_args(args),
        rbf_cutoff=_rbf_cutoff_from_args(args),
        graph_kwargs=_graph_kwargs_from_args(args),
        include_line_graph=include_line_graph,
        line_graph_kwargs=_line_graph_kwargs_from_args(args) if include_line_graph else None,
        graph_cache_dir=graph_cache_dir,
        graph_cache_namespace=args.graph_cache_namespace,
        overwrite_graph_cache=args.overwrite_graph_cache,
        cache_include_file_hash=not args.no_cache_file_hash,
    )
    result = precompute_graph_cache(
        dataset,
        max_samples=args.max_samples,
        fail_fast=not args.continue_on_error,
        progress_every=args.progress_every,
        print_fn=print,
        num_workers=args.num_workers,
    )
    summary = {
        "cache_dir": str(graph_cache_dir),
        "cache_namespace": args.graph_cache_namespace,
        "cache_key_schema": 2,
        "include_line_graph": include_line_graph,
        "config": vars(args).copy(),
        "result": result.to_dict(),
    }

    print(
        "graph cache precompute complete: "
        f"requested={result.requested} "
        f"hits={result.cache_hits} "
        f"built={result.graphs_built} "
        f"failed={result.failed}"
    )
    if result.failed:
        print("failures:")
        for failure in result.failures[:20]:
            print(
                f"  row={failure.index} material_id={failure.material_id} "
                f"{failure.error_type}: {failure.error_message}"
            )

    if args.output_json is not None:
        output_json = write_experiment_config(summary, args.output_json)
        print(f"wrote summary: {output_json}")


if __name__ == "__main__":
    main()
