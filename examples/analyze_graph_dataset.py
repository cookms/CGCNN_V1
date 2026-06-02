#!/usr/bin/env python
"""Analyze crystal graph sizes before training.

This script builds the same graph objects used by the training examples, then reports atom
counts, bond-edge counts, ALIGNN line-edge counts, distance ranges, and heuristic OOM risk
flags. It is useful before long CUDA runs, especially when using dense neighbor graphs or
uncapped ALIGNN-like line graphs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from materials_gnn.data.datasets import CrystalGraphDataset
from materials_gnn.data.graph_stats import (
    analyze_dataset_graphs,
    graph_stats_to_rows,
    summarize_graph_stats,
)
from materials_gnn.utils import write_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report crystal graph statistics before training")
    parser.add_argument("--csv", required=True, help="CSV with material_id,cif_path,target columns")
    parser.add_argument("--target", default="target", help="Target column name required by CrystalGraphDataset")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Periodic neighbor cutoff / default length scale in Angstrom")
    parser.add_argument(
        "--neighbor-strategy",
        choices=["cutoff", "knn", "voronoi", "adaptive_shell", "strain_consensus"],
        default="cutoff",
        help="Periodic graph construction rule",
    )
    parser.add_argument("--neighbor-k", type=int, default=12, help="K for --neighbor-strategy knn")
    parser.add_argument("--neighbor-max-radius", type=float, default=None, help="Search radius for knn/voronoi/adaptive_shell")
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
    parser.add_argument("--no-line-graph", action="store_true", help="Skip ALIGNN-like line-graph construction")
    parser.add_argument("--num-angle-rbf", type=int, default=32, help="Angle basis size")
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
    parser.add_argument("--max-samples", type=int, default=None, help="Analyze only the first N rows")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension planned for training memory estimate")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of GNN layers planned for training memory estimate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size planned for training memory estimate")
    parser.add_argument("--dtype-bytes", type=int, default=4, help="Bytes per scalar for memory estimate; fp32=4, fp16/bf16=2")
    parser.add_argument(
        "--activation-multiplier",
        type=float,
        default=8.0,
        help="Conservative multiplier for gradients, temporary tensors, and optimizer/autograd state",
    )
    parser.add_argument("--warn-atoms", type=float, default=512)
    parser.add_argument("--warn-edges", type=float, default=100_000)
    parser.add_argument("--warn-line-edges", type=float, default=250_000)
    parser.add_argument("--warn-line-edges-per-atom", type=float, default=1_000)
    parser.add_argument("--warn-estimated-batch-mb", type=float, default=2_000)
    parser.add_argument("--top-k", type=int, default=10, help="Number of riskiest structures to print")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="CPU worker processes for graph construction and analysis; 0 or 1 runs in the main process",
    )
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N rows; set <=0 to disable")
    parser.add_argument("--output-csv", default=None, help="Optional per-structure CSV output path")
    parser.add_argument("--output-json", default=None, help="Optional summary JSON output path")
    parser.add_argument("--cache-graphs", action="store_true", help="Cache graph tensors in RAM after first load")
    parser.add_argument("--graph-cache-dir", default=None, help="Directory for persistent precomputed graph cache")
    parser.add_argument("--graph-cache-namespace", default="graphs", help="Subdirectory namespace under --graph-cache-dir")
    parser.add_argument("--overwrite-graph-cache", action="store_true", help="Rebuild graphs even if cached entries exist")
    parser.add_argument(
        "--no-cache-file-hash",
        action="store_true",
        help="Use path/size/mtime invalidation instead of hashing CIF contents",
    )
    return parser.parse_args()


def _neighbor_kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
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


def _line_graph_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_angle_rbf": args.num_angle_rbf,
        "angle_basis_type": args.angle_basis,
        "use_cosine_basis": args.angle_basis_use_cosine,
        "max_outgoing_neighbors": args.max_line_neighbors,
        "max_line_edges": args.max_line_edges,
        "line_neighbor_selection": args.line_neighbor_selection,
    }


def _thresholds_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        "max_atoms": args.warn_atoms,
        "max_edges": args.warn_edges,
        "max_line_edges": args.warn_line_edges,
        "max_line_edges_per_atom": args.warn_line_edges_per_atom,
        "max_estimated_batch_mb": args.warn_estimated_batch_mb,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"graphs analyzed: {summary['count']}")
    if summary["count"] == 0:
        return
    print(
        "atoms: "
        f"median={summary['num_atoms']['median']:.0f} "
        f"max={summary['num_atoms']['max']:.0f}"
    )
    print(
        "edges: "
        f"median={summary['num_edges']['median']:.0f} "
        f"max={summary['num_edges']['max']:.0f}"
    )
    print(
        "line_edges: "
        f"median={summary['num_line_edges']['median']:.0f} "
        f"max={summary['num_line_edges']['max']:.0f}"
    )
    print(
        "estimated_batch_mb: "
        f"median={summary['estimated_batch_mb']['median']:.1f} "
        f"max={summary['estimated_batch_mb']['max']:.1f}"
    )
    print(f"risky graphs: {summary['num_risky_graphs']}")
    if summary["riskiest"]:
        print("\nriskiest structures:")
        for item in summary["riskiest"]:
            flags = ",".join(item["risk_flags"]) if item["risk_flags"] else "none"
            print(
                f"  {item['material_id']}: "
                f"atoms={item['num_atoms']} "
                f"edges={item['num_edges']} "
                f"line_edges={item['num_line_edges']} "
                f"dist=[{item['distance_min']}, {item['distance_max']}] "
                f"est_batch_mb={item['estimated_batch_mb']:.1f} "
                f"risk={flags}"
            )


def main() -> None:
    args = parse_args()
    include_line_graph = not args.no_line_graph
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
        cache_graphs=args.cache_graphs,
        graph_cache_dir=args.graph_cache_dir,
        graph_cache_namespace=args.graph_cache_namespace,
        overwrite_graph_cache=args.overwrite_graph_cache,
        cache_include_file_hash=not args.no_cache_file_hash,
    )
    thresholds = _thresholds_from_args(args)
    stats = analyze_dataset_graphs(
        dataset,
        max_samples=args.max_samples,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        dtype_bytes=args.dtype_bytes,
        activation_multiplier=args.activation_multiplier,
        thresholds=thresholds,
        num_workers=args.num_workers,
        progress_every=args.progress_every,
        print_fn=print,
    )
    rows = graph_stats_to_rows(stats)
    summary = summarize_graph_stats(stats, top_k=args.top_k)

    _print_summary(summary)

    if args.output_csv is not None:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f"wrote per-structure stats: {output_csv}")

    if args.output_json is not None:
        output_json = Path(args.output_json)
        write_experiment_config(
            {
                "analysis_config": vars(args).copy(),
                "thresholds": thresholds,
                "summary": summary,
            },
            output_json,
        )
        print(f"wrote summary: {output_json}")


if __name__ == "__main__":
    main()
