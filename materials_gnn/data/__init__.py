"""Data loading and preprocessing utilities."""

from materials_gnn.data.cache_precompute import (
    CachePrecomputeFailure,
    CachePrecomputeResult,
    precompute_graph_cache,
)
from materials_gnn.data.datamodules import make_crystal_dataloaders
from materials_gnn.data.datasets import (
    CrystalGraphDataset,
    GraphConstructionError,
    collate_graphs,
)
from materials_gnn.data.graph_cache import GraphCache
from materials_gnn.data.graph_stats import (
    GraphStats,
    analyze_dataset_graphs,
    compute_graph_stats,
    graph_stats_to_rows,
    summarize_graph_stats,
)
from materials_gnn.data.splits import split_dataset, train_val_test_split_indices
from materials_gnn.data.transforms import TargetNormalizer

__all__ = [
    "CachePrecomputeFailure",
    "CachePrecomputeResult",
    "CrystalGraphDataset",
    "GraphCache",
    "GraphConstructionError",
    "GraphStats",
    "TargetNormalizer",
    "analyze_dataset_graphs",
    "collate_graphs",
    "compute_graph_stats",
    "graph_stats_to_rows",
    "make_crystal_dataloaders",
    "precompute_graph_cache",
    "split_dataset",
    "summarize_graph_stats",
    "train_val_test_split_indices",
]
