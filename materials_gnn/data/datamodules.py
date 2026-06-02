"""Small DataLoader factory functions.

This module intentionally avoids a hard dependency on PyTorch Lightning. It provides the
same practical benefits for the prototype: reproducible splits, target normalization, and
consistent graph collation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from materials_gnn.data.datasets import CrystalGraphDataset, collate_graphs
from materials_gnn.training.device import dataloader_device_kwargs, resolve_device
from materials_gnn.data.splits import split_dataset


def make_crystal_dataloaders(
    csv_path: str | Path,
    *,
    target_column: str = "target",
    batch_size: int = 16,
    cutoff: float = 5.0,
    neighbor_strategy: str | Any | None = None,
    neighbor_kwargs: dict[str, Any] | None = None,
    rbf_cutoff: float | None = None,
    include_line_graph: bool = False,
    graph_kwargs: dict[str, Any] | None = None,
    line_graph_kwargs: dict[str, Any] | None = None,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    cache_graphs: bool = False,
    graph_cache_dir: str | Path | None = None,
    overwrite_graph_cache: bool = False,
    device: str = "auto",
) -> tuple[CrystalGraphDataset, DataLoader, DataLoader, DataLoader]:
    """Create dataset and train/validation/test loaders from a CSV file."""

    dataset = CrystalGraphDataset(
        csv_path,
        target_column=target_column,
        cutoff=cutoff,
        neighbor_strategy=neighbor_strategy,
        neighbor_kwargs=neighbor_kwargs,
        rbf_cutoff=rbf_cutoff,
        graph_kwargs=graph_kwargs,
        include_line_graph=include_line_graph,
        line_graph_kwargs=line_graph_kwargs,
        cache_graphs=cache_graphs,
        graph_cache_dir=graph_cache_dir,
        overwrite_graph_cache=overwrite_graph_cache,
    )
    train_set, val_set, test_set, (train_idx, _val_idx, _test_idx) = split_dataset(
        dataset,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )
    dataset.fit_target_normalizer(train_idx)

    loader_device_kwargs = dataloader_device_kwargs(resolve_device(device), num_workers=num_workers)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_graphs,
        **loader_device_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graphs,
        **loader_device_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graphs,
        **loader_device_kwargs,
    )
    return dataset, train_loader, val_loader, test_loader
