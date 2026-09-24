"""In-memory structure datasets used by Matbench benchmarking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from materials_gnn.data.graph_cache import GraphCache, stable_hash
from materials_gnn.data.transforms import TargetNormalizer
from materials_gnn.featurization.crystal_graph import structure_to_bond_graph
from materials_gnn.featurization.line_graph import add_line_graph


def structure_fingerprint(structure: Any) -> str:
    """Return a deterministic fingerprint for a pymatgen-like Structure."""
    if not hasattr(structure, "as_dict"):
        raise TypeError("MatbenchStructureDataset expects pymatgen Structure-like objects")
    payload = structure.as_dict()
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MatbenchStructureDataset(Dataset):
    """Adapt in-memory pymatgen structures to the repository graph pipeline.

    This deliberately calls the same structure_to_bond_graph and add_line_graph
    functions used by the CSV/CIF workflow. Targets may be omitted for held-out
    prediction; dummy zeros are returned only because the shared collator expects
    a scalar y field. Training code must never use held-out dummy targets.
    """

    def __init__(
        self,
        structures: Sequence[Any],
        targets: Sequence[float] | Tensor | None = None,
        *,
        sample_ids: Sequence[str] | None = None,
        task_name: str = "matbench",
        cutoff: float = 5.0,
        neighbor_strategy: str | Any | None = None,
        neighbor_kwargs: dict[str, Any] | None = None,
        rbf_cutoff: float | None = None,
        graph_kwargs: dict[str, Any] | None = None,
        include_line_graph: bool = False,
        line_graph_kwargs: dict[str, Any] | None = None,
        target_normalizer: TargetNormalizer | None = None,
        cache_graphs: bool = False,
        graph_cache_dir: str | Path | None = None,
        graph_cache_namespace: str = "matbench_graphs",
        overwrite_graph_cache: bool = False,
    ) -> None:
        self.structures = list(structures)
        self.targets = None if targets is None else torch.as_tensor(targets, dtype=torch.float32).view(-1)
        if self.targets is not None and len(self.targets) != len(self.structures):
            raise ValueError("targets must have the same length as structures")
        self.sample_ids = (
            [str(i) for i in range(len(self.structures))]
            if sample_ids is None else [str(value) for value in sample_ids]
        )
        if len(self.sample_ids) != len(self.structures):
            raise ValueError("sample_ids must have the same length as structures")

        self.task_name = task_name
        self.cutoff = cutoff
        self.neighbor_strategy = neighbor_strategy
        self.neighbor_kwargs = dict(neighbor_kwargs or {})
        if isinstance(neighbor_strategy, str) and neighbor_strategy.lower().strip() == "voronoi":
            self.neighbor_kwargs.setdefault("failure_policy", "raise")
        self.rbf_cutoff = rbf_cutoff
        self.graph_kwargs = dict(graph_kwargs or {})
        self.include_line_graph = include_line_graph
        self.line_graph_kwargs = dict(line_graph_kwargs or {})
        self.target_normalizer = target_normalizer
        self.cache_graphs = cache_graphs
        self.graph_cache = (
            GraphCache(graph_cache_dir, namespace=graph_cache_namespace)
            if graph_cache_dir is not None else None
        )
        self.overwrite_graph_cache = overwrite_graph_cache
        self._graph_cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.structures)

    def _strategy_name(self) -> str | None:
        if self.neighbor_strategy is None or isinstance(self.neighbor_strategy, str):
            return self.neighbor_strategy
        return f"{self.neighbor_strategy.__class__.__module__}.{self.neighbor_strategy.__class__.__qualname__}"

    def _cache_identity(self, index: int) -> dict[str, Any]:
        # Task/sample labels are included as human-identifying namespace information,
        # while training-only choices are intentionally absent.
        return {
            "cache_schema": 1,
            "task": self.task_name,
            "sample_id": self.sample_ids[index],
            "structure_sha256": structure_fingerprint(self.structures[index]),
            "cutoff": self.cutoff,
            "neighbor_strategy": self._strategy_name(),
            "neighbor_kwargs": self.neighbor_kwargs,
            "rbf_cutoff": self.rbf_cutoff,
            "graph_kwargs": self.graph_kwargs,
            "include_line_graph": self.include_line_graph,
            "line_graph_kwargs": self.line_graph_kwargs,
        }

    def graph_cache_key(self, index: int) -> str | None:
        if self.graph_cache is None:
            return None
        return self.graph_cache.key(self._cache_identity(index))

    def is_graph_cached(self, index: int) -> bool:
        key = self.graph_cache_key(index)
        return bool(self.graph_cache is not None and key and self.graph_cache.exists(key))

    def _build_graph(self, index: int) -> dict[str, Any]:
        if self.cache_graphs and index in self._graph_cache:
            return self._graph_cache[index]

        identity = self._cache_identity(index)
        key = self.graph_cache.key(identity) if self.graph_cache is not None else None
        if self.graph_cache is not None and key is not None and not self.overwrite_graph_cache:
            cached = self.graph_cache.load(key)
            if cached is not None:
                if self.cache_graphs:
                    self._graph_cache[index] = cached
                return cached

        graph = structure_to_bond_graph(
            self.structures[index],
            cutoff=self.cutoff,
            rbf_cutoff=self.rbf_cutoff,
            neighbor_strategy=self.neighbor_strategy,
            neighbor_kwargs=self.neighbor_kwargs,
            **self.graph_kwargs,
        )
        if self.include_line_graph:
            graph = add_line_graph(graph, **self.line_graph_kwargs)

        if self.graph_cache is not None and key is not None:
            self.graph_cache.save(key, graph, metadata=identity)
        if self.cache_graphs:
            self._graph_cache[index] = graph
        return graph

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.targets is None:
            target = torch.tensor(0.0, dtype=torch.float32)
        else:
            target = self.targets[index].clone()
            if self.target_normalizer is not None:
                target = self.target_normalizer.transform(target)
        return {
            "graph": self._build_graph(index),
            "y": target,
            "material_id": self.sample_ids[index],
        }

    def get_targets(self, indices: Sequence[int] | None = None) -> Tensor:
        if self.targets is None:
            raise ValueError("This dataset has no targets")
        if indices is None:
            return self.targets.clone()
        return self.targets[torch.as_tensor(list(indices), dtype=torch.long)].clone()

    def fit_target_normalizer(self, indices: Sequence[int] | None = None) -> TargetNormalizer:
        normalizer = TargetNormalizer.from_tensor(self.get_targets(indices))
        self.target_normalizer = normalizer
        return normalizer

    def set_target_normalizer(self, normalizer: TargetNormalizer | None) -> None:
        self.target_normalizer = normalizer

    def precompute(self, indices: Sequence[int] | None = None) -> dict[str, int]:
        selected = range(len(self)) if indices is None else indices
        built = reused = 0
        for index in selected:
            was_cached = self.is_graph_cached(int(index))
            self._build_graph(int(index))
            if was_cached:
                reused += 1
            else:
                built += 1
        return {"built": built, "reused": reused, "total": built + reused}


def graph_diagnostics(graph: dict[str, Any]) -> dict[str, float | int]:
    """Return compact graph-size diagnostics for error analysis."""
    num_nodes = int(graph["num_nodes"])
    num_edges = int(graph["edge_index"].shape[1])
    degrees = torch.bincount(graph["edge_index"][0], minlength=num_nodes)
    return {
        "num_atoms": num_nodes,
        "num_edges": num_edges,
        "mean_neighbors": float(degrees.float().mean().item()) if num_nodes else 0.0,
        "max_neighbors": int(degrees.max().item()) if degrees.numel() else 0,
        "num_line_edges": int(graph["line_edge_index"].shape[1]) if "line_edge_index" in graph else 0,
    }
