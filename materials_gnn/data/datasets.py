"""Datasets and collation for crystal graph learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from materials_gnn.data.graph_cache import GraphCache, file_fingerprint
from materials_gnn.data.transforms import TargetNormalizer
from materials_gnn.featurization.crystal_graph import structure_to_bond_graph
from materials_gnn.featurization.line_graph import add_line_graph


class GraphConstructionError(RuntimeError):
    """Graph-build failure annotated with the originating dataset row and structure."""


class CrystalGraphDataset(Dataset):
    """CSV-backed dataset for CIF-to-graph supervised learning.

    Expected CSV format can be as simple as::

        material_id,cif_path,target
        mp-149,data/cifs/Si.cif,-5.42

    CIF files are converted lazily in ``__getitem__``. This keeps startup simple and lets
    researchers vary graph-construction settings such as cutoff or RBF size without
    rebuilding a separate database.
    """

    def __init__(
        self,
        csv_path: str | Path,
        *,
        target_column: str = "target",
        cif_path_column: str = "cif_path",
        id_column: str = "material_id",
        root: str | Path | None = None,
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
        graph_cache_namespace: str = "graphs",
        overwrite_graph_cache: bool = False,
        cache_include_file_hash: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.root = Path(root) if root is not None else self.csv_path.parent
        self.table = pd.read_csv(self.csv_path)
        self.target_column = target_column
        self.cif_path_column = cif_path_column
        self.id_column = id_column
        self.cutoff = cutoff
        self.neighbor_strategy = neighbor_strategy
        self.neighbor_kwargs = dict(neighbor_kwargs or {})
        if isinstance(neighbor_strategy, str) and neighbor_strategy.lower().strip() == "voronoi":
            # Keep the topology-changing policy explicit in cache identities. This also
            # prevents legacy cached empty/partial Voronoi graphs from being reused under
            # the new fail-safe default.
            self.neighbor_kwargs.setdefault("failure_policy", "raise")
        self.rbf_cutoff = rbf_cutoff
        self.graph_kwargs = graph_kwargs or {}
        self.include_line_graph = include_line_graph
        self.line_graph_kwargs = line_graph_kwargs or {}
        self.target_normalizer = target_normalizer
        self.cache_graphs = cache_graphs
        self.graph_cache = GraphCache(graph_cache_dir, namespace=graph_cache_namespace) if graph_cache_dir is not None else None
        self.overwrite_graph_cache = overwrite_graph_cache
        self.cache_include_file_hash = cache_include_file_hash
        self._graph_cache: dict[int, dict[str, Tensor | int | float | str]] = {}

        for column in [target_column, cif_path_column]:
            if column not in self.table.columns:
                raise ValueError(f"CSV is missing required column {column!r}")
        if id_column not in self.table.columns:
            self.table[id_column] = [str(i) for i in range(len(self.table))]

        self.table[target_column] = pd.to_numeric(self.table[target_column])

    def __len__(self) -> int:
        return len(self.table)

    def _resolve_cif_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _load_structure(self, cif_path: Path) -> Any:
        try:
            from pymatgen.core import Structure
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("Install pymatgen to load CIF structures: `pip install pymatgen`") from exc
        return Structure.from_file(str(cif_path))

    def _cache_identity(self, index: int, cif_path: Path) -> dict[str, Any]:
        """Return the graph-only cache identity for a dataset row.

        Only inputs that can change the constructed graph belong in this payload.
        Labels, target column names, split settings, and training hyperparameters are
        intentionally excluded so the same CIF graph can be reused across target columns
        and experiments.
        """

        strategy_name = self.neighbor_strategy
        if strategy_name is not None and not isinstance(strategy_name, str):
            strategy_name = f"{strategy_name.__class__.__module__}.{strategy_name.__class__.__qualname__}"
        return {
            "cache_schema": 2,
            "file": file_fingerprint(cif_path, include_hash=self.cache_include_file_hash),
            "cutoff": self.cutoff,
            "neighbor_strategy": strategy_name,
            "neighbor_kwargs": self.neighbor_kwargs,
            "rbf_cutoff": self.rbf_cutoff,
            "graph_kwargs": self.graph_kwargs,
            "include_line_graph": self.include_line_graph,
            "line_graph_kwargs": self.line_graph_kwargs,
        }

    def _legacy_cache_payload(self, index: int, cif_path: Path) -> dict[str, Any]:
        """Return the v1 cache payload for best-effort migration of old cache files."""

        row = self.table.iloc[index]
        legacy = dict(self._cache_identity(index, cif_path))
        legacy["cache_schema"] = 1
        legacy["material_id"] = str(row[self.id_column])
        legacy["target_column"] = self.target_column
        return legacy

    def _cache_metadata(self, index: int, cif_path: Path) -> dict[str, Any]:
        """Return human-readable cache metadata saved alongside a graph."""

        row = self.table.iloc[index]
        metadata = dict(self._cache_identity(index, cif_path))
        metadata.update(
            {
                "material_id": str(row[self.id_column]),
                "csv_path": str(self.csv_path),
                "cif_path_column": self.cif_path_column,
                "id_column": self.id_column,
                "target_column": self.target_column,
            }
        )
        return metadata

    def graph_cache_key(self, index: int) -> str | None:
        """Return the current persistent-cache key for a row, or ``None`` if disabled."""

        if self.graph_cache is None:
            return None
        row = self.table.iloc[index]
        cif_path = self._resolve_cif_path(row[self.cif_path_column])
        return self.graph_cache.key(self._cache_identity(index, cif_path))

    def is_graph_cached(self, index: int) -> bool:
        """Return whether the current cache entry for ``index`` already exists."""

        key = self.graph_cache_key(index)
        return bool(self.graph_cache is not None and key is not None and self.graph_cache.exists(key))

    def _build_graph(self, index: int) -> dict[str, Tensor | int | float | str]:
        if self.cache_graphs and index in self._graph_cache:
            return self._graph_cache[index]

        row = self.table.iloc[index]
        cif_path = self._resolve_cif_path(row[self.cif_path_column])
        cache_identity = self._cache_identity(index, cif_path) if self.graph_cache is not None else None
        cache_metadata = self._cache_metadata(index, cif_path) if self.graph_cache is not None else None
        cache_key = self.graph_cache.key(cache_identity) if self.graph_cache is not None and cache_identity is not None else None

        if self.graph_cache is not None and cache_key is not None and not self.overwrite_graph_cache:
            cached_graph = self.graph_cache.load(cache_key)
            if cached_graph is not None:
                if self.cache_graphs:
                    self._graph_cache[index] = cached_graph
                return cached_graph

            # Best-effort migration path for cache files written by schema v1, whose
            # key included non-graph fields such as target_column and material_id.
            legacy_key = self.graph_cache.key(self._legacy_cache_payload(index, cif_path))
            legacy_graph = self.graph_cache.load(legacy_key)
            if legacy_graph is not None:
                self.graph_cache.save(cache_key, legacy_graph, metadata=cache_metadata)
                if self.cache_graphs:
                    self._graph_cache[index] = legacy_graph
                return legacy_graph

        try:
            structure = self._load_structure(cif_path)
            graph = structure_to_bond_graph(
                structure,
                cutoff=self.cutoff,
                rbf_cutoff=self.rbf_cutoff,
                neighbor_strategy=self.neighbor_strategy,
                neighbor_kwargs=self.neighbor_kwargs,
                **self.graph_kwargs,
            )
            if self.include_line_graph:
                graph = add_line_graph(graph, **self.line_graph_kwargs)
        except Exception as exc:
            material_id = str(row[self.id_column])
            strategy = self.neighbor_strategy
            if strategy is not None and not isinstance(strategy, str):
                strategy = f"{strategy.__class__.__module__}.{strategy.__class__.__qualname__}"
            raise GraphConstructionError(
                "Failed to construct graph for "
                f"dataset row {index}, material_id={material_id!r}, "
                f"cif_path={cif_path}, neighbor_strategy={strategy!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if self.graph_cache is not None and cache_key is not None:
            self.graph_cache.save(cache_key, graph, metadata=cache_metadata)
        if self.cache_graphs:
            self._graph_cache[index] = graph
        return graph

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.table.iloc[index]
        target = torch.tensor(float(row[self.target_column]), dtype=torch.float32)
        if self.target_normalizer is not None:
            target = self.target_normalizer.transform(target)
        return {
            "graph": self._build_graph(index),
            "y": target,
            "material_id": str(row[self.id_column]),
        }

    def get_targets(self, indices: list[int] | None = None) -> Tensor:
        """Return raw, unnormalized targets for all rows or selected indices."""

        if indices is None:
            values = self.table[self.target_column].to_numpy(dtype=float)
        else:
            values = self.table.iloc[indices][self.target_column].to_numpy(dtype=float)
        return torch.tensor(values, dtype=torch.float32)

    def fit_target_normalizer(self, indices: list[int] | None = None) -> TargetNormalizer:
        """Fit and attach a target normalizer from selected raw targets."""

        self.target_normalizer = TargetNormalizer.from_tensor(self.get_targets(indices))
        return self.target_normalizer

    def set_target_normalizer(self, normalizer: TargetNormalizer | None) -> None:
        self.target_normalizer = normalizer


def _cat_tensors(tensors: list[Tensor], *, dim: int, empty_shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
    if tensors:
        return torch.cat(tensors, dim=dim)
    return torch.empty(empty_shape, dtype=dtype)


def collate_graphs(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch variable-size crystal graphs into one disconnected graph.

    Node indices in each graph are offset before concatenation. Line-graph indices are
    offset by the number of directed bonds already present because line-graph nodes are
    original bond ids.
    """

    if not samples:
        raise ValueError("Cannot collate an empty batch")

    z_list: list[Tensor] = []
    edge_index_list: list[Tensor] = []
    edge_vec_list: list[Tensor] = []
    distance_list: list[Tensor] = []
    edge_attr_list: list[Tensor] = []
    edge_weight_list: list[Tensor] = []
    edge_unit_vec_list: list[Tensor] = []
    batch_list: list[Tensor] = []
    pos_list: list[Tensor] = []
    atom_attr_list: list[Tensor] = []

    line_edge_index_list: list[Tensor] = []
    line_edge_attr_list: list[Tensor] = []
    angle_list: list[Tensor] = []
    cosine_list: list[Tensor] = []

    y_list: list[Tensor] = []
    material_ids: list[str] = []

    node_offset = 0
    edge_offset = 0
    has_line_graph = any("line_edge_index" in sample["graph"] for sample in samples)
    has_edge_weight = any("edge_weight" in sample["graph"] for sample in samples)
    has_edge_unit_vec = any("edge_unit_vec" in sample["graph"] for sample in samples)
    has_atom_attr = any("atom_attr" in sample["graph"] for sample in samples)
    atom_attr_dim = next(
        (int(sample["graph"]["atom_attr"].shape[1]) for sample in samples if "atom_attr" in sample["graph"]),
        0,
    )

    for graph_id, sample in enumerate(samples):
        graph = sample["graph"]
        num_nodes = int(graph["num_nodes"])
        num_edges = int(graph["edge_index"].shape[1])  # type: ignore[index]

        z_list.append(graph["z"])  # type: ignore[arg-type]
        batch_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long))

        edge_index_list.append(graph["edge_index"] + node_offset)  # type: ignore[operator]
        edge_vec_list.append(graph["edge_vec"])  # type: ignore[arg-type]
        distance_list.append(graph["distance"])  # type: ignore[arg-type]
        edge_attr_list.append(graph["edge_attr"])  # type: ignore[arg-type]
        if has_edge_weight:
            if "edge_weight" in graph:
                edge_weight_list.append(graph["edge_weight"])  # type: ignore[arg-type]
            else:
                edge_weight_list.append(torch.ones((num_edges,), dtype=torch.float32))
        if has_edge_unit_vec:
            if "edge_unit_vec" in graph:
                edge_unit_vec_list.append(graph["edge_unit_vec"])  # type: ignore[arg-type]
            else:
                edge_unit_vec_list.append(
                    graph["edge_vec"] / graph["distance"].clamp_min(1e-12).unsqueeze(-1)  # type: ignore[operator, union-attr]
                )
        if has_atom_attr:
            if "atom_attr" in graph:
                atom_attr_list.append(graph["atom_attr"])  # type: ignore[arg-type]
            else:
                atom_attr_list.append(torch.zeros((num_nodes, atom_attr_dim), dtype=torch.float32))
        if "pos" in graph:
            pos_list.append(graph["pos"])  # type: ignore[arg-type]

        if has_line_graph:
            if "line_edge_index" in graph:
                line_edge_index_list.append(graph["line_edge_index"] + edge_offset)  # type: ignore[operator]
                line_edge_attr_list.append(graph["line_edge_attr"])  # type: ignore[arg-type]
                if "angle" in graph:
                    angle_list.append(graph["angle"])  # type: ignore[arg-type]
                if "cosine" in graph:
                    cosine_list.append(graph["cosine"])  # type: ignore[arg-type]
            else:
                # This should not happen for ALIGNN training, but keeping an empty fallback
                # makes mixed debugging batches easier to inspect.
                line_edge_index_list.append(torch.empty((2, 0), dtype=torch.long))
                angle_dim = line_edge_attr_list[0].shape[1] if line_edge_attr_list else 0
                line_edge_attr_list.append(torch.empty((0, angle_dim), dtype=torch.float32))
                angle_list.append(torch.empty((0,), dtype=torch.float32))
                cosine_list.append(torch.empty((0,), dtype=torch.float32))

        y_list.append(sample["y"].view(()))
        material_ids.append(sample["material_id"])
        node_offset += num_nodes
        edge_offset += num_edges

    edge_attr_dim = edge_attr_list[0].shape[1]
    batch: dict[str, Any] = {
        "z": torch.cat(z_list, dim=0),
        "edge_index": _cat_tensors(edge_index_list, dim=1, empty_shape=(2, 0), dtype=torch.long),
        "edge_vec": _cat_tensors(edge_vec_list, dim=0, empty_shape=(0, 3), dtype=torch.float32),
        "distance": _cat_tensors(distance_list, dim=0, empty_shape=(0,), dtype=torch.float32),
        "edge_attr": _cat_tensors(
            edge_attr_list,
            dim=0,
            empty_shape=(0, edge_attr_dim),
            dtype=torch.float32,
        ),
        "num_nodes": node_offset,
        "batch": torch.cat(batch_list, dim=0),
        "y": torch.stack(y_list, dim=0),
        "material_id": material_ids,
    }

    if has_edge_weight:
        batch["edge_weight"] = _cat_tensors(
            edge_weight_list,
            dim=0,
            empty_shape=(0,),
            dtype=torch.float32,
        )

    if has_edge_unit_vec:
        batch["edge_unit_vec"] = _cat_tensors(
            edge_unit_vec_list,
            dim=0,
            empty_shape=(0, 3),
            dtype=torch.float32,
        )

    if has_atom_attr:
        batch["atom_attr"] = _cat_tensors(
            atom_attr_list,
            dim=0,
            empty_shape=(0, atom_attr_dim),
            dtype=torch.float32,
        )

    if pos_list:
        batch["pos"] = torch.cat(pos_list, dim=0)

    if has_line_graph:
        angle_dim = line_edge_attr_list[0].shape[1] if line_edge_attr_list else 0
        batch["line_edge_index"] = _cat_tensors(
            line_edge_index_list,
            dim=1,
            empty_shape=(2, 0),
            dtype=torch.long,
        )
        batch["line_edge_attr"] = _cat_tensors(
            line_edge_attr_list,
            dim=0,
            empty_shape=(0, angle_dim),
            dtype=torch.float32,
        )
        if angle_list:
            batch["angle"] = _cat_tensors(
                angle_list,
                dim=0,
                empty_shape=(0,),
                dtype=torch.float32,
            )
        if cosine_list:
            batch["cosine"] = _cat_tensors(
                cosine_list,
                dim=0,
                empty_shape=(0,),
                dtype=torch.float32,
            )

    return batch
