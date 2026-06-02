"""Graph-size diagnostics for crystal-graph datasets.

These utilities are intentionally model-agnostic: they inspect already-built graph
dictionaries and compute sizes that are useful before launching a training run. The line
edge count is the most important memory signal for ALIGNN-like models because angle
message passing scales with the number of bond-bond triplets.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset

DEFAULT_OOM_THRESHOLDS: dict[str, float] = {
    "max_atoms": 512,
    "max_edges": 100_000,
    "max_line_edges": 250_000,
    "max_line_edges_per_atom": 1_000,
    "max_estimated_batch_mb": 2_000,
}


@dataclass(slots=True)
class GraphStats:
    """Compact per-structure graph diagnostics."""

    material_id: str
    num_atoms: int
    num_edges: int
    num_line_edges: int
    distance_min: float | None
    distance_mean: float | None
    distance_max: float | None
    out_degree_mean: float
    out_degree_max: int
    edges_per_atom: float
    line_edges_per_atom: float
    estimated_batch_mb: float
    risk_flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _distance_summary(distance: Tensor) -> tuple[float | None, float | None, float | None]:
    if distance.numel() == 0:
        return None, None, None
    values = distance.detach().float().cpu()
    return float(values.min().item()), float(values.mean().item()), float(values.max().item())


def _line_edge_count(graph: Mapping[str, Any]) -> int:
    line_edge_index = graph.get("line_edge_index")
    if isinstance(line_edge_index, Tensor):
        return int(line_edge_index.shape[1])
    num_line_edges = graph.get("num_line_edges")
    return int(num_line_edges) if num_line_edges is not None else 0


def estimate_batch_memory_mb(
    *,
    num_atoms: int,
    num_edges: int,
    num_line_edges: int,
    hidden_dim: int = 128,
    num_layers: int = 3,
    batch_size: int = 1,
    dtype_bytes: int = 4,
    activation_multiplier: float = 8.0,
) -> float:
    """Rough activation-memory estimate for one homogeneous batch.

    This is deliberately conservative and approximate. It is intended to rank structures
    and flag obvious OOM candidates, not to predict exact CUDA allocator usage.
    """

    hidden_states = (num_atoms + num_edges + num_line_edges) * max(hidden_dim, 1)
    layer_states = hidden_states * max(num_layers, 1)
    bytes_estimate = layer_states * max(dtype_bytes, 1) * max(batch_size, 1) * activation_multiplier
    return float(bytes_estimate / (1024**2))


def compute_graph_stats(
    graph: Mapping[str, Any],
    *,
    material_id: str = "",
    hidden_dim: int = 128,
    num_layers: int = 3,
    batch_size: int = 1,
    dtype_bytes: int = 4,
    activation_multiplier: float = 8.0,
    thresholds: Mapping[str, float] | None = None,
) -> GraphStats:
    """Compute graph-size and heuristic OOM diagnostics for one graph."""

    limits = dict(DEFAULT_OOM_THRESHOLDS)
    if thresholds is not None:
        limits.update({key: float(value) for key, value in thresholds.items()})

    num_atoms = int(graph["num_nodes"])
    edge_index = graph["edge_index"]
    if not isinstance(edge_index, Tensor):
        raise TypeError("graph['edge_index'] must be a torch.Tensor")
    num_edges = int(edge_index.shape[1])
    num_line_edges = _line_edge_count(graph)

    distance = graph.get("distance")
    if not isinstance(distance, Tensor):
        distance = torch.empty(0)
    distance_min, distance_mean, distance_max = _distance_summary(distance)

    if num_atoms > 0 and num_edges > 0:
        source = edge_index[0].detach().cpu()
        out_degree = torch.bincount(source, minlength=num_atoms).float()
        out_degree_mean = float(out_degree.mean().item())
        out_degree_max = int(out_degree.max().item())
    else:
        out_degree_mean = 0.0
        out_degree_max = 0

    edges_per_atom = num_edges / max(num_atoms, 1)
    line_edges_per_atom = num_line_edges / max(num_atoms, 1)
    estimated_batch_mb = estimate_batch_memory_mb(
        num_atoms=num_atoms,
        num_edges=num_edges,
        num_line_edges=num_line_edges,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        batch_size=batch_size,
        dtype_bytes=dtype_bytes,
        activation_multiplier=activation_multiplier,
    )

    risk_flags: list[str] = []
    if num_atoms > limits["max_atoms"]:
        risk_flags.append(f"atoms>{int(limits['max_atoms'])}")
    if num_edges > limits["max_edges"]:
        risk_flags.append(f"edges>{int(limits['max_edges'])}")
    if num_line_edges > limits["max_line_edges"]:
        risk_flags.append(f"line_edges>{int(limits['max_line_edges'])}")
    if line_edges_per_atom > limits["max_line_edges_per_atom"]:
        risk_flags.append(f"line_edges_per_atom>{limits['max_line_edges_per_atom']:.0f}")
    if estimated_batch_mb > limits["max_estimated_batch_mb"]:
        risk_flags.append(f"estimated_batch_mb>{limits['max_estimated_batch_mb']:.0f}")

    return GraphStats(
        material_id=material_id,
        num_atoms=num_atoms,
        num_edges=num_edges,
        num_line_edges=num_line_edges,
        distance_min=distance_min,
        distance_mean=distance_mean,
        distance_max=distance_max,
        out_degree_mean=out_degree_mean,
        out_degree_max=out_degree_max,
        edges_per_atom=edges_per_atom,
        line_edges_per_atom=line_edges_per_atom,
        estimated_batch_mb=estimated_batch_mb,
        risk_flags=risk_flags,
    )


_WORKER_DATASET: Dataset | None = None
_WORKER_STATS_KWARGS: dict[str, Any] | None = None


def _init_graph_stats_worker(dataset: Dataset, stats_kwargs: dict[str, Any]) -> None:
    global _WORKER_DATASET, _WORKER_STATS_KWARGS
    _WORKER_DATASET = dataset
    _WORKER_STATS_KWARGS = stats_kwargs
    # Persistent graph_cache_dir is the reusable cache. Avoid retaining every graph in
    # each worker process when analyzing large datasets.
    if hasattr(_WORKER_DATASET, "cache_graphs"):
        _WORKER_DATASET.cache_graphs = False
    if hasattr(_WORKER_DATASET, "_graph_cache"):
        _WORKER_DATASET._graph_cache = {}


def _analyze_dataset_index(dataset: Dataset, index: int, stats_kwargs: Mapping[str, Any]) -> tuple[int, GraphStats]:
    sample = dataset[index]
    stats = compute_graph_stats(
        sample["graph"],
        material_id=str(sample.get("material_id", index)),
        **stats_kwargs,
    )
    return index, stats


def _analyze_graph_stats_worker(index: int) -> tuple[int, GraphStats]:
    if _WORKER_DATASET is None or _WORKER_STATS_KWARGS is None:  # pragma: no cover - defensive guard
        raise RuntimeError("graph-stats worker was not initialized")
    return _analyze_dataset_index(_WORKER_DATASET, index, _WORKER_STATS_KWARGS)


def _report_analysis_progress(
    *,
    done: int,
    total: int,
    progress_every: int,
    print_fn: Callable[[str], None] | None,
) -> None:
    if print_fn is None or progress_every <= 0:
        return
    if done == total or done % progress_every == 0:
        print_fn(f"analyzed {done}/{total} graphs")


def analyze_dataset_graphs(
    dataset: Dataset,
    *,
    max_samples: int | None = None,
    hidden_dim: int = 128,
    num_layers: int = 3,
    batch_size: int = 1,
    dtype_bytes: int = 4,
    activation_multiplier: float = 8.0,
    thresholds: Mapping[str, float] | None = None,
    num_workers: int = 0,
    progress_every: int = 0,
    print_fn: Callable[[str], None] | None = None,
) -> list[GraphStats]:
    """Build and analyze graphs from a dataset that returns ``{'graph', 'material_id'}``.

    Set ``num_workers`` above 1 to construct/analyze graphs in parallel CPU worker
    processes. When the dataset was created with ``graph_cache_dir``, those workers use
    the same persistent cache as training, so analysis can also warm or reuse the cache.
    """

    limit = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    indices = list(range(limit))
    stats_kwargs: dict[str, Any] = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "dtype_bytes": dtype_bytes,
        "activation_multiplier": activation_multiplier,
        "thresholds": thresholds,
    }
    workers = max(int(num_workers), 0)

    if workers <= 1 or limit == 0:
        stats: list[GraphStats] = []
        for done, index in enumerate(indices, start=1):
            _, item = _analyze_dataset_index(dataset, index, stats_kwargs)
            stats.append(item)
            _report_analysis_progress(
                done=done,
                total=limit,
                progress_every=progress_every,
                print_fn=print_fn,
            )
        return stats

    indexed_stats: list[tuple[int, GraphStats]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_graph_stats_worker,
        initargs=(dataset, stats_kwargs),
    ) as pool:
        futures = [pool.submit(_analyze_graph_stats_worker, index) for index in indices]
        for done, future in enumerate(as_completed(futures), start=1):
            indexed_stats.append(future.result())
            _report_analysis_progress(
                done=done,
                total=limit,
                progress_every=progress_every,
                print_fn=print_fn,
            )

    indexed_stats.sort(key=lambda item: item[0])
    return [item for _, item in indexed_stats]


def graph_stats_to_rows(stats: Iterable[GraphStats]) -> list[dict[str, Any]]:
    """Convert ``GraphStats`` objects to flat dictionaries for CSV/JSON output."""

    rows: list[dict[str, Any]] = []
    for item in stats:
        row = item.to_dict()
        row["risk_flags"] = ";".join(item.risk_flags)
        rows.append(row)
    return rows


def summarize_graph_stats(stats: Iterable[GraphStats], *, top_k: int = 10) -> dict[str, Any]:
    """Aggregate per-graph diagnostics into a small summary dictionary."""

    items = list(stats)
    if not items:
        return {"count": 0, "riskiest": []}

    def values(name: str) -> list[float]:
        return [float(getattr(item, name)) for item in items]

    def numeric_summary(name: str) -> dict[str, float]:
        xs = values(name)
        tensor = torch.tensor(xs, dtype=torch.float32)
        return {
            "min": float(tensor.min().item()),
            "mean": float(tensor.mean().item()),
            "median": float(tensor.median().item()),
            "max": float(tensor.max().item()),
        }

    riskiest = sorted(
        items,
        key=lambda item: (len(item.risk_flags), item.estimated_batch_mb, item.num_line_edges, item.num_edges),
        reverse=True,
    )[:top_k]
    return {
        "count": len(items),
        "num_atoms": numeric_summary("num_atoms"),
        "num_edges": numeric_summary("num_edges"),
        "num_line_edges": numeric_summary("num_line_edges"),
        "edges_per_atom": numeric_summary("edges_per_atom"),
        "line_edges_per_atom": numeric_summary("line_edges_per_atom"),
        "estimated_batch_mb": numeric_summary("estimated_batch_mb"),
        "num_risky_graphs": sum(1 for item in items if item.risk_flags),
        "riskiest": [item.to_dict() for item in riskiest],
    }
