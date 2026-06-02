"""Utilities for precomputing persistent crystal graph caches."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class CachePrecomputeFailure:
    """Failure details for one dataset row during graph-cache precomputation."""

    index: int
    material_id: str
    error_type: str
    error_message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CachePrecomputeResult:
    """Summary returned by :func:`precompute_graph_cache`."""

    requested: int = 0
    cache_hits: int = 0
    graphs_built: int = 0
    failures: list[CachePrecomputeFailure] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return self.cache_hits + self.graphs_built

    @property
    def failed(self) -> int:
        return len(self.failures)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "succeeded": self.succeeded,
            "cache_hits": self.cache_hits,
            "graphs_built": self.graphs_built,
            "failed": self.failed,
            "failures": [failure.to_dict() for failure in self.failures],
        }


@dataclass(slots=True)
class _PrecomputeIndexResult:
    index: int
    material_id: str
    was_cached: bool = False
    built: bool = False
    failure: CachePrecomputeFailure | None = None


def _material_id_for(dataset: Any, index: int) -> str:
    table = getattr(dataset, "table", None)
    id_column = getattr(dataset, "id_column", None)
    if table is not None and id_column is not None and id_column in table.columns:
        return str(table.iloc[index][id_column])
    return str(index)


def _precompute_one(dataset: Any, index: int) -> _PrecomputeIndexResult:
    material_id = _material_id_for(dataset, index)
    try:
        was_cached = bool(dataset.is_graph_cached(index))
        _ = dataset[index]["graph"]
        built = not (was_cached and not getattr(dataset, "overwrite_graph_cache", False))
        return _PrecomputeIndexResult(
            index=index,
            material_id=material_id,
            was_cached=was_cached,
            built=built,
        )
    except Exception as exc:  # pragma: no cover - covered through public error handling
        return _PrecomputeIndexResult(
            index=index,
            material_id=material_id,
            failure=CachePrecomputeFailure(
                index=index,
                material_id=material_id,
                error_type=exc.__class__.__name__,
                error_message=str(exc),
            ),
        )


_WORKER_DATASET: Any | None = None


def _init_precompute_worker(dataset: Any) -> None:
    global _WORKER_DATASET
    _WORKER_DATASET = dataset
    # Persistent disk cache is the useful cross-process cache. Keep per-process RAM
    # caching off so large precompute jobs do not retain every graph in every worker.
    if hasattr(_WORKER_DATASET, "cache_graphs"):
        _WORKER_DATASET.cache_graphs = False
    if hasattr(_WORKER_DATASET, "_graph_cache"):
        _WORKER_DATASET._graph_cache = {}


def _precompute_worker(index: int) -> _PrecomputeIndexResult:
    if _WORKER_DATASET is None:  # pragma: no cover - defensive guard
        raise RuntimeError("precompute worker was not initialized")
    return _precompute_one(_WORKER_DATASET, index)


def _raise_precompute_failure(failure: CachePrecomputeFailure) -> None:
    raise RuntimeError(
        f"Failed to precompute graph cache for row {failure.index} "
        f"({failure.material_id}): {failure.error_message}"
    )


def _update_result(result: CachePrecomputeResult, item: _PrecomputeIndexResult, *, fail_fast: bool) -> None:
    if item.failure is not None:
        result.failures.append(item.failure)
        if fail_fast:
            _raise_precompute_failure(item.failure)
    elif item.built:
        result.graphs_built += 1
    else:
        result.cache_hits += 1


def _report_progress(
    *,
    done: int,
    total: int,
    result: CachePrecomputeResult,
    progress_every: int,
    print_fn: Callable[[str], None] | None,
) -> None:
    if print_fn is None or progress_every <= 0:
        return
    if done == total or done % progress_every == 0:
        print_fn(
            "precomputed "
            f"{done}/{total} graphs "
            f"(hits={result.cache_hits}, built={result.graphs_built}, failed={result.failed})"
        )


def precompute_graph_cache(
    dataset: Any,
    *,
    max_samples: int | None = None,
    fail_fast: bool = True,
    progress_every: int = 0,
    print_fn: Callable[[str], None] | None = None,
    num_workers: int = 0,
) -> CachePrecomputeResult:
    """Build and save persistent graph-cache entries for a dataset.

    The dataset must expose the ``CrystalGraphDataset`` cache interface and must be
    constructed with ``graph_cache_dir``. Cache hits are counted before graph loading so
    the result distinguishes graphs that were already warm from graphs built by this run.

    Parameters
    ----------
    dataset:
        Dataset with a persistent ``graph_cache`` configured.
    max_samples:
        Optional limit over the first N rows.
    fail_fast:
        Raise as soon as a row fails when true; otherwise record failures and continue.
    progress_every:
        Print progress every N finished rows when ``print_fn`` is provided.
    print_fn:
        Callable used for progress messages, usually ``print``.
    num_workers:
        Number of worker processes. ``0`` or ``1`` uses the current process. Multiprocessing
        is useful for large CIF datasets because graph construction is CPU-bound.
    """

    if getattr(dataset, "graph_cache", None) is None:
        raise ValueError("precompute_graph_cache requires a dataset with graph_cache_dir set")

    limit = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    result = CachePrecomputeResult(requested=limit)
    indices = list(range(limit))
    workers = max(int(num_workers), 0)

    if workers <= 1 or limit == 0:
        for done, index in enumerate(indices, start=1):
            item = _precompute_one(dataset, index)
            _update_result(result, item, fail_fast=fail_fast)
            _report_progress(
                done=done,
                total=limit,
                result=result,
                progress_every=progress_every,
                print_fn=print_fn,
            )
        return result

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_precompute_worker, initargs=(dataset,)) as pool:
        futures = [pool.submit(_precompute_worker, index) for index in indices]
        for done, future in enumerate(as_completed(futures), start=1):
            item = future.result()
            _update_result(result, item, fail_fast=fail_fast)
            _report_progress(
                done=done,
                total=limit,
                result=result,
                progress_every=progress_every,
                print_fn=print_fn,
            )

    return result
