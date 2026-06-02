"""Persistent graph cache for expensive crystal featurization.

Converting CIF files into periodic graphs can dominate wall-clock time when the same
structures are reused across many model experiments. The cache stores fully constructed
CPU graph tensors on disk, keyed by the input CIF content and the graph-construction
configuration. Training can still move the collated batches to CUDA later; the cache is
intentionally CPU/disk based so it is portable across machines and GPU types.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor


def _json_safe(value: Any) -> Any:
    """Convert common Python objects into stable JSON-compatible values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"__repr__": repr(value), "__class__": value.__class__.__qualname__}


def stable_hash(payload: Mapping[str, Any]) -> str:
    """Return a deterministic SHA256 hash for a JSON-like mapping."""

    text = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_fingerprint(path: str | Path, *, include_hash: bool = True) -> dict[str, Any]:
    """Describe a structure file for cache invalidation.

    The content hash is safest because CIF files can be edited without changing their
    names. It is also small compared with graph construction for most materials datasets.
    Set ``include_hash=False`` for very large structure files when mtime/size invalidation
    is sufficient.
    """

    path = Path(path)
    stat = path.stat()
    fingerprint: dict[str, Any] = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        fingerprint["sha256"] = digest.hexdigest()
    return fingerprint


def graph_to_cpu(graph: Mapping[str, Tensor | int | float | str]) -> dict[str, Tensor | int | float | str]:
    """Return a detached CPU copy of graph tensors for serialization."""

    cached: dict[str, Tensor | int | float | str] = {}
    for key, value in graph.items():
        if isinstance(value, Tensor):
            cached[key] = value.detach().cpu().clone()
        else:
            cached[key] = value
    return cached


class GraphCache:
    """Small file-backed cache for precomputed crystal graphs.

    Each graph is saved as ``<key>.pt`` below ``root/namespace``. ``torch.save`` is used
    because the graph is already a dictionary of PyTorch tensors and simple metadata.
    The cache never stores CUDA tensors; batches are moved to the selected device by the
    trainer after collation.
    """

    def __init__(self, root: str | Path, *, namespace: str = "graphs") -> None:
        self.root = Path(root)
        self.namespace = namespace
        self.cache_dir = self.root / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key(self, payload: Mapping[str, Any]) -> str:
        return stable_hash(payload)

    def path_for_key(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pt"

    def exists(self, key: str) -> bool:
        return self.path_for_key(key).exists()

    def load(self, key: str) -> dict[str, Tensor | int | float | str] | None:
        path = self.path_for_key(key)
        if not path.exists():
            return None
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # older PyTorch versions do not expose weights_only
            data = torch.load(path, map_location="cpu")
        if not isinstance(data, dict) or "graph" not in data:
            raise ValueError(f"Invalid graph cache entry: {path}")
        graph = data["graph"]
        if not isinstance(graph, dict):
            raise ValueError(f"Invalid graph payload in cache entry: {path}")
        return graph

    def save(
        self,
        key: str,
        graph: Mapping[str, Tensor | int | float | str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        path = self.path_for_key(key)
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        payload = {
            "graph": graph_to_cpu(graph),
            "metadata": _json_safe(dict(metadata or {})),
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        return path

    def clear(self) -> int:
        """Delete cache files and return the number removed."""

        count = 0
        for path in self.cache_dir.glob("*.pt"):
            path.unlink()
            count += 1
        return count
