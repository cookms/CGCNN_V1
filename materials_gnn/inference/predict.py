"""Inference helpers for CIF files and pymatgen structures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from materials_gnn.data.transforms import TargetNormalizer
from materials_gnn.featurization.crystal_graph import structure_to_bond_graph
from materials_gnn.featurization.line_graph import add_line_graph
from materials_gnn.training.trainer import move_batch_to_device


def graph_from_structure(
    structure: Any,
    *,
    cutoff: float = 5.0,
    neighbor_strategy: str | Any | None = None,
    neighbor_kwargs: dict[str, Any] | None = None,
    rbf_cutoff: float | None = None,
    include_line_graph: bool = False,
    graph_kwargs: dict[str, Any] | None = None,
    line_graph_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a single-structure graph with a batch vector for model inference."""

    graph = structure_to_bond_graph(
        structure,
        cutoff=cutoff,
        neighbor_strategy=neighbor_strategy,
        neighbor_kwargs=neighbor_kwargs,
        rbf_cutoff=rbf_cutoff,
        **(graph_kwargs or {}),
    )
    if include_line_graph:
        graph = add_line_graph(graph, **(line_graph_kwargs or {}))
    graph["batch"] = torch.zeros(int(graph["num_nodes"]), dtype=torch.long)
    return graph


@torch.no_grad()
def predict_structure(
    model: nn.Module,
    structure: Any,
    *,
    cutoff: float = 5.0,
    neighbor_strategy: str | Any | None = None,
    neighbor_kwargs: dict[str, Any] | None = None,
    rbf_cutoff: float | None = None,
    include_line_graph: bool = False,
    graph_kwargs: dict[str, Any] | None = None,
    line_graph_kwargs: dict[str, Any] | None = None,
    target_normalizer: TargetNormalizer | None = None,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Predict a scalar property for a ``pymatgen.Structure``."""

    model.eval()
    model.to(device)
    graph = graph_from_structure(
        structure,
        cutoff=cutoff,
        neighbor_strategy=neighbor_strategy,
        neighbor_kwargs=neighbor_kwargs,
        rbf_cutoff=rbf_cutoff,
        include_line_graph=include_line_graph,
        graph_kwargs=graph_kwargs,
        line_graph_kwargs=line_graph_kwargs,
    )
    graph = move_batch_to_device(graph, device)
    pred = model(graph).detach().cpu()
    if target_normalizer is not None:
        pred = target_normalizer.inverse_transform(pred)
    return pred


@torch.no_grad()
def predict_cif(
    model: nn.Module,
    cif_path: str | Path,
    *,
    cutoff: float = 5.0,
    neighbor_strategy: str | Any | None = None,
    neighbor_kwargs: dict[str, Any] | None = None,
    rbf_cutoff: float | None = None,
    include_line_graph: bool = False,
    graph_kwargs: dict[str, Any] | None = None,
    line_graph_kwargs: dict[str, Any] | None = None,
    target_normalizer: TargetNormalizer | None = None,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Load a CIF and predict a scalar property."""

    try:
        from pymatgen.core import Structure
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install pymatgen to load CIF structures: `pip install pymatgen`") from exc
    structure = Structure.from_file(str(cif_path))
    return predict_structure(
        model,
        structure,
        cutoff=cutoff,
        neighbor_strategy=neighbor_strategy,
        neighbor_kwargs=neighbor_kwargs,
        rbf_cutoff=rbf_cutoff,
        include_line_graph=include_line_graph,
        graph_kwargs=graph_kwargs,
        line_graph_kwargs=line_graph_kwargs,
        target_normalizer=target_normalizer,
        device=device,
    )
