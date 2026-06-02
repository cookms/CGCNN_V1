"""Line-graph construction for ALIGNN-like angle message passing.

A normal crystal graph has atoms as nodes and bonds as edges. In a line graph, each directed
bond becomes a node. A line-graph edge connects bond i->j to bond j->k, representing a
three-body path with an angle at center atom j. ALIGNN-style models use this construction
so bond states can be updated with angular information before atom states are updated.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from materials_gnn.featurization.basis import angle_basis_expansion


def build_line_graph(
    edge_index: Tensor,
    edge_vec: Tensor,
    num_nodes: int,
    *,
    num_angle_rbf: int = 32,
    angle_basis_type: str = "gaussian",
    angle_basis_kwargs: dict[str, Any] | None = None,
    use_cosine_basis: bool = False,
    skip_backtracking: bool = True,
    max_outgoing_neighbors: int | None = None,
    max_line_edges: int | None = None,
    line_neighbor_selection: str = "nearest",
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Build a directed line graph from a directed bond graph.

    Args:
        edge_index: Tensor of shape ``[2, num_edges]``. Column ``e`` is directed bond
            ``source -> destination``.
        edge_vec: Cartesian displacement vectors of shape ``[num_edges, 3]`` pointing from
            source atom to destination atom, including periodic image translations.
        num_nodes: Number of atoms in the original graph.
        num_angle_rbf: Number of basis functions for angle expansion.
        angle_basis_type: Static preprocessing basis for angles: ``gaussian``,
            ``bessel``, or ``fourier``. Use model-level ``learnable_gaussian`` when the
            angle basis should be trained end-to-end.
        angle_basis_kwargs: Extra options forwarded to the angle basis function.
        use_cosine_basis: If true, encode cosine values on ``[-1, 1]`` instead of angles
            in radians on ``[0, pi]``.
        skip_backtracking: If true, skip paths ``i -> j -> i``. These 180 degree paths are
            often less useful because they immediately return along the same site sequence.
        max_outgoing_neighbors: Optional cap on the number of outgoing ``j -> k`` bonds
            considered for each incoming ``i -> j`` bond. This is a practical memory control
            for ALIGNN-like models because line-graph edges scale roughly as the sum of
            squared coordination numbers.
        max_line_edges: Optional hard cap on the total number of line-graph edges for this
            crystal. When reached, construction stops adding more angle triplets. This is
            mainly a safety valve for very large structures or overly dense neighbor graphs.
        line_neighbor_selection: How to choose outgoing bonds when
            ``max_outgoing_neighbors`` is active. ``nearest`` keeps the shortest outgoing
            bonds. ``first`` keeps the stable original order.
        eps: Numerical stability constant for norms.

    Returns:
        Dictionary containing ``line_edge_index`` and ``line_edge_attr``. Diagnostic
        ``angle`` and ``cosine`` tensors are also returned for testing and analysis.
    """

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_vec.ndim != 2 or edge_vec.shape[1] != 3:
        raise ValueError("edge_vec must have shape [num_edges, 3]")
    if edge_index.shape[1] != edge_vec.shape[0]:
        raise ValueError("edge_index and edge_vec disagree on num_edges")
    if max_outgoing_neighbors is not None and max_outgoing_neighbors <= 0:
        raise ValueError("max_outgoing_neighbors must be positive when provided")
    if max_line_edges is not None and max_line_edges < 0:
        raise ValueError("max_line_edges must be non-negative when provided")
    if line_neighbor_selection not in {"nearest", "first"}:
        raise ValueError("line_neighbor_selection must be 'nearest' or 'first'")

    device = edge_vec.device
    dtype = edge_vec.dtype
    num_edges = int(edge_index.shape[1])

    sources = edge_index[0].detach().cpu().tolist()
    destinations = edge_index[1].detach().cpu().tolist()

    incoming: list[list[int]] = [[] for _ in range(num_nodes)]
    outgoing: list[list[int]] = [[] for _ in range(num_nodes)]
    for edge_id, (src, dst) in enumerate(zip(sources, destinations, strict=True)):
        if not (0 <= src < num_nodes and 0 <= dst < num_nodes):
            raise ValueError("edge_index contains node id outside [0, num_nodes)")
        incoming[dst].append(edge_id)
        outgoing[src].append(edge_id)

    line_pairs: list[tuple[int, int]] = []
    cosines: list[Tensor] = []
    angles: list[Tensor] = []

    for center in range(num_nodes):
        # Incoming bond is i -> j. The vector from center j back toward i is -edge_vec.
        for in_edge in incoming[center]:
            previous_site = sources[in_edge]
            v_in = -edge_vec[in_edge]
            in_norm = torch.linalg.norm(v_in).clamp_min(eps)

            # Outgoing bond is j -> k. The angle is between j->i and j->k.
            candidate_out_edges = [
                out_edge
                for out_edge in outgoing[center]
                if not (skip_backtracking and previous_site == destinations[out_edge])
            ]
            if max_outgoing_neighbors is not None and len(candidate_out_edges) > max_outgoing_neighbors:
                if line_neighbor_selection == "nearest":
                    candidate_out_edges = sorted(
                        candidate_out_edges,
                        key=lambda edge_id: float(torch.linalg.norm(edge_vec[edge_id]).detach().cpu()),
                    )
                candidate_out_edges = candidate_out_edges[:max_outgoing_neighbors]

            for out_edge in candidate_out_edges:
                if max_line_edges is not None and len(line_pairs) >= max_line_edges:
                    break

                v_out = edge_vec[out_edge]
                out_norm = torch.linalg.norm(v_out).clamp_min(eps)
                cos_theta = torch.dot(v_in, v_out) / (in_norm * out_norm)
                cos_theta = cos_theta.clamp(-1.0, 1.0)
                theta = torch.acos(cos_theta)

                line_pairs.append((in_edge, out_edge))
                cosines.append(cos_theta)
                angles.append(theta)

            if max_line_edges is not None and len(line_pairs) >= max_line_edges:
                break
        if max_line_edges is not None and len(line_pairs) >= max_line_edges:
            break

    if line_pairs:
        line_edge_index = torch.tensor(line_pairs, dtype=torch.long, device=device).t().contiguous()
        cosine = torch.stack(cosines).to(device=device, dtype=dtype)
        angle = torch.stack(angles).to(device=device, dtype=dtype)
    else:
        line_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        cosine = torch.empty((0,), dtype=dtype, device=device)
        angle = torch.empty((0,), dtype=dtype, device=device)

    basis_input = cosine if use_cosine_basis else angle
    line_edge_attr = angle_basis_expansion(
        basis_input,
        num_basis=num_angle_rbf,
        use_cosine=use_cosine_basis,
        basis_type=angle_basis_type,
        **(angle_basis_kwargs or {}),
    ).to(dtype=dtype, device=device)

    return {
        "line_edge_index": line_edge_index,
        "line_edge_attr": line_edge_attr,
        "angle": angle,
        "cosine": cosine,
        "num_line_edges": int(line_edge_index.shape[1]),
        "num_bond_edges": num_edges,
    }


def add_line_graph(graph: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Return a shallow copy of ``graph`` with ALIGNN-style line-graph fields added."""

    updated = dict(graph)
    updated.update(
        build_line_graph(
            graph["edge_index"],
            graph["edge_vec"],
            int(graph["num_nodes"]),
            **kwargs,
        )
    )
    return updated
