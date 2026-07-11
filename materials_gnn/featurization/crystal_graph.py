"""Periodic crystal graph construction with pymatgen.

Molecules can usually be represented by a finite graph. Crystals are different: atoms in
one unit cell interact with periodic images of atoms in neighboring unit cells. A periodic
neighbor graph therefore stores not only which two sites are connected, but also the image
translation used to reach the neighbor. The resulting displacement vector gives the real
Cartesian bond direction and length.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from materials_gnn.featurization.basis import radial_basis_expansion
from materials_gnn.featurization.elemental_features import ElementalDescriptorFeaturizer, parse_feature_names
from materials_gnn.featurization.neighbor_strategies import NeighborStrategy, make_neighbor_strategy


def _require_pymatgen() -> None:
    try:
        import pymatgen  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only when dependency absent
        raise ImportError(
            "pymatgen is required for crystal graph construction. Install with `pip install pymatgen`."
        ) from exc


def _site_atomic_number(site: Any) -> int:
    """Return a representative atomic number for ordered or lightly disordered sites."""

    try:
        return int(site.specie.Z)
    except Exception:
        # For disordered sites, use the most occupied species as a pragmatic first prototype.
        species_items = list(site.species.items())
        if not species_items:
            raise ValueError(f"Could not infer atomic number for site {site!r}")
        element, _occupancy = max(species_items, key=lambda item: float(item[1]))
        return int(element.Z)


def structure_to_bond_graph(
    structure: Any,
    *,
    cutoff: float = 5.0,
    num_rbf: int = 64,
    rbf_start: float = 0.0,
    rbf_cutoff: float | None = None,
    distance_basis_type: str = "gaussian",
    distance_basis_kwargs: dict[str, Any] | None = None,
    atom_feature_names: str | list[str] | tuple[str, ...] | None = None,
    atom_feature_kwargs: dict[str, Any] | None = None,
    neighbor_strategy: str | NeighborStrategy | None = None,
    neighbor_kwargs: dict[str, Any] | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor | int]:
    """Convert a ``pymatgen.Structure`` into a directed periodic bond graph.

    Args:
        structure: ``pymatgen.core.Structure`` object.
        cutoff: Backward-compatible default cutoff in Angstrom. If
            ``neighbor_strategy`` is omitted, this is the periodic neighbor cutoff. For
            non-cutoff strategies it is also used as the default radial-basis scale.
        num_rbf: Number of radial basis functions used to encode distances.
        rbf_start: First radial basis center.
        rbf_cutoff: Final distance-basis center. Leave as ``None`` to use ``cutoff``. For
            KNN or Voronoi experiments, set this to a fixed length scale that covers the
            expected graph distances across the whole dataset.
        distance_basis_type: Static preprocessing basis for distances: ``gaussian``,
            ``bessel``, or ``fourier``. Use model-level ``learnable_gaussian`` when the
            basis should be trained end-to-end from raw distances.
        distance_basis_kwargs: Extra options forwarded to the distance basis function.
        atom_feature_names: Optional descriptor names, comma-separated string, sequence,
            or ``"default"``. If supplied, the graph includes ``atom_attr``.
        atom_feature_kwargs: Options for ``ElementalDescriptorFeaturizer`` such as
            ``normalize`` or ``add_missing_indicators``.
        neighbor_strategy: ``None``/``"cutoff"`` for fixed cutoff neighbors, one of
            ``"knn"``, ``"voronoi"``, ``"adaptive_shell"``, or a custom strategy object
            implementing ``build(structure)``.
        neighbor_kwargs: Strategy-specific options such as ``{"k": 12}``,
            ``{"max_radius": 10.0}``, or Voronoi ``{"failure_policy": "raise"}``.
        dtype: Floating-point dtype for geometric tensors.

    Returns:
        Dictionary with atomic numbers, directed edge indices, periodic displacement
        vectors, distances, radial basis edge features, and ``num_nodes``. Optional
        ``edge_weight`` is included when the strategy provides geometric weights.
    """

    _require_pymatgen()
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if rbf_cutoff is None:
        rbf_cutoff = cutoff
    if rbf_cutoff <= rbf_start:
        raise ValueError("rbf_cutoff must be greater than rbf_start")

    num_nodes = len(structure)
    z = torch.tensor([_site_atomic_number(site) for site in structure], dtype=torch.long)

    strategy = make_neighbor_strategy(
        neighbor_strategy,
        cutoff=cutoff,
        strategy_kwargs=neighbor_kwargs,
    )
    neighbor_list = strategy.build(structure)
    center_indices = neighbor_list.center_indices
    neighbor_indices = neighbor_list.neighbor_indices
    image_vectors = neighbor_list.image_vectors
    distances = neighbor_list.distances

    if len(center_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_vec = torch.empty((0, 3), dtype=dtype)
        distance = torch.empty((0,), dtype=dtype)
        edge_attr = torch.empty((0, num_rbf), dtype=dtype)
    else:
        cart_coords = np.asarray(structure.cart_coords, dtype=float)
        lattice_matrix = np.asarray(structure.lattice.matrix, dtype=float)
        image_cart = image_vectors @ lattice_matrix

        # Directed edge i -> j points from the center atom i to the periodic image of j.
        displacement = cart_coords[neighbor_indices] + image_cart - cart_coords[center_indices]

        edge_index = torch.tensor(np.vstack([center_indices, neighbor_indices]), dtype=torch.long)
        edge_vec = torch.tensor(displacement, dtype=dtype)
        distance = torch.tensor(distances, dtype=dtype)
        edge_attr = radial_basis_expansion(
            distance,
            num_basis=num_rbf,
            start=rbf_start,
            cutoff=rbf_cutoff,
            basis_type=distance_basis_type,
            **(distance_basis_kwargs or {}),
        ).to(dtype=dtype)

    graph: dict[str, Tensor | int] = {
        "z": z,
        "edge_index": edge_index,
        "edge_vec": edge_vec,
        "distance": distance,
        "edge_attr": edge_attr,
        "edge_unit_vec": edge_vec / distance.clamp_min(1e-12).unsqueeze(-1) if distance.numel() else edge_vec,
        "num_nodes": num_nodes,
        "pos": torch.tensor(np.asarray(structure.cart_coords, dtype=float), dtype=dtype),
    }

    descriptor_names = parse_feature_names(atom_feature_names)
    if descriptor_names:
        descriptor_featurizer = ElementalDescriptorFeaturizer(
            descriptor_names,
            **(atom_feature_kwargs or {}),
        )
        graph["atom_attr"] = descriptor_featurizer(z).to(dtype=dtype)
    if neighbor_list.weights is not None:
        graph["edge_weight"] = torch.tensor(neighbor_list.weights, dtype=dtype)
    return graph


def cif_to_bond_graph(cif_path: str | Path, **kwargs: Any) -> dict[str, Tensor | int]:
    """Load a CIF file and convert it to a periodic bond graph."""

    _require_pymatgen()
    from pymatgen.core import Structure

    structure = Structure.from_file(str(cif_path))
    return structure_to_bond_graph(structure, **kwargs)
