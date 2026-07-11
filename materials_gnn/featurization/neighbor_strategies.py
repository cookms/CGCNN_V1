"""Interchangeable periodic neighbor graph construction strategies.

A crystal graph is only as good as its neighborhood definition. The same crystal can be
represented with a fixed distance cutoff, a fixed number of nearest neighbors, Voronoi
coordination, or more experimental constructions. Keeping this logic behind small strategy
classes makes graph topology a first-class research variable instead of a hard-coded
preprocessing detail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class NeighborList:
    """Periodic directed neighbors returned by a construction strategy.

    The arrays describe directed edges ``center_indices[e] -> neighbor_indices[e]``.
    ``image_vectors[e]`` is the integer lattice image of the neighbor atom used for that
    edge. For example, image ``[1, 0, 0]`` means the neighbor is in the adjacent unit cell
    displaced by one lattice vector along ``a``.

    ``weights`` is optional. Voronoi and experimental strategies can attach geometric
    confidence, face-area weight, or ensemble survival probability without changing the
    core model API. Existing models may ignore it, while future layers can consume it.
    """

    center_indices: np.ndarray
    neighbor_indices: np.ndarray
    image_vectors: np.ndarray
    distances: np.ndarray
    weights: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_indices", np.asarray(self.center_indices, dtype=np.int64))
        object.__setattr__(self, "neighbor_indices", np.asarray(self.neighbor_indices, dtype=np.int64))
        object.__setattr__(self, "image_vectors", np.asarray(self.image_vectors, dtype=float))
        object.__setattr__(self, "distances", np.asarray(self.distances, dtype=float))
        if self.weights is not None:
            object.__setattr__(self, "weights", np.asarray(self.weights, dtype=float))

        n = len(self.center_indices)
        if len(self.neighbor_indices) != n or len(self.image_vectors) != n or len(self.distances) != n:
            raise ValueError("NeighborList arrays must all have the same length")
        if self.image_vectors.ndim != 2 or self.image_vectors.shape[1] != 3:
            raise ValueError("image_vectors must have shape [num_edges, 3]")
        if self.weights is not None and len(self.weights) != n:
            raise ValueError("weights must have shape [num_edges]")

    @classmethod
    def empty(cls) -> "NeighborList":
        """Create an empty neighbor list with the correct array shapes."""

        return cls(
            center_indices=np.empty((0,), dtype=np.int64),
            neighbor_indices=np.empty((0,), dtype=np.int64),
            image_vectors=np.empty((0, 3), dtype=float),
            distances=np.empty((0,), dtype=float),
        )


@runtime_checkable
class NeighborStrategy(Protocol):
    """Protocol implemented by all periodic neighbor strategies."""

    name: str

    def build(self, structure: Any) -> NeighborList:
        """Return directed periodic neighbors for a ``pymatgen.Structure``."""


class VoronoiNeighborError(RuntimeError):
    """Raised when pymatgen cannot construct a complete Voronoi neighbor graph."""


VoronoiFailurePolicy = Literal["raise", "empty", "cutoff"]


def _call_get_neighbor_list(structure: Any, cutoff: float, *, exclude_self: bool = True) -> NeighborList:
    """Call pymatgen's neighbor-list API with version-tolerant arguments."""

    try:
        center_indices, neighbor_indices, image_vectors, distances = structure.get_neighbor_list(
            r=cutoff,
            exclude_self=exclude_self,
        )
    except TypeError:  # pragma: no cover - compatibility with older pymatgen releases
        center_indices, neighbor_indices, image_vectors, distances = structure.get_neighbor_list(cutoff)

    return NeighborList(
        center_indices=np.asarray(center_indices, dtype=np.int64),
        neighbor_indices=np.asarray(neighbor_indices, dtype=np.int64),
        image_vectors=np.asarray(image_vectors, dtype=float),
        distances=np.asarray(distances, dtype=float),
    )


@dataclass(frozen=True)
class CutoffNeighborStrategy:
    """Include every periodic neighbor within a fixed cutoff radius.

    This is the usual CGCNN-style default. It is simple, deterministic, and physically
    interpretable: every edge means an atom lies within ``cutoff`` Angstrom of a central
    atom after considering periodic images.
    """

    cutoff: float = 5.0
    exclude_self: bool = True
    name: str = "cutoff"

    def build(self, structure: Any) -> NeighborList:
        if self.cutoff <= 0:
            raise ValueError("cutoff must be positive")
        return _call_get_neighbor_list(structure, self.cutoff, exclude_self=self.exclude_self)


@dataclass(frozen=True)
class KNearestNeighborStrategy:
    """Keep the ``k`` shortest periodic neighbors around each atom.

    KNN graphs control the number of outgoing edges per atom, which can stabilize memory
    usage across dense and open frameworks. The search still starts from a periodic cutoff
    list; ``max_radius`` should be large enough that every site can find at least ``k``
    neighbors in the crystals being studied.
    """

    k: int = 12
    max_radius: float = 8.0
    min_radius: float = 4.0
    allow_incomplete: bool = True
    name: str = "knn"

    def build(self, structure: Any) -> NeighborList:
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.max_radius <= 0:
            raise ValueError("max_radius must be positive")
        if self.min_radius <= 0:
            raise ValueError("min_radius must be positive")
        if self.max_radius < self.min_radius:
            raise ValueError("max_radius must be greater than or equal to min_radius")

        # Expand the radius geometrically until all atoms have at least k candidates or the
        # maximum search radius is reached. This avoids surprising empty KNN graphs for
        # low-density crystals while keeping preprocessing deterministic.
        radius = min(self.min_radius, self.max_radius)
        neighbors = _call_get_neighbor_list(structure, radius, exclude_self=True)
        while radius < self.max_radius:
            counts = np.bincount(neighbors.center_indices, minlength=len(structure))
            if counts.size >= len(structure) and np.all(counts[: len(structure)] >= self.k):
                break
            radius = min(self.max_radius, radius * 1.5)
            neighbors = _call_get_neighbor_list(structure, radius, exclude_self=True)

        selected: list[int] = []
        for center in range(len(structure)):
            candidate_ids = np.where(neighbors.center_indices == center)[0]
            if candidate_ids.size == 0:
                if self.allow_incomplete:
                    continue
                raise ValueError(f"Atom {center} has no neighbors within max_radius={self.max_radius}")
            # Stable lexsort gives deterministic tie-breaking by distance, neighbor index,
            # image a, image b, image c. Ties can happen in high-symmetry crystals.
            order = np.lexsort(
                (
                    neighbors.image_vectors[candidate_ids, 2],
                    neighbors.image_vectors[candidate_ids, 1],
                    neighbors.image_vectors[candidate_ids, 0],
                    neighbors.neighbor_indices[candidate_ids],
                    neighbors.distances[candidate_ids],
                )
            )
            chosen = candidate_ids[order[: self.k]]
            if chosen.size < self.k and not self.allow_incomplete:
                raise ValueError(
                    f"Atom {center} has only {chosen.size} neighbors within max_radius={self.max_radius}; "
                    "increase max_radius or set allow_incomplete=True."
                )
            selected.extend(int(idx) for idx in chosen)

        if not selected:
            return NeighborList.empty()
        selected_array = np.asarray(selected, dtype=np.int64)
        return NeighborList(
            center_indices=neighbors.center_indices[selected_array],
            neighbor_indices=neighbors.neighbor_indices[selected_array],
            image_vectors=neighbors.image_vectors[selected_array],
            distances=neighbors.distances[selected_array],
            metadata={"search_radius": radius, "k": self.k},
        )


@dataclass(frozen=True)
class VoronoiNeighborStrategy:
    """Construct neighbors from periodic Voronoi coordination polyhedra.

    Voronoi neighbors share a Voronoi face with the central atom. This is often a more
    chemistry-aware notion of coordination than a global cutoff, especially when bond
    lengths vary strongly across elements or oxidation states. The returned ``weights`` are
    normalized Voronoi face weights from pymatgen when available.

    ``failure_policy`` applies to both pymatgen Voronoi failures and any center atom for
    which pymatgen returns no neighbors. The default, ``"raise"``, rejects the whole graph
    with contextual diagnostics. ``"empty"`` returns ``NeighborList.empty()`` and
    ``"cutoff"`` rebuilds the whole graph with ``CutoffNeighborStrategy``. A partially
    constructed Voronoi graph is never returned.
    """

    tol: float = 0.0
    cutoff: float = 10.0
    allow_pathological: bool = True
    name: str = "voronoi"
    failure_policy: VoronoiFailurePolicy = "raise"

    def _handle_failure(
        self,
        structure: Any,
        *,
        atom_index: int,
        detail: str,
        cause: Exception | None = None,
    ) -> NeighborList:
        structure_size = len(structure)
        message = (
            "Voronoi neighbor construction failed for "
            f"atom index {atom_index} in structure size {structure_size} "
            f"(cutoff={self.cutoff}, tolerance={self.tol}, "
            f"allow_pathological={self.allow_pathological}): {detail}. "
            "No partial Voronoi graph was returned."
        )
        if self.failure_policy == "empty":
            return NeighborList.empty()
        if self.failure_policy == "cutoff":
            return CutoffNeighborStrategy(cutoff=self.cutoff).build(structure)

        error = VoronoiNeighborError(message)
        if cause is not None:
            raise error from cause
        raise error

    def build(self, structure: Any) -> NeighborList:
        if self.cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if self.failure_policy not in {"raise", "empty", "cutoff"}:
            raise ValueError("failure_policy must be one of: raise, empty, cutoff")

        try:
            from pymatgen.analysis.local_env import VoronoiNN
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("VoronoiNeighborStrategy requires pymatgen") from exc

        nn_finder = VoronoiNN(
            tol=self.tol,
            cutoff=self.cutoff,
            allow_pathological=self.allow_pathological,
        )
        nn_info_by_center: list[list[dict[str, Any]]] = []
        missing_centers: list[int] = []
        structure_size = len(structure)
        for center in range(structure_size):
            try:
                center_info = list(nn_finder.get_nn_info(structure, center))
            except (RuntimeError, ValueError) as exc:
                return self._handle_failure(
                    structure,
                    atom_index=center,
                    detail=(
                        "pymatgen VoronoiNN.get_nn_info() raised "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    cause=exc,
                )
            nn_info_by_center.append(center_info)
            if not center_info:
                missing_centers.append(center)

        if missing_centers:
            if len(missing_centers) == structure_size:
                detail = "pymatgen VoronoiNN.get_nn_info() returned no neighbors for every atom"
            else:
                detail = (
                    "pymatgen VoronoiNN.get_nn_info() returned no neighbors for "
                    f"atom index {missing_centers[0]}"
                )
            return self._handle_failure(
                structure,
                atom_index=missing_centers[0],
                detail=detail,
            )

        centers: list[int] = []
        neighbors: list[int] = []
        images: list[np.ndarray] = []
        distances: list[float] = []
        weights: list[float] = []

        frac_coords = np.asarray(structure.frac_coords, dtype=float)
        cart_coords = np.asarray(structure.cart_coords, dtype=float)
        lattice_matrix = np.asarray(structure.lattice.matrix, dtype=float)

        for center, center_info in enumerate(nn_info_by_center):
            for info in center_info:
                neighbor_index = int(info.get("site_index"))
                image = info.get("image")
                site = info.get("site")

                if image is None:
                    # pymatgen's PeriodicNeighbor usually includes an image, but derive it
                    # from fractional coordinates when the key is absent for compatibility.
                    site_frac = np.asarray(site.frac_coords, dtype=float)
                    image_array = np.rint(site_frac - frac_coords[neighbor_index]).astype(int)
                else:
                    image_array = np.asarray(image, dtype=int)

                displacement = cart_coords[neighbor_index] + image_array @ lattice_matrix - cart_coords[center]
                centers.append(center)
                neighbors.append(neighbor_index)
                images.append(image_array)
                distances.append(float(np.linalg.norm(displacement)))
                weights.append(float(info.get("weight", 1.0)))

        if not centers:  # Only possible for an empty structure; nonempty failures are handled above.
            return NeighborList.empty()
        return NeighborList(
            center_indices=np.asarray(centers, dtype=np.int64),
            neighbor_indices=np.asarray(neighbors, dtype=np.int64),
            image_vectors=np.asarray(images, dtype=float),
            distances=np.asarray(distances, dtype=float),
            weights=np.asarray(weights, dtype=float),
        )


@dataclass(frozen=True)
class AdaptiveShellNeighborStrategy:
    """Experimental local-shell graph based on distance gaps around each atom.

    This strategy is intentionally research-facing. Instead of using one global cutoff or a
    fixed K, it sorts each atom's periodic neighbors and chooses a local coordination shell:
    keep at least ``min_neighbors``; then continue adding neighbors until a large distance
    gap appears, the ratio to the first neighbor exceeds ``max_shell_ratio``, or
    ``max_neighbors`` is reached.

    The motivation is that open frameworks, dense metals, and mixed-anion crystals often
    have very different natural coordination radii. A local shell rule may preserve the
    first chemically relevant coordination environment while avoiding graph-size explosion.
    It is provided as an experimental baseline, not as a claim of novelty or superiority.
    """

    max_radius: float = 8.0
    min_neighbors: int = 4
    max_neighbors: int = 24
    gap_scale: float = 1.35
    max_shell_ratio: float = 1.65
    allow_incomplete: bool = True
    name: str = "adaptive_shell"

    def build(self, structure: Any) -> NeighborList:
        if self.max_radius <= 0:
            raise ValueError("max_radius must be positive")
        if self.min_neighbors <= 0:
            raise ValueError("min_neighbors must be positive")
        if self.max_neighbors < self.min_neighbors:
            raise ValueError("max_neighbors must be greater than or equal to min_neighbors")
        if self.gap_scale <= 1.0:
            raise ValueError("gap_scale should be greater than 1.0")
        if self.max_shell_ratio <= 1.0:
            raise ValueError("max_shell_ratio should be greater than 1.0")

        candidates = _call_get_neighbor_list(structure, self.max_radius, exclude_self=True)
        selected: list[int] = []
        edge_weights: list[float] = []

        for center in range(len(structure)):
            candidate_ids = np.where(candidates.center_indices == center)[0]
            if candidate_ids.size == 0:
                if self.allow_incomplete:
                    continue
                raise ValueError(f"Atom {center} has no neighbors within max_radius={self.max_radius}")

            order = np.lexsort(
                (
                    candidates.image_vectors[candidate_ids, 2],
                    candidates.image_vectors[candidate_ids, 1],
                    candidates.image_vectors[candidate_ids, 0],
                    candidates.neighbor_indices[candidate_ids],
                    candidates.distances[candidate_ids],
                )
            )
            sorted_ids = candidate_ids[order]
            sorted_distances = candidates.distances[sorted_ids]
            first_distance = max(float(sorted_distances[0]), 1e-8)

            keep_count = min(self.min_neighbors, len(sorted_ids))
            for pos in range(keep_count, min(len(sorted_ids), self.max_neighbors)):
                prev_d = max(float(sorted_distances[pos - 1]), 1e-8)
                this_d = float(sorted_distances[pos])
                local_gap = this_d / prev_d
                shell_ratio = this_d / first_distance
                if local_gap > self.gap_scale or shell_ratio > self.max_shell_ratio:
                    break
                keep_count = pos + 1

            chosen = sorted_ids[:keep_count]
            if chosen.size < self.min_neighbors and not self.allow_incomplete:
                raise ValueError(
                    f"Atom {center} has only {chosen.size} neighbors within max_radius={self.max_radius}; "
                    "increase max_radius or set allow_incomplete=True."
                )
            selected.extend(int(idx) for idx in chosen)

            # Give closer neighbors larger optional weights. Future message-passing layers
            # can use this as a soft edge prior, but current models ignore it.
            chosen_d = candidates.distances[chosen]
            scale = max(float(np.median(chosen_d)), 1e-8)
            edge_weights.extend(np.exp(-chosen_d / scale).tolist())

        if not selected:
            return NeighborList.empty()
        selected_array = np.asarray(selected, dtype=np.int64)
        return NeighborList(
            center_indices=candidates.center_indices[selected_array],
            neighbor_indices=candidates.neighbor_indices[selected_array],
            image_vectors=candidates.image_vectors[selected_array],
            distances=candidates.distances[selected_array],
            weights=np.asarray(edge_weights, dtype=float),
        )


@dataclass(frozen=True)
class StrainJitterConsensusNeighborStrategy:
    """Experimental graph: neighbors stable under small virtual lattice strains.

    This is a deliberately speculative construction for research experiments. It builds
    several virtual copies of the crystal under tiny deterministic lattice strains, collects
    cutoff-neighbor edges from each copy, and keeps edges that survive in at least
    ``min_survival_fraction`` of those local perturbations. The optional ``edge_weight`` is
    the survival frequency.

    Why this might be useful: DFT-relaxed structures, finite-temperature structures, and
    near-degenerate polymorphs can have edges close to a hard cutoff boundary. A consensus
    graph gives the model a soft notion of which local contacts are robust to small
    geometric uncertainty. Treat this as an exploratory hypothesis rather than an
    established materials-GNN method.
    """

    cutoff: float = 5.0
    strain_epsilon: float = 0.02
    min_survival_fraction: float = 0.5
    include_unstrained: bool = True
    name: str = "strain_consensus"

    def _deformation_matrices(self) -> list[np.ndarray]:
        eps = self.strain_epsilon
        identity = np.eye(3)
        matrices: list[np.ndarray] = [identity] if self.include_unstrained else []

        # Deterministic axial and shear probes. These are not physical strain trajectories;
        # they are small local tests for whether graph connectivity is cutoff-fragile.
        for axis in range(3):
            plus = np.eye(3)
            minus = np.eye(3)
            plus[axis, axis] += eps
            minus[axis, axis] -= eps
            matrices.extend([plus, minus])
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            plus = np.eye(3)
            minus = np.eye(3)
            plus[a, b] += eps
            plus[b, a] += eps
            minus[a, b] -= eps
            minus[b, a] -= eps
            matrices.extend([plus, minus])
        return matrices

    def _deformed_structure(self, structure: Any, deformation: np.ndarray) -> Any:
        from pymatgen.core import Lattice, Structure

        lattice_matrix = np.asarray(structure.lattice.matrix, dtype=float)
        # pymatgen stores lattice vectors as rows. Right-multiplying by deformation.T
        # applies the small Cartesian deformation to those row vectors.
        new_lattice = lattice_matrix @ deformation.T
        species = [site.species for site in structure]
        return Structure(
            Lattice(new_lattice),
            species,
            np.asarray(structure.frac_coords, dtype=float),
            coords_are_cartesian=False,
            site_properties=getattr(structure, "site_properties", None),
        )

    def build(self, structure: Any) -> NeighborList:
        if self.cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if self.strain_epsilon < 0:
            raise ValueError("strain_epsilon must be nonnegative")
        if not (0.0 < self.min_survival_fraction <= 1.0):
            raise ValueError("min_survival_fraction must be in (0, 1]")

        counts: dict[tuple[int, int, tuple[int, int, int]], int] = {}
        matrices = self._deformation_matrices()
        for deformation in matrices:
            variant = structure if np.allclose(deformation, np.eye(3)) else self._deformed_structure(structure, deformation)
            neighbors = _call_get_neighbor_list(variant, self.cutoff, exclude_self=True)
            for center, neighbor, image in zip(
                neighbors.center_indices,
                neighbors.neighbor_indices,
                neighbors.image_vectors,
                strict=True,
            ):
                image_key = tuple(int(v) for v in np.rint(image).astype(int).tolist())
                key = (int(center), int(neighbor), image_key)
                counts[key] = counts.get(key, 0) + 1

        if not counts:
            return NeighborList.empty()

        threshold = int(np.ceil(self.min_survival_fraction * len(matrices)))
        cart_coords = np.asarray(structure.cart_coords, dtype=float)
        lattice_matrix = np.asarray(structure.lattice.matrix, dtype=float)

        rows: list[tuple[int, int, tuple[int, int, int], float, float]] = []
        for (center, neighbor, image_key), count in counts.items():
            if count < threshold:
                continue
            image = np.asarray(image_key, dtype=float)
            displacement = cart_coords[neighbor] + image @ lattice_matrix - cart_coords[center]
            distance = float(np.linalg.norm(displacement))
            weight = float(count / len(matrices))
            rows.append((center, neighbor, image_key, distance, weight))

        if not rows:
            return NeighborList.empty()
        rows.sort(key=lambda row: (row[0], row[3], row[1], row[2]))
        return NeighborList(
            center_indices=np.asarray([row[0] for row in rows], dtype=np.int64),
            neighbor_indices=np.asarray([row[1] for row in rows], dtype=np.int64),
            image_vectors=np.asarray([row[2] for row in rows], dtype=float),
            distances=np.asarray([row[3] for row in rows], dtype=float),
            weights=np.asarray([row[4] for row in rows], dtype=float),
            metadata={
                "num_strain_probes": len(matrices),
                "min_survival_fraction": self.min_survival_fraction,
            },
        )


_STRATEGY_ALIASES = {
    "cutoff": CutoffNeighborStrategy,
    "radius": CutoffNeighborStrategy,
    "knn": KNearestNeighborStrategy,
    "k_nearest": KNearestNeighborStrategy,
    "k-nearest": KNearestNeighborStrategy,
    "voronoi": VoronoiNeighborStrategy,
    "adaptive_shell": AdaptiveShellNeighborStrategy,
    "adaptive-shell": AdaptiveShellNeighborStrategy,
    "shell": AdaptiveShellNeighborStrategy,
    "strain_consensus": StrainJitterConsensusNeighborStrategy,
    "strain-consensus": StrainJitterConsensusNeighborStrategy,
    "robust_cutoff": StrainJitterConsensusNeighborStrategy,
}


def make_neighbor_strategy(
    strategy: str | NeighborStrategy | None,
    *,
    cutoff: float = 5.0,
    strategy_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> NeighborStrategy:
    """Create a neighbor strategy from a string name or return a custom object.

    Args:
        strategy: Strategy name, existing strategy object, or ``None``. ``None`` defaults
            to ``CutoffNeighborStrategy(cutoff=cutoff)`` for backward compatibility.
        cutoff: Default cutoff used by the cutoff strategy and as a sensible minimum radius
            for KNN when the caller does not supply explicit values.
        strategy_kwargs: Strategy-specific constructor arguments supplied as a mapping.
            Use this form when an option such as ``cutoff`` has the same name as a factory
            argument.
        **kwargs: Additional strategy-specific constructor arguments. These must not
            duplicate keys in ``strategy_kwargs``.
    """

    constructor_kwargs = dict(strategy_kwargs or {})
    duplicate_keys = constructor_kwargs.keys() & kwargs.keys()
    if duplicate_keys:
        duplicates = ", ".join(sorted(duplicate_keys))
        raise TypeError(f"Strategy arguments provided more than once: {duplicates}")
    constructor_kwargs.update(kwargs)

    if strategy is None:
        constructor_kwargs.setdefault("cutoff", cutoff)
        return CutoffNeighborStrategy(**constructor_kwargs)
    if not isinstance(strategy, str):
        if not isinstance(strategy, NeighborStrategy):
            raise TypeError("Custom neighbor_strategy must implement build(structure) -> NeighborList")
        return strategy

    key = strategy.lower().strip()
    if key not in _STRATEGY_ALIASES:
        valid = ", ".join(sorted(_STRATEGY_ALIASES))
        raise ValueError(f"Unknown neighbor strategy {strategy!r}. Valid names: {valid}")

    cls = _STRATEGY_ALIASES[key]
    if cls is CutoffNeighborStrategy:
        constructor_kwargs.setdefault("cutoff", cutoff)
    elif cls is KNearestNeighborStrategy:
        constructor_kwargs.setdefault("min_radius", cutoff)
        constructor_kwargs.setdefault("max_radius", max(8.0, cutoff))
    elif cls is AdaptiveShellNeighborStrategy:
        constructor_kwargs.setdefault("max_radius", max(8.0, cutoff))
    elif cls is StrainJitterConsensusNeighborStrategy:
        constructor_kwargs.setdefault("cutoff", cutoff)
    return cls(**constructor_kwargs)  # type: ignore[return-value]
