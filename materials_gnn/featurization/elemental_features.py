"""Atom featurizers and optional elemental descriptor tables.

Atomic number embeddings are still the default because they are simple, trainable, and do
not assume a particular chemistry prior. For materials informatics experiments, however,
it is often useful to concatenate known elemental descriptors: electronegativity, periodic
row/group, covalent radius, valence-electron count, electron affinity, polarizability,
magnetic moment, ionization energy, and related properties.

Descriptor values are looked up by atomic number and normalized over the periodic table.
Missing values are filled with zero after normalization and, by default, accompanied by a
missing-value indicator. This is important because some requested properties are not
well-defined or consistently tabulated for every element. For example, atomic magnetic
moments depend strongly on chemical environment; this package only uses a tabulated value
when pymatgen exposes one and otherwise marks the descriptor as missing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

DEFAULT_ELEMENTAL_FEATURES: tuple[str, ...] = (
    "electronegativity",
    "group",
    "period",
    "covalent_radius",
    "valence_electrons",
    "electron_affinity",
    "polarizability",
    "magnetic_moment",
    "ionization_energy",
)

FEATURE_ALIASES: Mapping[str, str] = {
    "x": "electronegativity",
    "en": "electronegativity",
    "row": "period",
    "atomic_row": "period",
    "atomic_group": "group",
    "covalent_radius_angstrom": "covalent_radius",
    "nvalence": "valence_electrons",
    "valence": "valence_electrons",
    "ea": "electron_affinity",
    "first_ionization_energy": "ionization_energy",
    "ie": "ionization_energy",
    "atomic_polarizability": "polarizability",
    "spin_moment": "magnetic_moment",
}


def normalize_feature_name(name: str) -> str:
    """Normalize user-facing descriptor names and aliases."""

    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return FEATURE_ALIASES.get(key, key)


def parse_feature_names(names: Sequence[str] | str | None) -> tuple[str, ...]:
    """Parse ``None``, ``'default'``, comma-separated strings, or sequences."""

    if names is None:
        return ()
    if isinstance(names, str):
        if names.strip().lower() in {"", "none", "off", "false"}:
            return ()
        if names.strip().lower() == "default":
            return DEFAULT_ELEMENTAL_FEATURES
        names = [item.strip() for item in names.split(",") if item.strip()]
    parsed = tuple(normalize_feature_name(name) for name in names)
    unknown = [name for name in parsed if name not in AVAILABLE_ELEMENTAL_FEATURES]
    if unknown:
        valid = ", ".join(sorted(AVAILABLE_ELEMENTAL_FEATURES))
        raise ValueError(f"Unknown elemental descriptor(s) {unknown}. Valid options: {valid}")
    return parsed


def _float_or_none(value: Any) -> float | None:
    """Convert numeric or unit-bearing values to float, returning ``None`` if unavailable."""

    if value is None:
        return None
    try:
        if isinstance(value, str) and value.strip() in {"", "no data", "None"}:
            return None
        result = float(value)
        if np.isfinite(result):
            return result
    except Exception:
        pass
    try:
        # Some pymatgen objects are unit-bearing wrappers with magnitude-like attributes.
        for attr in ("value", "magnitude", "real"):
            if hasattr(value, attr):
                result = float(getattr(value, attr))
                if np.isfinite(result):
                    return result
    except Exception:
        return None
    return None


def _data_value(element: Any, *keys: str) -> float | None:
    data = getattr(element, "data", {}) or {}
    for key in keys:
        if key in data:
            value = _float_or_none(data[key])
            if value is not None:
                return value
    # Pymatgen data keys vary across versions; use case-insensitive matching as a fallback.
    lower_to_key = {str(k).lower(): k for k in data}
    for key in keys:
        matched = lower_to_key.get(key.lower())
        if matched is not None:
            value = _float_or_none(data[matched])
            if value is not None:
                return value
    return None


def _attr_value(element: Any, *attrs: str) -> float | None:
    for attr in attrs:
        try:
            value = getattr(element, attr)
        except Exception:
            continue
        value = value() if callable(value) else value
        value_f = _float_or_none(value)
        if value_f is not None:
            return value_f
    return None


def _electronegativity(element: Any) -> float | None:
    return _attr_value(element, "X")


def _group(element: Any) -> float | None:
    return _attr_value(element, "group")


def _period(element: Any) -> float | None:
    return _attr_value(element, "row")


def _covalent_radius(element: Any) -> float | None:
    # Prefer explicitly covalent radius when the pymatgen data table has it. Fall back to
    # atomic radius because older pymatgen versions expose that more consistently.
    return _data_value(element, "Covalent radius", "Covalent Radius") or _attr_value(
        element,
        "covalent_radius",
        "atomic_radius",
        "atomic_radius_calculated",
    )


def _valence_electrons(element: Any) -> float | None:
    direct = _attr_value(element, "NValence", "nvalence") or _data_value(
        element,
        "NValence",
        "Number of valence electrons",
        "Valence electrons",
    )
    if direct is not None:
        return direct

    # Fallback: count electrons in the outermost principal quantum shell. This is a useful
    # simple descriptor, though not a full chemistry model for transition metals.
    try:
        config = getattr(element, "full_electronic_structure")
    except Exception:
        config = None
    if config:
        try:
            max_n = max(int(item[0]) for item in config)
            return float(sum(float(item[2]) for item in config if int(item[0]) == max_n))
        except Exception:
            return None
    return None


def _electron_affinity(element: Any) -> float | None:
    return _attr_value(element, "electron_affinity") or _data_value(
        element,
        "Electron affinity",
        "Electron Affinity",
    )


def _polarizability(element: Any) -> float | None:
    return _attr_value(element, "atomic_polarizability", "polarizability") or _data_value(
        element,
        "Atomic polarizability",
        "Atomic Polarizability",
        "Polarizability",
    )


def _magnetic_moment(element: Any) -> float | None:
    # Environment-independent magnetic moments are not generally well-defined. Use only
    # tabulated values when available; otherwise the missing indicator carries the signal.
    return _attr_value(element, "magnetic_moment") or _data_value(
        element,
        "Magnetic moment",
        "Magnetic Moment",
    )


def _ionization_energy(element: Any) -> float | None:
    try:
        energies = getattr(element, "ionization_energies")
        if energies:
            value = _float_or_none(energies[0])
            if value is not None:
                return value
    except Exception:
        pass
    return _data_value(element, "First ionization energy", "Ionization energy", "Ionization Energy")


AVAILABLE_ELEMENTAL_FEATURES: Mapping[str, Callable[[Any], float | None]] = {
    "electronegativity": _electronegativity,
    "group": _group,
    "period": _period,
    "covalent_radius": _covalent_radius,
    "valence_electrons": _valence_electrons,
    "electron_affinity": _electron_affinity,
    "polarizability": _polarizability,
    "magnetic_moment": _magnetic_moment,
    "ionization_energy": _ionization_energy,
}


@dataclass(frozen=True)
class ElementalDescriptorConfig:
    """Configuration for periodic-table descriptor lookup."""

    feature_names: Sequence[str] | str = DEFAULT_ELEMENTAL_FEATURES
    normalize: bool = True
    missing_value: float = 0.0
    add_missing_indicators: bool = True
    max_atomic_number: int = 118


class AtomFeaturizer(ABC, nn.Module):
    """Base interface for atom featurizers."""

    @abstractmethod
    def forward(self, atomic_numbers: Tensor) -> Tensor:
        """Return atom feature matrix of shape ``[num_atoms, feature_dim]``."""


class AtomicNumberEmbedding(AtomFeaturizer):
    """Learned embedding table indexed by atomic number."""

    def __init__(self, embedding_dim: int, max_atomic_number: int = 118) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_atomic_number + 1, embedding_dim, padding_idx=0)

    @property
    def output_dim(self) -> int:
        return self.embedding.embedding_dim

    def forward(self, atomic_numbers: Tensor) -> Tensor:
        z = atomic_numbers.long()
        if torch.any(z < 0):
            raise ValueError("Atomic numbers must be non-negative")
        return self.embedding(z)


class ElementalDescriptorFeaturizer(AtomFeaturizer):
    """Lookup normalized elemental descriptors by atomic number.

    The descriptor table is a non-trainable buffer. This keeps static chemistry priors
    separate from learned atomic-number embeddings while making it easy to change the list
    of included features through configuration.
    """

    def __init__(
        self,
        feature_names: Sequence[str] | str = DEFAULT_ELEMENTAL_FEATURES,
        *,
        normalize: bool = True,
        missing_value: float = 0.0,
        add_missing_indicators: bool = True,
        max_atomic_number: int = 118,
    ) -> None:
        super().__init__()
        self.feature_names = parse_feature_names(feature_names)
        self.normalize = normalize
        self.missing_value = float(missing_value)
        self.add_missing_indicators = bool(add_missing_indicators)
        self.max_atomic_number = int(max_atomic_number)
        table, missing_mask = self._build_table()
        self.register_buffer("feature_table", torch.tensor(table, dtype=torch.float32))
        self.register_buffer("missing_mask", torch.tensor(missing_mask, dtype=torch.float32))
        if self.add_missing_indicators:
            full_table = np.concatenate([table, missing_mask], axis=1)
        else:
            full_table = table
        self.register_buffer("full_feature_table", torch.tensor(full_table, dtype=torch.float32))
        self.output_dim = int(full_table.shape[1])

    @classmethod
    def from_config(cls, config: ElementalDescriptorConfig) -> "ElementalDescriptorFeaturizer":
        return cls(
            config.feature_names,
            normalize=config.normalize,
            missing_value=config.missing_value,
            add_missing_indicators=config.add_missing_indicators,
            max_atomic_number=config.max_atomic_number,
        )

    def _build_table(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.feature_names:
            return (
                np.zeros((self.max_atomic_number + 1, 0), dtype=np.float32),
                np.zeros((self.max_atomic_number + 1, 0), dtype=np.float32),
            )
        try:
            from pymatgen.core.periodic_table import Element
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "pymatgen is required for elemental descriptors. Install with `pip install pymatgen`."
            ) from exc

        raw = np.full((self.max_atomic_number + 1, len(self.feature_names)), np.nan, dtype=float)
        for z in range(1, self.max_atomic_number + 1):
            try:
                element = Element.from_Z(z)
            except Exception:
                continue
            for j, name in enumerate(self.feature_names):
                value = AVAILABLE_ELEMENTAL_FEATURES[name](element)
                raw[z, j] = np.nan if value is None else value

        missing = np.isnan(raw).astype(np.float32)
        if self.normalize:
            valid = np.isfinite(raw[1:])
            counts = valid.sum(axis=0)
            sums = np.where(valid, raw[1:], 0.0).sum(axis=0)
            means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
            centered = np.where(valid, raw[1:] - means[None, :], 0.0)
            variances = np.divide(
                (centered**2).sum(axis=0),
                counts,
                out=np.ones_like(sums),
                where=counts > 0,
            )
            stds = np.sqrt(variances)
            stds = np.where((np.isfinite(stds)) & (stds > 1e-12), stds, 1.0)
            raw = (raw - means[None, :]) / stds[None, :]
        table = np.where(np.isnan(raw), self.missing_value, raw).astype(np.float32)
        table[0, :] = 0.0
        missing[0, :] = 1.0
        return table, missing

    def forward(self, atomic_numbers: Tensor) -> Tensor:
        z = atomic_numbers.long()
        if torch.any(z < 0) or torch.any(z > self.max_atomic_number):
            raise ValueError(f"Atomic numbers must be in [0, {self.max_atomic_number}]")
        return self.full_feature_table.to(device=z.device)[z]

    def feature_labels(self) -> list[str]:
        labels = list(self.feature_names)
        if self.add_missing_indicators:
            labels.extend(f"{name}_missing" for name in self.feature_names)
        return labels


class AtomFeatureEncoder(AtomFeaturizer):
    """Combine learned atomic-number embeddings with optional descriptor features.

    ``atom_attr`` can be supplied by the graph builder for future site-specific or
    task-specific atom features. If no external atom attributes are supplied, the encoder
    can look up configured periodic-table descriptors directly from atomic numbers.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        max_atomic_number: int = 118,
        descriptor_names: Sequence[str] | str | None = None,
        descriptor_kwargs: Mapping[str, Any] | None = None,
        external_feature_dim: int | None = None,
        combine: str = "concat_project",
    ) -> None:
        super().__init__()
        self.embedding = AtomicNumberEmbedding(embedding_dim, max_atomic_number=max_atomic_number)
        self.combine = combine
        descriptor_names_parsed = parse_feature_names(descriptor_names)
        self.descriptor_featurizer: ElementalDescriptorFeaturizer | None = None
        descriptor_dim = 0
        if descriptor_names_parsed:
            kwargs = dict(descriptor_kwargs or {})
            kwargs.setdefault("max_atomic_number", max_atomic_number)
            self.descriptor_featurizer = ElementalDescriptorFeaturizer(descriptor_names_parsed, **kwargs)
            descriptor_dim = self.descriptor_featurizer.output_dim
        elif external_feature_dim is not None:
            descriptor_dim = int(external_feature_dim)

        self.descriptor_dim = descriptor_dim
        if descriptor_dim > 0:
            if combine == "concat_project":
                self.merge = nn.Sequential(nn.Linear(embedding_dim + descriptor_dim, embedding_dim), nn.SiLU())
            elif combine == "sum_project":
                self.descriptor_projection = nn.Linear(descriptor_dim, embedding_dim)
                self.merge = None
            else:
                raise ValueError("combine must be 'concat_project' or 'sum_project'")
        else:
            self.merge = None
        self.output_dim = embedding_dim

    def forward(self, atomic_numbers: Tensor, atom_attr: Tensor | None = None) -> Tensor:  # type: ignore[override]
        embedding = self.embedding(atomic_numbers)
        descriptors = atom_attr
        if descriptors is None and self.descriptor_featurizer is not None:
            descriptors = self.descriptor_featurizer(atomic_numbers)
        if descriptors is None:
            return embedding
        descriptors = descriptors.to(device=embedding.device, dtype=embedding.dtype)
        if self.descriptor_dim and descriptors.shape[-1] != self.descriptor_dim:
            raise ValueError(
                f"Expected atom_attr dimension {self.descriptor_dim}, got {descriptors.shape[-1]}"
            )
        if self.combine == "concat_project":
            if self.merge is None:
                return embedding
            return self.merge(torch.cat([embedding, descriptors], dim=-1))
        return embedding + self.descriptor_projection(descriptors)  # type: ignore[attr-defined]
