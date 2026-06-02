from __future__ import annotations

import pytest
import torch

from materials_gnn.featurization import (
    ElementalDescriptorFeaturizer,
    angle_basis_expansion,
    make_basis_expansion,
    radial_basis_expansion,
)


def test_static_basis_expansions_have_requested_shape() -> None:
    distances = torch.tensor([1.0, 2.0, 3.0])
    for basis_type in ["gaussian", "bessel", "fourier"]:
        expanded = radial_basis_expansion(
            distances,
            num_basis=7,
            start=0.0,
            cutoff=5.0,
            basis_type=basis_type,
        )
        assert expanded.shape == (3, 7)
        assert torch.isfinite(expanded).all()

    angles = torch.tensor([0.0, 1.57, 3.14])
    expanded_angle = angle_basis_expansion(angles, num_basis=5, basis_type="fourier")
    assert expanded_angle.shape == (3, 5)


def test_learnable_gaussian_basis_has_trainable_parameters() -> None:
    basis = make_basis_expansion(
        "learnable_gaussian",
        start=0.0,
        cutoff=5.0,
        num_basis=4,
    )
    values = torch.tensor([1.0, 2.0])
    out = basis(values)
    assert out.shape == (2, 4)
    assert any(parameter.requires_grad for parameter in basis.parameters())


def test_elemental_descriptor_featurizer_if_pymatgen_available() -> None:
    pytest.importorskip("pymatgen")
    featurizer = ElementalDescriptorFeaturizer(
        ["electronegativity", "group", "period", "valence_electrons"],
        add_missing_indicators=True,
    )
    z = torch.tensor([6, 8, 14])
    features = featurizer(z)
    assert features.shape == (3, 8)
    assert torch.isfinite(features).all()
    assert featurizer.feature_labels()[-1] == "valence_electrons_missing"
