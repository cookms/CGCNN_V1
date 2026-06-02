from __future__ import annotations

import pytest

from materials_gnn.featurization.crystal_graph import structure_to_bond_graph
from materials_gnn.featurization.line_graph import add_line_graph


@pytest.mark.skipif(pytest.importorskip("pymatgen", reason="pymatgen is optional in CI") is None, reason="pymatgen missing")
def test_structure_to_bond_graph_and_line_graph() -> None:
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(5.43)
    structure = Structure(lattice, ["Si", "Si"], [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])

    graph = structure_to_bond_graph(structure, cutoff=4.0, num_rbf=12)
    graph = add_line_graph(graph, num_angle_rbf=6)

    assert graph["z"].shape[0] == 2
    assert graph["edge_index"].shape[0] == 2
    assert graph["edge_attr"].shape[1] == 12
    assert graph["line_edge_index"].shape[0] == 2
    assert graph["line_edge_attr"].shape[1] == 6
