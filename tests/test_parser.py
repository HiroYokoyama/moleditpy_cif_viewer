import numpy as np

from cif_viewer.parser import (
    celleditpy_cell_axis_segments,
    expand_supercell,
    parse_cif,
    parse_cif_number,
    supercell_edges,
)


NACL_CIF = """
data_NaCl
_cell_length_a    5.6402(2)
_cell_length_b    5.6402(2)
_cell_length_c    5.6402(2)
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Na1 Na 0 0 0 1
Cl1 Cl 0.5 0.5 0.5 1
"""

BOUNDARY_CIF = """
data_boundary
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.95 0 0
C2 C 0.05 0 0
"""


def test_parse_number_removes_uncertainty():
    assert parse_cif_number("5.6402(2)") == 5.6402


def test_parse_cif_cell_and_atoms():
    structure = parse_cif(NACL_CIF)

    assert structure.name == "NaCl"
    assert structure.cell_lengths == (5.6402, 5.6402, 5.6402)
    assert len(structure.atoms) == 2
    assert structure.atoms[0].element == "Na"
    np.testing.assert_allclose(structure.atoms[1].cart, [2.8201, 2.8201, 2.8201])


def test_expand_supercell_repeats_atoms():
    structure = parse_cif(NACL_CIF)
    atoms, bonds = expand_supercell(structure, (2, 1, 1))

    assert len(atoms) == 4
    assert atoms[2].image == (1, 0, 0)
    np.testing.assert_allclose(atoms[2].position, [5.6402, 0.0, 0.0])
    assert bonds == []


def test_expand_supercell_keeps_boundary_molecule_connected_by_default():
    structure = parse_cif(BOUNDARY_CIF)
    atoms, bonds = expand_supercell(structure, (1, 1, 1))

    assert bonds == [(0, 1)]
    np.testing.assert_allclose(atoms[0].position, [9.5, 0.0, 0.0])
    np.testing.assert_allclose(atoms[1].position, [10.5, 0.0, 0.0])


def test_expand_supercell_can_leave_atoms_inside_cell():
    structure = parse_cif(BOUNDARY_CIF)
    atoms, bonds = expand_supercell(structure, (1, 1, 1), keep_connected=False)

    assert bonds == []
    np.testing.assert_allclose(atoms[1].position, [0.5, 0.0, 0.0])


def test_supercell_edges_scale_lattice():
    structure = parse_cif(NACL_CIF)
    edges = supercell_edges(structure.lattice, (2, 1, 1))

    assert len(edges) == 12
    assert any(np.allclose(end - start, [11.2804, 0.0, 0.0]) for start, end in edges)


def test_celleditpy_axis_segments_use_abc_colors_and_labels():
    structure = parse_cif(NACL_CIF)
    segments = celleditpy_cell_axis_segments(structure.lattice)

    assert len(segments) == 12
    assert [(segment[2], segment[3]) for segment in segments[:3]] == [
        ("red", "a"),
        ("green", "b"),
        ("blue", "c"),
    ]
    np.testing.assert_allclose(segments[0][1], structure.lattice[0])


def test_cartesian_atom_site_loop_is_supported():
    cif = """
data_cart
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_Cartn_x
_atom_site_Cartn_y
_atom_site_Cartn_z
C1 1.0 2.0 3.0
"""
    structure = parse_cif(cif)

    assert structure.atoms[0].element == "C"
    np.testing.assert_allclose(structure.atoms[0].fract, [0.1, 0.2, 0.3])
