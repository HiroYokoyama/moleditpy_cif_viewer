import numpy as np

from cif_viewer.parser import (
    celleditpy_cell_axis_segments,
    expand_supercell,
    parse_cif,
    parse_cif_number,
    supercell_edges,
    write_supercell_cif,
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


def test_write_supercell_cif(tmp_path):
    structure = parse_cif(NACL_CIF)
    export_path = tmp_path / "supercell.cif"
    write_supercell_cif(str(export_path), structure, (2, 1, 1), keep_connected=False)

    # Parse the exported supercell
    exported_struct = parse_cif(export_path.read_text(encoding="utf-8"))
    assert exported_struct.cell_lengths == (11.2804, 5.6402, 5.6402)
    assert len(exported_struct.atoms) == 4
    # The first atom (Na1_0_0_0)
    assert exported_struct.atoms[0].element == "Na"
    np.testing.assert_allclose(exported_struct.atoms[0].fract, [0.0, 0.0, 0.0])

    # Check all fractional x coordinates: they should be 0.0, 0.25, 0.5, 0.75
    frac_xs = sorted(atom.fract[0] for atom in exported_struct.atoms)
    np.testing.assert_allclose(frac_xs, [0.0, 0.25, 0.5, 0.75])


def test_write_cif_no_cell(tmp_path):
    from cif_viewer.parser import CifStructure, CifAtom

    atom = CifAtom(
        label="C1",
        element="C",
        fract=np.array([0.0, 0.0, 0.0]),
        cart=np.array([1.2, 3.4, 5.6]),
        occupancy=1.0,
    )
    structure = CifStructure(
        name="NoCellStruct",
        cell_lengths=(0.0, 0.0, 0.0),
        cell_angles=(90.0, 90.0, 90.0),
        lattice=None,
        atoms=(atom,),
    )
    export_path = tmp_path / "nocell.cif"
    write_supercell_cif(str(export_path), structure, (1, 1, 1), keep_connected=False)

    content = export_path.read_text(encoding="utf-8")
    assert "data_molecule" in content
    assert "_cell_length_a" not in content
    assert "_atom_site_Cartn_x" in content
    assert "_atom_site_fract_x" not in content
    assert "C1_0_0_0" in content
    assert "1.200000 3.400000 5.600000" in content


def test_render_atoms_to_rdkit_mol():
    from cif_viewer.rdkit_bridge import render_atoms_to_rdkit_mol
    from cif_viewer.parser import RenderAtom

    atoms = [
        RenderAtom("C1", "C", 0, (0, 0, 0), np.array([0.0, 0.0, 0.0])),
        RenderAtom("H1", "H", 1, (0, 0, 0), np.array([1.0, 0.0, 0.0])),
    ]
    bonds = [(0, 1)]
    mol = render_atoms_to_rdkit_mol(atoms, bonds)
    assert mol.GetNumAtoms() == 2
    assert mol.GetAtomWithIdx(0).GetSymbol() == "C"
    assert mol.GetAtomWithIdx(1).GetSymbol() == "H"
    assert mol.GetNumBonds() == 1

    bond = mol.GetBondBetweenAtoms(0, 1)
    assert bond is not None

    conf = mol.GetConformer()
    np.testing.assert_allclose(conf.GetAtomPosition(0), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(conf.GetAtomPosition(1), [1.0, 0.0, 0.0])


def test_render_atoms_to_rdkit_mol_with_error():
    from cif_viewer.rdkit_bridge import render_atoms_to_rdkit_mol
    from cif_viewer.parser import RenderAtom

    # A valence-violating molecule (Carbon with 5 single bonds)
    atoms = [
        RenderAtom("C1", "C", 0, (0, 0, 0), np.array([0.0, 0.0, 0.0])),
        RenderAtom("C2", "C", 1, (0, 0, 0), np.array([1.0, 0.0, 0.0])),
        RenderAtom("C3", "C", 2, (0, 0, 0), np.array([0.0, 1.0, 0.0])),
        RenderAtom("C4", "C", 3, (0, 0, 0), np.array([0.0, -1.0, 0.0])),
        RenderAtom("C5", "C", 4, (0, 0, 0), np.array([-1.0, 0.0, 0.0])),
        RenderAtom("C6", "C", 5, (0, 0, 0), np.array([1.0, 1.0, 0.0])),
    ]
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    mol = render_atoms_to_rdkit_mol(atoms, bonds, determine_bond_order=True)
    assert mol.HasProp("_bond_order_error")
    assert "could not find valid bond ordering" in mol.GetProp("_bond_order_error")


def test_parse_cif_file(tmp_path):
    from cif_viewer.parser import parse_cif_file

    cif_file = tmp_path / "nacl.cif"
    cif_file.write_text(NACL_CIF, encoding="utf-8")
    structure = parse_cif_file(str(cif_file))
    assert structure.name == "NaCl"
    assert len(structure.atoms) == 2


def test_covalent_radius_and_normalize_element():
    from cif_viewer.parser import covalent_radius, normalize_element

    assert covalent_radius("C") == 0.76
    assert covalent_radius("h") == 0.31
    assert covalent_radius("Xx") == 0.77  # fallback
    assert normalize_element(" c ") == "C"
    assert normalize_element("Fe3+") == "Fe"
    assert normalize_element("123") == "X"


def test_parse_cif_number_errors():
    import pytest

    with pytest.raises(ValueError):
        parse_cif_number("?")
    with pytest.raises(ValueError):
        parse_cif_number(".")


def test_parse_cif_missing_atoms():
    import pytest

    bad_cif = """
data_bad
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
"""
    with pytest.raises(ValueError):
        parse_cif(bad_cif)


def test_cell_vectors_invalid():
    import pytest
    from cif_viewer.parser import cell_vectors

    # Gamma 180 or 0 makes sin(gamma) = 0
    with pytest.raises(ValueError):
        cell_vectors((10.0, 10.0, 10.0), (90.0, 90.0, 180.0))
    # Inconsistent angles
    with pytest.raises(ValueError):
        cell_vectors((10.0, 10.0, 10.0), (30.0, 30.0, 120.0))


def test_parse_disorder_and_refinement_metadata():
    disorder_cif = """
data_DisorderTest
_cell_length_a    10.0
_cell_length_b    10.0
_cell_length_c    10.0
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
_space_group_name_h-m_alt 'P 21/c'
_space_group_crystal_system monoclinic
_chemical_formula_sum 'C10 H15 N O'
_refine_ls_r_factor_gt 0.045
_refine_ls_wr_factor_ref 0.120
_refine_ls_goodness_of_fit_ref 1.05

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_disorder_group
_atom_site_disorder_assembly
C1 C 0.1 0.1 0.1 1 A
C2 C 0.2 0.2 0.2 2 A
C3 C 0.3 0.3 0.3 . .
"""
    structure = parse_cif(disorder_cif)
    assert structure.space_group == "P 21/c"
    assert structure.crystal_system == "monoclinic"
    assert structure.formula == "C10 H15 N O"
    assert structure.r1 == "0.045"
    assert structure.wr2 == "0.120"
    assert structure.goof == "1.05"

    assert len(structure.atoms) == 3
    assert structure.atoms[0].disorder_group == "1"
    assert structure.atoms[0].disorder_assembly == "A"
    assert structure.atoms[0].disorder_key == "A_1"

    assert structure.atoms[1].disorder_group == "2"
    assert structure.atoms[1].disorder_assembly == "A"
    assert structure.atoms[1].disorder_key == "A_2"

    assert structure.atoms[2].disorder_group is None
    assert structure.atoms[2].disorder_assembly is None
    assert structure.atoms[2].disorder_key is None


def test_multiline_semicolon_text_field_is_parsed():
    """Standard CIF text fields ('_tag' on its own line, followed by a value
    delimited by ';' lines) must be read as the full multi-line text, not
    lost or truncated to a stray ';' character."""
    cif = """
data_TextFieldTest
_cell_length_a    10.0
_cell_length_b    10.0
_cell_length_c    10.0
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
_chemical_formula_structural
;
C10 H15 N O,
a simple test compound
;
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.1 0.1 0.1
"""
    structure = parse_cif(cif)
    assert structure.formula == "C10 H15 N O,\na simple test compound"


def test_parse_cif_file_pymatgen_disorder_and_metadata(tmp_path):
    from cif_viewer.parser import parse_cif_file_pymatgen

    cif_content = """
data_DisorderPymatgen
_cell_length_a    10.0
_cell_length_b    10.0
_cell_length_c    10.0
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
_space_group_name_h-m_alt 'P 21/c'
_space_group_crystal_system monoclinic
_chemical_formula_sum 'C10 H15 N O'
_refine_ls_r_factor_gt 0.045
_refine_ls_wr_factor_ref 0.120
_refine_ls_goodness_of_fit_ref 1.05

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_disorder_group
_atom_site_disorder_assembly
C1 C 0.1 0.1 0.1 1 A
C2 C 0.2 0.2 0.2 2 A
C3 C 0.3 0.3 0.3 . .
"""
    cif_file = tmp_path / "disorder.cif"
    cif_file.write_text(cif_content, encoding="utf-8")

    structures = parse_cif_file_pymatgen(str(cif_file))
    assert len(structures) == 1
    structure = structures[0]

    assert structure.space_group == "P 21/c"
    assert structure.crystal_system == "monoclinic"
    assert structure.formula == "C10 H15 N O"
    assert structure.r1 == "0.045"
    assert structure.wr2 == "0.120"
    assert structure.goof == "1.05"

    assert len(structure.atoms) == 3
    # Check that disorder assemblies and groups map correctly for the parsed atoms
    # We sort by label or element just in case order differs in pymatgen
    atoms_map = {a.label: a for a in structure.atoms}
    assert "C1" in atoms_map
    assert atoms_map["C1"].disorder_group == "1"
    assert atoms_map["C1"].disorder_assembly == "A"
    assert atoms_map["C1"].disorder_key == "A_1"

    assert "C2" in atoms_map
    assert atoms_map["C2"].disorder_group == "2"
    assert atoms_map["C2"].disorder_assembly == "A"
    assert atoms_map["C2"].disorder_key == "A_2"

    assert "C3" in atoms_map
    assert atoms_map["C3"].disorder_group is None
    assert atoms_map["C3"].disorder_assembly is None
    assert atoms_map["C3"].disorder_key is None


def test_disorder_part_bond_exclusion():
    cif = """
data_disorder_bonds
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
_atom_site_disorder_group
_atom_site_disorder_assembly
C1 C 0.5 0.5 0.5 1 A
C2 C 0.5 0.5 0.6 2 A
C3 C 0.5 0.5 0.4 . .
"""
    structure = parse_cif(cif)
    atoms, bonds = expand_supercell(structure, (1, 1, 1), keep_connected=False)

    assert len(structure.atoms) == 3
    # C1 and C2 are in different groups (A_1 vs A_2), so they should not form a bond.
    # C1 (index 0) and C3 (index 2, framework atom) should form a bond.
    assert (0, 1) not in bonds
    assert (1, 0) not in bonds
    assert (0, 2) in bonds or (2, 0) in bonds


def test_write_supercell_cif_with_disorder_filtering(tmp_path):
    cif = """
data_disorder_export
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
_atom_site_disorder_group
_atom_site_disorder_assembly
C1 C 0.5 0.5 0.5 1 A
C2 C 0.5 0.5 0.6 2 A
C3 C 0.5 0.5 0.4 . .
"""
    structure = parse_cif(cif)
    export_path = tmp_path / "supercell_disorder.cif"
    write_supercell_cif(
        str(export_path),
        structure,
        (1, 1, 1),
        keep_connected=False,
        selected_disorder_key="A_1",
    )

    exported = parse_cif(export_path.read_text(encoding="utf-8"))
    atoms_map = {a.label.split("_")[0]: a for a in exported.atoms}
    assert "C1" in atoms_map
    assert "C3" in atoms_map
    assert "C2" not in atoms_map


def test_user_disorder_loop_structure(tmp_path):
    cif_content = """
data_user_disorder
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
loop_
  _atom_site_label
  _atom_site_type_symbol
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  _atom_site_U_iso_or_equiv
  _atom_site_adp_type
  _atom_site_occupancy
  _atom_site_site_symmetry_order
  _atom_site_calc_flag
  _atom_site_refinement_flags_posn
  _atom_site_refinement_flags_adp
  _atom_site_refinement_flags_occupancy
  _atom_site_disorder_assembly
  _atom_site_disorder_group
C1 C 0.1 0.1 0.1 0.05 Uani 1.0 1 d . . . A 1
C2 C 0.2 0.2 0.2 0.05 Uani 1.0 1 d . . . A 2
C3 C 0.3 0.3 0.3 0.05 Uani 1.0 1 d . . . . .
"""
    # 1. Pure Python parser
    struct = parse_cif(cif_content)
    assert len(struct.atoms) == 3
    assert struct.atoms[0].disorder_assembly == "A"
    assert struct.atoms[0].disorder_group == "1"
    assert struct.atoms[1].disorder_assembly == "A"
    assert struct.atoms[1].disorder_group == "2"
    assert struct.atoms[2].disorder_assembly is None
    assert struct.atoms[2].disorder_group is None

    # 2. Pymatgen parser
    cif_file = tmp_path / "user_disorder.cif"
    cif_file.write_text(cif_content, encoding="utf-8")
    from cif_viewer.parser import parse_cif_file_pymatgen

    structures = parse_cif_file_pymatgen(str(cif_file))
    assert len(structures) == 1
    struct_pm = structures[0]
    atoms_map = {a.label: a for a in struct_pm.atoms}

    assert atoms_map["C1"].disorder_assembly == "A"
    assert atoms_map["C1"].disorder_group == "1"
    assert atoms_map["C2"].disorder_assembly == "A"
    assert atoms_map["C2"].disorder_group == "2"
    assert atoms_map["C3"].disorder_assembly is None
    assert atoms_map["C3"].disorder_group is None


def test_parse_cif_extended_metadata():
    cif_content = """
data_metadata_test
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
_cell_formula_units_z 4
_cell_formula_units_zprime 1.5
_refine_absolute_configuration_flack 0.02(5)
_space_group_it_number 14
loop_
  _atom_site_label
  _atom_site_type_symbol
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
C1 C 0.1 0.1 0.1
"""
    struct = parse_cif(cif_content)
    assert struct.z == "4"
    assert struct.z_prime == "1.5"
    assert struct.flack == "0.02(5)"
    assert struct.space_group_number == "14"


def test_parse_cif_file_pymatgen_extended_metadata(tmp_path):
    cif_content = """
data_metadata_test
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
_cell_formula_units_z 4
_cell_formula_units_zprime 1.5
_refine_ls_abs_structure_Flack 0.02(5)
_space_group_it_number 14
_refine_ls_R_factor_all 0.0665
_refine_ls_R_factor_gt 0.0624
_refine_ls_wR_factor_gt 0.1760
_refine_ls_wR_factor_ref 0.1798
loop_
  _atom_site_label
  _atom_site_type_symbol
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
C1 C 0.1 0.1 0.1
"""
    cif_file = tmp_path / "metadata_test.cif"
    cif_file.write_text(cif_content, encoding="utf-8")
    from cif_viewer.parser import parse_cif_file_pymatgen

    structures = parse_cif_file_pymatgen(str(cif_file))
    assert len(structures) == 1
    struct = structures[0]
    assert struct.z == "4"
    assert struct.z_prime == "1.5"
    assert struct.flack == "0.02(5)"
    assert struct.space_group_number == "14"
    assert struct.r1_all == "0.0665"
    assert struct.r1_gt == "0.0624"
    assert struct.wr2_gt == "0.1760"
    assert struct.wr2_all == "0.1798"


def test_grow_molecules():
    from cif_viewer.parser import grow_molecules, parse_cif

    cif = """
data_water
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_h-m_alt 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
O1 O 0.5 0.5 0.5
H1 H 0.4 0.5 0.5
H2 H 0.6 0.5 0.5
"""
    struct = parse_cif(cif)
    atoms, bonds = grow_molecules(struct)
    assert len(atoms) == 3
    assert len(bonds) == 2


def test_grow_molecules_boundary():
    from cif_viewer.parser import grow_molecules, parse_cif

    cif = """
data_boundary_grow
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_h-m_alt 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.95 0.5 0.5
C2 C 0.05 0.5 0.5
"""
    struct = parse_cif(cif)
    atoms, bonds = grow_molecules(struct)
    assert len(atoms) == 2
    assert len(bonds) == 1


def test_grow_molecules_keeps_all_in_cell_symmetry_components():
    from cif_viewer.parser import grow_molecules, parse_cif
    import numpy as np

    cif = """
data_p21_test
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P 1 21 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.1 0.1 0.1
"""
    struct = parse_cif(cif)
    atoms, bonds = grow_molecules(struct)

    # Under P 1 21 1 there are 2 symmetry operations, so the cell physically
    # contains 2 atoms. Per ALGORITHMS.md §2 Step 4, every component
    # overlapping the unit-cell box is kept — including the screw-axis image
    # with no identity-op atom (the old is_asym filter dropped it).
    assert len(atoms) == 2
    assert all(a.label == "C1" for a in atoms)
    positions = sorted(tuple(np.round(a.position, 6)) for a in atoms)
    # identity: (0.1,0.1,0.1) -> (1,1,1); 21 screw along b:
    # (-x, y+1/2, -z) -> (0.9, 0.6, 0.9) -> (9,6,9)
    assert np.allclose(positions[0], (1.0, 1.0, 1.0))
    assert np.allclose(positions[1], (9.0, 6.0, 9.0))
    # exactly one of the two carries the identity-op flag
    assert sum(a.is_original_asym for a in atoms) == 1


def test_grow_molecules_p1bar_keeps_inversion_image():
    from cif_viewer.parser import grow_molecules, parse_cif

    cif = """
data_p1bar_test
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P -1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.1 0.1 0.1
"""
    struct = parse_cif(cif)
    atoms, bonds = grow_molecules(struct)

    # P -1 has Z=2: identity at (0.1,0.1,0.1) and inversion at (0.9,0.9,0.9).
    # Both are physically present in the cell and must both be returned.
    assert len(atoms) == 2


def test_grow_molecules_polymer():
    from cif_viewer.parser import grow_molecules, parse_cif

    # A 1D polymer chain along x-axis
    cif = """
data_polymer
_cell_length_a 2.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_h-m_alt 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.2 0.5 0.5
C2 C 0.8 0.5 0.5
"""
    struct = parse_cif(cif)
    atoms, bonds = grow_molecules(struct)

    # The polymer is detected, triggering the fallback 1x1x1 supercell representation.
    assert len(atoms) == 2
    # Verify that only a single bond is formed, rather than duplicate/periodic loop bonds between the same two atoms
    assert len(bonds) == 1


def test_infer_bonds_fallback(monkeypatch):
    from cif_viewer.parser import infer_bonds, RenderAtom
    import numpy as np

    def mock_determine_connectivity(*args, **kwargs):
        raise RuntimeError("Simulated RDKit error")

    try:
        import rdkit.Chem.rdDetermineBonds as rdb

        monkeypatch.setattr(rdb, "DetermineConnectivity", mock_determine_connectivity)
    except (ImportError, ModuleNotFoundError):
        pass

    atoms = [
        RenderAtom(
            label="C1",
            element="C",
            base_index=0,
            image=(0, 0, 0),
            position=np.array([0.0, 0.0, 0.0]),
        ),
        RenderAtom(
            label="C2",
            element="C",
            base_index=1,
            image=(0, 0, 0),
            position=np.array([1.5, 0.0, 0.0]),
        ),
    ]
    bonds = infer_bonds(atoms)
    assert len(bonds) == 1


def test_grow_molecules_rotates_adp():
    import tempfile
    import os
    import numpy as np
    from cif_viewer.parser import grow_molecules, parse_cif_file_pymatgen

    cif = """
data_adp_rot
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_h-m_alt 'P 2'
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
'-x, y, -z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.1 0.0 0.0
C2 C 0.1 0.25 0.0

loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_23
_atom_site_aniso_U_13
_atom_site_aniso_U_12
C1 0.05 0.02 0.03 0.005 0.006 0.007
C2 0.05 0.02 0.03 0.005 0.006 0.007
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as f:
        f.write(cif)
        temp_path = f.name

    try:
        structures = parse_cif_file_pymatgen(temp_path)
        struct = structures[0]
        assert struct.u_cart is not None

        atoms, bonds = grow_molecules(struct)

        found_rotated = False
        for atom in atoms:
            assert atom.u_cart is not None
            if np.allclose(atom.u_cart[0, 1], -0.007, atol=1e-5):
                np.testing.assert_allclose(atom.u_cart[1, 2], -0.005, atol=1e-5)
                found_rotated = True

        assert found_rotated, (
            "Symmetry-rotated ellipsoid was not generated or not correctly rotated"
        )
    finally:
        os.remove(temp_path)


def test_grow_molecules_cross_boundary_connected():
    """Regression: a P1 molecule straddling the cell boundary must be returned
    as a single connected component, not two disconnected fragments."""
    from cif_viewer.parser import grow_molecules, parse_cif
    import numpy as np

    # Three-atom chain: C1-C2-C3.  C2 and C3 are near x=0 and C1 is near x=1,
    # so C1 bonds to C2 via the periodic boundary.  The old code produced two
    # components (C1 alone and C2-C3) because tx=0 copies could not bridge.
    cif = """
data_cross_boundary
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_h-m_alt 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.97 0.5 0.5
C2 C 0.03 0.5 0.5
C3 C 0.18 0.5 0.5
"""
    struct = parse_cif(cif)
    atoms, bonds = grow_molecules(struct)

    # All three atoms must come back as one molecule
    assert len(atoms) == 3, f"Expected 3 atoms, got {len(atoms)}"
    assert len(bonds) == 2, f"Expected 2 bonds, got {len(bonds)}"

    # The molecule must be spatially contiguous (max pairwise distance < cell/2)
    positions = np.array([a.position for a in atoms])
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            assert dist < 5.0, (
                f"Atoms {i} and {j} are {dist:.2f} Å apart — molecule is split"
            )


def test_grow_molecules_inversion_cross_boundary():
    """Regression: a molecule on an inversion centre near the cell boundary
    must be returned whole and connected, not as two half-molecules."""
    from cif_viewer.parser import grow_molecules, parse_cif
    import numpy as np

    # P-1: inversion maps (x,y,z) -> (-x,-y,-z).
    # Asymmetric unit atom near x=0.97 → inversion image near x=0.03.
    # The two halves must bond and be returned as one molecule.
    cif = """
data_inv_boundary
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P -1'
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
'-x, -y, -z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.97 0.5 0.5
C2 C 0.82 0.5 0.5
"""
    struct = parse_cif(cif)
    atoms, bonds = grow_molecules(struct)

    # Inversion generates 4 sites total; the two that make the molecule
    # centred near x≈0 should all be contiguous.
    assert len(atoms) == 4, f"Expected 4 atoms, got {len(atoms)}"
    assert len(bonds) >= 3, f"Expected ≥3 bonds, got {len(bonds)}"

    positions = np.array([a.position for a in atoms])
    centroid = positions.mean(axis=0)
    for p in positions:
        assert np.linalg.norm(p - centroid) < 5.0, (
            "Molecule is not contiguous — fragment is too far from centroid"
        )


def test_grow_molecules_no_duplicate_images():
    """Regression: a molecule near the cell boundary must appear exactly once,
    not as two overlapping copies (one in the central cell, one from an image)."""
    from cif_viewer.parser import grow_molecules, parse_cif

    # Simple 2-atom molecule straddling x=0/1
    cif = """
data_no_dup
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_h-m_alt 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.96 0.5 0.5
C2 C 0.04 0.5 0.5
"""
    struct = parse_cif(cif)
    atoms, bonds = grow_molecules(struct)

    # Must return exactly one complete molecule (2 atoms, 1 bond), not two
    assert len(atoms) == 2, (
        f"Expected 2 atoms (one molecule), got {len(atoms)} — possible duplicate image"
    )
    assert len(bonds) == 1, f"Expected 1 bond, got {len(bonds)}"


# ---------------------------------------------------------------------------
# Tests for is_polymer_structure() and the four grow_molecules() bug fixes
# introduced in v0.10.0.
# ---------------------------------------------------------------------------

_POLYMER_1D_CIF = """\
data_polymer_1d
_cell_length_a 2.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_h-m_alt 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.2 0.5 0.5
C2 C 0.8 0.5 0.5
"""

_MOLECULE_CIF = """\
data_molecule
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_h-m_alt 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
O1 O 0.5 0.5 0.5
H1 H 0.4 0.5 0.5
H2 H 0.6 0.5 0.5
"""

_RING_POLYMER_CIF = """\
data_ring_polymer
_cell_length_a 3.0
_cell_length_b 3.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_h-m_alt 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.1 0.5 0.5
C2 C 0.9 0.5 0.5
"""


_CROSS_BOUNDARY_MOLECULE_CIF = """\
data_cross_boundary
_cell_length_a 5.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_h-m_alt 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.1 0.5 0.5
C2 C 0.9 0.5 0.5
"""

_3D_FRAMEWORK_CIF = """\
data_3d_framework
_cell_length_a 1.5
_cell_length_b 1.5
_cell_length_c 1.5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_h-m_alt 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.0 0.0 0.0
"""


def test_is_polymer_structure_detects_polymer():
    """is_polymer_structure() returns True when the BFS finds a topological ring crossing the cell."""
    from cif_viewer.parser import is_polymer_structure

    struct = parse_cif(_POLYMER_1D_CIF)
    assert is_polymer_structure(struct) is True


def test_is_polymer_structure_returns_false_for_molecule():
    """is_polymer_structure() returns False for an ordinary discrete molecule."""
    from cif_viewer.parser import is_polymer_structure

    struct = parse_cif(_MOLECULE_CIF)
    assert is_polymer_structure(struct) is False


def test_is_polymer_structure_returns_false_for_disconnected_ionic():
    """is_polymer_structure() returns False for NaCl (no covalent bonds at all)."""
    from cif_viewer.parser import is_polymer_structure

    struct = parse_cif(NACL_CIF)
    assert is_polymer_structure(struct) is False


def test_is_polymer_structure_returns_false_for_cross_boundary_molecule():
    """A discrete molecule that lies across the unit cell boundary has bonds that cross the boundary,
    but it does not form an infinite topological ring. is_polymer_structure() must return False."""
    from cif_viewer.parser import is_polymer_structure

    struct = parse_cif(_CROSS_BOUNDARY_MOLECULE_CIF)
    assert is_polymer_structure(struct) is False


def test_is_polymer_structure_detects_3d_framework():
    """A 3D framework where an atom bonds to its own periodic images must be detected as a polymer."""
    from cif_viewer.parser import is_polymer_structure

    struct = parse_cif(_3D_FRAMEWORK_CIF)
    assert is_polymer_structure(struct) is True


def test_grow_molecules_polymer_returns_in_cell_atoms():
    """Bug 4 fix: grow_molecules() must not discard polymer atoms when the
    identity symop is not the one that produced them. The 1D polymer should
    return both asymmetric-unit atoms without dropping either one."""
    struct = parse_cif(_POLYMER_1D_CIF)

    from cif_viewer.parser import grow_molecules

    atoms, bonds = grow_molecules(struct)

    # Both asymmetric-unit atoms must be present.
    assert len(atoms) == 2, (
        f"Expected 2 atoms from 1D polymer asym unit, got {len(atoms)}"
    )
    # The single periodic bond within the unit cell must be present.
    assert len(bonds) == 1, f"Expected 1 bond, got {len(bonds)}"


def test_grow_molecules_bfs_positions_are_contiguous():
    """Bug 2 fix: atom positions from grow_molecules() must be spatially
    contiguous for a polymer chain. If the BFS offset accumulation is wrong,
    atoms end up at positions more than one cell-length apart."""
    struct = parse_cif(_POLYMER_1D_CIF)

    from cif_viewer.parser import grow_molecules

    atoms, _ = grow_molecules(struct)

    positions = np.array([a.position for a in atoms])
    # Cell a = 2.0 A; the two atoms are 0.2 and 0.8 of that = 0.4 and 1.6 A.
    # Maximum separation within one molecule must be less than half the cell.
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            assert dist < 1.5, (
                f"Atoms {i} and {j} are {dist:.3f} A apart -- "
                "BFS offset may be wrong (Bug 2)"
            )


def test_grow_molecules_identity_op_detection_robust():
    """Bug 1 fix: identity-op detection must work even for a space group
    where the identity is not listed as the trivial 'x,y,z' string but is
    still functionally the identity.  Here we verify that atoms generated
    from the identity operation are correctly flagged so the molecule is
    retained in the output."""
    cif = """\
data_p1_identity
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_h-m_alt 'P 1'
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.5 0.5 0.5
H1 H 0.4 0.5 0.5
"""
    struct = parse_cif(cif)

    from cif_viewer.parser import grow_molecules

    atoms, bonds = grow_molecules(struct)

    # Both atoms must be returned; if identity detection fails the molecule
    # is dropped entirely.
    assert len(atoms) == 2, (
        f"Expected 2 atoms, got {len(atoms)} -- "
        "identity-op detection may have failed (Bug 1)"
    )
    assert len(bonds) == 1, f"Expected 1 bond, got {len(bonds)}"
