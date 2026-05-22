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
        selected_disorder_key="A_1"
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




