import os
import tempfile

from cif_viewer.parser import parse_cif_file_pymatgen


def test_parse_cif_file_pymatgen_multi_structure_and_adp():
    cif_content = """
data_struct1
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.0 0.0 0.0
C2 C 0.5 0.5 0.5

loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_23
_atom_site_aniso_U_13
_atom_site_aniso_U_12
C1 0.01 0.02 0.03 0.0 0.0 0.0
C2 0.02 0.02 0.02 0.0 0.0 0.0

data_struct2
_cell_length_a 6.0
_cell_length_b 6.0
_cell_length_c 6.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
O1 O 0.0 0.0 0.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as f:
        f.write(cif_content)
        temp_path = f.name

    try:
        structures = parse_cif_file_pymatgen(temp_path)
        assert len(structures) == 2
        assert structures[0].name == "struct1"
        assert structures[1].name == "struct2"

        # Verify ADPs
        assert structures[0].u_cart is not None
        assert structures[0].u_cart.shape == (2, 3, 3)
        assert structures[1].u_cart is None
    finally:
        os.remove(temp_path)


def test_grow_molecules_adp_rotation():
    import numpy as np

    cif_content = """
data_adp_rot
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90

_symmetry_space_group_name_h-m 'P -1'
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
C1 C 0.07 0.0 0.0

loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_23
_atom_site_aniso_U_13
_atom_site_aniso_U_12
C1 0.05 0.01 0.01 0.0 0.0 0.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as f:
        f.write(cif_content)
        temp_path = f.name

    try:
        from cif_viewer.parser import grow_molecules

        structures = parse_cif_file_pymatgen(temp_path)
        struct = structures[0]

        # Verify that asymmetric unit atoms have the correct u_cart values
        assert struct.asymmetric_atoms is not None
        assert len(struct.asymmetric_atoms) == 1
        c1_asym = struct.asymmetric_atoms[0]

        assert c1_asym.u_cart is not None
        assert np.allclose(c1_asym.u_cart[0, 0], 0.05)

        # Grow structure
        import dataclasses

        struct_to_grow = dataclasses.replace(
            struct, atoms=struct.asymmetric_atoms, is_asymmetric_unit_only=True
        )
        grown_atoms, grown_bonds = grow_molecules(struct_to_grow)
        assert len(grown_atoms) > 0

        # Find some symmetry generated C1 (where is_original_asym is False)
        c1_syms = [a for a in grown_atoms if a.label == "C1" and not a.is_original_asym]
        assert len(c1_syms) > 0
        for c1_sym in c1_syms:
            assert c1_sym.u_cart is not None
            # Under inversion, R = -I, so R_cart @ U @ R_cart.T = (-I) @ U @ (-I) = U.
            # Thus, the size and shape should be preserved exactly.
            assert np.allclose(c1_sym.u_cart[0, 0], 0.05)
            assert np.allclose(c1_sym.u_cart[1, 1], 0.01)

    finally:
        os.remove(temp_path)
