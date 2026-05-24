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
