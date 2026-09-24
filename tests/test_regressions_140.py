"""Regressions fixed in 1.4.0."""

import logging

import numpy as np
import pytest

from cif_viewer import parser as P
from cif_viewer.viewer_xrd import make_pymatgen_structure

P21C = """data_{name}
_cell_length_a 5.0
_cell_length_b 6.0
_cell_length_c 7.0
_cell_angle_alpha 90
_cell_angle_beta 100
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 21/c'
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
'-x, y+1/2, -z+1/2'
'-x, -y, -z'
'x, -y+1/2, z+1/2'
"""

MIXED = """data_mixed
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Fe1 Fe 0 0 0 0.3
Ni1 Ni 0 0 0 0.7
O1 O 0.5 0.5 0.5 1.0
"""

DISORDER = P21C.replace("{name}", "disorder") + """loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_disorder_assembly
_atom_site_disorder_group
C1 C 0.1 0.2 0.3 1 . .
C2A C 0.3 0.2 0.3 0.6 A 1
C2B C 0.3 0.25 0.35 0.4 A 2
"""


def _write(tmp_path, text, name="x.cif"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# -- built-in parser (the fallback when pymatgen cannot read a file) --------


def test_builtin_parser_reads_primed_labels():
    """Nucleotide and sugar labels such as C1' made the shell lexer give up."""
    text = P21C.replace("{name}", "primed") + (
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "C1' C 0.1 0.2 0.3\nO5'' O 0.3 0.2 0.3 # sugar oxygen\n"
    )
    structure = P.parse_cif(text)
    assert [a.label for a in structure.atoms] == ["C1'", "O5''"]
    assert [a.element for a in structure.atoms] == ["C", "O"]


def test_builtin_parser_still_reads_quoted_values():
    assert P._split_cif_line("'P 21/c' \"x, y, z\" C1 # note") == ["P 21/c", "x, y, z", "C1"]
    assert P._strip_comment("C1' C 0.1 # comment") == "C1' C 0.1 "
    assert P._strip_comment("_name 'no # here'") == "_name 'no # here'"


def test_builtin_parser_skips_an_aniso_loop_listed_first():
    text = P21C.replace("{name}", "aniso") + (
        "loop_\n_atom_site_aniso_label\n_atom_site_aniso_U_11\n_atom_site_aniso_U_22\n"
        "C1 0.02 0.02\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "C1 C 0.1 0.2 0.3\n"
    )
    structure = P.parse_cif(text)
    assert len(structure.atoms) == 1
    np.testing.assert_allclose(structure.atoms[0].fract, [0.1, 0.2, 0.3])


# -- pymatgen path -----------------------------------------------------------


def test_partial_occupancies_are_kept(tmp_path):
    """Sites have no occupancy attribute; reading one always gave 1.0."""
    structure = P.parse_cif_file_pymatgen(_write(tmp_path, DISORDER))[0]
    occupancy = {atom.label: atom.occupancy for atom in structure.atoms}
    assert occupancy["C1"] == pytest.approx(1.0)
    assert occupancy["C2A"] == pytest.approx(0.6)
    assert occupancy["C2B"] == pytest.approx(0.4)


def test_mixed_site_keeps_every_element(tmp_path):
    structure = P.parse_cif_file_pymatgen(_write(tmp_path, MIXED))[0]
    mixed = [atom for atom in structure.atoms if atom.species]
    assert len(mixed) == 1
    atom = mixed[0]
    assert atom.element == "Ni"  # the majority element is the one drawn
    assert dict(atom.species) == pytest.approx({"Fe": 0.3, "Ni": 0.7})


def test_ordered_sites_are_unchanged(tmp_path):
    structure = P.parse_cif_file_pymatgen(_write(tmp_path, MIXED))[0]
    oxygen = [atom for atom in structure.atoms if atom.element == "O"]
    assert len(oxygen) == 1
    assert oxygen[0].species is None
    assert oxygen[0].occupancy == pytest.approx(1.0)


def test_powder_pattern_sees_the_whole_mixed_site(tmp_path):
    structure = P.parse_cif_file_pymatgen(_write(tmp_path, MIXED))[0]
    composition = make_pymatgen_structure(structure).composition
    assert composition["Fe"] == pytest.approx(0.3)
    assert composition["Ni"] == pytest.approx(0.7)
    assert composition["O"] == pytest.approx(1.0)


def test_powder_pattern_weights_disorder_parts(tmp_path):
    """With all parts shown, two half-sites must not scatter as two full atoms."""
    structure = P.parse_cif_file_pymatgen(_write(tmp_path, DISORDER))[0]
    composition = make_pymatgen_structure(structure).composition
    # 4 x C1 + 4 x 0.6 C2A + 4 x 0.4 C2B = 8 carbons, not 12
    assert composition["C"] == pytest.approx(8.0)


def test_export_writes_each_element_of_a_mixed_site(tmp_path):
    structure = P.parse_cif_file_pymatgen(_write(tmp_path, MIXED))[0]
    out = tmp_path / "out.cif"
    P.write_supercell_cif(str(out), structure, (1, 1, 1), keep_connected=False)
    rows = [line.split() for line in out.read_text().splitlines() if line.startswith(("Ni1", "O1"))]
    by_element = {row[1]: float(row[-1]) for row in rows}
    assert by_element == pytest.approx({"Fe": 0.3, "Ni": 0.7, "O": 1.0})
    assert len({row[0] for row in rows}) == len(rows)  # labels stay unique


def test_a_block_without_atoms_is_not_reported_as_an_error(tmp_path, caplog):
    text = "data_global\n_publ_section_title 'x'\n\n" + DISORDER
    with caplog.at_level(logging.ERROR):
        structures = P.parse_cif_file_pymatgen(_write(tmp_path, text))
    assert [s.name for s in structures] == ["disorder"]
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


# -- thermal ellipsoids -------------------------------------------------------


def test_isotropic_atom_does_not_borrow_another_ellipsoid():
    from cif_viewer import atom_covariance

    u = np.diag([0.05, 0.02, 0.01])

    class _Structure:
        u_cart = np.array([u, u])

    isotropic = P.RenderAtom("C9", "C", 1, (0, 0, 0), np.zeros(3), u_cart=None)
    anisotropic = P.RenderAtom("C1", "C", 0, (0, 0, 0), np.zeros(3), u_cart=u)
    assert atom_covariance(isotropic, _Structure()) is None
    np.testing.assert_allclose(atom_covariance(anisotropic, _Structure()), u)


# -- render errors ------------------------------------------------------------


def test_a_failed_render_is_reported(qtbot):
    """Render errors always arrive with no atoms and used to vanish silently."""
    from cif_viewer.viewer import CifViewerWidget

    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget._on_render_data_ready([], [], None, "boom\ntraceback ...", (1, 1, 1))
    assert widget.summary_label.text() == "Could not build the view: boom"
