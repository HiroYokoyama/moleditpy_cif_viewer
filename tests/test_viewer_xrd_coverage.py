"""Additional coverage tests for cif_viewer.viewer_xrd, targeting branches not
exercised by tests/test_plugin_integration.py (SpaceGroup lookup failure,
disorder-aware symmetry expansion duplicates, all radiation sources, empty
disorder selections, calculation/export error paths)."""

import numpy as np

from cif_viewer.parser import CifAtom, CifStructure
from cif_viewer.viewer_xrd import PowderPatternDialog, make_pymatgen_structure


def _p1_structure(space_group=None, is_asym=False, atoms=None):
    lattice = np.eye(3) * 5.0
    if atoms is None:
        atoms = (
            CifAtom(
                label="C1",
                element="C",
                fract=np.array([0.1, 0.2, 0.3]),
                cart=np.array([0.5, 1.0, 1.5]),
                occupancy=1.0,
            ),
        )
    return CifStructure(
        name="test",
        cell_lengths=(5.0, 5.0, 5.0),
        cell_angles=(90.0, 90.0, 90.0),
        lattice=lattice,
        atoms=atoms,
        space_group=space_group,
        is_asymmetric_unit_only=is_asym,
    )


def test_make_pymatgen_structure_spacegroup_lookup_failure_falls_back(monkeypatch):
    struct = _p1_structure(space_group="not a real space group!!", is_asym=True)

    # SpaceGroup() should raise for a bogus symbol; the except branch logs and
    # leaves symops=None so we fall back to using atoms directly.
    pm = make_pymatgen_structure(struct)
    assert len(pm) == 1


def test_make_pymatgen_structure_symops_disorder_occupancy_forced_to_one():
    atom = CifAtom(
        label="C1",
        element="C",
        fract=np.array([0.2, 0.3, 0.4]),
        cart=np.array([1.0, 1.5, 2.0]),
        occupancy=0.5,
        disorder_group="1",
        disorder_assembly="A",
    )
    struct = _p1_structure(space_group="P -1", is_asym=True, atoms=(atom,))

    pm = make_pymatgen_structure(struct, selected_disorder_key="A_1")
    # P-1 has 2 symmetry operations (identity + inversion); disordered atom's
    # occupancy should be forced to 1.0 since its part was explicitly selected.
    assert len(pm) == 2
    assert pm[0].species.num_atoms == 1.0


def test_make_pymatgen_structure_symops_deduplicates_special_position():
    # An atom sitting at the origin is its own inversion image under P-1, so
    # the second symmetry operation must be detected as a duplicate.
    atom = CifAtom(
        label="C1",
        element="C",
        fract=np.array([0.0, 0.0, 0.0]),
        cart=np.array([0.0, 0.0, 0.0]),
        occupancy=1.0,
    )
    struct = _p1_structure(space_group="P -1", is_asym=True, atoms=(atom,))
    pm = make_pymatgen_structure(struct)
    assert len(pm) == 1


def test_powder_pattern_dialog_no_atoms_in_disorder_group_shows_message(qtbot):
    from cif_viewer.parser import parse_cif

    cif_content = """data_disorder_only
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
_atom_site_disorder_group
C1 C 0.1 0.1 0.1 1
C2 C 0.2 0.2 0.2 1
"""
    structure = parse_cif(cif_content)
    # Select a disorder key that matches none of the atoms -> empty structure.
    dialog = PowderPatternDialog(structure, selected_disorder_key="nonexistent")
    qtbot.addWidget(dialog)
    assert dialog.last_xrd is None
    assert dialog.last_profile_x is None


def test_powder_pattern_dialog_all_radiation_sources(qtbot):
    from cif_viewer.parser import parse_cif

    cif_content = """data_xrd_sources
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
C1 C 0.1 0.1 0.1
"""
    structure = parse_cif(cif_content)
    dialog = PowderPatternDialog(structure)
    qtbot.addWidget(dialog)

    expected = {
        "MoKa": 0.71073,
        "CrKa": 2.29100,
        "FeKa": 1.9373,
        "CoKa": 1.7902,
    }
    for source, wavelength in expected.items():
        dialog.source_combo.setCurrentText(source)
        assert np.isclose(dialog.wavelength_spin.value(), wavelength, atol=1e-3)
        assert not dialog.wavelength_spin.isEnabled()

    # Cycle back to CuKa explicitly (its handler is otherwise never invoked
    # since it's already the combo's initial value).
    dialog.source_combo.setCurrentText("Custom")
    dialog.source_combo.setCurrentText("CuKa")
    assert np.isclose(dialog.wavelength_spin.value(), 1.54184, atol=1e-3)
    assert not dialog.wavelength_spin.isEnabled()


def test_powder_pattern_dialog_calculation_exception_shows_message(qtbot, monkeypatch):
    from cif_viewer.parser import parse_cif

    cif_content = """data_xrd_err
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
C1 C 0.1 0.1 0.1
"""
    structure = parse_cif(cif_content)
    dialog = PowderPatternDialog(structure)
    qtbot.addWidget(dialog)

    def raise_get_pattern(self, *a, **k):
        raise RuntimeError("XRD calculation exploded")

    monkeypatch.setattr(
        "cif_viewer.viewer_xrd.XRDCalculator.get_pattern", raise_get_pattern
    )
    dialog.calculate_and_plot()  # should not raise; draws the error text instead
    assert dialog.last_xrd is None


def test_export_csv_no_data_warns(qtbot, monkeypatch):
    from cif_viewer.parser import parse_cif

    cif_content = """data_xrd_nodata
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
C1 C 0.1 0.1 0.1
"""
    structure = parse_cif(cif_content)
    dialog = PowderPatternDialog(structure)
    qtbot.addWidget(dialog)
    dialog.last_xrd = None

    warnings = []
    monkeypatch.setattr(
        "cif_viewer.viewer_xrd.QMessageBox.warning",
        lambda *a, **k: warnings.append(True),
    )
    dialog._export_csv()
    assert warnings


def test_export_csv_write_exception_shows_critical(qtbot, monkeypatch, tmp_path):
    from cif_viewer.parser import parse_cif
    from PyQt6.QtWidgets import QMessageBox

    cif_content = """data_xrd_writeerr
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
C1 C 0.1 0.1 0.1
C2 C 0.2 0.2 0.2
"""
    structure = parse_cif(cif_content)
    dialog = PowderPatternDialog(structure)
    qtbot.addWidget(dialog)
    assert dialog.last_xrd is not None

    class MockQMessageBox:
        StandardButton = QMessageBox.StandardButton
        ButtonRole = QMessageBox.ButtonRole

        def __init__(self, parent=None):
            self._buttons = []

        def setWindowTitle(self, title):
            pass

        def setText(self, text):
            pass

        def addButton(self, arg1, role=None):
            self._buttons.append(arg1)
            return arg1

        def exec(self):
            pass

        def clickedButton(self):
            return self._buttons[0]  # "Simulated Profile (Curve)"

        @staticmethod
        def warning(*a, **k):
            pass

        @staticmethod
        def information(*a, **k):
            pass

        @staticmethod
        def critical(*a, **k):
            critical_calls.append(True)

    critical_calls = []
    monkeypatch.setattr("cif_viewer.viewer_xrd.QMessageBox", MockQMessageBox)
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(tmp_path / "out.csv"), "CSV Files (*.csv)"),
    )

    def raise_open(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", raise_open)
    dialog._export_csv()
    assert critical_calls
