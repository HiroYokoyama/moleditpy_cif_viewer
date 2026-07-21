"""Additional coverage tests for cif_viewer.viewer, targeting logic branches
not exercised by tests/test_plugin_integration.py (packing/export edge cases,
camera axis edge cases, settings error paths, thread run() body, etc.)."""

import os
import logging

import numpy as np
import pytest

from cif_viewer.parser import CifStructure, CifAtom
from cif_viewer.viewer import (
    CifViewerWidget,
    RenderThread,
    _run_render_calculation,
)


SIMPLE_CIF = """data_simple
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
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


class FakePlotter:
    def __init__(self):
        self.lines = []
        self.labels = []
        self.removed = []
        self.rendered = False
        self.camera_reset = False
        self.cleared = False

    def add_lines(self, points, **kwargs):
        self.lines.append((np.asarray(points), kwargs))

    def add_point_labels(self, *args, **kwargs):
        self.labels.append((args, kwargs))

    def remove_actor(self, name):
        self.removed.append(name)

    def clear(self):
        self.cleared = True

    def render(self):
        self.rendered = True

    def reset_camera(self):
        self.camera_reset = True


class StubContext:
    def __init__(self, plotter=None):
        self._plotter = plotter
        self.main_window = None
        self.status_messages = []

    @property
    def plotter(self):
        return self._plotter

    def get_main_window(self):
        return self.main_window

    def show_status_message(self, msg, duration=0):
        self.status_messages.append(msg)

    def mark_project_modified(self):
        pass


def _flat_structure(n=2):
    lattice = np.eye(3) * 10.0
    atoms = tuple(
        CifAtom(
            f"C{i}",
            "C",
            np.array([i / 10.0, 0.0, 0.0]),
            np.array([float(i), 0.0, 0.0]),
        )
        for i in range(n)
    )
    return CifStructure(
        "flat", (10.0, 10.0, 10.0), (90.0, 90.0, 90.0), lattice, atoms
    )


# ---------------------------------------------------------------------------
# _run_render_calculation direct-call coverage (packing branch, RemoveHs error)
# ---------------------------------------------------------------------------


def test_run_render_calculation_packing_branch():
    struct = _flat_structure(2)
    atoms, bonds, mol = _run_render_calculation(
        struct,
        "Packing",
        None,
        (2, 1, 1),
        True,
        0.45,
        False,
        300,
        True,
        True,
    )
    assert len(atoms) > 0
    assert mol.GetProp("_from_cif_viewer") == "1"


def test_run_render_calculation_remove_hs_failure(monkeypatch):
    struct = _flat_structure(2)

    def raise_remove_hs(mol):
        raise ValueError("boom")

    monkeypatch.setattr("rdkit.Chem.RemoveHs", raise_remove_hs)

    atoms, bonds, mol = _run_render_calculation(
        struct,
        "Asymmetric Unit",
        None,
        (1, 1, 1),
        True,
        0.45,
        False,
        300,
        False,
        True,
    )
    # RemoveHs failed, so mol keeps its original (unfiltered) atom count,
    # but last_rendered_atoms is still filtered in python.
    assert all(a.element != "H" for a in atoms)


# ---------------------------------------------------------------------------
# RenderThread.run() body -- call synchronously (not .start()) so coverage
# tracing (which does not follow real QThread execution) sees it.
# ---------------------------------------------------------------------------


def test_render_thread_run_success():
    struct = _flat_structure(2)
    thread = RenderThread(
        struct, "Asymmetric Unit", None, (1, 1, 1), True, 0.45, False, 300, True, True
    )
    received = []
    thread.result_ready.connect(lambda *args: received.append(args))
    thread.run()
    assert len(received) == 1
    atoms, bonds, mol, err = received[0]
    assert err == ""
    assert mol is not None


def test_render_thread_run_failure(monkeypatch):
    struct = _flat_structure(2)
    thread = RenderThread(
        struct, "Asymmetric Unit", None, (1, 1, 1), True, 0.45, False, 300, True, True
    )

    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("cif_viewer.viewer._run_render_calculation", boom)
    received = []
    thread.result_ready.connect(lambda *args: received.append(args))
    thread.run()
    assert len(received) == 1
    atoms, bonds, mol, err = received[0]
    assert atoms == []
    assert mol is None
    assert "kaboom" in err


# ---------------------------------------------------------------------------
# Color button picker
# ---------------------------------------------------------------------------


def test_create_color_button_pick_color(qtbot, monkeypatch):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)

    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import QColorDialog

    monkeypatch.setattr(
        QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor("#abcdef"))
    )
    saved = []
    monkeypatch.setattr(widget, "save_settings", lambda *a: saved.append(True))
    rendered = []
    monkeypatch.setattr(widget, "render", lambda *a: rendered.append(True))

    widget.color_axis_a.click()

    assert widget.color_axis_a.property("color_hex") == "#abcdef"
    assert saved
    assert rendered


def test_create_color_button_pick_color_invalid(qtbot, monkeypatch):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)

    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import QColorDialog

    monkeypatch.setattr(
        QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor())
    )
    prior = widget.color_axis_a.property("color_hex")
    widget.color_axis_a.click()
    assert widget.color_axis_a.property("color_hex") == prior


# ---------------------------------------------------------------------------
# _export_supercell edge cases
# ---------------------------------------------------------------------------


def test_export_supercell_no_structure(qtbot, monkeypatch):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    warned = []
    monkeypatch.setattr(
        "cif_viewer.viewer.QMessageBox.warning",
        lambda *a, **k: warned.append(True),
    )
    widget._export_supercell()
    assert warned


def test_export_supercell_cancel_dialog(qtbot, monkeypatch, tmp_path):
    from cif_viewer.parser import parse_cif

    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget.structure = parse_cif(SIMPLE_CIF)

    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
        lambda *a, **k: ("", ""),
    )
    # Should just return, no exception
    widget._export_supercell()


def test_export_supercell_asymmetric_unit(qtbot, monkeypatch, tmp_path):
    from cif_viewer.parser import parse_cif

    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget.structure = parse_cif(SIMPLE_CIF)
    widget.radio_asym.setChecked(True)

    out_path = tmp_path / "asym_export.cif"
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(out_path), ""),
    )
    infos = []
    monkeypatch.setattr(
        "cif_viewer.viewer.QMessageBox.information",
        lambda *a, **k: infos.append(True),
    )
    widget._export_supercell()
    assert out_path.exists()
    assert infos


def test_export_supercell_packing(qtbot, monkeypatch, tmp_path):
    from cif_viewer.parser import parse_cif

    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget.structure = parse_cif(SIMPLE_CIF)
    widget.radio_pack.setChecked(True)
    widget.repeat_a.setValue(2)

    out_path = tmp_path / "pack_export.cif"
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(out_path), ""),
    )
    infos = []
    monkeypatch.setattr(
        "cif_viewer.viewer.QMessageBox.information",
        lambda *a, **k: infos.append(True),
    )
    widget._export_supercell()
    assert out_path.exists()
    assert infos


def test_export_supercell_exception(qtbot, monkeypatch, tmp_path):
    from cif_viewer.parser import parse_cif

    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget.structure = parse_cif(SIMPLE_CIF)
    widget.radio_pack.setChecked(True)

    out_path = tmp_path / "fail_export.cif"
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(out_path), ""),
    )

    def boom(*a, **k):
        raise RuntimeError("write failed")

    monkeypatch.setattr("cif_viewer.viewer.write_supercell_cif", boom, raising=False)

    # write_supercell_cif is imported locally inside the method, so patch the
    # parser module attribute it resolves against.
    import cif_viewer.parser as parser_mod

    monkeypatch.setattr(parser_mod, "write_supercell_cif", boom)

    criticals = []
    monkeypatch.setattr(
        "cif_viewer.viewer.QMessageBox.critical",
        lambda *a, **k: criticals.append(True),
    )
    widget._export_supercell()
    assert criticals
    assert not out_path.exists()


# ---------------------------------------------------------------------------
# View mode helpers
# ---------------------------------------------------------------------------


def test_get_current_view_mode_default_fallback(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    # An exclusive QButtonGroup refuses to leave zero buttons checked via the
    # UI, so temporarily relax exclusivity to exercise the defensive fallback.
    widget.view_mode_group.setExclusive(False)
    widget.radio_mol.setChecked(False)
    widget.radio_asym.setChecked(False)
    widget.radio_pack.setChecked(False)
    assert widget._get_current_view_mode() == "Whole Molecule"
    widget.view_mode_group.setExclusive(True)


def test_set_current_view_mode_whole_and_packing(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget._set_current_view_mode("Packing")
    assert widget.radio_pack.isChecked()
    assert widget.show_cell.isEnabled()
    assert widget.show_axes.isEnabled()

    widget._set_current_view_mode("Asymmetric Unit")
    assert widget.radio_asym.isChecked()

    widget._set_current_view_mode("Whole Molecule")
    assert widget.radio_mol.isChecked()
    assert not widget.show_cell.isEnabled()


def test_on_tab_changed_noop(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget.tabs.setCurrentIndex(1)
    widget.tabs.setCurrentIndex(0)


def test_on_supercell_spin_changed_switches_to_packing(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget.radio_mol.setChecked(True)
    widget._on_supercell_spin_changed()
    assert widget.radio_pack.isChecked()


# ---------------------------------------------------------------------------
# view_from_axis edge cases
# ---------------------------------------------------------------------------


def test_view_from_axis_no_structure(qtbot):
    widget = CifViewerWidget(context=StubContext(FakePlotter()))
    qtbot.addWidget(widget)
    widget.structure = None
    widget.view_from_axis("a")  # should just return


def test_view_from_axis_no_plotter(qtbot):
    widget = CifViewerWidget(context=StubContext(None))
    qtbot.addWidget(widget)
    widget.structure = _flat_structure()
    widget.view_from_axis("a")  # should just return, no plotter


def test_view_from_axis_negative_b_and_c(qtbot):
    plotter = FakePlotter()
    widget = CifViewerWidget(context=StubContext(plotter))
    qtbot.addWidget(widget)
    widget.structure = _flat_structure()

    widget.view_from_axis("-b")
    widget.view_from_axis("-c")
    assert plotter.camera_reset
    assert plotter.rendered


def test_view_from_axis_invalid_name_returns(qtbot):
    plotter = FakePlotter()
    widget = CifViewerWidget(context=StubContext(plotter))
    qtbot.addWidget(widget)
    widget.structure = _flat_structure()
    widget.view_from_axis("q")
    assert not plotter.rendered


def test_view_from_axis_zero_direction_norm(qtbot):
    plotter = FakePlotter()
    widget = CifViewerWidget(context=StubContext(plotter))
    qtbot.addWidget(widget)

    class ZeroAStructure:
        lattice = np.array(
            [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        )

    widget.structure = ZeroAStructure()
    widget.view_from_axis("a")
    assert not plotter.rendered


def test_view_from_axis_zero_up_vector_uses_default(qtbot):
    plotter = FakePlotter()
    widget = CifViewerWidget(context=StubContext(plotter))
    qtbot.addWidget(widget)

    class ZeroUpStructure:
        # For axis "a", view_up = lattice[2]; make it zero.
        lattice = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.0]]
        )

    widget.structure = ZeroUpStructure()
    widget.repeat_a.setValue(1)
    widget.repeat_b.setValue(1)
    widget.repeat_c.setValue(1)
    widget.view_from_axis("a")
    assert plotter.rendered


def test_view_from_axis_camera_position_setter_raises_falls_back(qtbot):
    class RaisingCameraPositionPlotter(FakePlotter):
        class _Camera:
            def __init__(self):
                self.position = None
                self.focal_point = None
                self.up = None

        def __init__(self):
            super().__init__()
            self.camera = self._Camera()

        @property
        def camera_position(self):
            return None

        @camera_position.setter
        def camera_position(self, value):
            raise RuntimeError("cannot set")

    plotter = RaisingCameraPositionPlotter()
    widget = CifViewerWidget(context=StubContext(plotter))
    qtbot.addWidget(widget)
    widget.structure = _flat_structure()
    widget.view_from_axis("a")
    assert plotter.camera.position is not None


def test_view_from_axis_camera_position_and_fallback_both_raise(qtbot):
    class DoublyRaisingPlotter(FakePlotter):
        class _Camera:
            @property
            def position(self):
                return None

            @position.setter
            def position(self, value):
                raise RuntimeError("camera attribute set failed too")

        def __init__(self):
            super().__init__()
            self.camera = self._Camera()

        @property
        def camera_position(self):
            return None

        @camera_position.setter
        def camera_position(self, value):
            raise RuntimeError("cannot set")

    plotter = DoublyRaisingPlotter()
    widget = CifViewerWidget(context=StubContext(plotter))
    qtbot.addWidget(widget)
    widget.structure = _flat_structure()
    widget.view_from_axis("a")  # both fallbacks fail; should not raise


def test_view_from_axis_reset_camera_raises(qtbot):
    class RaisingResetPlotter(FakePlotter):
        def reset_camera(self):
            raise RuntimeError("cannot reset")

    plotter = RaisingResetPlotter()
    widget = CifViewerWidget(context=StubContext(plotter))
    qtbot.addWidget(widget)
    widget.structure = _flat_structure()
    widget.view_from_axis("a")  # should not raise


# ---------------------------------------------------------------------------
# _choose_file
# ---------------------------------------------------------------------------


def test_choose_file_opens_selected_path(qtbot, monkeypatch, tmp_path):
    cif_file = tmp_path / "chosen.cif"
    cif_file.write_text(SIMPLE_CIF, encoding="utf-8")

    widget = CifViewerWidget(context=StubContext(FakePlotter()))
    qtbot.addWidget(widget)

    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(cif_file), ""),
    )
    loaded = []
    monkeypatch.setattr(widget, "load_cif", lambda p: loaded.append(p))
    widget._choose_file()
    assert loaded == [str(cif_file)]


def test_choose_file_cancelled(qtbot, monkeypatch):
    widget = CifViewerWidget(context=StubContext(FakePlotter()))
    qtbot.addWidget(widget)
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getOpenFileName", lambda *a, **k: ("", "")
    )
    loaded = []
    monkeypatch.setattr(widget, "load_cif", lambda p: loaded.append(p))
    widget._choose_file()
    assert loaded == []


# ---------------------------------------------------------------------------
# _structure_selected / multi-structure table
# ---------------------------------------------------------------------------


MULTI_CIF = """
data_alpha
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

data_beta
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


def test_multi_structure_load_and_select(qtbot, tmp_path):
    cif_file = tmp_path / "multi.cif"
    cif_file.write_text(MULTI_CIF, encoding="utf-8")

    widget = CifViewerWidget(context=StubContext(FakePlotter()))
    qtbot.addWidget(widget)
    widget.show()
    widget.load_cif(str(cif_file))

    assert len(widget.all_structures) == 2
    assert widget.structure_table.isVisible()
    assert widget.structure.name == "alpha"

    widget.structure_table.selectRow(1)
    assert widget.structure.name == "beta"


def test_structure_selected_no_selection_noop(qtbot):
    widget = CifViewerWidget(context=StubContext(FakePlotter()))
    qtbot.addWidget(widget)
    widget.all_structures = []
    widget._structure_selected()  # should not raise


# ---------------------------------------------------------------------------
# polymer UI update
# ---------------------------------------------------------------------------


def test_update_polymer_ui_no_structure(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget.structure = None
    widget._update_polymer_ui()  # early return, no crash


def test_update_polymer_ui_disables_and_switches_off_whole_molecule(qtbot, monkeypatch):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget.structure = _flat_structure()

    monkeypatch.setattr("cif_viewer.viewer.is_polymer_structure", lambda *a, **k: True)
    assert widget.radio_mol.isChecked()  # default mode
    widget._update_polymer_ui()
    assert widget._was_polymer is True
    assert not widget.radio_mol.isEnabled()
    assert widget.radio_asym.isChecked()  # switched away from Whole Molecule


def test_update_polymer_ui_recovers_after_polymer(qtbot, monkeypatch):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget.structure = _flat_structure()

    calls = iter([True, False])
    monkeypatch.setattr(
        "cif_viewer.viewer.is_polymer_structure", lambda *a, **k: next(calls)
    )

    widget.radio_asym.setChecked(True)
    widget._update_polymer_ui()
    assert widget._was_polymer is True
    assert not widget.radio_mol.isEnabled()

    widget.radio_asym.setChecked(True)
    widget._update_polymer_ui()
    assert widget._was_polymer is False
    assert widget.radio_mol.isChecked()


def test_update_polymer_ui_exception_falls_back(qtbot, monkeypatch):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget.structure = _flat_structure()

    def boom(*a, **k):
        raise ValueError("nope")

    monkeypatch.setattr("cif_viewer.viewer.is_polymer_structure", boom)
    widget._update_polymer_ui()
    assert widget.polymer_warning_label.isHidden() or not widget.polymer_warning_label.isVisible()
    assert widget.radio_mol.isEnabled()
    assert widget._was_polymer is False


# ---------------------------------------------------------------------------
# _update_info_ui / _simulate_powder_pattern with no structure
# ---------------------------------------------------------------------------


def test_update_disorder_ui_no_structure_hides_combo(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget.disorder_label.setVisible(True)
    widget.disorder_combo.setVisible(True)
    widget.structure = None
    widget._update_disorder_ui()
    assert not widget.disorder_label.isVisible()
    assert not widget.disorder_combo.isVisible()


def test_update_info_ui_no_structure_resets_labels(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget.info_formula.setText("something")
    widget.structure = None
    widget._update_info_ui()
    assert widget.info_formula.text() == "N/A"
    assert not widget.simulate_xrd_btn.isEnabled()


def test_simulate_powder_pattern_no_structure_noop(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget.structure = None
    widget._simulate_powder_pattern()  # early return, no dialog created


def test_simulate_powder_pattern_with_disorder_selection(qtbot, monkeypatch):
    from cif_viewer.parser import parse_cif

    disorder_cif = """data_disorder
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
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
C2 C 0.2 0.2 0.2 2
"""
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget.structure = parse_cif(disorder_cif)
    widget._update_disorder_ui()
    widget.disorder_combo.setCurrentIndex(1)

    created = []

    class FakeDialog:
        def __init__(self, structure, selected_key, parent):
            created.append((structure, selected_key))

        def exec(self):
            return 1

    monkeypatch.setattr("cif_viewer.viewer_xrd.PowderPatternDialog", FakeDialog)
    widget._simulate_powder_pattern()
    assert len(created) == 1
    assert created[0][1] == "1"


# ---------------------------------------------------------------------------
# load_settings error paths
# ---------------------------------------------------------------------------


def test_load_settings_probability_backward_compat_scale(qtbot, tmp_path, monkeypatch):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)

    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"probability": 1.54}', encoding="utf-8")
    monkeypatch.setattr(widget, "_settings_path", lambda: str(settings_file))

    widget.load_settings()
    assert widget.probability_spin.value() == 50.0


def test_load_settings_probability_parse_exception(qtbot, tmp_path, monkeypatch):
    import cif_viewer.viewer as viewer_mod

    widget = CifViewerWidget()
    qtbot.addWidget(widget)

    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"probability": 42.0}', encoding="utf-8")
    monkeypatch.setattr(widget, "_settings_path", lambda: str(settings_file))

    def raising_search(*a, **k):
        raise ValueError("regex boom")

    monkeypatch.setattr(viewer_mod.re, "search", raising_search)
    widget.load_settings()  # Should not raise; exception caught and logged


def test_load_settings_malformed_json_logs_error(qtbot, tmp_path, monkeypatch):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)

    settings_file = tmp_path / "settings.json"
    settings_file.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(widget, "_settings_path", lambda: str(settings_file))

    widget.load_settings()  # Should not raise


def test_save_settings_write_failure_logged(qtbot, monkeypatch, caplog):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)

    def raise_open(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", raise_open)
    with caplog.at_level(logging.ERROR):
        widget.save_settings()
    assert any("Failed to save settings" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# load_cif failure paths
# ---------------------------------------------------------------------------


def test_load_cif_both_parsers_fail(qtbot, monkeypatch, tmp_path):
    bad_file = tmp_path / "bad.cif"
    bad_file.write_text("this is not a cif file at all", encoding="utf-8")

    widget = CifViewerWidget(context=StubContext(FakePlotter()))
    qtbot.addWidget(widget)

    def raise_pymatgen(path):
        raise ValueError("pymatgen cannot parse this")

    def raise_builtin(path):
        raise ValueError("builtin parser cannot parse this either")

    monkeypatch.setattr("cif_viewer.viewer.parse_cif_file_pymatgen", raise_pymatgen)
    monkeypatch.setattr("cif_viewer.viewer.parse_cif_file", raise_builtin)

    criticals = []
    monkeypatch.setattr(
        "cif_viewer.viewer.QMessageBox.critical",
        lambda *a, **k: criticals.append(a),
    )
    widget.load_cif(str(bad_file))
    assert criticals
    assert widget.structure is None


def test_load_cif_pymatgen_fails_builtin_succeeds(qtbot, monkeypatch, tmp_path):
    cif_file = tmp_path / "fallback.cif"
    cif_file.write_text(SIMPLE_CIF, encoding="utf-8")

    widget = CifViewerWidget(context=StubContext(FakePlotter()))
    qtbot.addWidget(widget)

    def raise_pymatgen(path):
        raise ValueError("pymatgen refuses this CIF")

    monkeypatch.setattr("cif_viewer.viewer.parse_cif_file_pymatgen", raise_pymatgen)
    widget.load_cif(str(cif_file))
    assert widget.structure is not None
    assert widget.structure.name == "simple"


def test_load_cif_empty_structures_list(qtbot, monkeypatch, tmp_path):
    cif_file = tmp_path / "empty.cif"
    cif_file.write_text(SIMPLE_CIF, encoding="utf-8")

    widget = CifViewerWidget(context=StubContext(FakePlotter()))
    qtbot.addWidget(widget)

    monkeypatch.setattr(
        "cif_viewer.viewer.parse_cif_file_pymatgen", lambda path: []
    )
    criticals = []
    monkeypatch.setattr(
        "cif_viewer.viewer.QMessageBox.critical",
        lambda *a, **k: criticals.append(a),
    )
    widget.load_cif(str(cif_file))
    assert criticals


def test_load_cif_updates_init_manager_path(qtbot, tmp_path):
    cif_file = tmp_path / "with_mw.cif"
    cif_file.write_text(SIMPLE_CIF, encoding="utf-8")

    class FakeInitManager:
        current_file_path = None

    class FakeMainWindow:
        def __init__(self):
            self.init_manager = FakeInitManager()

    context = StubContext(FakePlotter())
    context.main_window = FakeMainWindow()

    widget = CifViewerWidget(context=context)
    qtbot.addWidget(widget)
    widget.load_cif(str(cif_file))

    assert context.main_window.init_manager.current_file_path == str(cif_file)


# ---------------------------------------------------------------------------
# closeEvent / clear_view / render_overlays_only / render()
# ---------------------------------------------------------------------------


def test_close_event_terminates_active_render_thread(qtbot):
    from PyQt6.QtGui import QCloseEvent

    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    struct = _flat_structure()
    thread = RenderThread(
        struct, "Asymmetric Unit", None, (1, 1, 1), True, 0.45, False, 300, True, True
    )
    widget._render_thread = thread
    widget.closeEvent(QCloseEvent())
    assert widget._render_thread is None


def test_clear_view_no_plotter_returns(qtbot):
    widget = CifViewerWidget(context=StubContext(None))
    qtbot.addWidget(widget)
    widget.overlay_actor_names = ["a1"]
    widget.clear_view()
    assert widget.overlay_actor_names == ["a1"]  # untouched, early return


def test_render_now_no_structure_returns(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget.structure = None
    widget._render_now()  # early return, no crash


def test_clear_view_handles_remove_and_render_exceptions(qtbot):
    class RaisingPlotter:
        def remove_actor(self, name):
            raise RuntimeError("remove failed")

        def render(self):
            raise RuntimeError("render failed")

    widget = CifViewerWidget(context=StubContext(RaisingPlotter()))
    qtbot.addWidget(widget)
    widget.overlay_actor_names = ["a1"]
    widget.clear_view()  # should not raise
    assert widget.overlay_actor_names == []


def test_render_overlays_only_no_structure(qtbot):
    widget = CifViewerWidget(context=StubContext(FakePlotter()))
    qtbot.addWidget(widget)
    widget.structure = None
    widget.render_overlays_only()  # early return


def test_render_overlays_only_no_plotter(qtbot):
    widget = CifViewerWidget(context=StubContext(None))
    qtbot.addWidget(widget)
    widget.structure = _flat_structure()
    widget.render_overlays_only()  # early return, no plotter


def test_render_overlays_only_draws_cell_when_packing(qtbot):
    plotter = FakePlotter()
    widget = CifViewerWidget(context=StubContext(plotter))
    qtbot.addWidget(widget)
    widget.structure = _flat_structure()
    widget.radio_pack.setChecked(True)
    widget.show_cell.setChecked(True)
    widget.render_overlays_only()
    assert plotter.rendered
    assert plotter.lines


def test_render_overlays_only_render_exception_logged(qtbot):
    class RaisingRenderPlotter(FakePlotter):
        def render(self):
            raise RuntimeError("render boom")

    widget = CifViewerWidget(context=StubContext(RaisingRenderPlotter()))
    qtbot.addWidget(widget)
    widget.structure = _flat_structure()
    widget.render_overlays_only()  # should not raise


def test_render_noop_without_structure(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget.structure = None
    widget.render()
    assert not widget.render_timer.isActive()


# ---------------------------------------------------------------------------
# _render_now: space-group-operations exception fallback, mid-size async path
# ---------------------------------------------------------------------------


def test_render_now_whole_molecule_symops_lookup_fails(qtbot, monkeypatch):
    from cif_viewer.parser import parse_cif

    widget = CifViewerWidget(context=StubContext(FakePlotter()))
    qtbot.addWidget(widget)
    widget.structure = parse_cif(SIMPLE_CIF)
    widget.radio_mol.setChecked(True)

    def boom(*a, **k):
        raise RuntimeError("symops lookup failed")

    monkeypatch.setattr("cif_viewer.parser.get_space_group_operations", boom)
    widget._render_now()  # should not raise despite the lookup failure


def test_render_now_sync_calculation_exception(qtbot, monkeypatch):
    from cif_viewer.parser import parse_cif

    widget = CifViewerWidget(context=StubContext(FakePlotter()))
    qtbot.addWidget(widget)
    widget.structure = parse_cif(SIMPLE_CIF)

    def boom(*a, **k):
        raise RuntimeError("calc failed")

    monkeypatch.setattr("cif_viewer.viewer._run_render_calculation", boom)
    widget._render_now()  # exception should be caught -> _on_render_data_ready([], ..., err)
    assert getattr(widget, "last_rendered_atoms", None) != []  # early-returned, unset


def test_render_now_progress_dialog_cancel_already_disconnected(qtbot, monkeypatch):
    """Covers the TypeError branch when on_cancel's disconnect() finds no
    connected slots left (e.g. a second/rapid cancel)."""
    import os
    import time
    from rdkit import Chem

    stop_event = False

    def slow_run_calc(*args, **kwargs):
        for _ in range(40):
            if stop_event:
                break
            time.sleep(0.05)
        return [], [], Chem.Mol()

    import cif_viewer.viewer as viewer_mod

    monkeypatch.setattr(viewer_mod, "_run_render_calculation", slow_run_calc)
    monkeypatch.setattr(
        "cif_viewer.viewer.render_atoms_to_rdkit_mol",
        lambda *args, **kwargs: Chem.Mol(),
    )

    widget = CifViewerWidget(context=StubContext(FakePlotter()))
    qtbot.addWidget(widget)

    lattice = np.eye(3) * 10.0
    atoms = tuple(
        CifAtom(
            f"C{i}", "C", np.array([i / 10.0, 0.0, 0.0]), np.array([float(i), 0.0, 0.0])
        )
        for i in range(10)
    )
    widget.structure = CifStructure(
        "large", (10.0, 10.0, 10.0), (90.0, 90.0, 90.0), lattice, atoms
    )
    widget.radio_pack.blockSignals(True)
    widget.radio_pack.setChecked(True)
    widget.radio_pack.blockSignals(False)
    for spin in (widget.repeat_a, widget.repeat_b, widget.repeat_c):
        spin.blockSignals(True)
        spin.setValue(3)
        spin.blockSignals(False)

    monkeypatch.setattr(widget, "_draw_with_moleditpy", lambda mol: None)
    monkeypatch.setenv("MOLEDITPY_HEADLESS", "0")
    monkeypatch.setenv("QT_QPA_PLATFORM", "windows")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    from PyQt6.QtWidgets import QProgressDialog

    monkeypatch.setattr(QProgressDialog, "show", lambda self_dialog: None)
    monkeypatch.setattr(QProgressDialog, "close", lambda self_dialog: None)

    widget._render_now()
    qtbot.waitUntil(lambda: widget._render_thread is not None, timeout=2000)

    def mock_terminate():
        nonlocal stop_event
        stop_event = True

    monkeypatch.setattr(widget._render_thread, "terminate", mock_terminate)

    # Pre-disconnect so on_cancel's own disconnect() call finds nothing left,
    # forcing the TypeError branch.
    widget._render_thread.result_ready.disconnect()

    # Find the QProgressDialog child and trigger cancellation.
    from PyQt6.QtWidgets import QProgressDialog as _QPD

    dialogs = widget.findChildren(_QPD)
    assert dialogs
    dialogs[0].canceled.emit()

    qtbot.waitUntil(lambda: widget._render_thread is None, timeout=3000)


def test_render_now_mid_size_async_no_progress_dialog(qtbot, monkeypatch):
    from rdkit import Chem

    monkeypatch.setattr(
        "cif_viewer.viewer.render_atoms_to_rdkit_mol",
        lambda *args, **kwargs: Chem.Mol(),
    )

    widget = CifViewerWidget(context=StubContext(FakePlotter()))
    qtbot.addWidget(widget)

    lattice = np.eye(3) * 10.0
    atoms = tuple(
        CifAtom(
            f"C{i}",
            "C",
            np.array([i / 10.0, 0.0, 0.0]),
            np.array([float(i), 0.0, 0.0]),
        )
        for i in range(6)
    )
    widget.structure = CifStructure(
        "mid", (10.0, 10.0, 10.0), (90.0, 90.0, 90.0), lattice, atoms
    )

    widget.radio_pack.blockSignals(True)
    widget.radio_pack.setChecked(True)
    widget.radio_pack.blockSignals(False)
    for spin in (widget.repeat_a, widget.repeat_b, widget.repeat_c):
        spin.blockSignals(True)
        spin.setValue(3)
        spin.blockSignals(False)

    monkeypatch.setattr(widget, "_draw_with_moleditpy", lambda mol: None)
    monkeypatch.setenv("MOLEDITPY_HEADLESS", "0")
    monkeypatch.setenv("QT_QPA_PLATFORM", "windows")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    widget._render_now()
    qtbot.waitUntil(
        lambda: getattr(widget, "last_rendered_atoms", None) is not None, timeout=5000
    )
    assert len(widget.last_rendered_atoms) == 6 * 27


# ---------------------------------------------------------------------------
# _on_render_data_ready branches
# ---------------------------------------------------------------------------


def test_on_render_data_ready_terminate_error_returns(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget._on_render_data_ready([], [], None, "terminate requested", (1, 1, 1))
    assert not hasattr(widget, "last_rendered_atoms") or widget.last_rendered_atoms == []


def test_on_render_data_ready_error_with_atoms_shows_critical(qtbot, monkeypatch):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    criticals = []
    monkeypatch.setattr(
        "cif_viewer.viewer.QMessageBox.critical",
        lambda *a, **k: criticals.append(a),
    )
    fake_atoms = ["atom"]
    widget._on_render_data_ready(fake_atoms, [], None, "boom error", (1, 1, 1))
    assert criticals


def test_on_render_data_ready_draw_exception_shows_critical(qtbot, monkeypatch):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    criticals = []
    monkeypatch.setattr(
        "cif_viewer.viewer.QMessageBox.critical",
        lambda *a, **k: criticals.append(a),
    )
    monkeypatch.setattr(
        widget,
        "_draw_with_moleditpy",
        lambda mol: (_ for _ in ()).throw(RuntimeError("draw failed")),
    )
    widget.structure = _flat_structure()
    widget._on_render_data_ready(["atom"], [], object(), "", (1, 1, 1))
    assert criticals


def test_on_render_data_ready_full_flow_packing_cell_overlay(qtbot):
    from cif_viewer.parser import parse_cif
    from rdkit import Chem

    plotter = FakePlotter()
    widget = CifViewerWidget(context=StubContext(plotter))
    qtbot.addWidget(widget)
    widget.structure = parse_cif(SIMPLE_CIF)
    widget.radio_pack.setChecked(True)
    widget.show_cell.setChecked(True)

    mol = Chem.Mol()
    widget._on_render_data_ready([], [], mol, "", (1, 1, 1))
    qtbot.wait(200)
    assert plotter.rendered
    assert plotter.lines


def test_on_render_data_ready_draw_overlays_runtime_error_deleted(qtbot):
    from cif_viewer.parser import parse_cif
    from rdkit import Chem

    class DeletedRuntimeErrorPlotter(FakePlotter):
        def render(self):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    widget = CifViewerWidget(context=StubContext(DeletedRuntimeErrorPlotter()))
    qtbot.addWidget(widget)
    widget.structure = parse_cif(SIMPLE_CIF)

    mol = Chem.Mol()
    widget._on_render_data_ready([], [], mol, "", (1, 1, 1))
    qtbot.wait(200)  # should not raise/crash despite the "deleted" RuntimeError


def test_on_render_data_ready_draw_overlays_generic_exception_logged(qtbot):
    from cif_viewer.parser import parse_cif
    from rdkit import Chem

    class BoomPlotter(FakePlotter):
        def render(self):
            raise ValueError("some other failure")

    widget = CifViewerWidget(context=StubContext(BoomPlotter()))
    qtbot.addWidget(widget)
    widget.structure = parse_cif(SIMPLE_CIF)

    mol = Chem.Mol()
    widget._on_render_data_ready([], [], mol, "", (1, 1, 1))
    qtbot.wait(200)


def test_on_render_data_ready_draw_overlays_runtime_error_not_deleted_logged(qtbot):
    from cif_viewer.parser import parse_cif
    from rdkit import Chem

    class OtherRuntimeErrorPlotter(FakePlotter):
        def render(self):
            raise RuntimeError("some unrelated runtime failure")

    widget = CifViewerWidget(context=StubContext(OtherRuntimeErrorPlotter()))
    qtbot.addWidget(widget)
    widget.structure = parse_cif(SIMPLE_CIF)

    mol = Chem.Mol()
    widget._on_render_data_ready([], [], mol, "", (1, 1, 1))
    qtbot.wait(200)  # should log the error but not crash


def test_on_render_data_ready_qtimer_singleshot_failure_falls_back_sync(
    qtbot, monkeypatch
):
    from cif_viewer.parser import parse_cif
    from rdkit import Chem

    plotter = FakePlotter()
    widget = CifViewerWidget(context=StubContext(plotter))
    qtbot.addWidget(widget)
    widget.structure = parse_cif(SIMPLE_CIF)

    def boom(*a, **k):
        raise RuntimeError("no event loop")

    monkeypatch.setattr("cif_viewer.viewer.QTimer.singleShot", boom)
    mol = Chem.Mol()
    widget._on_render_data_ready([], [], mol, "", (1, 1, 1))
    # Fallback path calls draw_overlays_and_render() synchronously
    assert plotter.rendered


def test_on_render_data_ready_bond_order_failure_message(qtbot):
    from cif_viewer.parser import parse_cif
    from rdkit import Chem

    widget = CifViewerWidget(context=StubContext(FakePlotter()))
    qtbot.addWidget(widget)
    widget.structure = parse_cif(SIMPLE_CIF)
    widget.determine_bond_order.setChecked(True)
    assert widget.determine_bond_order.isEnabled()

    mol = Chem.Mol()
    mol.SetProp("_bond_order_error", "could not determine bonds")
    widget._on_render_data_ready(["a1", "a2"], [], mol, "", (1, 1, 1))
    assert "Bond order determination failed" in widget.summary_label.text()


# ---------------------------------------------------------------------------
# _draw_with_moleditpy fallback (no context)
# ---------------------------------------------------------------------------


def test_draw_with_moleditpy_no_context_uses_main_window(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)

    calls = []

    class FakeMW:
        def draw_molecule_3d(self, mol):
            calls.append(mol)

    monkeypatch_mw = FakeMW()
    widget._main_window = lambda: monkeypatch_mw
    widget._draw_with_moleditpy("themol")
    assert calls == ["themol"]


def test_draw_with_moleditpy_no_context_no_main_window(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget._main_window = lambda: None
    widget._draw_with_moleditpy("themol")  # no-op, no crash


# ---------------------------------------------------------------------------
# _enter_viewer_mode branches
# ---------------------------------------------------------------------------


def test_enter_viewer_mode_context_method_raises_falls_through(qtbot):
    class RaisingContext(StubContext):
        def enter_3d_mode(self):
            raise RuntimeError("boom")

    context = RaisingContext(FakePlotter())
    widget = CifViewerWidget(context=context)
    qtbot.addWidget(widget)
    widget._enter_viewer_mode()  # should not raise; falls through to main_window


def test_enter_viewer_mode_context_method_succeeds_returns(qtbot):
    calls = []

    class SucceedingContext(StubContext):
        def enter_3d_mode(self):
            calls.append(True)

    context = SucceedingContext(FakePlotter())
    widget = CifViewerWidget(context=context)
    qtbot.addWidget(widget)
    widget._enter_viewer_mode()
    assert calls == [True]


def test_enter_viewer_mode_no_context_ui_manager_method(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)

    calls = []

    class FakeUIManager:
        def enter_3d_viewer_mode(self):
            calls.append(True)

    class FakeMW:
        ui_manager = FakeUIManager()

    widget._main_window = lambda: FakeMW()
    widget._enter_viewer_mode()
    assert calls == [True]


def test_enter_viewer_mode_ui_manager_method_raises_continues(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)

    class FakeUIManager:
        def enter_3d_viewer_mode(self):
            raise RuntimeError("boom")

    class FakeMW:
        ui_manager = FakeUIManager()
        splitter = None

    widget._main_window = lambda: FakeMW()
    widget._enter_viewer_mode()  # should not raise


def test_enter_viewer_mode_splitter_fallback(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)

    class FakeSplitter:
        def __init__(self):
            self.sizes = None

        def setSizes(self, sizes):
            self.sizes = sizes

    class FakeMW:
        pass

    mw = FakeMW()
    mw.splitter = FakeSplitter()
    widget._main_window = lambda: mw
    widget._enter_viewer_mode()
    assert mw.splitter.sizes == [0, 1]


def test_enter_viewer_mode_splitter_raises_logged(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)

    class FakeSplitter:
        def setSizes(self, sizes):
            raise RuntimeError("boom")

    class FakeMW:
        pass

    mw = FakeMW()
    mw.splitter = FakeSplitter()
    widget._main_window = lambda: mw
    widget._enter_viewer_mode()  # should not raise


def test_enter_viewer_mode_no_main_window(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget._main_window = lambda: None
    widget._enter_viewer_mode()  # no-op


# ---------------------------------------------------------------------------
# _plotter / _main_window fallbacks
# ---------------------------------------------------------------------------


def test_plotter_falls_back_to_main_window_plotter_attr(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)

    class FakeMW:
        plotter = "the-plotter"

    widget._main_window = lambda: FakeMW()
    assert widget._plotter() == "the-plotter"


def test_plotter_falls_back_to_view_3d_manager_plotter(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)

    class FakeView3DManager:
        plotter = "manager-plotter"

    class FakeMW:
        view_3d_manager = FakeView3DManager()

    widget._main_window = lambda: FakeMW()
    assert widget._plotter() == "manager-plotter"


def test_plotter_returns_none_when_nothing_available(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    widget._main_window = lambda: None
    assert widget._plotter() is None


def test_main_window_uses_parent_chain_without_context(qtbot):
    from PyQt6.QtWidgets import QWidget

    grandparent = QWidget()
    parent = QWidget(grandparent)
    widget = CifViewerWidget(parent=parent)
    qtbot.addWidget(grandparent)
    assert widget._main_window() is grandparent


def test_main_window_none_when_no_parent(qtbot):
    widget = CifViewerWidget()
    qtbot.addWidget(widget)
    assert widget._main_window() is None
