"""Extra coverage for cif_viewer/__init__.py (EllipsoidWorkerThread + initialize()).

Owns only this file; does not touch existing test modules.
"""
import os
import sys
import types

import numpy as np
import pytest
from rdkit import Chem

from cif_viewer import EllipsoidWorkerThread, initialize
from tests.test_plugin_integration import StubContext, StubMainWindow, _install_fake_qt


# ---------------------------------------------------------------------------
# EllipsoidWorkerThread.run() -- drive it synchronously (no .start()/thread).
# ---------------------------------------------------------------------------


def _make_atom(pos, element, cov, has_cov):
    return {"pos": pos, "element": element, "cov": cov, "has_cov": has_cov}


def test_ellipsoid_worker_run_mixed_atoms_emits_meshes():
    # cov chosen so eigh(...) returns an eigenvector matrix with det < 0,
    # exercising the "flip first eigenvector" branch.
    cov = np.array(
        [
            [0.22684606, -0.01247749, 0.05889865],
            [-0.01247749, 0.23555363, 0.09598517],
            [0.05889865, 0.09598517, 0.12966435],
        ]
    )
    assert np.linalg.det(np.linalg.eigh(cov)[1]) < 0

    atoms_data = [
        _make_atom([0.0, 0.0, 0.0], "O", cov, True),
        _make_atom([1.0, 0.0, 0.0], "H", None, False),
        _make_atom([0.0, 1.0, 0.0], "C", None, False),
    ]
    col = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])

    thread = EllipsoidWorkerThread(
        atoms_data, 8, 1.5382, 0.2, 1.0, col, True, "black", 2
    )

    results = []
    thread.result_ready.connect(lambda *a: results.append(a))
    thread.run()

    assert len(results) == 1
    merged, fallback, h_glyphs, rings, err = results[0]
    assert err == ""
    assert merged is not None
    assert h_glyphs is not None
    assert fallback is not None
    assert rings is not None


def test_ellipsoid_worker_run_no_rings_and_short_col_list():
    # show_rings False -> rings_mesh stays None; col shorter than atoms_data
    # exercises the "index < len(self.col)" fallback color branch.
    atoms_data = [
        _make_atom([0.0, 0.0, 0.0], "N", None, False),
    ]
    thread = EllipsoidWorkerThread(atoms_data, 8, 1.5382, 0.2, 1.0, [], False, "red", 1)

    results = []
    thread.result_ready.connect(lambda *a: results.append(a))
    thread.run()

    merged, fallback, h_glyphs, rings, err = results[0]
    assert err == ""
    assert merged is None
    assert h_glyphs is None
    assert fallback is not None
    assert rings is None


def test_ellipsoid_worker_run_bad_cov_logs_warning_and_continues(caplog):
    # cov with wrong shape makes np.linalg.eigh raise -> caught, logged,
    # atom simply dropped (not added to fallback/h lists either).
    atoms_data = [
        _make_atom([0.0, 0.0, 0.0], "O", np.array([1.0, 2.0]), True),
    ]
    thread = EllipsoidWorkerThread(atoms_data, 8, 1.5382, 0.2, 1.0, [], False, "red", 1)

    results = []
    thread.result_ready.connect(lambda *a: results.append(a))
    with caplog.at_level("WARNING"):
        thread.run()

    merged, fallback, h_glyphs, rings, err = results[0]
    assert err == ""
    assert merged is None
    assert any("draw ellipsoid" in r.message for r in caplog.records)


def test_ellipsoid_worker_run_outer_exception_reports_error():
    # self.col=None breaks "self.col[index]" indexing outside the inner
    # try/except -> hits the outer except at the bottom of run().
    atoms_data = [_make_atom([0.0, 0.0, 0.0], "C", None, False)]
    thread = EllipsoidWorkerThread(
        atoms_data, 8, 1.5382, 0.2, 1.0, None, False, "red", 1
    )

    results = []
    thread.result_ready.connect(lambda *a: results.append(a))
    thread.run()

    merged, fallback, h_glyphs, rings, err = results[0]
    assert merged is None and fallback is None and h_glyphs is None and rings is None
    assert err != ""
    assert "Traceback" in err


def test_ellipsoid_worker_run_hydrogen_uses_rdkit_periodic_table():
    atoms_data = [_make_atom([0.0, 0.0, 0.0], "H", None, False)]
    thread = EllipsoidWorkerThread(
        atoms_data, 8, 1.5382, 0.2, 1.0, [[1, 1, 1]], False, "red", 1
    )
    results = []
    thread.result_ready.connect(lambda *a: results.append(a))
    thread.run()
    merged, fallback, h_glyphs, rings, err = results[0]
    assert err == ""
    assert h_glyphs is not None


# ---------------------------------------------------------------------------
# Helpers shared by the initialize()-callback tests below.
# ---------------------------------------------------------------------------


class _FakeDockWidget:
    def __init__(self, title, parent):
        self.title = title
        self.parent = parent
        self._widget = None
        self.shown = False
        self.raised = False
        self.visibilityChanged = _FakeSignal()

    def setAllowedAreas(self, areas):
        pass

    def setWidget(self, widget):
        self._widget = widget

    def widget(self):
        return self._widget

    def show(self):
        self.shown = True

    def hide(self):
        self.shown = False

    def raise_(self):
        self.raised = True

    def isVisible(self):
        return self.shown


class _FakeSignal:
    def __init__(self):
        self._callbacks = []

    def connect(self, cb):
        self._callbacks.append(cb)

    def emit(self, *a, **kw):
        for cb in self._callbacks:
            cb(*a, **kw)


class _FakeViewerWidget:
    def __init__(self, dock, context):
        self.loaded = []
        self.cleared = False

    def load_cif(self, path):
        self.loaded.append(path)

    def clear_view(self):
        self.cleared = True


def _install_fake_viewer_module(monkeypatch, widget_cls=_FakeViewerWidget):
    fake_viewer = types.ModuleType("cif_viewer.viewer")
    fake_viewer.CifViewerWidget = widget_cls
    monkeypatch.setitem(sys.modules, "cif_viewer.viewer", fake_viewer)


# ---------------------------------------------------------------------------
# hooked_draw camera-reset / QTimer scheduling branches
# ---------------------------------------------------------------------------


def test_hooked_draw_reset_camera_exception_is_swallowed(monkeypatch):
    _install_fake_qt(monkeypatch)
    _install_fake_viewer_module(monkeypatch)

    context = StubContext()

    def boom():
        raise RuntimeError("camera broke")

    context.reset_3d_camera = boom

    class FakeVM:
        def draw_molecule_3d(self, mol):
            pass

    vm = FakeVM()
    context.main_window.view_3d_manager = vm

    initialize(context)
    context.menu_actions[0][1]()  # show_panel() -> hooks vm

    dock = context.get_window("cif_viewer_panel")
    dock.isVisible = lambda: True
    widget = dock.widget()
    widget.structure = object()
    widget._reset_camera_on_next_render = True
    widget.render_overlays_only = lambda: None

    class Mol:
        def HasProp(self, name):
            return name == "_from_cif_viewer"

    vm.draw_molecule_3d(Mol())
    assert widget._reset_camera_on_next_render is True  # unchanged: exception path


def test_hooked_draw_qtimer_schedule_failure_calls_overlays_directly(monkeypatch):
    _install_fake_qt(monkeypatch)
    _install_fake_viewer_module(monkeypatch)

    context = StubContext()

    class FakeVM:
        def draw_molecule_3d(self, mol):
            pass

    vm = FakeVM()
    context.main_window.view_3d_manager = vm

    initialize(context)
    context.menu_actions[0][1]()

    dock = context.get_window("cif_viewer_panel")
    dock.isVisible = lambda: True
    widget = dock.widget()
    widget.structure = object()
    calls = []
    widget.render_overlays_only = lambda: calls.append(True)

    def bad_singleshot(ms, cb):
        raise RuntimeError("no event loop")

    import PyQt6.QtCore as qtcore

    monkeypatch.setattr(qtcore, "QTimer", types.SimpleNamespace(singleShot=bad_singleshot))

    class Mol:
        def HasProp(self, name):
            return name == "_from_cif_viewer"

    vm.draw_molecule_3d(Mol())
    assert calls == [True]


# ---------------------------------------------------------------------------
# show_panel(file_path=...) when dock already exists
# ---------------------------------------------------------------------------


def test_show_panel_with_file_path_when_dock_already_exists(monkeypatch):
    _install_fake_qt(monkeypatch)
    _install_fake_viewer_module(monkeypatch)

    context = StubContext()
    initialize(context)
    open_from_menu = context.menu_actions[0][1]

    open_from_menu()  # creates the dock
    dock = context.get_window("cif_viewer_panel")
    dock.shown = False
    dock.raised = False

    file_opener = context.file_openers[0][1]
    file_opener("second.cif")

    assert dock.shown is True
    assert dock.raised is True
    assert dock.widget().loaded == ["second.cif"]


def test_open_file_qtimer_schedule_failure_falls_back_to_direct_call(monkeypatch):
    _install_fake_qt(monkeypatch)
    _install_fake_viewer_module(monkeypatch)

    context = StubContext()
    initialize(context)

    def bad_singleshot(ms, cb):
        raise RuntimeError("no loop")

    import PyQt6.QtCore as qtcore

    monkeypatch.setattr(qtcore, "QTimer", types.SimpleNamespace(singleShot=bad_singleshot))

    file_opener = context.file_openers[0][1]
    result = file_opener("direct.cif")
    assert result is True

    dock = context.get_window("cif_viewer_panel")
    assert dock.widget().loaded == ["direct.cif"]


def test_revert_hook_noop_when_not_hooked(monkeypatch):
    _install_fake_qt(monkeypatch)
    _install_fake_viewer_module(monkeypatch)

    context = StubContext()

    class FakeVM:
        def draw_molecule_3d(self, mol):
            pass

    vm = FakeVM()
    context.main_window.view_3d_manager = vm
    initialize(context)

    open_from_menu = context.menu_actions[0][1]
    open_from_menu()
    dock = context.get_window("cif_viewer_panel")

    # Hide twice in a row: second hide() finds _cif_viewer_hooked already False
    # and revert_hook() should just return without changing draw_molecule_3d.
    dock.visibilityChanged.emit(False)
    current = vm.draw_molecule_3d
    dock.visibilityChanged.emit(False)
    assert vm.draw_molecule_3d is current


# ---------------------------------------------------------------------------
# draw_ellipsoid_model: guard branches (invalid mol / missing ADP / plotter None)
# ---------------------------------------------------------------------------


def _base_mw():
    mw = StubMainWindow()
    return mw


def test_draw_ellipsoid_model_invalid_mol_resets_style_via_view_3d_manager(
    monkeypatch,
):
    _install_fake_qt(monkeypatch)
    context = StubContext()
    initialize(context)
    draw_callback = context.styles["Thermal Ellipsoids"]

    class FakeVM:
        def __init__(self):
            self.style = None

        def set_3d_style(self, style):
            self.style = style

    mw = _base_mw()
    mw.view_3d_manager = FakeVM()

    draw_callback(mw, None)
    assert mw.view_3d_manager.style == "ball_and_stick"


def test_draw_ellipsoid_model_no_plotter_anywhere_returns(monkeypatch):
    _install_fake_qt(monkeypatch)
    context = StubContext()
    initialize(context)
    draw_callback = context.styles["Thermal Ellipsoids"]

    mw = _base_mw()  # no plotter, no view_3d_manager

    class Mol:
        def HasProp(self, name):
            return name == "_from_cif_viewer"

    # Should return quietly (no exception) since plotter stays None.
    draw_callback(mw, Mol())


def test_draw_ellipsoid_model_no_adp_falls_back_with_view_3d_manager(monkeypatch):
    _install_fake_qt(monkeypatch)
    context = StubContext()
    initialize(context)
    draw_callback = context.styles["Thermal Ellipsoids"]

    class FakeVM:
        def __init__(self):
            self.style = None

        def set_3d_style(self, style):
            self.style = style

    mw = _base_mw()
    mw.plotter = object()
    mw.view_3d_manager = FakeVM()

    context.register_window("cif_viewer_panel", _FakeDockWidget("CIF Viewer", None))
    context.get_window("cif_viewer_panel").setWidget(types.SimpleNamespace(structure=None))

    class Mol:
        def HasProp(self, name):
            return name == "_from_cif_viewer"

    draw_callback(mw, Mol())
    assert mw.view_3d_manager.style == "ball_and_stick"


# ---------------------------------------------------------------------------
# draw_ellipsoid_model: full render path with lots of knobs exercised.
# ---------------------------------------------------------------------------


class _MockAtom:
    def __init__(self, label, element, base_index):
        self.label = label
        self.element = element
        self.base_index = base_index


class _MockStructure:
    def __init__(self, u_cart):
        self.u_cart = u_cart


class _MockCheckBox:
    def __init__(self, checked):
        self._checked = checked

    def isChecked(self):
        return self._checked


class _MockColorButton:
    def __init__(self, hexval="#ff00ff"):
        self._hex = hexval

    def property(self, name):
        return self._hex if name == "color_hex" else None


class _MockSpinBox:
    def __init__(self, v):
        self._v = v

    def value(self):
        return self._v


class _MockPlotter:
    def __init__(self, camera_position=None, raise_on_camera_set=False,
                 raise_on_reset_camera=False, has_reset_camera=True):
        self.cleared = False
        self.background = None
        self.lights = []
        self.meshes = []
        self.rendered = False
        self._camera_position = camera_position
        self._raise_on_camera_set = raise_on_camera_set
        self._raise_on_reset_camera = raise_on_reset_camera
        self._has_reset_camera = has_reset_camera

    def clear(self):
        self.cleared = True

    def set_background(self, color):
        self.background = color

    def add_light(self, light):
        self.lights.append(light)

    def add_mesh(self, mesh, *args, **kwargs):
        self.meshes.append((mesh, kwargs))
        return types.SimpleNamespace(
            GetProperty=lambda: types.SimpleNamespace(SetEdgeOpacity=lambda o: None)
        )

    def render(self):
        self.rendered = True

    @property
    def camera_position(self):
        return self._camera_position

    @camera_position.setter
    def camera_position(self, value):
        if self._raise_on_camera_set:
            raise RuntimeError("cannot set camera")
        self._camera_position = value

    def reset_camera(self):
        if not self._has_reset_camera:
            raise AttributeError("no reset_camera")
        if self._raise_on_reset_camera:
            raise RuntimeError("reset failed")


def _make_mol(elements, positions):
    editable = Chem.EditableMol(Chem.Mol())
    z_map = {"O": 8, "H": 1, "C": 6, "N": 7}
    for el in elements:
        editable.AddAtom(Chem.Atom(z_map[el]))
    mol = editable.GetMol()
    conf = Chem.Conformer(len(elements))
    for i, p in enumerate(positions):
        conf.SetAtomPosition(i, p)
    mol.AddConformer(conf)
    mol.SetProp("_from_cif_viewer", "1")
    return mol


def _make_widget(**overrides):
    defaults = dict(
        structure=_MockStructure(
            np.array(
                [
                    [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
                    [[0.0, 0.0, 0.0]] * 3,
                    [[0.0, 0.0, 0.0]] * 3,
                ]
            )
        ),
        probability_spin=_MockSpinBox(50.0),
        h_scale_spin=_MockSpinBox(20.0),
        show_ellipsoid_rings=_MockCheckBox(True),
        color_ellipsoid_rings=_MockColorButton(),
        ellipsoid_ring_width=_MockSpinBox(2),
        last_rendered_atoms=[
            _MockAtom("O1", "O", 0),
            _MockAtom("H1", "H", 1),
            _MockAtom("C1", "C", 2),
        ],
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _setup(monkeypatch, mw, widget, mol=None):
    _install_fake_qt(monkeypatch)
    context = StubContext()
    initialize(context)
    draw_callback = context.styles["Thermal Ellipsoids"]
    context.register_window("cif_viewer_panel", _FakeDockWidget("CIF Viewer", None))
    context.get_window("cif_viewer_panel").setWidget(widget)
    if mol is None:
        mol = _make_mol(["O", "H", "C"], [(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    return draw_callback, mol


def test_draw_ellipsoid_model_defaults_when_no_init_manager(monkeypatch):
    mw = _base_mw()
    mw.plotter = _MockPlotter()
    widget = _make_widget()
    draw_callback, mol = _setup(monkeypatch, mw, widget)

    draw_callback(mw, mol)

    assert mw.plotter.background == "#919191"
    assert len(mw.plotter.lights) == 1


def test_draw_ellipsoid_model_probability_edge_values_return_default_scale(
    monkeypatch,
):
    mw = _base_mw()
    mw.plotter = _MockPlotter()
    widget = _make_widget(probability_spin=_MockSpinBox(0.0))
    draw_callback, mol = _setup(monkeypatch, mw, widget)

    draw_callback(mw, mol)
    assert any(
        kw.get("name") == "cif_viewer_ellipsoids" for _, kw in mw.plotter.meshes
    )


def test_draw_ellipsoid_model_probability_combo_paren_match(monkeypatch):
    mw = _base_mw()
    mw.plotter = _MockPlotter()
    combo = types.SimpleNamespace(currentText=lambda: "68.3% (0.9916)")
    widget = _make_widget(probability_combo=combo)
    del widget.probability_spin
    draw_callback, mol = _setup(monkeypatch, mw, widget)
    draw_callback(mw, mol)
    assert mw.plotter.cleared


def test_draw_ellipsoid_model_probability_combo_plain_number_match(monkeypatch):
    mw = _base_mw()
    mw.plotter = _MockPlotter()
    combo = types.SimpleNamespace(currentText=lambda: "sigma 1.5 level")
    widget = _make_widget(probability_combo=combo)
    del widget.probability_spin
    draw_callback, mol = _setup(monkeypatch, mw, widget)
    draw_callback(mw, mol)
    assert mw.plotter.cleared


def test_draw_ellipsoid_model_probability_combo_no_match_uses_default(monkeypatch):
    mw = _base_mw()
    mw.plotter = _MockPlotter()
    combo = types.SimpleNamespace(currentText=lambda: "no digits here")
    widget = _make_widget(probability_combo=combo)
    del widget.probability_spin
    draw_callback, mol = _setup(monkeypatch, mw, widget)
    draw_callback(mw, mol)
    assert mw.plotter.cleared


def test_draw_ellipsoid_model_probability_combo_bad_float_falls_back(monkeypatch):
    mw = _base_mw()
    mw.plotter = _MockPlotter()
    combo = types.SimpleNamespace(currentText=lambda: "sigma (1.2.3)")
    widget = _make_widget(probability_combo=combo)
    del widget.probability_spin
    draw_callback, mol = _setup(monkeypatch, mw, widget)
    draw_callback(mw, mol)
    assert mw.plotter.cleared


def test_draw_ellipsoid_model_atoms_beyond_mol_count_are_skipped(monkeypatch):
    mw = _base_mw()
    mw.plotter = _MockPlotter()
    widget = _make_widget(
        last_rendered_atoms=[
            _MockAtom("O1", "O", 0),
            _MockAtom("H1", "H", 1),
            _MockAtom("C1", "C", 2),
            _MockAtom("N1", "N", 0),  # index 3 has no matching mol atom
        ]
    )
    draw_callback, mol = _setup(monkeypatch, mw, widget)
    draw_callback(mw, mol)
    assert mw.plotter.cleared


def test_draw_ellipsoid_model_camera_restore_success_and_failure(monkeypatch):
    mw = _base_mw()
    mw.plotter = _MockPlotter(camera_position=("pos", "focal", "up"))
    widget = _make_widget()
    draw_callback, mol = _setup(monkeypatch, mw, widget)
    draw_callback(mw, mol)  # success path (line 552-553)

    mw2 = _base_mw()
    mw2.plotter = _MockPlotter(
        camera_position=("pos", "focal", "up"), raise_on_camera_set=True
    )
    widget2 = _make_widget()
    draw_callback2, mol2 = _setup(monkeypatch, mw2, widget2)
    draw_callback2(mw2, mol2)  # exception path (line 554-555)


def test_draw_ellipsoid_model_camera_reset_success_and_failure(monkeypatch):
    mw = _base_mw()
    mw.plotter = _MockPlotter(camera_position=None)
    widget = _make_widget()
    widget._reset_camera_on_next_render = True
    draw_callback, mol = _setup(monkeypatch, mw, widget)
    draw_callback(mw, mol)
    assert widget._reset_camera_on_next_render is False

    mw2 = _base_mw()
    mw2.plotter = _MockPlotter(camera_position=None, raise_on_reset_camera=True)
    widget2 = _make_widget()
    draw_callback2, mol2 = _setup(monkeypatch, mw2, widget2)
    draw_callback2(mw2, mol2)  # logged and swallowed


def test_draw_ellipsoid_model_render_overlays_exception_is_swallowed(monkeypatch):
    mw = _base_mw()
    mw.plotter = _MockPlotter()
    widget = _make_widget()

    def boom():
        raise RuntimeError("overlay render failed")

    widget.render_overlays_only = boom
    draw_callback, mol = _setup(monkeypatch, mw, widget)
    draw_callback(mw, mol)
    assert mw.plotter.cleared


def test_draw_ellipsoid_model_render_axes_view3d_exception_falls_back_to_plotter_render(
    monkeypatch,
):
    mw = _base_mw()
    mw.plotter = _MockPlotter()
    widget = _make_widget()

    class FakeVM:
        def apply_3d_settings(self, redraw=True):
            raise RuntimeError("boom")

    mw.view_3d_manager = FakeVM()
    draw_callback, mol = _setup(monkeypatch, mw, widget)
    draw_callback(mw, mol)
    assert mw.plotter.rendered is True


def test_draw_ellipsoid_model_non_testing_schedules_or_falls_back(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("MOLEDITPY_HEADLESS", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)

    mw = _base_mw()
    mw.plotter = _MockPlotter()
    widget = _make_widget()
    draw_callback, mol = _setup(monkeypatch, mw, widget)
    draw_callback(mw, mol)  # QTimer.singleShot scheduled (line 649-651)


def test_draw_ellipsoid_model_non_testing_qtimer_failure_falls_back(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("MOLEDITPY_HEADLESS", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)

    mw = _base_mw()
    mw.plotter = _MockPlotter()
    widget = _make_widget()
    draw_callback, mol = _setup(monkeypatch, mw, widget)

    import PyQt6.QtCore as qtcore

    def bad_singleshot(ms, cb):
        raise RuntimeError("no loop")

    monkeypatch.setattr(qtcore, "QTimer", types.SimpleNamespace(singleShot=bad_singleshot))
    draw_callback(mw, mol)
    assert mw.plotter.rendered is True


def test_draw_ellipsoid_model_rdkit_and_vdw_import_failures(monkeypatch):
    mw = _base_mw()
    mw.plotter = _MockPlotter()
    widget = _make_widget()
    draw_callback, mol = _setup(monkeypatch, mw, widget)

    monkeypatch.setitem(sys.modules, "rdkit", None)
    monkeypatch.setitem(sys.modules, "moleditpy.constants", None)
    monkeypatch.setitem(sys.modules, "moleditpy.utils.constants", None)

    draw_callback(mw, mol)
    assert mw.plotter.cleared


def test_draw_ellipsoid_model_negative_determinant_eigenvectors(monkeypatch):
    mw = _base_mw()
    mw.plotter = _MockPlotter()
    cov = np.array(
        [
            [0.22684606, -0.01247749, 0.05889865],
            [-0.01247749, 0.23555363, 0.09598517],
            [0.05889865, 0.09598517, 0.12966435],
        ]
    )
    widget = _make_widget(
        structure=_MockStructure(np.array([cov])),
        last_rendered_atoms=[_MockAtom("O1", "O", 0)],
    )
    mol = _make_mol(["O"], [(0, 0, 0)])
    draw_callback, mol = _setup(monkeypatch, mw, widget, mol=mol)
    draw_callback(mw, mol)
    assert any(
        kw.get("name") == "cif_viewer_ellipsoids" for _, kw in mw.plotter.meshes
    )


def test_draw_ellipsoid_model_bad_cov_shape_logged_and_skipped(monkeypatch, capsys):
    mw = _base_mw()
    mw.plotter = _MockPlotter()
    bad_cov = np.array([1.0, 2.0])  # wrong shape -> eigh raises
    widget = _make_widget(
        structure=_MockStructure(np.array([bad_cov], dtype=object)),
        last_rendered_atoms=[_MockAtom("O1", "O", 0)],
    )
    mol = _make_mol(["O"], [(0, 0, 0)])
    draw_callback, mol = _setup(monkeypatch, mw, widget, mol=mol)
    draw_callback(mw, mol)  # should not raise; ellipsoid skipped, error printed
    assert mw.plotter.cleared


def test_draw_ellipsoid_model_hydrogen_rvdw_exception_uses_default(monkeypatch):
    mw = _base_mw()
    mw.plotter = _MockPlotter()
    widget = _make_widget()
    draw_callback, mol = _setup(monkeypatch, mw, widget)

    from rdkit import Chem as RealChem

    real_pt = RealChem.GetPeriodicTable

    class BadPT:
        def GetRvdw(self, symbol):
            raise RuntimeError("no data")

    monkeypatch.setattr(RealChem, "GetPeriodicTable", lambda: BadPT())
    try:
        draw_callback(mw, mol)
    finally:
        monkeypatch.setattr(RealChem, "GetPeriodicTable", real_pt)
    assert any(kw.get("name") == "cif_viewer_h_atoms" for _, kw in mw.plotter.meshes)


# ---------------------------------------------------------------------------
# Threaded rendering path (use_threading branch, lines ~816-860)
# ---------------------------------------------------------------------------


class _FakeThreadSignal:
    def __init__(self):
        self._cb = None

    def connect(self, cb):
        self._cb = cb

    def disconnect(self):
        if self._cb is None:
            raise RuntimeError("nothing connected")
        self._cb = None

    def emit(self, *a):
        if self._cb:
            self._cb(*a)


class _FakeEllipsoidThread:
    instances = []

    def __init__(self, *args, **kwargs):
        self.result_ready = _FakeThreadSignal()
        self.started = False
        self.terminated = False
        self.waited = False
        _FakeEllipsoidThread.instances.append(self)

    def start(self):
        self.started = True
        self.result_ready.emit(None, None, None, None, "")

    def terminate(self):
        self.terminated = True

    def wait(self):
        self.waited = True


def test_draw_ellipsoid_model_threaded_path_used_for_many_atoms(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("MOLEDITPY_HEADLESS", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)

    import cif_viewer

    _FakeEllipsoidThread.instances.clear()
    monkeypatch.setattr(cif_viewer, "EllipsoidWorkerThread", _FakeEllipsoidThread)

    mw = _base_mw()
    mw.plotter = _MockPlotter()

    n = 101
    elements = ["C"] * n
    positions = [(float(i), 0.0, 0.0) for i in range(n)]
    atoms = [_MockAtom(f"C{i}", "C", i) for i in range(n)]
    u_cart = np.array([[[0.0, 0.0, 0.0]] * 3] * n)

    widget = _make_widget(
        structure=_MockStructure(u_cart),
        last_rendered_atoms=atoms,
    )
    mol = _make_mol(elements, positions)
    draw_callback, mol = _setup(monkeypatch, mw, widget, mol=mol)

    draw_callback(mw, mol)

    assert len(_FakeEllipsoidThread.instances) == 1
    assert _FakeEllipsoidThread.instances[0].started is True
    assert mw._ellipsoid_thread is None  # cleared by on_ready


def test_draw_ellipsoid_model_threaded_path_terminates_previous_thread(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("MOLEDITPY_HEADLESS", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)

    import cif_viewer

    _FakeEllipsoidThread.instances.clear()
    monkeypatch.setattr(cif_viewer, "EllipsoidWorkerThread", _FakeEllipsoidThread)

    mw = _base_mw()
    mw.plotter = _MockPlotter()

    n = 101
    elements = ["C"] * n
    positions = [(float(i), 0.0, 0.0) for i in range(n)]
    atoms = [_MockAtom(f"C{i}", "C", i) for i in range(n)]
    u_cart = np.array([[[0.0, 0.0, 0.0]] * 3] * n)

    widget = _make_widget(structure=_MockStructure(u_cart), last_rendered_atoms=atoms)
    mol = _make_mol(elements, positions)
    draw_callback, mol = _setup(monkeypatch, mw, widget, mol=mol)

    old_thread = _FakeEllipsoidThread()
    mw._ellipsoid_thread = old_thread

    draw_callback(mw, mol)

    assert old_thread.terminated is True
    assert old_thread.waited is True


def test_draw_ellipsoid_model_threaded_path_disconnect_exception_swallowed(
    monkeypatch,
):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("MOLEDITPY_HEADLESS", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)

    import cif_viewer

    _FakeEllipsoidThread.instances.clear()
    monkeypatch.setattr(cif_viewer, "EllipsoidWorkerThread", _FakeEllipsoidThread)

    mw = _base_mw()
    mw.plotter = _MockPlotter()

    n = 101
    elements = ["C"] * n
    positions = [(float(i), 0.0, 0.0) for i in range(n)]
    atoms = [_MockAtom(f"C{i}", "C", i) for i in range(n)]
    u_cart = np.array([[[0.0, 0.0, 0.0]] * 3] * n)

    widget = _make_widget(structure=_MockStructure(u_cart), last_rendered_atoms=atoms)
    mol = _make_mol(elements, positions)
    draw_callback, mol = _setup(monkeypatch, mw, widget, mol=mol)

    old_thread = _FakeEllipsoidThread()
    old_thread.result_ready._cb = None  # disconnect() will raise (nothing connected)
    mw._ellipsoid_thread = old_thread

    draw_callback(mw, mol)  # should not raise despite disconnect() failing
    assert old_thread.terminated is True
