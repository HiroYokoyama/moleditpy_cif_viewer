import sys
import types

from cif_viewer import initialize
from tests.test_parser import NACL_CIF


class StubContext:
    def __init__(self):
        self.menu_actions = []
        self.file_openers = []
        self.drop_handlers = []
        self.reset_handlers = []
        self.windows = {}
        self.main_window = StubMainWindow()
        self.styles = {}
        self._plotter = None

    @property
    def plotter(self):
        return self._plotter

    @plotter.setter
    def plotter(self, value):
        self._plotter = value

    def get_main_window(self):
        return self.main_window

    def add_menu_action(self, path, callback, text=None, icon=None, shortcut=None):
        self.menu_actions.append((path, callback, text, icon, shortcut))

    def register_file_opener(self, extension, callback, priority=0):
        self.file_openers.append((extension, callback, priority))

    def register_drop_handler(self, callback, priority=0):
        self.drop_handlers.append((callback, priority))

    def register_document_reset_handler(self, callback):
        self.reset_handlers.append(callback)

    def get_window(self, window_id):
        return self.windows.get(window_id)

    def register_window(self, window_id, window):
        self.windows[window_id] = window

    def register_3d_style(self, name, callback):
        self.styles[name] = callback

    def show_status_message(self, msg, duration=0):
        pass


def test_initialize_registers_visualization_entry_points():
    context = StubContext()

    initialize(context)

    assert context.menu_actions[0][0] == "View/CIF Viewer Panel"
    assert context.file_openers[0][0] == ".cif"
    assert context.file_openers[0][2] == 20
    assert context.drop_handlers[0][1] == 20
    assert len(context.reset_handlers) == 1


def test_drop_handler_only_claims_cif_files(monkeypatch):
    context = StubContext()
    opened = []

    initialize(context)
    callback = context.drop_handlers[0][0]

    action = context.menu_actions[0][1]
    assert callable(action)

    _install_fake_qt(monkeypatch)
    fake_viewer = types.ModuleType("cif_viewer.viewer")
    fake_viewer.CifViewerWidget = _FakeViewerWidget
    monkeypatch.setitem(sys.modules, "cif_viewer.viewer", fake_viewer)
    _FakeViewerWidget.loaded.clear()

    assert callback("sample.txt") is False
    assert callback("sample.cif") is True
    assert _FakeViewerWidget.loaded == ["sample.cif"]
    assert "cif_viewer_panel" in context.windows
    assert context.main_window.docks[0][0] == 2


class StubMainWindow:
    def __init__(self):
        self.docks = []

    def addDockWidget(self, area, dock):
        self.docks.append((area, dock))


class _FakeViewerWidget:
    loaded = []

    def __init__(self, dock, context):
        self.cleared = False

    def load_cif(self, path):
        self.loaded.append(path)

    def clear_view(self):
        self.cleared = True


class FakeSignal:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs):
        for cb in self._callbacks:
            cb(*args, **kwargs)


class _FakeDockWidget:
    def __init__(self, title, parent):
        self.title = title
        self.parent = parent
        self._widget = None
        self.shown = False
        self.raised = False
        self.visibilityChanged = FakeSignal()

    def setAllowedAreas(self, areas):
        self.allowed_areas = areas

    def setWidget(self, widget):
        self._widget = widget

    def widget(self):
        return self._widget

    def show(self):
        self.shown = True
        self.visibilityChanged.emit(True)

    def hide(self):
        self.shown = False
        self.visibilityChanged.emit(False)

    def raise_(self):
        self.raised = True

    def isVisible(self):
        return self.shown


def _install_fake_qt(monkeypatch):
    qtcore = types.ModuleType("PyQt6.QtCore")
    qtwidgets = types.ModuleType("PyQt6.QtWidgets")
    qtgui = types.ModuleType("PyQt6.QtGui")
    pyqt6 = types.ModuleType("PyQt6")

    class DockWidgetArea:
        LeftDockWidgetArea = 1
        RightDockWidgetArea = 2

    class Qt:
        pass

    Qt.DockWidgetArea = DockWidgetArea

    class FakeQColor:
        def __init__(self, *args):
            pass

        def redF(self):
            return 0.5

        def greenF(self):
            return 0.5

        def blueF(self):
            return 0.5

    class FakeQTimer:
        @staticmethod
        def singleShot(ms, cb):
            cb()

    qtcore.Qt = Qt
    qtcore.QTimer = FakeQTimer
    qtwidgets.QDockWidget = _FakeDockWidget
    qtgui.QColor = FakeQColor

    class FakeQFont:
        class Weight:
            Bold = 75

        def __init__(self, *args):
            pass

    qtgui.QFont = FakeQFont

    monkeypatch.setitem(sys.modules, "PyQt6", pyqt6)
    monkeypatch.setitem(sys.modules, "PyQt6.QtCore", qtcore)
    monkeypatch.setitem(sys.modules, "PyQt6.QtWidgets", qtwidgets)
    monkeypatch.setitem(sys.modules, "PyQt6.QtGui", qtgui)


def test_draw_ellipsoid_model(monkeypatch):
    _install_fake_qt(monkeypatch)
    import numpy as np
    from rdkit import Chem
    from cif_viewer import initialize

    # 1. Setup a stub context & register style
    context = StubContext()
    initialize(context)
    assert "Thermal Ellipsoids" in context.styles
    draw_callback = context.styles["Thermal Ellipsoids"]

    # Mock elements / widgets
    class MockAtom:
        def __init__(self, label, element, base_index):
            self.label = label
            self.element = element
            self.base_index = base_index
            self.image = (0, 0, 0)
            self.position = np.array([0.0, 0.0, 0.0])

    class MockStructure:
        def __init__(self):
            # Atom 0: Has ADP data
            # Atom 1: Hydrogen, lacks ADP data (cov is all zeros)
            # Atom 2: Carbon, lacks ADP data
            self.u_cart = np.array(
                [
                    [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],  # ADP data
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # No ADP
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # No ADP
                ]
            )
            self.lattice = np.eye(3)

    class MockDoubleSpinBox:
        def value(self):
            return 50.0

    class MockCheckBox:
        def __init__(self, checked):
            self._checked = checked

        def isChecked(self):
            return self._checked

    class MockWidget:
        def __init__(self):
            self.structure = MockStructure()
            self.probability_spin = MockDoubleSpinBox()
            self.h_scale_spin = MockDoubleSpinBox()
            self.show_ellipsoid_rings = MockCheckBox(True)

            class MockColorButton:
                def property(self, name):
                    if name == "color_hex":
                        return "#ff00ff"
                    return None

            self.color_ellipsoid_rings = MockColorButton()

            class MockSpinBox:
                def value(self):
                    return 5

            self.ellipsoid_ring_width = MockSpinBox()
            self.last_rendered_atoms = [
                MockAtom("O1", "O", 0),
                MockAtom("H1", "H", 1),
                MockAtom("C1", "C", 2),
            ]

    # Setup MainWindow mock
    class MockSettings:
        def get(self, key, default=None):
            if key == "ball_stick_atom_scale":
                return 1.2
            if key == "ball_stick_resolution":
                return 16
            return default

    class MockInitManager:
        def __init__(self):
            self.settings = MockSettings()

    class MockPlotter:
        def __init__(self):
            self.cleared = False
            self.background = None
            self.lights = []
            self.meshes = []
            self.lines = []
            self.rendered = False

        def clear(self):
            self.cleared = True

        def set_background(self, color):
            self.background = color

        def add_light(self, light):
            self.lights.append(light)

        def add_mesh(self, mesh, *args, **kwargs):
            self.meshes.append((mesh, kwargs))

            # Mock return value (Actor)
            class MockActor:
                def GetProperty(self):
                    class MockProp:
                        def SetEdgeOpacity(self, o):
                            pass

                    return MockProp()

            return MockActor()

        def add_lines(self, points, *args, **kwargs):
            self.lines.append((points, kwargs))

        def render(self):
            self.rendered = True

    class MockView3DManager:
        def __init__(self):
            self.apply_3d_settings_called = False
            self.apply_3d_settings_redraw = None

        def apply_3d_settings(self, redraw=True):
            self.apply_3d_settings_called = True
            self.apply_3d_settings_redraw = redraw

    mw = StubMainWindow()
    mw.plotter = MockPlotter()
    mw.init_manager = MockInitManager()
    mw.view_3d_manager = MockView3DManager()

    widget = MockWidget()
    context.register_window("cif_viewer_panel", _FakeDockWidget("CIF Viewer", None))
    context.get_window("cif_viewer_panel").setWidget(widget)

    # Setup RDKit Molecule with exactly 3 atoms
    editable_mol = Chem.EditableMol(Chem.Mol())
    editable_mol.AddAtom(Chem.Atom(8))  # Oxygen (index 0)
    editable_mol.AddAtom(Chem.Atom(1))  # Hydrogen (index 1)
    editable_mol.AddAtom(Chem.Atom(6))  # Carbon (index 2)
    mol = editable_mol.GetMol()

    # Add coordinates conformer
    conformer = Chem.Conformer(3)
    conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conformer.SetAtomPosition(1, (1.0, 0.0, 0.0))
    conformer.SetAtomPosition(2, (0.0, 1.0, 0.0))
    mol.AddConformer(conformer)
    mol.SetProp("_from_cif_viewer", "1")

    # Call the style handler
    draw_callback(mw, mol)

    # Verify rendering results
    assert mw.plotter.cleared
    # We should have meshes added
    assert len(mw.plotter.meshes) > 0

    # Verify apply_3d_settings was called on view_3d_manager to draw axis widget last
    assert mw.view_3d_manager.apply_3d_settings_called
    assert mw.view_3d_manager.apply_3d_settings_redraw is False

    # Let's inspect the opacity and properties of the added meshes
    ellipsoid_mesh = None
    h_mesh = None
    c_mesh = None
    rings_mesh = None
    for mesh, kwargs in mw.plotter.meshes:
        if kwargs.get("name") == "cif_viewer_ellipsoids":
            ellipsoid_mesh = (mesh, kwargs)
        elif kwargs.get("name") == "cif_viewer_ellipsoid_rings":
            rings_mesh = (mesh, kwargs)
        elif kwargs.get("name") == "cif_viewer_h_atoms":
            h_mesh = (mesh, kwargs)
        elif kwargs.get("name") == "cif_viewer_fallback_atoms":
            c_mesh = (mesh, kwargs)

    assert ellipsoid_mesh is not None
    assert ellipsoid_mesh[1]["opacity"] == 1.0

    assert rings_mesh is not None
    assert rings_mesh[1]["color"] == "#ff00ff"
    assert rings_mesh[1]["line_width"] == 5

    assert h_mesh is not None
    assert h_mesh[1]["opacity"] == 1.0

    assert c_mesh is not None
    assert c_mesh[1]["opacity"] == 1.0


def test_menu_action_toggles_dock_widget(monkeypatch):
    _install_fake_qt(monkeypatch)

    # Mock viewer widget class
    fake_viewer = types.ModuleType("cif_viewer.viewer")
    fake_viewer.CifViewerWidget = _FakeViewerWidget
    monkeypatch.setitem(sys.modules, "cif_viewer.viewer", fake_viewer)

    context = StubContext()
    initialize(context)

    action = context.menu_actions[0][1]  # open_from_menu / show_panel
    assert callable(action)

    # First click: Dock doesn't exist, should create it, show it and register it.
    action()
    assert "cif_viewer_panel" in context.windows
    dock = context.windows["cif_viewer_panel"]
    assert dock.shown is True
    assert dock.raised is True

    # Setup visibility controls on fake dock
    dock._visible = True
    dock.isVisible = lambda: dock._visible

    def hide():
        dock._visible = False

    def show():
        dock._visible = True

    dock.hide = hide
    dock.show = show

    # Second click: Dock exists and is visible -> should hide it
    action()
    assert dock._visible is False

    # Third click: Dock exists and is hidden -> should show it
    action()
    assert dock._visible is True


def test_draw_molecule_3d_hook_and_overlays(monkeypatch):
    _install_fake_qt(monkeypatch)

    # Setup mock viewer widget that registers when render_overlays_only is called
    class MockViewerWidget:
        def __init__(self, dock, context):
            self.dock = dock
            self.context = context
            self.structure = "dummy_structure"
            self.overlays_rendered = False
            self.cleared = False

        def render_overlays_only(self):
            self.overlays_rendered = True

        def clear_view(self):
            self.cleared = True

    fake_viewer = types.ModuleType("cif_viewer.viewer")
    fake_viewer.CifViewerWidget = MockViewerWidget
    monkeypatch.setitem(sys.modules, "cif_viewer.viewer", fake_viewer)

    context = StubContext()

    # Add view_3d_manager to the StubMainWindow
    class FakeView3DManager:
        def __init__(self):
            self.drawn_molecule = None

        def draw_molecule_3d(self, mol):
            self.drawn_molecule = mol

    vm = FakeView3DManager()
    context.main_window.view_3d_manager = vm

    # Initialize the plugin
    initialize(context)

    # Open the panel
    show_panel_action = context.menu_actions[0][1]
    show_panel_action()

    # Verify the hook is installed on view_3d_manager.draw_molecule_3d
    assert vm._cif_viewer_hooked is True
    assert vm.draw_molecule_3d != FakeView3DManager.draw_molecule_3d

    # Get the dock and mock its visibility to True
    dock = context.get_window("cif_viewer_panel")
    dock._visible = True
    dock.isVisible = lambda: dock._visible

    class MockMolecule:
        def HasProp(self, name):
            return name == "_from_cif_viewer"

    mol = MockMolecule()

    # Call the hooked draw_molecule_3d with a valid CIF molecule
    vm.draw_molecule_3d(mol)

    # Verify that the original draw was called
    assert vm.drawn_molecule == mol

    # Verify that render_overlays_only was triggered
    widget = dock.widget()
    assert widget.overlays_rendered is True
    assert widget.cleared is False

    # Call the hooked draw_molecule_3d with a non-CIF molecule (e.g. a string)
    vm.draw_molecule_3d("not_cif_mol")

    # Verify that clear_view was called
    assert widget.cleared is True


def test_draw_molecule_3d_hook_visibility_toggling(monkeypatch):
    _install_fake_qt(monkeypatch)

    class MockViewerWidget:
        def __init__(self, dock, context):
            self.dock = dock
            self.context = context
            self.structure = "dummy_structure"

    fake_viewer = types.ModuleType("cif_viewer.viewer")
    fake_viewer.CifViewerWidget = MockViewerWidget
    monkeypatch.setitem(sys.modules, "cif_viewer.viewer", fake_viewer)

    context = StubContext()

    class FakeView3DManager:
        def __init__(self):
            self.drawn_molecule = None

        def draw_molecule_3d(self, mol):
            self.drawn_molecule = mol

    vm = FakeView3DManager()
    context.main_window.view_3d_manager = vm

    initialize(context)

    # Hook shouldn't be active initially
    assert getattr(vm, "_cif_viewer_hooked", False) is False

    # Open panel (shows dock)
    show_panel_action = context.menu_actions[0][1]
    show_panel_action()

    # Hook should now be active
    assert vm._cif_viewer_hooked is True
    assert vm.draw_molecule_3d != FakeView3DManager.draw_molecule_3d

    # Hide the dock panel
    dock = context.get_window("cif_viewer_panel")
    dock.hide()

    # Hook should be reverted
    assert vm._cif_viewer_hooked is False
    assert vm.draw_molecule_3d == vm._orig_draw_molecule_3d

    # Show the dock panel again
    dock.show()

    # Hook should be active again
    assert vm._cif_viewer_hooked is True
    assert vm.draw_molecule_3d != vm._orig_draw_molecule_3d


def test_cif_viewer_widget_initialization(qtbot):
    from cif_viewer.viewer import CifViewerWidget

    context = StubContext()
    widget = CifViewerWidget(context=context)
    qtbot.addWidget(widget)

    assert widget.structure is None
    assert widget.repeat_a.value() == 1
    assert widget.show_bonds.isChecked() is True
    assert widget.determine_bond_order.isChecked() is False
    assert widget._get_current_view_mode() == "Whole Molecule"
    assert widget.show_cell.isEnabled() is False
    assert widget.show_axes.isEnabled() is False


def test_cif_viewer_widget_load_cif(qtbot, tmp_path):
    from cif_viewer.viewer import CifViewerWidget

    cif_file = tmp_path / "test.cif"
    cif_file.write_text(NACL_CIF, encoding="utf-8")

    context = StubContext()
    widget = CifViewerWidget(context=context)
    qtbot.addWidget(widget)

    # Set repetitions to non-1 values
    widget.repeat_a.setValue(3)
    widget.repeat_b.setValue(2)
    widget.repeat_c.setValue(4)

    widget.load_cif(str(cif_file))

    assert widget.structure is not None
    assert widget.structure.name == "NaCl"
    assert widget.file_label.text() == "test.cif"
    assert len(widget.all_structures) == 1

    # Verify that repetitions were reset to 1
    assert widget.repeat_a.value() == 1
    assert widget.repeat_b.value() == 1
    assert widget.repeat_c.value() == 1


def test_cif_viewer_widget_settings_and_ui_actions(qtbot, tmp_path, monkeypatch):
    from cif_viewer.viewer import CifViewerWidget

    context = StubContext()
    widget = CifViewerWidget(context=context)
    qtbot.addWidget(widget)

    # Test setting path redirection using monkeypatch
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(widget, "_settings_path", lambda: str(settings_file))

    # Modify some UI elements
    widget.show_bonds.setChecked(False)
    widget.show_hydrogens.setChecked(False)
    widget.keep_connected.setChecked(False)
    widget.fix_h_size.setChecked(False)
    widget.probability_spin.setValue(45.0)
    widget.h_scale_spin.setValue(25.0)
    widget.ellipsoid_ring_width.setValue(4)
    widget._set_button_color(widget.color_ellipsoid_rings, "#123456")

    widget.save_settings()
    assert settings_file.exists()

    # Reset widget state with signals blocked so we don't trigger toggled signals and overwrite settings
    widget.show_bonds.blockSignals(True)
    widget.show_bonds.setChecked(True)
    widget.show_bonds.blockSignals(False)

    widget.load_settings()
    assert widget.show_bonds.isChecked() is False
    assert widget.show_hydrogens.isChecked() is False
    assert widget.keep_connected.isChecked() is False
    assert widget.fix_h_size.isChecked() is False
    assert widget.probability_spin.value() == 45.0
    assert widget.h_scale_spin.value() == 25.0
    assert widget.ellipsoid_ring_width.value() == 4
    assert widget.color_ellipsoid_rings.property("color_hex") == "#123456"


def test_cif_viewer_widget_reset_to_defaults(qtbot, tmp_path, monkeypatch):
    from cif_viewer.viewer import CifViewerWidget

    context = StubContext()
    widget = CifViewerWidget(context=context)
    qtbot.addWidget(widget)

    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(widget, "_settings_path", lambda: str(settings_file))

    # Modify UI values away from defaults
    widget.show_bonds.setChecked(False)
    widget.determine_bond_order.setChecked(True)
    widget.show_hydrogens.setChecked(False)
    widget.keep_connected.setChecked(False)
    widget.show_cell.setChecked(False)
    widget.show_axes.setChecked(False)
    widget.show_ellipsoid_rings.setChecked(False)
    widget.fix_h_size.setChecked(False)
    widget.probability_spin.setValue(45.0)
    widget.h_scale_spin.setValue(25.0)
    widget.axis_width.setValue(8)
    widget.axis_font_size.setValue(15)
    widget.ellipsoid_ring_width.setValue(5)
    widget._set_button_color(widget.color_ellipsoid_rings, "#ffffff")

    # Trigger reset to defaults
    widget.reset_to_defaults()

    # Verify values are restored to defaults
    assert widget.show_bonds.isChecked() is True
    assert widget.determine_bond_order.isChecked() is False
    assert widget.show_hydrogens.isChecked() is True
    assert widget.keep_connected.isChecked() is True
    assert widget.show_cell.isChecked() is True
    assert widget.show_axes.isChecked() is True
    assert widget.show_ellipsoid_rings.isChecked() is True
    assert widget.fix_h_size.isChecked() is True
    assert widget.show_cell.isEnabled() is False
    assert widget.show_axes.isEnabled() is False
    assert widget.probability_spin.value() == 50.0
    assert widget.h_scale_spin.value() == 20.0
    assert widget.axis_width.value() == 5
    assert widget.axis_font.currentText() == "arial"
    assert widget.axis_font_size.value() == 20
    assert widget.color_axis_a.property("color_hex") == "#ff0000"
    assert widget.color_ellipsoid_rings.property("color_hex") == "#000000"
    assert widget.ellipsoid_ring_width.value() == 2

    # Check that settings file was updated with default values too
    assert settings_file.exists()
    import json

    with open(settings_file, "r") as f:
        data = json.load(f)
    assert data["show_bonds"] is True
    assert data["determine_bond_order"] is False
    assert data["show_hydrogens"] is True
    assert data["fix_h_size"] is True
    assert data["probability"] == 50.0
    assert data["color_axis_a"] == "#ff0000"
    assert data["color_ellipsoid_rings"] == "#000000"
    assert data["ellipsoid_ring_width"] == 2


def test_cif_viewer_widget_reset_supercell(qtbot):
    from cif_viewer.viewer import CifViewerWidget

    widget = CifViewerWidget()
    qtbot.addWidget(widget)

    widget.repeat_a.setValue(3)
    widget.repeat_b.setValue(2)
    widget.reset_supercell()
    assert widget.repeat_a.value() == 1
    assert widget.repeat_b.value() == 1

    widget.set_supercell_preset(2)
    assert widget.repeat_a.value() == 2
    assert widget.repeat_b.value() == 2
    assert widget.repeat_c.value() == 2

    widget.set_supercell_preset(3)
    assert widget.repeat_a.value() == 3
    assert widget.repeat_b.value() == 3
    assert widget.repeat_c.value() == 3


def test_cif_viewer_widget_switch_to_ellipsoids(qtbot):
    from cif_viewer.viewer import CifViewerWidget

    context = StubContext()

    class FakeView3DManager:
        def __init__(self):
            self.style = None

        def set_3d_style(self, style):
            self.style = style

    class MockAction:
        def __init__(self, text):
            self._text = text
            self._checked = False

        def text(self):
            return self._text

        def setChecked(self, checked):
            self._checked = checked

        def isChecked(self):
            return self._checked

    class MockMenu:
        def __init__(self, actions):
            self._actions = actions

        def actions(self):
            return self._actions

    class MockStyleButton:
        def __init__(self, menu):
            self._menu = menu

        def menu(self):
            return self._menu

    class MockInitManager:
        def __init__(self, style_button):
            self.style_button = style_button

    action_other = MockAction("Ball and Stick")
    action_ellip = MockAction("Thermal Ellipsoids")
    menu = MockMenu([action_other, action_ellip])
    style_button = MockStyleButton(menu)
    init_manager = MockInitManager(style_button)

    context.main_window.init_manager = init_manager
    context.main_window.view_3d_manager = FakeView3DManager()

    widget = CifViewerWidget(context=context)
    qtbot.addWidget(widget)

    widget._switch_to_ellipsoids()
    assert context.main_window.view_3d_manager.style == "Thermal Ellipsoids"
    assert action_ellip.isChecked() is True
    assert action_other.isChecked() is False


def test_handle_reset(monkeypatch):
    _install_fake_qt(monkeypatch)
    context = StubContext()
    initialize(context)

    reset_callback = context.reset_handlers[0]

    class FakeWidget:
        def __init__(self):
            self.cleared = False

        def clear_view(self):
            self.cleared = True

    dock = _FakeDockWidget("CIF Viewer", None)
    widget = FakeWidget()
    dock.setWidget(widget)
    context.register_window("cif_viewer_panel", dock)

    reset_callback()
    assert widget.cleared is True


def test_cif_viewer_widget_clear_view(qtbot):
    from cif_viewer.viewer import CifViewerWidget

    context = StubContext()

    class FakePlotter:
        def __init__(self):
            self.removed = []
            self.rendered = False

        def remove_actor(self, name):
            self.removed.append(name)

        def render(self):
            self.rendered = True

    context.plotter = FakePlotter()
    widget = CifViewerWidget(context=context)
    qtbot.addWidget(widget)

    widget.overlay_actor_names = ["actor1", "actor2"]
    widget.clear_view()

    assert context.plotter.removed == ["actor1", "actor2"]
    assert len(widget.overlay_actor_names) == 0
    assert context.plotter.rendered is True


def test_cif_viewer_widget_show_hydrogens_filter(qtbot, tmp_path):
    from cif_viewer.viewer import CifViewerWidget

    cif_file = tmp_path / "test_h.cif"
    # A simple CIF containing H and C
    h_cif = """data_H_test
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
H1 H 0.1 0.1 0.1
"""
    cif_file.write_text(h_cif, encoding="utf-8")

    context = StubContext()

    class FakePlotter:
        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

        def add_lines(self, *args, **kwargs):
            pass

        def add_point_labels(self, *args, **kwargs):
            pass

        def render(self):
            pass

        def reset_camera(self):
            pass

    context.plotter = FakePlotter()

    widget = CifViewerWidget(context=context)
    qtbot.addWidget(widget)
    widget.load_cif(str(cif_file))
    widget.radio_asym.setChecked(True)

    # Run _render_now with show_hydrogens = True (default)
    widget.show_hydrogens.setChecked(True)
    widget._render_now()
    assert len(widget.last_rendered_atoms) == 2

    # Run _render_now with show_hydrogens = False
    widget.show_hydrogens.setChecked(False)
    widget._render_now()
    assert len(widget.last_rendered_atoms) == 1
    assert widget.last_rendered_atoms[0].element == "C"


def test_draw_ellipsoid_model_fix_h_size(monkeypatch):
    _install_fake_qt(monkeypatch)
    import numpy as np
    from rdkit import Chem
    from cif_viewer import initialize

    context = StubContext()
    initialize(context)
    draw_callback = context.styles["Thermal Ellipsoids"]

    class MockAtom:
        def __init__(self, label, element, base_index):
            self.label = label
            self.element = element
            self.base_index = base_index
            self.image = (0, 0, 0)
            self.position = np.array([0.0, 0.0, 0.0])

    class MockStructure:
        def __init__(self):
            # Both Carbon (0) and Hydrogen (1) have ADP covariance matrices
            self.u_cart = np.array(
                [
                    [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],  # C0 ADP
                    [[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]],  # H1 ADP
                ]
            )
            self.lattice = np.eye(3)

    class MockDoubleSpinBox:
        def value(self):
            return 50.0

    class MockCheckBox:
        def __init__(self, checked):
            self._checked = checked

        def isChecked(self):
            return self._checked

    class MockWidget:
        def __init__(self, fix_h_size):
            self.structure = MockStructure()
            self.probability_spin = MockDoubleSpinBox()
            self.h_scale_spin = MockDoubleSpinBox()
            self.show_ellipsoid_rings = MockCheckBox(True)
            self.fix_h_size = MockCheckBox(fix_h_size)
            self.last_rendered_atoms = [
                MockAtom("C1", "C", 0),
                MockAtom("H1", "H", 1),
            ]

    class MockSettings:
        def get(self, key, default=None):
            return default

    class MockInitManager:
        def __init__(self):
            self.settings = MockSettings()

    class MockPlotter:
        def __init__(self):
            self.cleared = False
            self.meshes = []
            self.background = None
            self.lights = []

        def clear(self):
            self.cleared = True

        def set_background(self, color):
            self.background = color

        def add_light(self, light):
            self.lights.append(light)

        def render(self):
            pass

        def add_mesh(self, mesh, *args, **kwargs):
            self.meshes.append((mesh, kwargs))

            class MockActor:
                def GetProperty(self):
                    class MockProp:
                        def SetEdgeOpacity(self, o):
                            pass

                    return MockProp()

            return MockActor()

    mw = StubMainWindow()
    mw.plotter = MockPlotter()
    mw.init_manager = MockInitManager()

    # Create editable mol with Carbon and Hydrogen
    editable_mol = Chem.EditableMol(Chem.Mol())
    editable_mol.AddAtom(Chem.Atom(6))
    editable_mol.AddAtom(Chem.Atom(1))
    mol = editable_mol.GetMol()
    conformer = Chem.Conformer(2)
    conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conformer.SetAtomPosition(1, (1.0, 0.0, 0.0))
    mol.AddConformer(conformer)
    mol.SetProp("_from_cif_viewer", "1")

    # Case 1: fix_h_size = True
    widget_fixed = MockWidget(fix_h_size=True)
    context.register_window("cif_viewer_panel", _FakeDockWidget("CIF Viewer", None))
    context.get_window("cif_viewer_panel").setWidget(widget_fixed)

    draw_callback(mw, mol)

    # Let's see the added meshes. Since fix_h_size=True, hydrogen must fall back to fixed sphere:
    # "cif_viewer_h_atoms" should exist, and "cif_viewer_ellipsoids" should only contain the carbon.
    h_mesh_fixed = any(
        kw.get("name") == "cif_viewer_h_atoms" for mesh, kw in mw.plotter.meshes
    )
    assert h_mesh_fixed is True

    # Case 2: fix_h_size = False
    mw.plotter = MockPlotter()  # Reset plotter
    widget_ellipsoid = MockWidget(fix_h_size=False)
    context.get_window("cif_viewer_panel").setWidget(widget_ellipsoid)

    draw_callback(mw, mol)

    # Since fix_h_size=False, hydrogen must NOT fall back to fixed sphere, it should be drawn as an ellipsoid.
    # So "cif_viewer_h_atoms" should NOT exist.
    h_mesh_fixed = any(
        kw.get("name") == "cif_viewer_h_atoms" for mesh, kw in mw.plotter.meshes
    )
    assert h_mesh_fixed is False


def test_cif_viewer_widget_view_from_axis(qtbot):
    from cif_viewer.viewer import CifViewerWidget
    import numpy as np

    class MockCamera:
        def __init__(self):
            self.position = [0, 0, 0]
            self.focal_point = [0, 0, 0]
            self.up = [0, 0, 1]

    class MockPlotter:
        def __init__(self):
            self.camera = MockCamera()
            self._camera_position = None
            self.rendered = False
            self.camera_reset = False

        @property
        def camera_position(self):
            return self._camera_position

        @camera_position.setter
        def camera_position(self, value):
            self._camera_position = value
            self.camera.position = value[0]
            self.camera.focal_point = value[1]
            self.camera.up = value[2]

        def reset_camera(self):
            self.camera_reset = True

        def render(self):
            self.rendered = True

    plotter = MockPlotter()
    context = StubContext()
    context.plotter = plotter

    widget = CifViewerWidget(context=context)
    qtbot.addWidget(widget)

    class FakeStructure:
        def __init__(self):
            self.lattice = np.eye(3)
            self.atoms = []

    widget.structure = FakeStructure()

    # Test a axis
    widget.view_from_axis("a")
    assert plotter.camera_position is not None
    pos, focal, up = plotter.camera_position
    assert np.allclose(focal, [0.5, 0.5, 0.5])
    # direction from focal point to camera is +a (which is [1, 0, 0])
    assert np.allclose(pos - focal, [2.0, 0.0, 0.0])
    assert np.allclose(up, [0.0, 0.0, 1.0])
    assert plotter.camera_reset is True
    assert plotter.rendered is True
    assert widget._reset_camera_on_next_render is False

    # Test b axis
    widget.view_from_axis("b")
    pos, focal, up = plotter.camera_position
    assert np.allclose(pos - focal, [0.0, 2.0, 0.0])
    assert np.allclose(up, [0.0, 0.0, 1.0])

    # Test c axis
    widget.view_from_axis("c")
    pos, focal, up = plotter.camera_position
    assert np.allclose(pos - focal, [0.0, 0.0, 2.0])
    assert np.allclose(up, [0.0, 1.0, 0.0])

    # Test -a axis
    widget.view_from_axis("-a")
    pos, focal, up = plotter.camera_position
    assert np.allclose(pos - focal, [-2.0, 0.0, 0.0])
    assert np.allclose(up, [0.0, 0.0, 1.0])


def test_cif_viewer_widget_disorder_and_info(qtbot, tmp_path):
    from cif_viewer.viewer import CifViewerWidget

    cif_content = """data_disorder_info
_space_group_name_h-m_alt 'P 21/c'
_space_group_it_number 14
_space_group_crystal_system monoclinic
_chemical_formula_sum 'C10 H15 N O'
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_cell_volume 1000.0
_cell_formula_units_z 4
_cell_formula_units_zprime 1.5
_refine_absolute_configuration_flack 0.02(5)
_exptl_crystal_density_diffrn 1.25
_exptl_absorpt_coefficient_mu 0.12
_exptl_crystal_f_000 240
_diffrn_ambient_temperature 293
_diffrn_radiation_wavelength 0.71073
_exptl_crystal_size_max 0.3
_exptl_crystal_size_mid 0.2
_exptl_crystal_size_min 0.1
_diffrn_reflns_theta_min 2.5
_diffrn_reflns_theta_max 27.5
_diffrn_reflns_limit_h_min -12
_diffrn_reflns_limit_h_max 12
_diffrn_reflns_limit_k_min -12
_diffrn_reflns_limit_k_max 12
_diffrn_reflns_limit_l_min -12
_diffrn_reflns_limit_l_max 12
_diffrn_reflns_number 5000
_refine_ls_number_reflns 4500
_diffrn_reflns_av_r_equivalents 0.032
_diffrn_measured_fraction_theta_max 0.99
_refine_ls_structure_factor_coef Fsqd
_refine_ls_number_parameters 150
_refine_ls_number_restraints 0
_refine_ls_shift/su_max 0.002
_refine_diff_density_max 0.25
_refine_diff_density_min -0.25
_refine_ls_r_factor_gt 0.045
_refine_ls_wr_factor_ref 0.120
_refine_ls_r_factor_all 0.055
_refine_ls_wr_factor_all 0.130
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
    cif_file = tmp_path / "disorder_info.cif"
    cif_file.write_text(cif_content, encoding="utf-8")

    context = StubContext()

    class FakePlotter:
        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

        def add_lines(self, *args, **kwargs):
            pass

        def add_point_labels(self, *args, **kwargs):
            pass

        def render(self):
            pass

        def reset_camera(self):
            pass

    context.plotter = FakePlotter()

    widget = CifViewerWidget(context=context)
    qtbot.addWidget(widget)
    widget.show()
    widget.load_cif(str(cif_file))
    widget.radio_asym.setChecked(True)

    # 1. Verify Info Tab Labels
    assert widget.info_formula.text() == "C10 H15 N O"
    assert widget.info_space_group.text() == "P 21/c"
    assert widget.info_space_group_number.text() == "14"
    assert widget.info_crystal_system.text() == "monoclinic"
    assert widget.info_cell_a.text() == "10.0"
    assert widget.info_cell_b.text() == "10.0"
    assert widget.info_cell_c.text() == "10.0"
    assert widget.info_cell_alpha.text() == "90"
    assert widget.info_cell_beta.text() == "90"
    assert widget.info_cell_gamma.text() == "90"
    assert widget.info_volume.text() == "1000.0"
    assert widget.info_z.text() == "4"
    assert widget.info_z_prime.text() == "1.5"
    assert widget.info_density.text() == "1.25"
    assert widget.info_mu.text() == "0.12"
    assert widget.info_f000.text() == "240"

    assert widget.info_temp.text() == "293"
    assert widget.info_wavelength.text() == "0.71073"
    assert widget.info_crystal_size.text() == "0.3 x 0.2 x 0.1"
    assert widget.info_theta_range.text() == "2.5 to 27.5"
    assert widget.info_hkl_ranges.text() == "h: -12/12, k: -12/12, l: -12/12"
    assert widget.info_reflns_collected.text() == "5000"
    assert widget.info_reflns_unique.text() == "4500"
    assert widget.info_r_int.text() == "0.032"
    assert widget.info_completeness.text() == "0.99"

    assert widget.info_refinement_method.text() == "Fsqd"
    assert widget.info_num_reflns.text() == "4500"
    assert widget.info_num_params.text() == "150"
    assert widget.info_num_restraints.text() == "0"
    assert widget.info_goof.text() == "1.05"
    assert widget.info_r1.text() == "0.045"
    assert widget.info_wr2.text() == "0.120"
    assert widget.info_r1_all.text() == "0.055"
    assert widget.info_wr2_all.text() == "0.130"
    assert widget.info_max_shift.text() == "0.002"
    assert widget.info_flack.text() == "0.02(5)"
    assert widget.info_diff_peak_hole.text() == "0.25 / -0.25"

    assert widget.simulate_xrd_btn.isEnabled() is True

    # 2. Verify Disorder Dropdown UI visibility and items
    assert widget.disorder_label.isVisible() is True
    assert widget.disorder_combo.isVisible() is True
    assert widget.disorder_combo.count() == 3
    assert widget.disorder_combo.itemText(0) == "All Parts"
    assert widget.disorder_combo.itemText(1) == "Part 1"
    assert widget.disorder_combo.itemText(2) == "Part 2"

    # 3. Verify render filtering
    # Index 0: All parts
    widget.disorder_combo.setCurrentIndex(0)
    widget._render_now()
    assert len(widget.last_rendered_atoms) == 3

    # Index 1: Part A_1 (C1 and C3, C2 should be filtered out)
    widget.disorder_combo.setCurrentIndex(1)
    widget._render_now()
    assert len(widget.last_rendered_atoms) == 2
    atoms_map = {a.label: a for a in widget.last_rendered_atoms}
    assert "C1" in atoms_map
    assert "C3" in atoms_map
    assert "C2" not in atoms_map

    # Index 2: Part A_2 (C2 and C3, C1 should be filtered out)
    widget.disorder_combo.setCurrentIndex(2)
    widget._render_now()
    assert len(widget.last_rendered_atoms) == 2
    atoms_map = {a.label: a for a in widget.last_rendered_atoms}
    assert "C2" in atoms_map
    assert "C3" in atoms_map
    assert "C1" not in atoms_map


def test_cif_viewer_widget_export_with_disorder(qtbot, tmp_path, monkeypatch):
    from cif_viewer.viewer import CifViewerWidget

    cif_content = """data_export
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
_atom_site_disorder_assembly
C1 C 0.1 0.1 0.1 1 A
C2 C 0.2 0.2 0.2 2 A
C3 C 0.3 0.3 0.3 . .
"""
    cif_file = tmp_path / "export_disorder.cif"
    cif_file.write_text(cif_content, encoding="utf-8")

    context = StubContext()

    class FakePlotter:
        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

        def add_lines(self, *args, **kwargs):
            pass

        def add_point_labels(self, *args, **kwargs):
            pass

        def render(self):
            pass

        def reset_camera(self):
            pass

    context.plotter = FakePlotter()

    widget = CifViewerWidget(context=context)
    qtbot.addWidget(widget)
    widget.show()
    widget.load_cif(str(cif_file))

    # Select Part A_1 (index 1)
    widget.disorder_combo.setCurrentIndex(1)

    # Mock QFileDialog.getSaveFileName to return our target export path
    export_path = tmp_path / "exported_supercell.cif"
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (
            str(export_path),
            "Crystallographic Information Files (*.cif)",
        ),
    )

    # Mock QMessageBox.information
    info_calls = []
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QMessageBox.information",
        lambda *args, **kwargs: info_calls.append(args),
    )

    widget._export_supercell()
    assert export_path.exists()

    # Read the exported CIF and check atoms.
    # C2 (disorder Part A_2) should not be exported since we selected Part A_1.
    from cif_viewer.parser import parse_cif

    exported_struct = parse_cif(export_path.read_text(encoding="utf-8"))
    exported_labels = {a.label.split("_")[0] for a in exported_struct.atoms}
    assert "C1" in exported_labels
    assert "C3" in exported_labels
    assert "C2" not in exported_labels


def test_powder_pattern_dialog(qtbot):
    from cif_viewer.viewer_xrd import PowderPatternDialog, make_pymatgen_structure
    from cif_viewer.parser import parse_cif
    import numpy as np

    cif_content = """data_xrd
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
_atom_site_disorder_assembly
C1 C 0.1 0.1 0.1 1 A
C2 C 0.2 0.2 0.2 2 A
C3 C 0.3 0.3 0.3 . .
"""
    structure = parse_cif(cif_content)

    # Test make_pymatgen_structure helper
    pm_struct_all = make_pymatgen_structure(structure, selected_disorder_key=None)
    assert len(pm_struct_all) == 3
    # Check partial occupancies (since disorder key is None, they keep their values, defaulting to 1.0 here)
    assert pm_struct_all[0].species.num_atoms == 1.0

    pm_struct_part1 = make_pymatgen_structure(structure, selected_disorder_key="A_1")
    assert len(pm_struct_part1) == 2
    # Disordered atoms in the selected group should have their occupancy normalized to 1.0
    assert pm_struct_part1[0].species.num_atoms == 1.0

    # Test symmetry expansion helper
    from cif_viewer.parser import CifStructure, CifAtom

    lattice = np.eye(3) * 5.0
    cif_atom = CifAtom(
        label="C1",
        element="C",
        fract=np.array([0.1, 0.2, 0.3]),
        cart=np.array([0.5, 1.0, 1.5]),
        occupancy=0.8,
        disorder_group="1",
        disorder_assembly="A",
    )
    asym_struct = CifStructure(
        name="test_sym",
        cell_lengths=(5.0, 5.0, 5.0),
        cell_angles=(90.0, 90.0, 90.0),
        lattice=lattice,
        atoms=(cif_atom,),
        space_group="P -1",  # P-1 has inversion symmetry: (x,y,z) and (-x,-y,-z)
        is_asymmetric_unit_only=True,
    )
    pm_struct_sym = make_pymatgen_structure(asym_struct)
    # P-1 should expand 1 atom to 2 atoms
    assert len(pm_struct_sym) == 2
    assert pm_struct_sym[0].species.num_atoms == 0.8

    dialog = PowderPatternDialog(structure, selected_disorder_key="A_1")
    qtbot.addWidget(dialog)

    # Verify defaults
    assert dialog.source_combo.currentText() == "CuKa"
    assert np.isclose(dialog.wavelength_spin.value(), 1.5418)
    assert dialog.wavelength_spin.isEnabled() is False

    # Verify new pattern style controls
    assert dialog.pattern_type_combo.currentText() == "Profile (Gaussian)"
    assert dialog.fwhm_spin.isEnabled() is True
    assert dialog.show_sticks_chk.isEnabled() is True

    # Test change source to MoKa
    dialog.source_combo.setCurrentText("MoKa")
    assert np.isclose(dialog.wavelength_spin.value(), 0.7107)
    assert dialog.wavelength_spin.isEnabled() is False

    # Test change source to Custom
    dialog.source_combo.setCurrentText("Custom")
    assert dialog.wavelength_spin.isEnabled() is True
    dialog.wavelength_spin.setValue(1.8)

    # Test changing min/max 2-theta
    dialog.min_theta_spin.setValue(10.0)
    dialog.max_theta_spin.setValue(80.0)

    # Test changing pattern style to Sticks Only
    dialog.pattern_type_combo.setCurrentText("Sticks Only")
    assert dialog.fwhm_spin.isEnabled() is False
    assert dialog.show_sticks_chk.isEnabled() is False

    # Test changing pattern style to Profile (Lorentzian)
    dialog.pattern_type_combo.setCurrentText("Profile (Lorentzian)")
    assert dialog.fwhm_spin.isEnabled() is True
    dialog.fwhm_spin.setValue(0.15)

    # Test toggle peak labels and show sticks
    dialog.label_peaks_chk.setChecked(False)
    dialog.show_sticks_chk.setChecked(False)

    # Verify that the calculation ran and updated the plot without errors
    assert len(dialog.fig.axes) == 1


def test_cif_viewer_widget_simulate_powder_pattern_trigger(
    qtbot, tmp_path, monkeypatch
):
    from cif_viewer.viewer import CifViewerWidget

    cif_content = """data_xrd_trig
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
    cif_file = tmp_path / "xrd_trig.cif"
    cif_file.write_text(cif_content, encoding="utf-8")

    context = StubContext()

    class FakePlotter:
        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

        def add_lines(self, *args, **kwargs):
            pass

        def add_point_labels(self, *args, **kwargs):
            pass

        def render(self):
            pass

        def reset_camera(self):
            pass

    context.plotter = FakePlotter()

    widget = CifViewerWidget(context=context)
    qtbot.addWidget(widget)
    widget.load_cif(str(cif_file))

    dialog_instantiated = []
    dialog_exec_called = []

    from cif_viewer.viewer_xrd import PowderPatternDialog

    orig_init = PowderPatternDialog.__init__

    def mock_init(self, cif_structure, selected_disorder_key=None, parent=None):
        orig_init(self, cif_structure, selected_disorder_key, parent)
        dialog_instantiated.append((cif_structure, selected_disorder_key))

    def mock_exec(self):
        dialog_exec_called.append(True)
        return 1

    monkeypatch.setattr(PowderPatternDialog, "__init__", mock_init)
    monkeypatch.setattr(PowderPatternDialog, "exec", mock_exec)

    # Click the simulate XRD button
    widget.simulate_xrd_btn.click()

    assert len(dialog_instantiated) == 1
    assert dialog_instantiated[0][0].name == "xrd_trig"
    assert dialog_instantiated[0][1] is None
    assert len(dialog_exec_called) == 1


def test_powder_pattern_dialog_export_csv(qtbot, tmp_path, monkeypatch):
    from cif_viewer.viewer_xrd import PowderPatternDialog
    from cif_viewer.parser import parse_cif
    from PyQt6.QtWidgets import QMessageBox

    cif_content = """data_xrd
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

    # Verify calculation has completed and populated xrd & profile properties
    assert dialog.last_xrd is not None
    assert len(dialog.last_xrd.x) > 0
    assert dialog.last_profile_x is not None
    assert dialog.last_profile_y is not None

    # Mock QMessageBox to intercept the choice box
    class MockQMessageBox:
        StandardButton = QMessageBox.StandardButton
        ButtonRole = QMessageBox.ButtonRole

        simulated_clicked_button_name = "Simulated Profile (Curve)"
        simulated_cancel = False

        def __init__(self, parent=None):
            self._buttons = []
            self._clicked_btn = None

        def setWindowTitle(self, title):
            pass

        def setText(self, text):
            pass

        def addButton(self, arg1, role=None):
            self._buttons.append(arg1)
            return arg1

        def exec(self):
            if MockQMessageBox.simulated_cancel:
                for btn in self._buttons:
                    if btn == QMessageBox.StandardButton.Cancel:
                        self._clicked_btn = btn
                        return
            else:
                for btn in self._buttons:
                    if btn == MockQMessageBox.simulated_clicked_button_name:
                        self._clicked_btn = btn
                        return
            self._clicked_btn = None

        def clickedButton(self):
            return self._clicked_btn

        @staticmethod
        def warning(parent, title, text):
            pass

        @staticmethod
        def information(parent, title, text):
            pass

        @staticmethod
        def critical(parent, title, text):
            pass

    # Override QMessageBox class reference
    monkeypatch.setattr("cif_viewer.viewer_xrd.QMessageBox", MockQMessageBox)

    # 1. Export as Simulated Profile (Curve)
    export_path_profile = tmp_path / "profile_out.csv"
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(export_path_profile), "CSV Files (*.csv)"),
    )

    MockQMessageBox.simulated_clicked_button_name = "Simulated Profile (Curve)"
    MockQMessageBox.simulated_cancel = False

    dialog._export_csv()

    # Verify profile file is written
    assert export_path_profile.exists()
    content = export_path_profile.read_text(encoding="utf-8")
    assert "2-Theta (deg),Intensity" in content
    # Verify we have float pairs
    lines = content.strip().split("\n")
    assert len(lines) > 2
    header = lines[0]
    first_data = lines[1].split(",")
    float(first_data[0])
    float(first_data[1])

    # 2. Export as Bragg Peaks (Sticks)
    export_path_sticks = tmp_path / "sticks_out.csv"
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(export_path_sticks), "CSV Files (*.csv)"),
    )

    MockQMessageBox.simulated_clicked_button_name = "Bragg Peaks (Sticks)"
    MockQMessageBox.simulated_cancel = False

    dialog._export_csv()

    # Verify sticks file is written
    assert export_path_sticks.exists()
    content_sticks = export_path_sticks.read_text(encoding="utf-8")
    assert "2-Theta (deg),Intensity,d-spacing (A),HKL" in content_sticks
    lines_sticks = content_sticks.strip().split("\n")
    assert len(lines_sticks) > 1
    # Check HKL layout: e.g. contains parentheses
    assert "(" in lines_sticks[1] and ")" in lines_sticks[1]

    # 3. Cancel Choice Dialog
    export_path_cancelled = tmp_path / "cancelled_out.csv"
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(export_path_cancelled), "CSV Files (*.csv)"),
    )

    MockQMessageBox.simulated_cancel = True

    dialog._export_csv()
    assert not export_path_cancelled.exists()

    # 4. Cancel File Dialog (Choose Profile but cancel file dialog)
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getSaveFileName", lambda *args, **kwargs: ("", "")
    )
    MockQMessageBox.simulated_clicked_button_name = "Simulated Profile (Curve)"
    MockQMessageBox.simulated_cancel = False

    # This shouldn't raise any exception, and should just return
    dialog._export_csv()


def test_cif_viewer_widget_camera_reset_flag_on_toggles(qtbot):
    from cif_viewer.viewer import CifViewerWidget

    context = StubContext()
    widget = CifViewerWidget(context=context)
    qtbot.addWidget(widget)

    # Initially, _reset_camera_on_next_render is True (from __init__)
    assert widget._reset_camera_on_next_render is True

    # Clear it
    widget._reset_camera_on_next_render = False

    # 1. Toggle view mode
    widget.radio_pack.setChecked(True)
    assert widget._reset_camera_on_next_render is True

    widget._reset_camera_on_next_render = False
    widget.radio_mol.setChecked(True)
    assert widget._reset_camera_on_next_render is True

    widget._reset_camera_on_next_render = False

    # 2. Toggle disorder
    widget._on_disorder_changed()
    assert widget._reset_camera_on_next_render is True

    widget._reset_camera_on_next_render = False

    # 3. Reset defaults
    widget.reset_to_defaults()
    assert widget._reset_camera_on_next_render is True


def test_cif_viewer_auto_determine_bond_order_based_on_atom_count(qtbot, monkeypatch):
    from cif_viewer.viewer import CifViewerWidget
    import numpy as np

    context = StubContext()
    widget = CifViewerWidget(context=context)
    qtbot.addWidget(widget)

    # Mock plotter so _render_now doesn't return early
    class FakePlotter:
        def clear(self):
            pass

        def render(self):
            pass

        def reset_camera(self):
            pass

    context.plotter = FakePlotter()

    # Track arguments passed to render_atoms_to_rdkit_mol
    captured_kwargs = {}
    from rdkit import Chem

    def mock_render_atoms_to_rdkit_mol(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return Chem.Mol()

    monkeypatch.setattr(
        "cif_viewer.viewer.render_atoms_to_rdkit_mol",
        mock_render_atoms_to_rdkit_mol,
    )
    monkeypatch.setattr(widget, "_draw_with_moleditpy", lambda *args, **kwargs: None)

    class DummyAtom:
        def __init__(self):
            self.element = "C"
            self.position = np.array([0.0, 0.0, 0.0])
            self.disorder_group = None
            self.disorder_key = None

    import dataclasses

    @dataclasses.dataclass
    class DummyStructure:
        atoms: list
        asymmetric_atoms: list
        lattice: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(3))

    widget.radio_mol.setChecked(True)

    # Case 1: determine_bond_order unchecked -> determine_bond_order should be False even for 100 atoms
    widget.determine_bond_order.setChecked(False)
    monkeypatch.setattr(
        "cif_viewer.parser.grow_molecules",
        lambda *args, **kwargs: ([DummyAtom() for _ in range(100)], []),
    )
    widget.structure = DummyStructure(
        atoms=[DummyAtom() for _ in range(100)],
        asymmetric_atoms=[DummyAtom() for _ in range(100)],
    )
    widget._render_now()
    assert captured_kwargs.get("determine_bond_order") is False

    # Case 2: determine_bond_order checked and <= 300 atoms (e.g. 100 atoms) -> determine_bond_order should be True
    captured_kwargs.clear()
    widget.determine_bond_order.setChecked(True)
    widget._render_now()
    assert captured_kwargs.get("determine_bond_order") is True
    assert "skipped" not in widget.summary_label.text()

    # Case 3: determine_bond_order checked and > 300 atoms (e.g. 301 atoms) -> determine_bond_order should be False and warning shown
    captured_kwargs.clear()
    monkeypatch.setattr(
        "cif_viewer.parser.grow_molecules",
        lambda *args, **kwargs: ([DummyAtom() for _ in range(301)], []),
    )
    widget.structure = DummyStructure(
        atoms=[DummyAtom() for _ in range(301)],
        asymmetric_atoms=[DummyAtom() for _ in range(301)],
    )
    widget._render_now()
    assert captured_kwargs.get("determine_bond_order") is False
    # Checkbox itself should NOT be disabled or unchecked
    assert widget.determine_bond_order.isChecked() is True
    assert widget.determine_bond_order.isEnabled() is True
    assert "skipped: over 300 atoms limit" in widget.summary_label.text()
