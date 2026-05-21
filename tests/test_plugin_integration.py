import sys
import types

from cif_viewer import initialize


class StubContext:
    def __init__(self):
        self.menu_actions = []
        self.file_openers = []
        self.drop_handlers = []
        self.reset_handlers = []
        self.windows = {}
        self.main_window = StubMainWindow()
        self.styles = {}

    @property
    def plotter(self):
        return None

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


class _FakeDockWidget:
    def __init__(self, title, parent):
        self.title = title
        self.parent = parent
        self._widget = None
        self.shown = False
        self.raised = False

    def setAllowedAreas(self, areas):
        self.allowed_areas = areas

    def setWidget(self, widget):
        self._widget = widget

    def widget(self):
        return self._widget

    def show(self):
        self.shown = True

    def raise_(self):
        self.raised = True


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
        def redF(self): return 0.5
        def greenF(self): return 0.5
        def blueF(self): return 0.5

    qtcore.Qt = Qt
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
            self.u_cart = np.array([
                [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],  # ADP data
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # No ADP
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]   # No ADP
            ])
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
            self.last_rendered_atoms = [
                MockAtom("O1", "O", 0),
                MockAtom("H1", "H", 1),
                MockAtom("C1", "C", 2)
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

    mw = StubMainWindow()
    mw.plotter = MockPlotter()
    mw.init_manager = MockInitManager()
    
    widget = MockWidget()
    context.register_window("cif_viewer_panel", _FakeDockWidget("CIF Viewer", None))
    context.get_window("cif_viewer_panel").setWidget(widget)

    # Setup RDKit Molecule with exactly 3 atoms
    editable_mol = Chem.EditableMol(Chem.Mol())
    editable_mol.AddAtom(Chem.Atom(8)) # Oxygen (index 0)
    editable_mol.AddAtom(Chem.Atom(1)) # Hydrogen (index 1)
    editable_mol.AddAtom(Chem.Atom(6)) # Carbon (index 2)
    mol = editable_mol.GetMol()

    # Add coordinates conformer
    conformer = Chem.Conformer(3)
    conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conformer.SetAtomPosition(1, (1.0, 0.0, 0.0))
    conformer.SetAtomPosition(2, (0.0, 1.0, 0.0))
    mol.AddConformer(conformer)

    # Call the style handler
    draw_callback(mw, mol)

    # Verify rendering results
    assert mw.plotter.cleared
    # We should have meshes added
    assert len(mw.plotter.meshes) > 0

    # Let's inspect the opacity and properties of the added meshes
    ellipsoid_mesh = None
    h_mesh = None
    c_mesh = None
    for mesh, kwargs in mw.plotter.meshes:
        if kwargs.get("name") == "cif_viewer_ellipsoid_0":
            ellipsoid_mesh = (mesh, kwargs)
        elif kwargs.get("name") == "cif_viewer_h_atoms":
            h_mesh = (mesh, kwargs)
        elif kwargs.get("name") == "cif_viewer_fallback_atoms":
            c_mesh = (mesh, kwargs)

    assert ellipsoid_mesh is not None
    assert ellipsoid_mesh[1]["opacity"] == 1.0

    assert h_mesh is not None
    assert h_mesh[1]["opacity"] == 1.0

    assert c_mesh is not None
    assert c_mesh[1]["opacity"] == 1.0
