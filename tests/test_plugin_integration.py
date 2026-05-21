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
    pyqt6 = types.ModuleType("PyQt6")

    class DockWidgetArea:
        LeftDockWidgetArea = 1
        RightDockWidgetArea = 2

    class Qt:
        pass
    Qt.DockWidgetArea = DockWidgetArea

    qtcore.Qt = Qt
    qtwidgets.QDockWidget = _FakeDockWidget
    monkeypatch.setitem(sys.modules, "PyQt6", pyqt6)
    monkeypatch.setitem(sys.modules, "PyQt6.QtCore", qtcore)
    monkeypatch.setitem(sys.modules, "PyQt6.QtWidgets", qtwidgets)
