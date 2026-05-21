PLUGIN_NAME = "CIF Viewer"
PLUGIN_VERSION = "0.1.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "Visualization-only CIF crystal structure viewer with unit-cell and "
    "supercell rendering for MoleditPy."
)
PLUGIN_DEPENDENCIES = ["ase", "numpy", "PyQt6", "pyvista", "rdkit"]

WINDOW_ID = "cif_viewer_panel"


def initialize(context):
    """Register the CIF viewer with MoleditPy."""

    def show_panel(file_path=None):
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QDockWidget

        from .viewer import CifViewerWidget

        main_window = context.get_main_window()
        dock = context.get_window(WINDOW_ID) if hasattr(context, "get_window") else None
        if dock is None:
            dock = QDockWidget("CIF Viewer", main_window)
            dock.setAllowedAreas(
                Qt.DockWidgetArea.LeftDockWidgetArea
                | Qt.DockWidgetArea.RightDockWidgetArea
            )
            widget = CifViewerWidget(dock, context)
            dock.setWidget(widget)
            if main_window is not None and hasattr(main_window, "addDockWidget"):
                main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
            if hasattr(context, "register_window"):
                context.register_window(WINDOW_ID, dock)
        else:
            widget = dock.widget()

        dock.show()
        dock.raise_()
        if file_path:
            widget.load_cif(file_path)

    def open_from_menu():
        show_panel()

    def open_file(file_path):
        try:
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(0, lambda path=file_path: show_panel(path))
        except Exception:
            show_panel(file_path)
        return True

    def handle_drop(file_path):
        if str(file_path).lower().endswith(".cif"):
            open_file(file_path)
            return True
        return False

    def handle_reset():
        dock = context.get_window(WINDOW_ID) if hasattr(context, "get_window") else None
        if dock is not None and dock.widget() is not None:
            dock.widget().clear_view()

    context.add_menu_action("View/CIF Viewer Panel", open_from_menu)

    if hasattr(context, "register_file_opener"):
        context.register_file_opener(".cif", open_file, priority=20)
    if hasattr(context, "register_drop_handler"):
        context.register_drop_handler(handle_drop, priority=20)
    if hasattr(context, "register_document_reset_handler"):
        context.register_document_reset_handler(handle_reset)
