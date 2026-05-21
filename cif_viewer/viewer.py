from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .parser import (
    CifStructure,
    celleditpy_cell_axis_segments,
    expand_supercell,
    parse_cif_file,
)
from .rdkit_bridge import render_atoms_to_rdkit_mol


class CifViewerWidget(QWidget):
    """Right-dock control panel for visualization-only CIF rendering."""

    def __init__(self, parent=None, context=None):
        super().__init__(parent)
        self.context = context
        self.structure: Optional[CifStructure] = None
        self.current_path: Optional[str] = None
        self.overlay_actor_names = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        file_row = QHBoxLayout()
        self.file_label = QLabel("No CIF loaded")
        self.file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        open_button = QPushButton("Open CIF...")
        open_button.clicked.connect(self._choose_file)
        file_row.addWidget(self.file_label, 1)
        file_row.addWidget(open_button)
        layout.addLayout(file_row)

        supercell_group = QGroupBox("Supercell")
        form = QFormLayout(supercell_group)
        self.repeat_a = self._repeat_spin()
        self.repeat_b = self._repeat_spin()
        self.repeat_c = self._repeat_spin()
        form.addRow("a repeats", self.repeat_a)
        form.addRow("b repeats", self.repeat_b)
        form.addRow("c repeats", self.repeat_c)
        reset_supercell_button = QPushButton("Reset Supercell")
        reset_supercell_button.clicked.connect(self.reset_supercell)
        form.addRow(reset_supercell_button)
        layout.addWidget(supercell_group)

        options = QWidget()
        options_row = QHBoxLayout(options)
        options_row.setContentsMargins(0, 0, 0, 0)
        self.show_bonds = QCheckBox("Bonds")
        self.show_bonds.setChecked(True)
        self.keep_connected = QCheckBox("Keep connected")
        self.keep_connected.setChecked(True)
        self.show_cell = QCheckBox("Cell")
        self.show_cell.setChecked(True)
        self.show_axes = QCheckBox("a/b/c axes")
        self.show_axes.setChecked(True)
        for checkbox in (self.show_bonds, self.keep_connected, self.show_cell, self.show_axes):
            checkbox.toggled.connect(self.render)
            options_row.addWidget(checkbox)
        layout.addWidget(options)

        axis_group = QGroupBox("Cell Axis")
        axis_form = QFormLayout(axis_group)
        self.axis_width = QSpinBox()
        self.axis_width.setRange(1, 12)
        self.axis_width.setValue(5)
        self.axis_width.valueChanged.connect(self.render)
        self.axis_font = QComboBox()
        self.axis_font.addItems(["arial", "courier", "times"])
        self.axis_font.currentTextChanged.connect(self.render)
        self.axis_font_size = QSpinBox()
        self.axis_font_size.setRange(8, 48)
        self.axis_font_size.setValue(20)
        self.axis_font_size.valueChanged.connect(self.render)
        axis_form.addRow("Axis width", self.axis_width)
        axis_form.addRow("Font", self.axis_font)
        axis_form.addRow("Font size", self.axis_font_size)
        layout.addWidget(axis_group)

        self.summary_label = QLabel(
            "Load a CIF file to visualize the completed unit cell and supercell."
        )
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        button_row = QHBoxLayout()
        render_button = QPushButton("Render")
        render_button.clicked.connect(self.render)
        clear_button = QPushButton("Clear Overlay")
        clear_button.clicked.connect(self.clear_view)
        button_row.addStretch(1)
        button_row.addWidget(render_button)
        button_row.addWidget(clear_button)
        layout.addLayout(button_row)

    def _repeat_spin(self):
        spin = QSpinBox()
        spin.setRange(1, 8)
        spin.setValue(1)
        spin.valueChanged.connect(self.render)
        return spin

    def reset_supercell(self):
        for spin in (self.repeat_a, self.repeat_b, self.repeat_c):
            spin.blockSignals(True)
            spin.setValue(1)
            spin.blockSignals(False)
        self.render()

    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open CIF",
            "",
            "Crystallographic Information Files (*.cif);;All Files (*)",
        )
        if path:
            self.load_cif(path)

    def load_cif(self, path: str):
        try:
            self.structure = parse_cif_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "CIF Viewer", f"Could not read CIF file:\n{exc}")
            return

        self.current_path = path
        self.file_label.setText(os.path.basename(path))
        self._enter_viewer_mode()
        self.render()

    def clear_view(self):
        plotter = self._plotter()
        if plotter is None:
            return
        for name in self.overlay_actor_names:
            try:
                plotter.remove_actor(name)
            except Exception:
                pass
        self.overlay_actor_names.clear()
        try:
            plotter.render()
        except Exception:
            pass

    def render(self):
        if self.structure is None:
            return

        plotter = self._plotter()
        if plotter is None:
            QMessageBox.warning(self, "CIF Viewer", "MoleditPy 3D plotter is not available.")
            return

        self.clear_view()
        repeats = (self.repeat_a.value(), self.repeat_b.value(), self.repeat_c.value())
        atoms, bonds = expand_supercell(
            self.structure,
            repeats,
            keep_connected=self.keep_connected.isChecked(),
        )
        mol_bonds = bonds if self.show_bonds.isChecked() else []
        try:
            mol = render_atoms_to_rdkit_mol(atoms, mol_bonds)
        except Exception as exc:
            QMessageBox.critical(self, "CIF Viewer", f"Could not build RDKit view:\n{exc}")
            return

        self._draw_with_moleditpy(mol)
        plotter = self._plotter()
        if plotter is None:
            return

        if self.show_cell.isChecked():
            self._draw_cell_overlay(plotter, repeats)

        try:
            plotter.reset_camera()
            if hasattr(plotter, "camera"):
                plotter.camera.focal_point = self._cell_center(repeats)
            plotter.render()
        except Exception:
            pass

        self.summary_label.setText(
            f"{len(self.structure.atoms)} completed unit-cell atoms, "
            f"{len(atoms)} rendered atoms, {len(bonds)} inferred bonds, "
            f"supercell {repeats[0]} x {repeats[1]} x {repeats[2]}."
        )
        if self.context and hasattr(self.context, "show_status_message"):
            self.context.show_status_message("CIF Viewer rendered with MoleditPy 3D style.", 3000)

    def _draw_with_moleditpy(self, mol):
        if self.context is not None and hasattr(self.context, "draw_molecule_3d"):
            try:
                self.context.current_mol = mol
            except Exception:
                self.context.draw_molecule_3d(mol)
            return
        main_window = self._main_window()
        if main_window is not None and hasattr(main_window, "draw_molecule_3d"):
            if hasattr(main_window, "view_3d_manager"):
                try:
                    main_window.view_3d_manager.current_mol = mol
                except Exception:
                    pass
            main_window.draw_molecule_3d(mol)

    def _draw_cell_overlay(self, plotter, repeats):
        lattice = np.asarray(self.structure.lattice, dtype=float)
        for index, (start, end, color, label) in enumerate(
            celleditpy_cell_axis_segments(lattice)
        ):
            name = f"cif_viewer_cell_line_{index}"
            is_axis = bool(label and self.show_axes.isChecked())
            width = self.axis_width.value() if is_axis else max(1, self.axis_width.value() - 2)
            line_color = color if is_axis else "white"
            plotter.add_lines(np.array([start, end]), color=line_color, width=width, name=name)
            self.overlay_actor_names.append(name)
            if is_axis:
                label_name = f"cif_viewer_cell_axis_label_{label}"
                plotter.add_point_labels(
                    [end],
                    [label],
                    point_size=0,
                    font_size=self.axis_font_size.value(),
                    text_color=color,
                    font_family=self.axis_font.currentText(),
                    bold=True,
                    always_visible=True,
                    shape=None,
                    name=label_name,
                )
                self.overlay_actor_names.append(label_name)

    def _cell_center(self, repeats):
        lattice = np.asarray(self.structure.lattice, dtype=float)
        return np.sum(lattice, axis=0) / 2.0

    def _enter_viewer_mode(self):
        main_window = self._main_window()
        if main_window is None:
            return
        ui_manager = getattr(main_window, "ui_manager", None)
        if ui_manager is not None and hasattr(ui_manager, "_enter_3d_viewer_ui_mode"):
            try:
                ui_manager._enter_3d_viewer_ui_mode()
                return
            except Exception:
                pass
        for attr in ("splitter", "main_splitter", "central_splitter"):
            splitter = getattr(main_window, attr, None)
            if splitter is not None and hasattr(splitter, "setSizes"):
                try:
                    splitter.setSizes([0, 1])
                    return
                except Exception:
                    pass

    def _plotter(self):
        if self.context is not None:
            plotter = getattr(self.context, "plotter", None)
            if plotter is not None:
                return plotter
        main_window = self._main_window()
        if main_window is not None and hasattr(main_window, "plotter"):
            return main_window.plotter
        if main_window is not None and hasattr(main_window, "view_3d_manager"):
            return getattr(main_window.view_3d_manager, "plotter", None)
        return None

    def _main_window(self):
        if self.context is not None and hasattr(self.context, "get_main_window"):
            return self.context.get_main_window()
        parent = self.parent()
        return parent.parent() if parent is not None and hasattr(parent, "parent") else None


CifViewerDialog = CifViewerWidget
