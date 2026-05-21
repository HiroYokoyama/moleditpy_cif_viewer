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
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)

from .parser import (
    CifStructure,
    celleditpy_cell_axis_segments,
    expand_supercell,
    parse_cif_file,
    parse_cif_file_pymatgen,
)
from .rdkit_bridge import render_atoms_to_rdkit_mol


class CifViewerWidget(QWidget):
    """Right-dock control panel for visualization-only CIF rendering."""

    def __init__(self, parent=None, context=None):
        super().__init__(parent)
        self.context = context
        self.all_structures = []
        self.structure: Optional[CifStructure] = None
        self.current_path: Optional[str] = None
        self.overlay_actor_names = []
        self._build_ui()
        self.load_settings()

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

        self.structure_table = QTableWidget()
        self.structure_table.setColumnCount(2)
        self.structure_table.setHorizontalHeaderLabels(["Structure", "Atoms"])
        self.structure_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.structure_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.structure_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.structure_table.setFixedHeight(120)
        self.structure_table.itemSelectionChanged.connect(self._structure_selected)
        self.structure_table.setVisible(False)
        layout.addWidget(self.structure_table)

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
        self.show_ellipsoids = QCheckBox("Ellipsoids")
        self.show_ellipsoids.setChecked(True)
        for checkbox in (self.show_bonds, self.keep_connected, self.show_cell, self.show_axes, self.show_ellipsoids):
            checkbox.toggled.connect(self.render)
            checkbox.toggled.connect(self.save_settings)
            options_row.addWidget(checkbox)
            
        self.probability_combo = QComboBox()
        self.probability_combo.addItems(["50% (1.54)", "90% (2.50)", "95% (2.79)", "99% (3.40)"])
        self.probability_combo.setCurrentText("50% (1.54)")
        self.probability_combo.currentTextChanged.connect(self.render)
        self.probability_combo.currentTextChanged.connect(self.save_settings)
        options_row.addWidget(self.probability_combo)
        layout.addWidget(options)

        axis_group = QGroupBox("Cell Axis")
        axis_form = QFormLayout(axis_group)
        self.axis_width = QSpinBox()
        self.axis_width.setRange(1, 12)
        self.axis_width.setValue(5)
        self.axis_width.valueChanged.connect(self.render)
        self.axis_width.valueChanged.connect(self.save_settings)
        self.axis_font = QComboBox()
        self.axis_font.addItems(["arial", "courier", "times"])
        self.axis_font.currentTextChanged.connect(self.render)
        self.axis_font.currentTextChanged.connect(self.save_settings)
        self.axis_font_size = QSpinBox()
        self.axis_font_size.setRange(8, 48)
        self.axis_font_size.setValue(20)
        self.axis_font_size.valueChanged.connect(self.render)
        self.axis_font_size.valueChanged.connect(self.save_settings)
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

    def _structure_selected(self):
        selected_ranges = self.structure_table.selectedRanges()
        if not selected_ranges or not self.all_structures:
            return
        row = selected_ranges[0].topRow()
        if 0 <= row < len(self.all_structures):
            self.structure = self.all_structures[row]
            self.render()

    def _settings_path(self):
        return os.path.join(os.path.dirname(__file__), "settings.json")

    def load_settings(self):
        path = self._settings_path()
        if not os.path.exists(path):
            return
        import json
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.show_bonds.blockSignals(True)
            self.keep_connected.blockSignals(True)
            self.show_cell.blockSignals(True)
            self.show_axes.blockSignals(True)
            self.show_ellipsoids.blockSignals(True)
            self.probability_combo.blockSignals(True)
            self.axis_width.blockSignals(True)
            self.axis_font.blockSignals(True)
            self.axis_font_size.blockSignals(True)
            
            if "show_bonds" in data:
                self.show_bonds.setChecked(bool(data["show_bonds"]))
            if "keep_connected" in data:
                self.keep_connected.setChecked(bool(data["keep_connected"]))
            if "show_cell" in data:
                self.show_cell.setChecked(bool(data["show_cell"]))
            if "show_axes" in data:
                self.show_axes.setChecked(bool(data["show_axes"]))
            if "show_ellipsoids" in data:
                self.show_ellipsoids.setChecked(bool(data["show_ellipsoids"]))
            if "probability" in data:
                idx = self.probability_combo.findText(str(data["probability"]))
                if idx >= 0:
                    self.probability_combo.setCurrentIndex(idx)
            if "axis_width" in data:
                self.axis_width.setValue(int(data["axis_width"]))
            if "axis_font" in data:
                idx = self.axis_font.findText(str(data["axis_font"]))
                if idx >= 0:
                    self.axis_font.setCurrentIndex(idx)
            if "axis_font_size" in data:
                self.axis_font_size.setValue(int(data["axis_font_size"]))
        except Exception:
            pass
        finally:
            self.show_bonds.blockSignals(False)
            self.keep_connected.blockSignals(False)
            self.show_cell.blockSignals(False)
            self.show_axes.blockSignals(False)
            self.show_ellipsoids.blockSignals(False)
            self.probability_combo.blockSignals(False)
            self.axis_width.blockSignals(False)
            self.axis_font.blockSignals(False)
            self.axis_font_size.blockSignals(False)

    def save_settings(self, *args):
        path = self._settings_path()
        import json
        data = {
            "show_bonds": self.show_bonds.isChecked(),
            "keep_connected": self.keep_connected.isChecked(),
            "show_cell": self.show_cell.isChecked(),
            "show_axes": self.show_axes.isChecked(),
            "show_ellipsoids": self.show_ellipsoids.isChecked(),
            "probability": self.probability_combo.currentText(),
            "axis_width": self.axis_width.value(),
            "axis_font": self.axis_font.currentText(),
            "axis_font_size": self.axis_font_size.value(),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception:
            pass

    def load_cif(self, path: str):
        try:
            self.all_structures = parse_cif_file_pymatgen(path)
        except Exception:
            try:
                self.all_structures = [parse_cif_file(path)]
            except Exception as exc:
                QMessageBox.critical(self, "CIF Viewer", f"Could not read CIF file:\n{exc}")
                return
                
        if not self.all_structures:
            QMessageBox.critical(self, "CIF Viewer", "No valid structures found in CIF file.")
            return

        self.current_path = path
        self.file_label.setText(os.path.basename(path))
        
        self.structure_table.blockSignals(True)
        self.structure_table.setRowCount(len(self.all_structures))
        for i, struct in enumerate(self.all_structures):
            name_item = QTableWidgetItem(struct.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            atoms_item = QTableWidgetItem(str(len(struct.atoms)))
            atoms_item.setFlags(atoms_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.structure_table.setItem(i, 0, name_item)
            self.structure_table.setItem(i, 1, atoms_item)
            
        if len(self.all_structures) > 1:
            self.structure_table.setVisible(True)
            self.structure_table.selectRow(0)
            self.structure = self.all_structures[0]
        else:
            self.structure_table.setVisible(False)
            self.structure = self.all_structures[0]
            
        self.structure_table.blockSignals(False)
        self._enter_viewer_mode()
        
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self.render)

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

        if self.show_ellipsoids.isChecked() and self.structure.u_cart is not None:
            self._draw_ellipsoid_overlay(plotter, atoms)

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

    def _draw_ellipsoid_overlay(self, plotter, atoms):
        import pyvista as pv
        import re
        
        ELEMENT_COLORS = {
            "H": "#FFFFFF", "He": "#D9FFFF", "Li": "#CC80FF", "Be": "#C2FF00",
            "B": "#FFB5B5", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
            "F": "#90E050", "Ne": "#B3E3F5", "Na": "#AB5CF2", "Mg": "#8AFF00",
            "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#E6C800",
            "Cl": "#1FF01F", "Ar": "#80D1E3", "K": "#8F40D4", "Ca": "#3DFF00",
            "Fe": "#E06633", "Cu": "#C88033", "Zn": "#7D80B0",
        }
        
        prob_text = self.probability_combo.currentText()
        match = re.search(r"\(([\d\.]+)\)", prob_text)
        scale_factor = float(match.group(1)) if match else 1.54

        for index, atom in enumerate(atoms):
            base_idx = atom.base_index
            if base_idx >= len(self.structure.u_cart):
                continue
            cov = self.structure.u_cart[base_idx]
            if cov is None or np.allclose(cov, 0.0):
                continue
            
            try:
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                eigenvalues = np.maximum(eigenvalues, 1e-5)
                radii = np.sqrt(eigenvalues) * scale_factor
                
                ellipsoid = pv.ParametricEllipsoid(
                    xradius=radii[0],
                    yradius=radii[1],
                    zradius=radii[2],
                    u_res=20,
                    v_res=20
                )
                
                rotation_matrix = np.eye(4)
                rotation_matrix[:3, :3] = eigenvectors
                ellipsoid.transform(rotation_matrix, inplace=True)
                ellipsoid.translate(atom.position, inplace=True)
                
                color = ELEMENT_COLORS.get(atom.element, "#00FFFF")
                name = f"cif_viewer_ellipsoid_{index}"
                
                plotter.add_mesh(
                    ellipsoid,
                    color=color,
                    opacity=1.0,
                    name=name,
                    style='surface'
                )
                self.overlay_actor_names.append(name)
                
                for i in range(3):
                    vec = eigenvectors[:, i] * radii[i]
                    start_pt = atom.position - vec
                    end_pt = atom.position + vec
                    line_name = f"cif_viewer_ellipsoid_axis_{index}_{i}"
                    plotter.add_lines(
                        np.array([start_pt, end_pt]),
                        color="black",
                        width=2,
                        name=line_name
                    )
                    self.overlay_actor_names.append(line_name)
                    
            except Exception as exc:
                print(f"Error drawing ellipsoid for {atom.label}: {exc}")

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
