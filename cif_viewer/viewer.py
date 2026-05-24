from __future__ import annotations

import os
import logging
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
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
    QTabWidget,
    QColorDialog,
    QScrollArea,
    QRadioButton,
    QButtonGroup,
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
        self._reset_camera_on_next_render = True

        from PyQt6.QtCore import QTimer

        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self._render_now)

        self._build_ui()
        self.load_settings()

    def _create_color_button(self, default_hex):
        btn = QPushButton()
        btn.setFixedWidth(50)
        btn.setStyleSheet(
            f"background-color: {default_hex}; border: 1px solid #777; border-radius: 3px;"
        )
        btn.setProperty("color_hex", default_hex)

        def pick_color():
            current_hex = btn.property("color_hex")
            color = QColorDialog.getColor(QColor(current_hex), self, "Select Color")
            if color.isValid():
                hex_name = color.name()
                btn.setStyleSheet(
                    f"background-color: {hex_name}; border: 1px solid #777; border-radius: 3px;"
                )
                btn.setProperty("color_hex", hex_name)
                self.save_settings()
                self.render()

        btn.clicked.connect(pick_color)
        return btn

    def _set_button_color(self, btn, hex_color):
        btn.setProperty("color_hex", hex_color)
        btn.setStyleSheet(
            f"background-color: {hex_color}; border: 1px solid #777; border-radius: 3px;"
        )

    def _export_supercell(self):
        if self.structure is None:
            QMessageBox.warning(self, "Export CIF", "No CIF loaded to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Supercell CIF",
            "",
            "Crystallographic Information Files (*.cif)",
        )
        if not path:
            return

        try:
            repeats = (
                self.repeat_a.value(),
                self.repeat_b.value(),
                self.repeat_c.value(),
            )

            selected_key = None
            if (
                not self.disorder_combo.isHidden()
                and self.disorder_combo.currentIndex() > 0
            ):
                selected_key = self.disorder_combo.currentData()

            from .parser import write_supercell_cif

            write_supercell_cif(
                path,
                self.structure,
                repeats,
                keep_connected=self.keep_connected.isChecked(),
                selected_disorder_key=selected_key,
            )
            QMessageBox.information(
                self, "Export CIF", f"Successfully exported supercell to:\n{path}"
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Export CIF", f"Failed to export supercell:\n{exc}"
            )

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()

        # --- Tab 1: Structure ---
        struct_tab = QWidget()
        struct_layout = QVBoxLayout(struct_tab)

        file_row = QHBoxLayout()
        self.file_label = QLabel("No CIF loaded")
        self.file_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        open_button = QPushButton("Open CIF...")
        open_button.clicked.connect(self._choose_file)
        file_row.addWidget(self.file_label, 1)
        file_row.addWidget(open_button)
        struct_layout.addLayout(file_row)

        self.structure_table = QTableWidget()
        self.structure_table.setColumnCount(2)
        self.structure_table.setHorizontalHeaderLabels(["Structure", "Atoms"])
        self.structure_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.structure_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.structure_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.structure_table.setFixedHeight(120)
        self.structure_table.itemSelectionChanged.connect(self._structure_selected)
        self.structure_table.setVisible(False)
        struct_layout.addWidget(self.structure_table)

        # Disorder selection row
        disorder_row = QHBoxLayout()
        self.disorder_label = QLabel("Disorder Part:")
        self.disorder_combo = QComboBox()
        self.disorder_combo.currentTextChanged.connect(self._on_disorder_changed)
        disorder_row.addWidget(self.disorder_label)
        disorder_row.addWidget(self.disorder_combo, 1)
        self.disorder_label.setVisible(False)
        self.disorder_combo.setVisible(False)
        struct_layout.addLayout(disorder_row)

        # Display mode row
        view_row = QHBoxLayout()
        view_label = QLabel("Display Mode:")
        view_row.addWidget(view_label)

        self.view_mode_group = QButtonGroup(self)
        self.radio_mol = QRadioButton("Whole Molecule")
        self.radio_asym = QRadioButton("Asymmetric Unit")
        self.radio_pack = QRadioButton("Packing")

        self.view_mode_group.addButton(self.radio_mol)
        self.view_mode_group.addButton(self.radio_asym)
        self.view_mode_group.addButton(self.radio_pack)

        self.radio_mol.toggled.connect(self._on_view_mode_changed)
        self.radio_asym.toggled.connect(self._on_view_mode_changed)
        self.radio_pack.toggled.connect(self._on_view_mode_changed)

        view_row.addWidget(self.radio_mol)
        view_row.addWidget(self.radio_asym)
        view_row.addWidget(self.radio_pack)
        struct_layout.addLayout(view_row)

        self.determine_bond_order = QCheckBox("Determine bond order (RDKit)")
        self.determine_bond_order.setChecked(False)
        self.determine_bond_order.toggled.connect(self.render)
        self.determine_bond_order.toggled.connect(self.save_settings)
        struct_layout.addWidget(self.determine_bond_order)

        self.bond_order_disabled_label = QLabel(
            "  ⚠ Disabled: not available when \"All Parts\" is selected."
        )
        self.bond_order_disabled_label.setStyleSheet("color: gray; font-style: italic;")
        self.bond_order_disabled_label.setVisible(False)
        struct_layout.addWidget(self.bond_order_disabled_label)

        max_atoms_row = QHBoxLayout()
        max_atoms_label = QLabel("Max number of atoms:")
        self.max_atoms_spin = QSpinBox()
        self.max_atoms_spin.setRange(10, 5000)
        self.max_atoms_spin.setSingleStep(50)
        self.max_atoms_spin.setValue(300)
        self.max_atoms_spin.setToolTip(
            "Maximum number of atoms to allow for bond order determination."
        )
        self.max_atoms_spin.valueChanged.connect(self.render)
        self.max_atoms_spin.valueChanged.connect(self.save_settings)
        max_atoms_row.addWidget(max_atoms_label)
        max_atoms_row.addWidget(self.max_atoms_spin)
        max_atoms_row.addStretch(1)
        struct_layout.addLayout(max_atoms_row)

        struct_layout.addStretch(1)

        self.summary_label = QLabel(
            "Load a CIF file to visualize the completed unit cell and supercell."
        )
        self.summary_label.setWordWrap(True)
        struct_layout.addWidget(self.summary_label)

        self.export_button = QPushButton("Export Supercell CIF...")
        self.export_button.clicked.connect(self._export_supercell)
        struct_layout.addWidget(self.export_button)

        self.tabs.addTab(struct_tab, "Structure")

        # --- Tab 2: Info (Refinement & Cell Metadata) ---
        info_tab = QScrollArea()
        info_tab.setWidgetResizable(True)

        info_content = QWidget()
        info_content_layout = QVBoxLayout(info_content)
        info_content_layout.setContentsMargins(5, 5, 5, 5)
        info_content_layout.setSpacing(10)

        # 1. Crystal & Unit Cell Group
        group_cell = QGroupBox("Crystal & Unit Cell")
        layout_cell = QFormLayout(group_cell)
        layout_cell.setContentsMargins(5, 5, 5, 5)
        layout_cell.setSpacing(5)

        self.info_formula = QLabel("N/A")
        self.info_space_group = QLabel("N/A")
        self.info_space_group_number = QLabel("N/A")
        self.info_crystal_system = QLabel("N/A")
        self.info_cell_a = QLabel("N/A")
        self.info_cell_b = QLabel("N/A")
        self.info_cell_c = QLabel("N/A")
        self.info_cell_alpha = QLabel("N/A")
        self.info_cell_beta = QLabel("N/A")
        self.info_cell_gamma = QLabel("N/A")
        self.info_volume = QLabel("N/A")
        self.info_z = QLabel("N/A")
        self.info_z_prime = QLabel("N/A")
        self.info_density = QLabel("N/A")
        self.info_mu = QLabel("N/A")
        self.info_f000 = QLabel("N/A")

        layout_cell.addRow("Formula:", self.info_formula)
        layout_cell.addRow("Space Group:", self.info_space_group)
        layout_cell.addRow("Space Group No.:", self.info_space_group_number)
        layout_cell.addRow("Crystal System:", self.info_crystal_system)
        layout_cell.addRow("a (Å):", self.info_cell_a)
        layout_cell.addRow("b (Å):", self.info_cell_b)
        layout_cell.addRow("c (Å):", self.info_cell_c)
        layout_cell.addRow("α (°):", self.info_cell_alpha)
        layout_cell.addRow("β (°):", self.info_cell_beta)
        layout_cell.addRow("γ (°):", self.info_cell_gamma)
        layout_cell.addRow("Volume (Å³):", self.info_volume)
        layout_cell.addRow("Z:", self.info_z)
        layout_cell.addRow("Z':", self.info_z_prime)
        layout_cell.addRow("Density (calc. g/cm³):", self.info_density)
        layout_cell.addRow("μ (mm⁻¹):", self.info_mu)
        layout_cell.addRow("F(000):", self.info_f000)

        # 2. Data Collection Group
        group_data = QGroupBox("Data Collection")
        layout_data = QFormLayout(group_data)
        layout_data.setContentsMargins(5, 5, 5, 5)
        layout_data.setSpacing(5)

        self.info_temp = QLabel("N/A")
        self.info_wavelength = QLabel("N/A")
        self.info_crystal_size = QLabel("N/A")
        self.info_theta_range = QLabel("N/A")
        self.info_hkl_ranges = QLabel("N/A")
        self.info_reflns_collected = QLabel("N/A")
        self.info_reflns_unique = QLabel("N/A")
        self.info_r_int = QLabel("N/A")
        self.info_completeness = QLabel("N/A")

        layout_data.addRow("Temperature (K):", self.info_temp)
        layout_data.addRow("Wavelength (Å):", self.info_wavelength)
        layout_data.addRow("Crystal Size (mm):", self.info_crystal_size)
        layout_data.addRow("θ Range (°):", self.info_theta_range)
        layout_data.addRow("Index Ranges:", self.info_hkl_ranges)
        layout_data.addRow("Reflns Collected:", self.info_reflns_collected)
        layout_data.addRow("Reflns Unique:", self.info_reflns_unique)
        layout_data.addRow("R(int):", self.info_r_int)
        layout_data.addRow("Completeness:", self.info_completeness)

        # 3. Refinement Group
        group_refine = QGroupBox("Refinement")
        layout_refine = QFormLayout(group_refine)
        layout_refine.setContentsMargins(5, 5, 5, 5)
        layout_refine.setSpacing(5)

        self.info_refinement_method = QLabel("N/A")
        self.info_num_reflns = QLabel("N/A")
        self.info_num_params = QLabel("N/A")
        self.info_num_restraints = QLabel("N/A")
        self.info_goof = QLabel("N/A")
        self.info_r1 = QLabel("N/A")
        self.info_wr2 = QLabel("N/A")
        self.info_r1_all = QLabel("N/A")
        self.info_wr2_all = QLabel("N/A")
        self.info_max_shift = QLabel("N/A")
        self.info_flack = QLabel("N/A")
        self.info_diff_peak_hole = QLabel("N/A")

        layout_refine.addRow("Refinement Method:", self.info_refinement_method)
        layout_refine.addRow("No. Reflns:", self.info_num_reflns)
        layout_refine.addRow("No. Parameters:", self.info_num_params)
        layout_refine.addRow("No. Restraints:", self.info_num_restraints)
        layout_refine.addRow("GooF (S):", self.info_goof)
        layout_refine.addRow("R1 [I > 2σ(I)]:", self.info_r1)
        layout_refine.addRow("wR2 [I > 2σ(I)]:", self.info_wr2)
        layout_refine.addRow("R1 (all data):", self.info_r1_all)
        layout_refine.addRow("wR2 (all data):", self.info_wr2_all)
        layout_refine.addRow("Max Shift/s.u.:", self.info_max_shift)
        layout_refine.addRow("Flack Parameter:", self.info_flack)
        layout_refine.addRow("Diff Peak/Hole (e/Å³):", self.info_diff_peak_hole)

        # Enable text selection
        all_info_labels = [
            self.info_formula,
            self.info_space_group,
            self.info_space_group_number,
            self.info_crystal_system,
            self.info_cell_a,
            self.info_cell_b,
            self.info_cell_c,
            self.info_cell_alpha,
            self.info_cell_beta,
            self.info_cell_gamma,
            self.info_volume,
            self.info_z,
            self.info_z_prime,
            self.info_density,
            self.info_mu,
            self.info_f000,
            self.info_temp,
            self.info_wavelength,
            self.info_crystal_size,
            self.info_theta_range,
            self.info_hkl_ranges,
            self.info_reflns_collected,
            self.info_reflns_unique,
            self.info_r_int,
            self.info_completeness,
            self.info_refinement_method,
            self.info_num_reflns,
            self.info_num_params,
            self.info_num_restraints,
            self.info_goof,
            self.info_r1,
            self.info_wr2,
            self.info_r1_all,
            self.info_wr2_all,
            self.info_max_shift,
            self.info_flack,
            self.info_diff_peak_hole,
        ]
        for lbl in all_info_labels:
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        info_content_layout.addWidget(group_cell)
        info_content_layout.addWidget(group_data)
        info_content_layout.addWidget(group_refine)

        self.simulate_xrd_btn = QPushButton("Simulate Powder Pattern (XRD)...")
        self.simulate_xrd_btn.clicked.connect(self._simulate_powder_pattern)
        self.simulate_xrd_btn.setEnabled(False)
        info_content_layout.addWidget(self.simulate_xrd_btn)

        info_tab.setWidget(info_content)
        self.tabs.addTab(info_tab, "Info")

        # --- Tab 2: Supercell ---
        supercell_tab = QWidget()
        supercell_layout = QFormLayout(supercell_tab)
        self.repeat_a = self._repeat_spin()
        self.repeat_b = self._repeat_spin()
        self.repeat_c = self._repeat_spin()
        supercell_layout.addRow("a repeats", self.repeat_a)
        supercell_layout.addRow("b repeats", self.repeat_b)
        supercell_layout.addRow("c repeats", self.repeat_c)

        self.keep_connected = QCheckBox("Keep molecules connected")
        self.keep_connected.setChecked(True)
        self.keep_connected.toggled.connect(self.render)
        self.keep_connected.toggled.connect(self.save_settings)
        supercell_layout.addRow(self.keep_connected)

        self.show_bonds = QCheckBox("Show bonds")
        self.show_bonds.setChecked(True)
        self.show_bonds.toggled.connect(self.render)
        self.show_bonds.toggled.connect(self.save_settings)
        supercell_layout.addRow(self.show_bonds)

        self.show_hydrogens = QCheckBox("Show hydrogen atoms")
        self.show_hydrogens.setChecked(True)
        self.show_hydrogens.toggled.connect(self.render)
        self.show_hydrogens.toggled.connect(self.save_settings)
        supercell_layout.addRow(self.show_hydrogens)

        reset_supercell_button = QPushButton("Reset Supercell")
        reset_supercell_button.clicked.connect(self.reset_supercell)
        supercell_layout.addRow(reset_supercell_button)

        preset_row = QHBoxLayout()
        btn_222 = QPushButton("2x2x2")
        btn_222.clicked.connect(lambda: self.set_supercell_preset(2))
        btn_333 = QPushButton("3x3x3")
        btn_333.clicked.connect(lambda: self.set_supercell_preset(3))
        preset_row.addWidget(btn_222)
        preset_row.addWidget(btn_333)
        supercell_layout.addRow("Presets:", preset_row)

        self.tabs.addTab(supercell_tab, "Supercell")

        # --- Tab 3: Thermal Ellipsoids ---
        ellipsoids_tab = QWidget()
        ellipsoids_layout = QFormLayout(ellipsoids_tab)

        self.show_ellipsoid_rings = QCheckBox("Show circles")
        self.show_ellipsoid_rings.setChecked(True)
        self.show_ellipsoid_rings.toggled.connect(self.render)
        self.show_ellipsoid_rings.toggled.connect(self.save_settings)
        ellipsoids_layout.addRow(self.show_ellipsoid_rings)

        self.color_ellipsoid_rings = self._create_color_button("#000000")
        ellipsoids_layout.addRow("Circle color", self.color_ellipsoid_rings)

        self.ellipsoid_ring_width = QSpinBox()
        self.ellipsoid_ring_width.setRange(1, 10)
        self.ellipsoid_ring_width.setValue(2)
        self.ellipsoid_ring_width.valueChanged.connect(self.render)
        self.ellipsoid_ring_width.valueChanged.connect(self.save_settings)
        ellipsoids_layout.addRow("Circle width", self.ellipsoid_ring_width)

        self.fix_h_size = QCheckBox("Fix hydrogen atom size")
        self.fix_h_size.setChecked(True)
        self.fix_h_size.toggled.connect(self.render)
        self.fix_h_size.toggled.connect(self.save_settings)
        ellipsoids_layout.addRow(self.fix_h_size)

        self.probability_spin = QDoubleSpinBox()
        self.probability_spin.setRange(1.0, 99.9)
        self.probability_spin.setSingleStep(1.0)
        self.probability_spin.setValue(50.0)
        self.probability_spin.setDecimals(1)
        self.probability_spin.editingFinished.connect(self.render)
        self.probability_spin.editingFinished.connect(self.save_settings)
        ellipsoids_layout.addRow("Probability (%):", self.probability_spin)

        self.h_scale_spin = QDoubleSpinBox()
        self.h_scale_spin.setRange(1.0, 100.0)
        self.h_scale_spin.setSingleStep(5.0)
        self.h_scale_spin.setValue(20.0)
        self.h_scale_spin.setDecimals(1)
        self.h_scale_spin.editingFinished.connect(self.render)
        self.h_scale_spin.editingFinished.connect(self.save_settings)
        ellipsoids_layout.addRow("H Scale (% VDW):", self.h_scale_spin)

        self.switch_style_btn = QPushButton("Switch to Ellipsoids Style")
        self.switch_style_btn.clicked.connect(self._switch_to_ellipsoids)
        ellipsoids_layout.addRow(self.switch_style_btn)

        self.tabs.addTab(ellipsoids_tab, "Ellipsoids")

        # --- Tab 4: Cell / Axes ---
        cell_axes_tab = QWidget()
        cell_axes_layout = QFormLayout(cell_axes_tab)

        self.show_cell = QCheckBox("Show unit cell")
        self.show_cell.setChecked(True)
        self.show_cell.toggled.connect(self.render)
        self.show_cell.toggled.connect(self.save_settings)
        cell_axes_layout.addRow(self.show_cell)

        self.show_axes = QCheckBox("a/b/c axes")
        self.show_axes.setChecked(True)
        self.show_axes.toggled.connect(self.render)
        self.show_axes.toggled.connect(self.save_settings)
        cell_axes_layout.addRow(self.show_axes)

        self.axis_width = QSpinBox()
        self.axis_width.setRange(1, 12)
        self.axis_width.setValue(5)
        self.axis_width.valueChanged.connect(self.render)
        self.axis_width.valueChanged.connect(self.save_settings)
        cell_axes_layout.addRow("Axis width", self.axis_width)

        self.axis_font = QComboBox()
        self.axis_font.addItems(["arial", "courier", "times"])
        self.axis_font.currentTextChanged.connect(self.render)
        self.axis_font.currentTextChanged.connect(self.save_settings)
        cell_axes_layout.addRow("Font", self.axis_font)

        self.axis_font_size = QSpinBox()
        self.axis_font_size.setRange(8, 48)
        self.axis_font_size.setValue(20)
        self.axis_font_size.valueChanged.connect(self.render)
        self.axis_font_size.valueChanged.connect(self.save_settings)
        cell_axes_layout.addRow("Font size", self.axis_font_size)

        # View from axis buttons
        view_grid = QGridLayout()
        btn_a = QPushButton("a")
        btn_a.clicked.connect(lambda: self.view_from_axis("a"))
        btn_b = QPushButton("b")
        btn_b.clicked.connect(lambda: self.view_from_axis("b"))
        btn_c = QPushButton("c")
        btn_c.clicked.connect(lambda: self.view_from_axis("c"))

        btn_na = QPushButton("-a")
        btn_na.clicked.connect(lambda: self.view_from_axis("-a"))
        btn_nb = QPushButton("-b")
        btn_nb.clicked.connect(lambda: self.view_from_axis("-b"))
        btn_nc = QPushButton("-c")
        btn_nc.clicked.connect(lambda: self.view_from_axis("-c"))

        view_grid.addWidget(btn_a, 0, 0)
        view_grid.addWidget(btn_b, 0, 1)
        view_grid.addWidget(btn_c, 0, 2)
        view_grid.addWidget(btn_na, 1, 0)
        view_grid.addWidget(btn_nb, 1, 1)
        view_grid.addWidget(btn_nc, 1, 2)
        cell_axes_layout.addRow("View axis:", view_grid)

        # Color settings
        self.color_axis_a = self._create_color_button("#ff0000")
        self.color_axis_b = self._create_color_button("#00ff00")
        self.color_axis_c = self._create_color_button("#0000ff")
        self.color_cell_edges = self._create_color_button("#ffffff")
        self.color_origin = self._create_color_button("#000000")

        cell_axes_layout.addRow("Axis A color", self.color_axis_a)
        cell_axes_layout.addRow("Axis B color", self.color_axis_b)
        cell_axes_layout.addRow("Axis C color", self.color_axis_c)
        cell_axes_layout.addRow("Cell Edges color", self.color_cell_edges)
        cell_axes_layout.addRow("Cell Origin color", self.color_origin)

        self.tabs.addTab(cell_axes_tab, "Cell / Axes")

        layout.addWidget(self.tabs)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Bottom buttons
        button_row = QHBoxLayout()
        render_button = QPushButton("Render")
        render_button.clicked.connect(self.render)
        clear_button = QPushButton("Clear Overlay")
        clear_button.clicked.connect(self.clear_view)
        reset_defaults_button = QPushButton("Reset Defaults")
        reset_defaults_button.clicked.connect(self.reset_to_defaults)
        button_row.addStretch(1)
        button_row.addWidget(render_button)
        button_row.addWidget(clear_button)
        button_row.addWidget(reset_defaults_button)
        layout.addLayout(button_row)

    def _get_current_view_mode(self) -> str:
        if self.radio_mol.isChecked():
            return "Whole Molecule"
        elif self.radio_asym.isChecked():
            return "Asymmetric Unit"
        elif self.radio_pack.isChecked():
            return "Packing"
        return "Whole Molecule"

    def _set_current_view_mode(self, mode: str):
        self.radio_asym.blockSignals(True)
        self.radio_mol.blockSignals(True)
        self.radio_pack.blockSignals(True)
        if mode == "Asymmetric Unit":
            self.radio_asym.setChecked(True)
        elif mode == "Whole Molecule":
            self.radio_mol.setChecked(True)
        elif mode == "Packing":
            self.radio_pack.setChecked(True)
        self.radio_asym.blockSignals(False)
        self.radio_mol.blockSignals(False)
        self.radio_pack.blockSignals(False)
        if hasattr(self, "show_cell"):
            self.show_cell.setEnabled(mode == "Packing")
        if hasattr(self, "show_axes"):
            self.show_axes.setEnabled(mode == "Packing")

    def _on_view_mode_changed(self):
        if not self.sender().isChecked():
            return
        mode = self._get_current_view_mode()
        if hasattr(self, "show_cell"):
            self.show_cell.setEnabled(mode == "Packing")
        if hasattr(self, "show_axes"):
            self.show_axes.setEnabled(mode == "Packing")
        self.save_settings()
        self._reset_camera_on_next_render = True
        self.render()

    def _on_disorder_changed(self):
        """Handle changes in disorder selection.

        Disables bond order determination when "All Parts" (None) is selected,
        because RDKit cannot process overlapping disordered atoms.
        """
        self._reset_camera_on_next_render = True
        # Determine if "All Parts" is selected (currentData returns None)
        data = self.disorder_combo.currentData()
        if data is None:
            # Gray out bond order option; preserve the checked state
            self.determine_bond_order.setEnabled(False)
            self.bond_order_disabled_label.setVisible(True)
        else:
            self.determine_bond_order.setEnabled(True)
            self.bond_order_disabled_label.setVisible(False)
        self.render()

    def _on_tab_changed(self, index):
        tab_text = self.tabs.tabText(index)
        if tab_text == "Supercell":
            if not self.radio_pack.isChecked():
                self.radio_pack.setChecked(True)

    def _repeat_spin(self):
        spin = QSpinBox()
        spin.setRange(1, 8)
        spin.setValue(1)
        spin.valueChanged.connect(self.render)
        return spin

    def set_supercell_preset(self, count: int):
        for spin in (self.repeat_a, self.repeat_b, self.repeat_c):
            spin.blockSignals(True)
            spin.setValue(count)
            spin.blockSignals(False)
        self.render()

    def reset_supercell(self):
        self.set_supercell_preset(1)

    def view_from_axis(self, axis_name: str):
        if self.structure is None:
            return
        plotter = self._plotter()
        if plotter is None:
            return

        lattice = np.asarray(self.structure.lattice, dtype=float)

        if axis_name == "a":
            direction = lattice[0]
            view_up = lattice[2]
        elif axis_name == "-a":
            direction = -lattice[0]
            view_up = lattice[2]
        elif axis_name == "b":
            direction = lattice[1]
            view_up = lattice[2]
        elif axis_name == "-b":
            direction = -lattice[1]
            view_up = lattice[2]
        elif axis_name == "c":
            direction = lattice[2]
            view_up = lattice[1]
        elif axis_name == "-c":
            direction = -lattice[2]
            view_up = lattice[1]
        else:
            return

        dir_norm = np.linalg.norm(direction)
        if dir_norm == 0:
            return
        dir_unit = direction / dir_norm

        up_norm = np.linalg.norm(view_up)
        if up_norm > 0:
            up_unit = view_up / up_norm
        else:
            up_unit = np.array([0.0, 0.0, 1.0])

        repeats = (self.repeat_a.value(), self.repeat_b.value(), self.repeat_c.value())
        focal_point = self._cell_center(repeats)

        # Calculate maximum cell dimension to place the camera outside
        scaled_a = lattice[0] * repeats[0]
        scaled_b = lattice[1] * repeats[1]
        scaled_c = lattice[2] * repeats[2]
        max_dim = max(
            np.linalg.norm(scaled_a), np.linalg.norm(scaled_b), np.linalg.norm(scaled_c)
        )
        distance = max_dim * 2.0
        if distance == 0:
            distance = 10.0

        camera_position = focal_point + dir_unit * distance

        try:
            plotter.camera_position = (camera_position, focal_point, up_unit)
        except Exception as exc:
            logging.debug("Failed to set camera_position in view_from_axis: %s", exc)
            try:
                if hasattr(plotter, "camera"):
                    plotter.camera.position = camera_position
                    plotter.camera.focal_point = focal_point
                    plotter.camera.up = up_unit
            except Exception as e:
                logging.debug("Failed to set individual camera attributes: %s", e)

        self._reset_camera_on_next_render = False

        try:
            plotter.reset_camera()
            plotter.render()
        except Exception as exc:
            logging.debug("Failed to reset camera or render in view_from_axis: %s", exc)

    def reset_to_defaults(self):
        self.show_bonds.blockSignals(True)
        self.show_hydrogens.blockSignals(True)
        self.keep_connected.blockSignals(True)
        self.show_cell.blockSignals(True)
        self.show_axes.blockSignals(True)
        self.show_ellipsoid_rings.blockSignals(True)
        self.fix_h_size.blockSignals(True)
        self.probability_spin.blockSignals(True)
        self.h_scale_spin.blockSignals(True)
        self.axis_width.blockSignals(True)
        self.axis_font.blockSignals(True)
        self.axis_font_size.blockSignals(True)
        self.ellipsoid_ring_width.blockSignals(True)
        self.radio_asym.blockSignals(True)
        self.radio_mol.blockSignals(True)
        self.radio_pack.blockSignals(True)
        self.determine_bond_order.blockSignals(True)
        self.max_atoms_spin.blockSignals(True)

        try:
            self.show_bonds.setChecked(True)
            self.show_hydrogens.setChecked(True)
            self.keep_connected.setChecked(True)
            self.show_cell.setChecked(True)
            self.show_axes.setChecked(True)
            self.show_ellipsoid_rings.setChecked(True)
            self.fix_h_size.setChecked(True)
            self.probability_spin.setValue(50.0)
            self.h_scale_spin.setValue(20.0)
            self.axis_width.setValue(5)
            self.ellipsoid_ring_width.setValue(2)
            self.determine_bond_order.setChecked(False)
            self.max_atoms_spin.setValue(300)

            idx = self.axis_font.findText("arial")
            if idx >= 0:
                self.axis_font.setCurrentIndex(idx)
            self.axis_font_size.setValue(20)

            self.radio_mol.setChecked(True)
            self.show_cell.setEnabled(False)
            self.show_axes.setEnabled(False)

            self._set_button_color(self.color_axis_a, "#ff0000")
            self._set_button_color(self.color_axis_b, "#00ff00")
            self._set_button_color(self.color_axis_c, "#0000ff")
            self._set_button_color(self.color_cell_edges, "#ffffff")
            self._set_button_color(self.color_origin, "#000000")
            self._set_button_color(self.color_ellipsoid_rings, "#000000")
        finally:
            self.show_bonds.blockSignals(False)
            self.show_hydrogens.blockSignals(False)
            self.keep_connected.blockSignals(False)
            self.show_cell.blockSignals(False)
            self.show_axes.blockSignals(False)
            self.show_ellipsoid_rings.blockSignals(False)
            self.fix_h_size.blockSignals(False)
            self.probability_spin.blockSignals(False)
            self.h_scale_spin.blockSignals(False)
            self.axis_width.blockSignals(False)
            self.axis_font.blockSignals(False)
            self.axis_font_size.blockSignals(False)
            self.ellipsoid_ring_width.blockSignals(False)
            self.radio_asym.blockSignals(False)
            self.radio_mol.blockSignals(False)
            self.radio_pack.blockSignals(False)
            self.determine_bond_order.blockSignals(False)
            self.max_atoms_spin.blockSignals(False)

        self.save_settings()
        self._reset_camera_on_next_render = True
        self.render()

    def _switch_to_ellipsoids(self):
        if self.context is not None:
            mw = self.context.get_main_window()
            if mw is not None:
                if hasattr(mw, "init_manager") and hasattr(
                    mw.init_manager, "style_button"
                ):
                    style_btn = mw.init_manager.style_button
                    if style_btn is not None and style_btn.menu() is not None:
                        for action in style_btn.menu().actions():
                            if action.text() == "Thermal Ellipsoids":
                                action.setChecked(True)
                                break
                if hasattr(mw, "view_3d_manager"):
                    mw.view_3d_manager.set_3d_style("Thermal Ellipsoids")

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
            self._update_disorder_ui()
            self._update_info_ui()
            self._reset_camera_on_next_render = True
            self.render()

    def _update_disorder_ui(self):
        if self.structure is None:
            self.disorder_label.setVisible(False)
            self.disorder_combo.setVisible(False)
            return

        keys = set()
        for atom in self.structure.atoms:
            key = atom.disorder_group
            if key is not None:
                keys.add(key)

        if keys:
            self.disorder_combo.blockSignals(True)
            self.disorder_combo.clear()
            self.disorder_combo.addItem("All Parts", None)
            for k in sorted(keys):
                self.disorder_combo.addItem(f"Part {k}", k)
            self.disorder_combo.blockSignals(False)

            self.disorder_label.setVisible(True)
            self.disorder_combo.setVisible(True)

            # Combo defaults to index 0 = "All Parts"; gray out bond order
            # (signals are blocked above so _on_disorder_changed won't fire)
            self.determine_bond_order.setEnabled(False)
            self.bond_order_disabled_label.setVisible(True)
        else:
            self.disorder_label.setVisible(False)
            self.disorder_combo.setVisible(False)
            # No disorder - bond order determination is allowed
            self.determine_bond_order.setEnabled(True)
            self.bond_order_disabled_label.setVisible(False)

    def _update_info_ui(self):
        if self.structure is None:
            all_info_labels = [
                self.info_formula,
                self.info_space_group,
                self.info_space_group_number,
                self.info_crystal_system,
                self.info_cell_a,
                self.info_cell_b,
                self.info_cell_c,
                self.info_cell_alpha,
                self.info_cell_beta,
                self.info_cell_gamma,
                self.info_volume,
                self.info_z,
                self.info_z_prime,
                self.info_density,
                self.info_mu,
                self.info_f000,
                self.info_temp,
                self.info_wavelength,
                self.info_crystal_size,
                self.info_theta_range,
                self.info_hkl_ranges,
                self.info_reflns_collected,
                self.info_reflns_unique,
                self.info_r_int,
                self.info_completeness,
                self.info_refinement_method,
                self.info_num_reflns,
                self.info_num_params,
                self.info_num_restraints,
                self.info_goof,
                self.info_r1,
                self.info_wr2,
                self.info_r1_all,
                self.info_wr2_all,
                self.info_max_shift,
                self.info_flack,
                self.info_diff_peak_hole,
            ]
            for lbl in all_info_labels:
                lbl.setText("N/A")
            self.simulate_xrd_btn.setEnabled(False)
            return

        s = self.structure
        self.info_formula.setText(s.formula or "N/A")
        self.info_space_group.setText(s.space_group or "N/A")
        self.info_space_group_number.setText(s.space_group_number or "N/A")
        self.info_crystal_system.setText(s.crystal_system or "N/A")
        self.info_cell_a.setText(
            s.cell_a_str or (f"{s.cell_lengths[0]:.4f}" if s.cell_lengths else "N/A")
        )
        self.info_cell_b.setText(
            s.cell_b_str or (f"{s.cell_lengths[1]:.4f}" if s.cell_lengths else "N/A")
        )
        self.info_cell_c.setText(
            s.cell_c_str or (f"{s.cell_lengths[2]:.4f}" if s.cell_lengths else "N/A")
        )
        self.info_cell_alpha.setText(
            s.cell_alpha_str or (f"{s.cell_angles[0]:.3f}" if s.cell_angles else "N/A")
        )
        self.info_cell_beta.setText(
            s.cell_beta_str or (f"{s.cell_angles[1]:.3f}" if s.cell_angles else "N/A")
        )
        self.info_cell_gamma.setText(
            s.cell_gamma_str or (f"{s.cell_angles[2]:.3f}" if s.cell_angles else "N/A")
        )
        self.info_volume.setText(s.volume or "N/A")
        self.info_z.setText(s.z or "N/A")
        self.info_z_prime.setText(s.z_prime or "N/A")
        self.info_density.setText(s.density or "N/A")
        self.info_mu.setText(s.mu or "N/A")
        self.info_f000.setText(s.f000 or "N/A")

        self.info_temp.setText(s.temp or "N/A")
        self.info_wavelength.setText(s.wavelength or "N/A")
        self.info_crystal_size.setText(s.crystal_size or "N/A")
        self.info_theta_range.setText(s.theta_range or "N/A")
        self.info_hkl_ranges.setText(s.hkl_ranges or "N/A")
        self.info_reflns_collected.setText(s.reflns_collected or "N/A")
        self.info_reflns_unique.setText(s.reflns_unique or "N/A")
        self.info_r_int.setText(s.r_int or "N/A")
        self.info_completeness.setText(s.completeness or "N/A")

        self.info_refinement_method.setText(s.refinement_method or "N/A")
        self.info_num_reflns.setText(s.num_reflns or "N/A")
        self.info_num_params.setText(s.num_params or "N/A")
        self.info_num_restraints.setText(s.num_restraints or "N/A")
        self.info_goof.setText(s.goof or "N/A")
        self.info_r1.setText(s.r1_gt or s.r1 or "N/A")
        self.info_wr2.setText(s.wr2_gt or s.wr2 or "N/A")
        self.info_r1_all.setText(s.r1_all or "N/A")
        self.info_wr2_all.setText(s.wr2_all or "N/A")
        self.info_max_shift.setText(s.max_shift or "N/A")
        self.info_flack.setText(s.flack or "N/A")
        self.info_diff_peak_hole.setText(s.diff_peak_hole or "N/A")

        self.simulate_xrd_btn.setEnabled(True)

    def _simulate_powder_pattern(self):
        if self.structure is None:
            return

        selected_key = None
        if (
            not self.disorder_combo.isHidden()
            and self.disorder_combo.currentIndex() > 0
        ):
            selected_key = self.disorder_combo.currentData()

        from .viewer_xrd import PowderPatternDialog

        dialog = PowderPatternDialog(self.structure, selected_key, self)
        dialog.exec()

    def _settings_path(self):
        return os.path.join(os.path.dirname(__file__), "settings.json")

    def load_settings(self):
        self._set_current_view_mode("Whole Molecule")
        path = self._settings_path()
        if not os.path.exists(path):
            return
        import json

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.show_bonds.blockSignals(True)
            self.show_hydrogens.blockSignals(True)
            self.keep_connected.blockSignals(True)
            self.show_cell.blockSignals(True)
            self.show_axes.blockSignals(True)
            self.show_ellipsoid_rings.blockSignals(True)
            self.fix_h_size.blockSignals(True)
            self.probability_spin.blockSignals(True)
            self.h_scale_spin.blockSignals(True)
            self.axis_width.blockSignals(True)
            self.axis_font.blockSignals(True)
            self.axis_font_size.blockSignals(True)
            self.ellipsoid_ring_width.blockSignals(True)
            self.radio_asym.blockSignals(True)
            self.radio_mol.blockSignals(True)
            self.radio_pack.blockSignals(True)
            self.determine_bond_order.blockSignals(True)
            self.max_atoms_spin.blockSignals(True)

            self._set_current_view_mode("Whole Molecule")

            if "show_bonds" in data:
                self.show_bonds.setChecked(bool(data["show_bonds"]))
            if "determine_bond_order" in data:
                self.determine_bond_order.setChecked(bool(data["determine_bond_order"]))
            if "max_atoms_for_bond_order" in data:
                self.max_atoms_spin.setValue(int(data["max_atoms_for_bond_order"]))
            if "show_hydrogens" in data:
                self.show_hydrogens.setChecked(bool(data["show_hydrogens"]))
            if "keep_connected" in data:
                self.keep_connected.setChecked(bool(data["keep_connected"]))
            if "show_cell" in data:
                self.show_cell.setChecked(bool(data["show_cell"]))
            if "show_axes" in data:
                self.show_axes.setChecked(bool(data["show_axes"]))
            if "h_scale" in data:
                self.h_scale_spin.setValue(float(data["h_scale"]))
            if "show_ellipsoid_rings" in data:
                self.show_ellipsoid_rings.setChecked(bool(data["show_ellipsoid_rings"]))
            if "fix_h_size" in data:
                self.fix_h_size.setChecked(bool(data["fix_h_size"]))
            if "probability" in data:
                try:
                    val_str = str(data["probability"])
                    import re

                    match = re.search(r"([\d\.]+)", val_str)
                    if match:
                        val = float(match.group(1))
                        # Backwards compatibility: if value is small (like 1.54),
                        # it was a scale factor. Default to 50%.
                        if val <= 10.0:
                            val = 50.0
                        self.probability_spin.setValue(val)
                except Exception as exc:
                    logging.debug("Failed to parse probability setting: %s", exc)
            if "axis_width" in data:
                self.axis_width.setValue(int(data["axis_width"]))
            if "axis_font" in data:
                idx = self.axis_font.findText(str(data["axis_font"]))
                if idx >= 0:
                    self.axis_font.setCurrentIndex(idx)
            if "axis_font_size" in data:
                self.axis_font_size.setValue(int(data["axis_font_size"]))
            if "color_axis_a" in data:
                self._set_button_color(self.color_axis_a, data["color_axis_a"])
            if "color_axis_b" in data:
                self._set_button_color(self.color_axis_b, data["color_axis_b"])
            if "color_axis_c" in data:
                self._set_button_color(self.color_axis_c, data["color_axis_c"])
            if "color_cell_edges" in data:
                self._set_button_color(self.color_cell_edges, data["color_cell_edges"])
            if "color_origin" in data:
                self._set_button_color(self.color_origin, data["color_origin"])
            if "color_ellipsoid_rings" in data:
                self._set_button_color(
                    self.color_ellipsoid_rings, data["color_ellipsoid_rings"]
                )
            if "ellipsoid_ring_width" in data:
                self.ellipsoid_ring_width.setValue(int(data["ellipsoid_ring_width"]))
        except Exception as e:
            logging.error("Failed to load settings: %s", e)
        finally:
            self.show_bonds.blockSignals(False)
            self.show_hydrogens.blockSignals(False)
            self.keep_connected.blockSignals(False)
            self.show_cell.blockSignals(False)
            self.show_axes.blockSignals(False)
            self.show_ellipsoid_rings.blockSignals(False)
            self.fix_h_size.blockSignals(False)
            self.probability_spin.blockSignals(False)
            self.h_scale_spin.blockSignals(False)
            self.axis_width.blockSignals(False)
            self.axis_font.blockSignals(False)
            self.axis_font_size.blockSignals(False)
            self.ellipsoid_ring_width.blockSignals(False)
            self.radio_asym.blockSignals(False)
            self.radio_mol.blockSignals(False)
            self.radio_pack.blockSignals(False)
            self.determine_bond_order.blockSignals(False)
            self.max_atoms_spin.blockSignals(False)

    def save_settings(self, *args):
        path = self._settings_path()
        import json

        data = {
            "show_bonds": self.show_bonds.isChecked(),
            "determine_bond_order": self.determine_bond_order.isChecked(),
            "max_atoms_for_bond_order": self.max_atoms_spin.value(),
            "show_hydrogens": self.show_hydrogens.isChecked(),
            "keep_connected": self.keep_connected.isChecked(),
            "show_cell": self.show_cell.isChecked(),
            "show_axes": self.show_axes.isChecked(),
            "show_ellipsoid_rings": self.show_ellipsoid_rings.isChecked(),
            "fix_h_size": self.fix_h_size.isChecked(),
            "probability": self.probability_spin.value(),
            "h_scale": self.h_scale_spin.value(),
            "axis_width": self.axis_width.value(),
            "axis_font": self.axis_font.currentText(),
            "axis_font_size": self.axis_font_size.value(),
            "color_axis_a": self.color_axis_a.property("color_hex"),
            "color_axis_b": self.color_axis_b.property("color_hex"),
            "color_axis_c": self.color_axis_c.property("color_hex"),
            "color_cell_edges": self.color_cell_edges.property("color_hex"),
            "color_origin": self.color_origin.property("color_hex"),
            "color_ellipsoid_rings": self.color_ellipsoid_rings.property("color_hex"),
            "ellipsoid_ring_width": self.ellipsoid_ring_width.value(),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logging.error("Failed to save settings: %s", e)

    def load_cif(self, path: str):
        for spin in (self.repeat_a, self.repeat_b, self.repeat_c):
            spin.blockSignals(True)
            spin.setValue(1)
            spin.blockSignals(False)

        try:
            self.all_structures = parse_cif_file_pymatgen(path)
        except Exception as e_pymatgen:
            logging.info(
                "pymatgen parser failed, falling back to built-in parser: %s",
                e_pymatgen,
            )
            try:
                self.all_structures = [parse_cif_file(path)]
            except Exception as exc:
                QMessageBox.critical(
                    self, "CIF Viewer", f"Could not read CIF file:\n{exc}"
                )
                logging.error("Failed to read CIF file %s: %s", path, exc)
                return

        if not self.all_structures:
            QMessageBox.critical(
                self, "CIF Viewer", "No valid structures found in CIF file."
            )
            return

        self._reset_camera_on_next_render = True
        self.current_path = path
        self.file_label.setText(os.path.basename(path))

        mw = self._main_window()
        if mw is not None:
            if hasattr(mw, "init_manager"):
                mw.init_manager.current_file_path = path
            if hasattr(mw, "state_manager") and hasattr(
                mw.state_manager, "update_window_title"
            ):
                mw.state_manager.update_window_title()

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

        self._update_disorder_ui()
        self._update_info_ui()

        self.structure_table.blockSignals(False)
        self._enter_viewer_mode()

        from PyQt6.QtCore import QTimer

        QTimer.singleShot(100, self.render)

    def clear_view(self, redraw=True):
        plotter = self._plotter()
        if plotter is None:
            return
        for name in self.overlay_actor_names:
            try:
                plotter.remove_actor(name)
            except Exception as exc:
                logging.debug("Failed to remove actor %s: %s", name, exc)
        self.overlay_actor_names.clear()
        if redraw:
            try:
                plotter.render()
            except Exception as exc:
                logging.debug("Failed to render in clear_view: %s", exc)

    def render_overlays_only(self):
        if self.structure is None:
            return
        plotter = self._plotter()
        if plotter is None:
            return

        self.clear_view(redraw=False)
        repeats = (self.repeat_a.value(), self.repeat_b.value(), self.repeat_c.value())
        if self.show_cell.isChecked() and self._get_current_view_mode() == "Packing":
            self._draw_cell_overlay(plotter, repeats)

        try:
            plotter.render()
        except Exception as exc:
            logging.debug("Failed to render overlays: %s", exc)

    def render(self):
        if self.structure is None:
            return
        self.render_timer.start(50)

    def _render_now(self):
        if self.structure is None:
            return

        plotter = self._plotter()
        if plotter is None:
            return

        self.clear_view()
        repeats = (self.repeat_a.value(), self.repeat_b.value(), self.repeat_c.value())

        view_mode = self._get_current_view_mode()
        if view_mode == "Asymmetric Unit" or view_mode == "Whole Molecule":
            base_atoms = (
                self.structure.asymmetric_atoms
                if self.structure.asymmetric_atoms is not None
                else self.structure.atoms
            )
        else:
            base_atoms = self.structure.atoms

        selected_key = None
        if (
            not self.disorder_combo.isHidden()
            and self.disorder_combo.currentIndex() > 0
        ):
            selected_key = self.disorder_combo.currentData()

        if selected_key is not None:
            base_atoms = [
                atom
                for atom in base_atoms
                if atom.disorder_group is None
                or atom.disorder_group == selected_key
                or atom.disorder_key == selected_key
            ]

        import dataclasses

        structure_to_render = dataclasses.replace(
            self.structure, atoms=tuple(base_atoms), asymmetric_atoms=tuple(base_atoms)
        )

        tol = 0.45

        if view_mode == "Whole Molecule":
            from .parser import grow_molecules

            atoms, bonds = grow_molecules(
                structure_to_render, selected_disorder_key=selected_key, tolerance=tol
            )
        else:
            atoms, bonds = expand_supercell(
                structure_to_render,
                repeats,
                keep_connected=self.keep_connected.isChecked(),
                tolerance=tol,
            )
        mol_bonds = bonds if self.show_bonds.isChecked() else []
        try:
            max_atoms = self.max_atoms_spin.value()
            determine_bo = self.determine_bond_order.isChecked() and (
                len(atoms) <= max_atoms
            )
            mol = render_atoms_to_rdkit_mol(
                atoms, mol_bonds, determine_bond_order=determine_bo
            )
            # Tag the molecule so custom styles and overlays know it's from CIF viewer
            mol.SetProp("_from_cif_viewer", "1")

            if not self.show_hydrogens.isChecked():
                from rdkit import Chem

                try:
                    mol = Chem.RemoveHs(mol)
                except Exception as e:
                    logging.debug("RemoveHs failed: %s", e)
                self.last_rendered_atoms = [
                    atom for atom in atoms if atom.element != "H"
                ]
            else:
                self.last_rendered_atoms = atoms
        except Exception as exc:
            QMessageBox.critical(
                self, "CIF Viewer", f"Could not build RDKit view:\n{exc}"
            )
            return

        self._draw_with_moleditpy(mol)

        def draw_overlays_and_render():
            plotter = self._plotter()
            if plotter is None:
                return

            self.clear_view()

            if (
                self.show_cell.isChecked()
                and self._get_current_view_mode() == "Packing"
            ):
                self._draw_cell_overlay(plotter, repeats)

            try:
                if getattr(self, "_reset_camera_on_next_render", True):
                    plotter.reset_camera()
                    self._reset_camera_on_next_render = False
                plotter.render()
            except Exception as exc:
                logging.error("Failed to reset camera or render cell: %s", exc)

        try:
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(100, draw_overlays_and_render)
        except Exception as exc:
            logging.warning(
                "Failed to schedule draw_overlays_and_render with QTimer: %s", exc
            )
            draw_overlays_and_render()

        summary_text = (
            f"{len(self.structure.atoms)} completed unit-cell atoms, "
            f"{len(atoms)} rendered atoms, {len(bonds)} inferred bonds, "
            f"supercell {repeats[0]} x {repeats[1]} x {repeats[2]}."
        )
        max_atoms = self.max_atoms_spin.value()
        if self.determine_bond_order.isChecked():
            if len(atoms) > max_atoms:
                summary_text += (
                    f" (Bond order determination skipped: over {max_atoms} atoms limit)"
                )
            elif mol.HasProp("_bond_order_error") and mol.GetProp("_bond_order_error"):
                err = mol.GetProp("_bond_order_error")
                summary_text += f" (Bond order determination failed: {err})"
        self.summary_label.setText(summary_text)
        if self.context and hasattr(self.context, "show_status_message"):
            self.context.show_status_message(
                "CIF Viewer rendered with MoleditPy 3D style.", 3000
            )

    def _draw_with_moleditpy(self, mol):
        if self.context is not None and hasattr(self.context, "draw_molecule_3d"):
            try:
                self.context.current_mol = mol
            except Exception as exc:
                logging.debug("Failed to assign context.current_mol directly: %s", exc)
                self.context.draw_molecule_3d(mol)
            return
        main_window = self._main_window()
        if main_window is not None and hasattr(main_window, "draw_molecule_3d"):
            if hasattr(main_window, "view_3d_manager"):
                try:
                    main_window.view_3d_manager.current_mol = mol
                except Exception as exc:
                    logging.debug(
                        "Failed to set main_window.view_3d_manager.current_mol: %s", exc
                    )
            main_window.draw_molecule_3d(mol)

    def _draw_cell_overlay(self, plotter, repeats):
        lattice = np.asarray(self.structure.lattice, dtype=float)
        color_a = self.color_axis_a.property("color_hex")
        color_b = self.color_axis_b.property("color_hex")
        color_c = self.color_axis_c.property("color_hex")
        color_edges = self.color_cell_edges.property("color_hex")
        color_ori = self.color_origin.property("color_hex")

        for index, (start, end, default_color, label) in enumerate(
            celleditpy_cell_axis_segments(lattice)
        ):
            name = f"cif_viewer_cell_line_{index}"
            is_axis = bool(label and self.show_axes.isChecked())
            width = (
                self.axis_width.value()
                if is_axis
                else max(1, self.axis_width.value() - 2)
            )

            if is_axis:
                if label == "a":
                    line_color = color_a
                elif label == "b":
                    line_color = color_b
                else:
                    line_color = color_c
            else:
                line_color = color_edges

            plotter.add_lines(
                np.array([start, end]), color=line_color, width=width, name=name
            )
            self.overlay_actor_names.append(name)
            if is_axis:
                label_name = f"cif_viewer_cell_axis_label_{label}"
                plotter.add_point_labels(
                    [end],
                    [label],
                    point_size=0,
                    font_size=self.axis_font_size.value(),
                    text_color=line_color,
                    font_family=self.axis_font.currentText(),
                    bold=True,
                    always_visible=True,
                    shape=None,
                    shape_opacity=0.0,
                    name=label_name,
                )
                self.overlay_actor_names.append(label_name)

        if self.show_axes.isChecked():
            origin_name = "cif_viewer_cell_origin_label"
            plotter.add_point_labels(
                [np.array([0.0, 0.0, 0.0])],
                ["O"],
                point_size=0,
                font_size=self.axis_font_size.value(),
                text_color=color_ori,
                font_family=self.axis_font.currentText(),
                bold=True,
                always_visible=True,
                shape=None,
                shape_opacity=0.0,
                name=origin_name,
            )
            self.overlay_actor_names.append(origin_name)

    def _cell_center(self, repeats):
        lattice = np.asarray(self.structure.lattice, dtype=float)
        scaled = lattice * np.array(repeats)[:, None]
        return np.sum(scaled, axis=0) / 2.0

    def _enter_viewer_mode(self):
        main_window = self._main_window()
        if main_window is None:
            return
        ui_manager = getattr(main_window, "ui_manager", None)
        if ui_manager is not None and hasattr(ui_manager, "_enter_3d_viewer_ui_mode"):
            try:
                ui_manager._enter_3d_viewer_ui_mode()
                return
            except Exception as exc:
                logging.debug("Failed to enter 3D viewer UI mode: %s", exc)
        for attr in ("splitter", "main_splitter", "central_splitter"):
            splitter = getattr(main_window, attr, None)
            if splitter is not None and hasattr(splitter, "setSizes"):
                try:
                    splitter.setSizes([0, 1])
                    return
                except Exception as exc:
                    logging.debug("Failed to set splitter sizes: %s", exc)

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
        return (
            parent.parent()
            if parent is not None and hasattr(parent, "parent")
            else None
        )


CifViewerDialog = CifViewerWidget
