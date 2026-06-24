from __future__ import annotations

import csv

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QComboBox,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QLabel,
    QWidget,
    QFileDialog,
    QMessageBox,
)
import logging

import numpy as np

# Matplotlib integration
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

# pymatgen XRD Calculator and structure utilities
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure


def make_pymatgen_structure(
    cif_structure, selected_disorder_key: str | None = None
) -> Structure:
    """Helper to convert a CifStructure to a pymatgen Structure, optionally filtering by disorder key."""
    # First, collect the base atoms to use
    base_atoms = []
    for atom in cif_structure.atoms:
        if selected_disorder_key is not None:
            if atom.disorder_group is not None:
                if (
                    atom.disorder_group != selected_disorder_key
                    and atom.disorder_key != selected_disorder_key
                ):
                    continue
        base_atoms.append(atom)

    species = []
    coords = []

    is_asymmetric = getattr(cif_structure, "is_asymmetric_unit_only", False)
    symops = None
    if is_asymmetric and cif_structure.space_group:
        try:
            from pymatgen.symmetry.groups import SpaceGroup

            sg = SpaceGroup(cif_structure.space_group)
            symops = sg.symmetry_ops
        except Exception as exc:
            logging.debug(
                "SpaceGroup lookup failed for %s: %s", cif_structure.space_group, exc
            )

    if symops:
        # Generate full unit cell by applying symmetry operations
        for atom in base_atoms:
            occ = atom.occupancy if atom.occupancy is not None else 1.0
            if selected_disorder_key is not None and atom.disorder_group is not None:
                occ = 1.0
            element_dict = {atom.element: occ}

            for op in symops:
                # Apply symmetry operation to fractional coordinates
                new_frac = op.operate(atom.fract)
                # Wrap to [0, 1)
                new_frac = np.mod(new_frac, 1.0)

                # Check for duplicates (special positions or already generated)
                is_duplicate = False
                for existing_coord, existing_spec in zip(coords, species):
                    diff = new_frac - existing_coord
                    diff_mod = diff - np.round(diff)
                    if np.allclose(diff_mod, 0.0, atol=1e-3):
                        if atom.element in existing_spec:
                            is_duplicate = True
                            break

                if not is_duplicate:
                    species.append(element_dict)
                    coords.append(new_frac)
    else:
        # Just use the atoms directly (they are already conventional cell or we have no symmetry operations)
        for atom in base_atoms:
            occ = atom.occupancy if atom.occupancy is not None else 1.0
            if selected_disorder_key is not None and atom.disorder_group is not None:
                occ = 1.0
            species.append({atom.element: occ})
            coords.append(atom.fract)

    return Structure(cif_structure.lattice, species, coords)


class PowderPatternDialog(QDialog):
    """Interactive dialog displaying the simulated powder diffraction pattern (XRD) of the crystal structure."""

    def __init__(
        self, cif_structure, selected_disorder_key: str | None = None, parent=None
    ):
        super().__init__(parent)
        self.cif_structure = cif_structure
        self.selected_disorder_key = selected_disorder_key

        self.last_xrd = None
        self.last_profile_x = None
        self.last_profile_y = None

        self.setWindowTitle("Simulate Powder Pattern (XRD)")
        self.resize(800, 600)

        self._build_ui()
        self.calculate_and_plot()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # Controls panel at the top
        controls_group = QWidget()
        controls_layout = QHBoxLayout(controls_group)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        form_layout = QFormLayout()

        self.source_combo = QComboBox()
        self.source_combo.addItems(["CuKa", "MoKa", "CrKa", "FeKa", "CoKa", "Custom"])
        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        form_layout.addRow("Radiation Source:", self.source_combo)

        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setRange(0.1, 10.0)
        self.wavelength_spin.setDecimals(4)
        self.wavelength_spin.setValue(1.54184)  # Default CuKa
        self.wavelength_spin.setEnabled(False)
        self.wavelength_spin.valueChanged.connect(self._on_params_changed)
        form_layout.addRow("Wavelength (Å):", self.wavelength_spin)

        controls_layout.addLayout(form_layout)

        form_layout_2 = QFormLayout()

        self.min_theta_spin = QDoubleSpinBox()
        self.min_theta_spin.setRange(0.0, 90.0)
        self.min_theta_spin.setValue(5.0)
        self.min_theta_spin.valueChanged.connect(self._on_params_changed)
        form_layout_2.addRow("Min 2-Theta (°):", self.min_theta_spin)

        self.max_theta_spin = QDoubleSpinBox()
        self.max_theta_spin.setRange(5.0, 180.0)
        self.max_theta_spin.setValue(90.0)
        self.max_theta_spin.valueChanged.connect(self._on_params_changed)
        form_layout_2.addRow("Max 2-Theta (°):", self.max_theta_spin)

        controls_layout.addLayout(form_layout_2)

        form_layout_3 = QFormLayout()

        self.pattern_type_combo = QComboBox()
        self.pattern_type_combo.addItems(
            ["Profile (Gaussian)", "Profile (Lorentzian)", "Sticks Only"]
        )
        self.pattern_type_combo.currentTextChanged.connect(
            self._on_pattern_type_changed
        )
        form_layout_3.addRow("Pattern Style:", self.pattern_type_combo)

        self.fwhm_spin = QDoubleSpinBox()
        self.fwhm_spin.setRange(0.01, 5.0)
        self.fwhm_spin.setDecimals(3)
        self.fwhm_spin.setSingleStep(0.01)
        self.fwhm_spin.setValue(0.1)
        self.fwhm_spin.valueChanged.connect(self._on_params_changed)
        form_layout_3.addRow("FWHM (2θ °):", self.fwhm_spin)

        self.show_sticks_chk = QCheckBox("Show peak sticks")
        self.show_sticks_chk.setChecked(True)
        self.show_sticks_chk.toggled.connect(self._on_params_changed)
        form_layout_3.addRow(self.show_sticks_chk)

        self.label_peaks_chk = QCheckBox("Label Peaks")
        self.label_peaks_chk.setChecked(True)
        self.label_peaks_chk.toggled.connect(self._on_params_changed)
        form_layout_3.addRow(self.label_peaks_chk)

        self.sym_label = QLabel()
        sg = getattr(self.cif_structure, "space_group", None) or "Unknown"
        sg_num = getattr(self.cif_structure, "space_group_number", None)
        sym_text = f"Space Group: {sg}"
        if sg_num:
            sym_text += f" (No. {sg_num})"
        self.sym_label.setText(sym_text)
        form_layout_3.addRow(self.sym_label)

        controls_layout.addLayout(form_layout_3)

        main_layout.addWidget(controls_group)

        # Matplotlib canvas and toolbar
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas, 1)

        # Close and Export buttons at bottom
        btn_layout = QHBoxLayout()
        self.export_csv_btn = QPushButton("Export CSV...")
        self.export_csv_btn.clicked.connect(self._export_csv)
        btn_layout.addWidget(self.export_csv_btn)

        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        main_layout.addLayout(btn_layout)

    def _on_source_changed(self, text: str):
        if text == "CuKa":
            self.wavelength_spin.setValue(1.54184)
            self.wavelength_spin.setEnabled(False)
        elif text == "MoKa":
            self.wavelength_spin.setValue(0.71073)
            self.wavelength_spin.setEnabled(False)
        elif text == "CrKa":
            self.wavelength_spin.setValue(2.29100)
            self.wavelength_spin.setEnabled(False)
        elif text == "FeKa":
            self.wavelength_spin.setValue(1.9373)
            self.wavelength_spin.setEnabled(False)
        elif text == "CoKa":
            self.wavelength_spin.setValue(1.7902)
            self.wavelength_spin.setEnabled(False)
        elif text == "Custom":
            self.wavelength_spin.setEnabled(True)
        self.calculate_and_plot()

    def _on_pattern_type_changed(self, text: str):
        is_profile = text.startswith("Profile")
        self.fwhm_spin.setEnabled(is_profile)
        self.show_sticks_chk.setEnabled(is_profile)
        self.calculate_and_plot()

    def _on_params_changed(self):
        self.calculate_and_plot()

    def calculate_and_plot(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        self.last_xrd = None
        self.last_profile_x = None
        self.last_profile_y = None

        try:
            # Build pymatgen structure
            pm_structure = make_pymatgen_structure(
                self.cif_structure, self.selected_disorder_key
            )
            if len(pm_structure) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No atoms selected in this disorder group.",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                self.canvas.draw()
                return

            # Instantiate XRDCalculator
            wavelength = self.wavelength_spin.value()
            calculator = XRDCalculator(wavelength=wavelength)

            # Get diffraction pattern
            min_theta = self.min_theta_spin.value()
            max_theta = self.max_theta_spin.value()

            xrd = calculator.get_pattern(
                pm_structure, two_theta_range=(min_theta, max_theta)
            )
            self.last_xrd = xrd

            pattern_style = self.pattern_type_combo.currentText()
            fwhm = self.fwhm_spin.value()
            show_sticks = self.show_sticks_chk.isChecked()

            # Always precompute the continuous profile on grid for potential export
            two_thetas = np.linspace(min_theta, max_theta, 2000)
            profile = np.zeros_like(two_thetas)

            profile_style = pattern_style
            if profile_style == "Sticks Only":
                profile_style = "Profile (Gaussian)"

            if profile_style == "Profile (Gaussian)":
                sigma = fwhm / 2.35482
                for x, y in zip(xrd.x, xrd.y):
                    profile += y * np.exp(-0.5 * ((two_thetas - x) / sigma) ** 2)
            elif profile_style == "Profile (Lorentzian)":
                gamma = fwhm / 2.0
                for x, y in zip(xrd.x, xrd.y):
                    profile += y * (gamma**2) / ((two_thetas - x) ** 2 + gamma**2)

            if len(profile) > 0 and np.max(profile) > 0:
                profile = (profile / np.max(profile)) * 100.0

            self.last_profile_x = two_thetas
            self.last_profile_y = profile

            if pattern_style.startswith("Profile"):
                ax.plot(
                    two_thetas,
                    profile,
                    color="#1f77b4",
                    linewidth=1.5,
                    label="Simulated Profile",
                )
                ax.fill_between(two_thetas, 0, profile, color="#1f77b4", alpha=0.15)

                if show_sticks and len(xrd.x) > 0:
                    ax.vlines(
                        xrd.x,
                        0,
                        xrd.y,
                        colors="#d62728",
                        linewidth=0.8,
                        alpha=0.6,
                        label="Bragg Peaks",
                    )
            else:
                # Sticks Only
                if len(xrd.x) > 0:
                    ax.vlines(
                        xrd.x,
                        0,
                        xrd.y,
                        colors="black",
                        linewidth=1.2,
                        label="Bragg Peaks",
                    )

            # Label Peaks if checked
            if self.label_peaks_chk.isChecked() and len(xrd.x) > 0:
                # Find peaks in range with intensity >= 5%
                in_range_peaks = []
                for x, y, hkls in zip(xrd.x, xrd.y, xrd.hkls):
                    if min_theta <= x <= max_theta and y >= 5.0:
                        in_range_peaks.append((x, y, hkls))

                # Sort by intensity descending and annotate the top 10
                in_range_peaks = sorted(
                    in_range_peaks, key=lambda k: k[1], reverse=True
                )[:10]

                for x, y, hkls in in_range_peaks:
                    hkl_tuples = [h["hkl"] for h in hkls]
                    label = ",".join("".join(map(str, hkl)) for hkl in hkl_tuples)

                    ax.annotate(
                        label,
                        xy=(x, y),
                        xytext=(0, 4),
                        textcoords="offset points",
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="darkred",
                    )

            # Formatting and labeling
            ax.set_xlabel(r"$2\theta$ ($^\circ$)")
            ax.set_ylabel("Intensity (scaled to 100)")
            ax.set_xlim(min_theta, max_theta)
            ax.set_ylim(0, 105)

            title_suffix = ""
            if self.selected_disorder_key is not None:
                title_suffix = f" (Part {self.selected_disorder_key})"
            ax.set_title(
                f"Simulated Powder Pattern - {self.cif_structure.name}{title_suffix}"
            )
            self.fig.tight_layout()

        except Exception as exc:
            ax.text(
                0.5,
                0.5,
                f"Calculation failed:\n{exc}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="red",
            )

        self.canvas.draw()

    def _export_csv(self):
        if self.last_xrd is None or len(self.last_xrd.x) == 0:
            QMessageBox.warning(
                self, "Export CSV", "No simulated diffraction data available to export."
            )
            return

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Export XRD Data")
        msg_box.setText("Select data to export:")

        profile_btn = msg_box.addButton(
            "Simulated Profile (Curve)", QMessageBox.ButtonRole.ActionRole
        )
        msg_box.addButton("Bragg Peaks (Sticks)", QMessageBox.ButtonRole.ActionRole)
        cancel_btn = msg_box.addButton(QMessageBox.StandardButton.Cancel)

        msg_box.exec()

        if msg_box.clickedButton() == cancel_btn:
            return

        export_profile = msg_box.clickedButton() == profile_btn

        default_filename = (
            f"{self.cif_structure.name}_profile.csv"
            if export_profile
            else f"{self.cif_structure.name}_sticks.csv"
        )

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export XRD Data as CSV",
            default_filename,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        try:
            if export_profile:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["2-Theta (deg)", "Intensity"])
                    for x, y in zip(self.last_profile_x, self.last_profile_y):
                        writer.writerow([f"{x:.4f}", f"{y:.4f}"])
            else:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        ["2-Theta (deg)", "Intensity", "d-spacing (A)", "HKL"]
                    )
                    for x, y, d, hkls in zip(
                        self.last_xrd.x,
                        self.last_xrd.y,
                        self.last_xrd.d_hkls,
                        self.last_xrd.hkls,
                    ):
                        hkl_str = ";".join(
                            f"({h['hkl'][0]},{h['hkl'][1]},{h['hkl'][2]})" for h in hkls
                        )
                        writer.writerow([f"{x:.4f}", f"{y:.4f}", f"{d:.4f}", hkl_str])

            QMessageBox.information(
                self, "Export CSV", f"Successfully exported to:\n{path}"
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Export CSV", f"Failed to export CSV file:\n{exc}"
            )
