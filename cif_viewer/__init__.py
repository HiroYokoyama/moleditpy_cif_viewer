import math
import os
import logging
import re
import traceback
import types

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QProgressDialog

PLUGIN_NAME = "CIF Viewer"
PLUGIN_VERSION = "1.2.2"
PLUGIN_AUTHOR = "HiroYokoyama"

PLUGIN_DESCRIPTION = (
    "Visualization-only CIF crystal structure viewer with unit-cell and "
    "supercell rendering for MoleditPy."
)
PLUGIN_DEPENDENCIES = ["numpy", "pymatgen", "PyQt6", "pyvista", "rdkit"]
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=4.0.0, <5.0.0"

WINDOW_ID = "cif_viewer_panel"


class EllipsoidWorkerThread(QThread):
    result_ready = pyqtSignal(object, object, object, object, str)

    def __init__(
        self,
        atoms_data,
        resolution,
        scale_factor,
        h_scale,
        atom_scale,
        col,
        show_rings,
        circle_color,
        circle_width,
    ):
        super().__init__()
        self.atoms_data = atoms_data
        self.resolution = resolution
        self.scale_factor = scale_factor
        self.h_scale = h_scale
        self.atom_scale = atom_scale
        self.col = col
        self.show_rings = show_rings
        self.circle_color = circle_color
        self.circle_width = circle_width

    def run(self):
        try:
            import pyvista as pv

            fallback_positions = []
            fallback_colors = []
            fallback_radii = []

            h_positions = []
            h_colors = []
            h_radii = []

            ellipsoid_meshes = []
            ellipsoid_circles_to_draw = []

            try:
                from rdkit import Chem

                pt = Chem.GetPeriodicTable()
            except ImportError:
                pt = None

            try:
                from moleditpy.constants import VDW_RADII
            except ImportError:
                try:
                    from moleditpy.utils.constants import VDW_RADII
                except ImportError:
                    VDW_RADII = {}

            for index, atom_info in enumerate(self.atoms_data):
                color_rgb = (
                    self.col[index] if index < len(self.col) else [0.5, 0.5, 0.5]
                )
                pos = atom_info["pos"]
                symbol = atom_info["element"]
                cov = atom_info["cov"]
                has_cov = atom_info["has_cov"]

                if has_cov:
                    try:
                        eigenvalues, eigenvectors = np.linalg.eigh(cov)
                        eigenvalues = np.maximum(eigenvalues, 1e-5)
                        radii = np.sqrt(eigenvalues) * self.scale_factor

                        if np.linalg.det(eigenvectors) < 0:
                            eigenvectors = eigenvectors.copy()
                            eigenvectors[:, 0] = -eigenvectors[:, 0]

                        ellipsoid = pv.Sphere(
                            radius=1.0,
                            theta_resolution=self.resolution,
                            phi_resolution=self.resolution,
                        )
                        ellipsoid.scale([radii[0], radii[1], radii[2]], inplace=True)

                        rotation_matrix = np.eye(4)
                        rotation_matrix[:3, :3] = eigenvectors
                        ellipsoid.transform(rotation_matrix, inplace=True)
                        ellipsoid.translate(pos, inplace=True)

                        ellipsoid.compute_normals(
                            auto_orient_normals=True, inplace=True
                        )
                        ellipsoid.point_data["colors"] = np.tile(
                            color_rgb, (ellipsoid.n_points, 1)
                        )
                        ellipsoid_meshes.append(ellipsoid)

                        ellipsoid_circles_to_draw.append(
                            (index, radii, eigenvectors, pos)
                        )
                    except Exception as exc:
                        logging.warning("draw ellipsoid for %s: %s", symbol, exc)
                else:
                    if symbol == "H":
                        h_vdw = 1.2
                        if pt is not None:
                            try:
                                h_vdw = pt.GetRvdw("H")
                            except Exception:
                                pass
                        rad = h_vdw * self.h_scale * self.atom_scale
                        h_positions.append(pos)
                        h_colors.append(color_rgb)
                        h_radii.append(rad)
                    else:
                        rad = VDW_RADII.get(symbol, 0.4) * self.atom_scale
                        fallback_positions.append(pos)
                        fallback_colors.append(color_rgb)
                        fallback_radii.append(rad)

            merged_ellipsoids = None
            if ellipsoid_meshes:
                merged_ellipsoids = pv.merge(ellipsoid_meshes)

            fallback_glyphs = None
            if fallback_positions:
                fallback_source = pv.PolyData(np.array(fallback_positions))
                fallback_source["colors"] = np.array(fallback_colors)
                fallback_source["radii"] = np.array(fallback_radii)
                fallback_glyphs = fallback_source.glyph(
                    scale="radii",
                    geom=pv.Sphere(
                        radius=1.0,
                        theta_resolution=self.resolution,
                        phi_resolution=self.resolution,
                    ),
                    orient=False,
                )

            h_glyphs = None
            if h_positions:
                h_source = pv.PolyData(np.array(h_positions))
                h_source["colors"] = np.array(h_colors)
                h_source["radii"] = np.array(h_radii)
                h_glyphs = h_source.glyph(
                    scale="radii",
                    geom=pv.Sphere(
                        radius=1.0,
                        theta_resolution=self.resolution,
                        phi_resolution=self.resolution,
                    ),
                    orient=False,
                )

            rings_mesh = None
            if self.show_rings and ellipsoid_circles_to_draw:
                all_ring_points = []
                all_ring_lines = []
                current_pt_idx = 0

                for index, radii, eigenvectors, pos in ellipsoid_circles_to_draw:
                    theta = np.linspace(0, 2 * np.pi, 37)
                    cos_t = np.cos(theta)
                    sin_t = np.sin(theta)
                    zero_t = np.zeros_like(theta)

                    circle_xy = np.column_stack((cos_t, sin_t, zero_t))
                    circle_yz = np.column_stack((zero_t, cos_t, sin_t))
                    circle_zx = np.column_stack((sin_t, zero_t, cos_t))

                    ring_radii = radii * 1.01

                    for base_circle in [circle_xy, circle_yz, circle_zx]:
                        pts = np.dot(base_circle * ring_radii, eigenvectors.T) + pos
                        all_ring_points.append(pts)

                        line_cells = np.empty(38, dtype=np.int32)
                        line_cells[0] = 37
                        line_cells[1:] = np.arange(
                            current_pt_idx, current_pt_idx + 37, dtype=np.int32
                        )
                        all_ring_lines.append(line_cells)

                        current_pt_idx += 37

                points_arr = np.concatenate(all_ring_points, axis=0)
                lines_arr = np.concatenate(all_ring_lines, axis=0)
                rings_mesh = pv.PolyData(points_arr, lines=lines_arr)

            self.result_ready.emit(
                merged_ellipsoids, fallback_glyphs, h_glyphs, rings_mesh, ""
            )
        except Exception as exc:
            err_trace = traceback.format_exc()
            self.result_ready.emit(None, None, None, None, f"{exc}\n{err_trace}")


def initialize(context):
    def apply_hook(vm, orig_draw):
        if getattr(vm, "_cif_viewer_hooked", False):
            return

        def hooked_draw(self_vm, mol, *args, **kwargs):
            orig_draw(mol, *args, **kwargs)
            dock_widget = context.get_window(WINDOW_ID)
            if dock_widget is not None and dock_widget.widget() is not None:
                w = dock_widget.widget()
                if (
                    mol is None
                    or not hasattr(mol, "HasProp")
                    or not mol.HasProp("_from_cif_viewer")
                ):
                    w.clear_view()
                    return
                if (
                    getattr(w, "structure", None) is not None
                    and dock_widget.isVisible()
                ):
                    if getattr(w, "_reset_camera_on_next_render", False):
                        try:
                            context.reset_3d_camera()
                            w._reset_camera_on_next_render = False
                        except Exception as e:
                            logging.debug(
                                "Failed to reset camera in hooked_draw: %s", e
                            )

                    try:
                        from PyQt6.QtCore import QTimer

                        QTimer.singleShot(50, w.render_overlays_only)
                    except Exception as e:
                        logging.debug("Failed to schedule render_overlays_only: %s", e)
                        w.render_overlays_only()

        vm.draw_molecule_3d = types.MethodType(hooked_draw, vm)
        vm._cif_viewer_hooked = True

    def revert_hook(vm, orig_draw):
        if not getattr(vm, "_cif_viewer_hooked", False):
            return
        vm.draw_molecule_3d = orig_draw
        vm._cif_viewer_hooked = False

    def show_panel(file_path=None):
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QDockWidget

        from .viewer import CifViewerWidget

        main_window = context.get_main_window()

        def update_hook_state(visible):
            if main_window is not None and hasattr(main_window, "view_3d_manager"):
                vm = main_window.view_3d_manager
                if not hasattr(vm, "_orig_draw_molecule_3d"):
                    vm._orig_draw_molecule_3d = vm.draw_molecule_3d
                if visible:
                    apply_hook(vm, vm._orig_draw_molecule_3d)
                else:
                    revert_hook(vm, vm._orig_draw_molecule_3d)

        dock = context.get_window(WINDOW_ID)
        if dock is None:
            dock = QDockWidget(f"CIF Viewer v{PLUGIN_VERSION}", main_window)
            dock.setAllowedAreas(
                Qt.DockWidgetArea.LeftDockWidgetArea
                | Qt.DockWidgetArea.RightDockWidgetArea
            )
            widget = CifViewerWidget(dock, context)
            dock.setWidget(widget)
            dock.visibilityChanged.connect(update_hook_state)

            if main_window is not None and hasattr(main_window, "addDockWidget"):
                main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
            context.register_window(WINDOW_ID, dock)

            dock.show()
            dock.raise_()
            update_hook_state(dock.isVisible())
        else:
            widget = dock.widget()
            if file_path is None:
                if dock.isVisible():
                    dock.hide()
                else:
                    dock.show()
                    dock.raise_()
            else:
                dock.show()
                dock.raise_()
            update_hook_state(dock.isVisible())

        if file_path:
            widget.load_cif(file_path)

    def open_from_menu():
        show_panel()

    def open_file(file_path):
        try:
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(0, lambda path=file_path: show_panel(path))
        except Exception as e:
            logging.debug("Failed to schedule show_panel: %s", e)
            show_panel(file_path)
        return True

    def handle_drop(file_path):
        if str(file_path).lower().endswith(".cif"):
            open_file(file_path)
            return True
        return False

    def handle_reset():
        dock = context.get_window(WINDOW_ID)
        if dock is not None and dock.widget() is not None:
            dock.widget().clear_view()

    def draw_ellipsoid_model(mw, mol):
        if (
            mol is None
            or not hasattr(mol, "HasProp")
            or not mol.HasProp("_from_cif_viewer")
        ):
            context.show_status_message(
                "Thermal Ellipsoids style is only available for CIF files opened via CIF Viewer.",
                4000,
            )
            if hasattr(mw, "view_3d_manager") and hasattr(
                mw.view_3d_manager, "set_3d_style"
            ):
                mw.view_3d_manager.set_3d_style("ball_and_stick")
            return

        plotter = getattr(mw, "plotter", None)
        if plotter is None and hasattr(mw, "view_3d_manager"):
            plotter = mw.view_3d_manager.plotter
        if plotter is None:
            return

        dock = context.get_window(WINDOW_ID)
        widget = dock.widget() if dock is not None else None

        has_adp = False
        if (
            widget is not None
            and getattr(widget, "structure", None) is not None
            and widget.structure.u_cart is not None
        ):
            if hasattr(widget, "last_rendered_atoms") and widget.last_rendered_atoms:
                has_adp = True

        if not has_adp:
            context.show_status_message(
                "No ADP data in current CIF. Falling back to Ball & Stick.", 4000
            )
            if hasattr(mw, "view_3d_manager") and hasattr(
                mw.view_3d_manager, "set_3d_style"
            ):
                mw.view_3d_manager.set_3d_style("ball_and_stick")
            return

        is_headless = (
            os.environ.get("MOLEDITPY_HEADLESS") == "1"
            or os.environ.get("QT_QPA_PLATFORM") == "offscreen"
            or "PYTEST_CURRENT_TEST" in os.environ
        )

        num_atoms = len(widget.last_rendered_atoms)
        use_threading = not is_headless and num_atoms > 100

        # Terminate any existing ellipsoid thread
        if hasattr(mw, "_ellipsoid_thread") and mw._ellipsoid_thread is not None:
            try:
                mw._ellipsoid_thread.result_ready.disconnect()
            except Exception:
                pass
            mw._ellipsoid_thread.terminate()
            mw._ellipsoid_thread.wait()
            mw._ellipsoid_thread = None

        should_restore_camera = True
        if widget is not None:
            should_restore_camera = not getattr(
                widget, "_reset_camera_on_next_render", False
            )

        camera_state = (
            getattr(plotter, "camera_position", None) if should_restore_camera else None
        )

        if hasattr(mw, "init_manager") and hasattr(mw.init_manager, "settings"):
            bg_color = mw.init_manager.settings.get("background_color", "#919191")
            is_lighting_enabled = mw.init_manager.settings.get("lighting_enabled", True)
            light_intensity = mw.init_manager.settings.get("light_intensity", 1.2)
            specular = mw.init_manager.settings.get("specular", 0.2)
            specular_power = mw.init_manager.settings.get("specular_power", 20)
        else:
            bg_color = "#919191"
            is_lighting_enabled = True
            light_intensity = 1.2
            specular = 0.2
            specular_power = 20

        mesh_props = dict(
            smooth_shading=True,
            specular=specular,
            specular_power=specular_power,
            lighting=is_lighting_enabled,
        )

        def get_scale_factor(p_percent):
            p = p_percent / 100.0
            if p <= 0 or p >= 1:
                return 1.5382

            def cdf(c):
                return math.erf(c / math.sqrt(2)) - math.sqrt(
                    2 / math.pi
                ) * c * math.exp(-(c**2) / 2)

            low, high = 0.0, 10.0
            for _ in range(50):
                mid = (low + high) / 2
                if cdf(mid) < p:
                    low = mid
                else:
                    high = mid
            return (low + high) / 2

        scale_factor = 1.5382
        if widget is not None:
            if hasattr(widget, "probability_spin"):
                scale_factor = get_scale_factor(widget.probability_spin.value())
            elif hasattr(widget, "probability_combo"):
                prob_text = widget.probability_combo.currentText()
                try:
                    match = re.search(r"\(([\d\.]+)\)", prob_text)
                    if match:
                        scale_factor = float(match.group(1))
                    else:
                        match = re.search(r"([\d\.]+)", prob_text)
                        scale_factor = float(match.group(1)) if match else 1.5382
                except Exception:
                    scale_factor = 1.5382

        h_scale = 0.2
        if widget is not None and hasattr(widget, "h_scale_spin"):
            h_scale = widget.h_scale_spin.value() / 100.0

        atom_scale = 1.0
        resolution = 16
        if hasattr(mw, "init_manager") and hasattr(mw.init_manager, "settings"):
            atom_scale = mw.init_manager.settings.get("ball_stick_atom_scale", 1.0)
            resolution = mw.init_manager.settings.get("ball_stick_resolution", 16)

        try:
            from moleditpy.constants import CPK_COLORS_PV as CPK_COLORS
        except ImportError:
            try:
                from moleditpy.utils.constants import CPK_COLORS_PV as CPK_COLORS
            except ImportError:
                CPK_COLORS = {}

        sym = [a.GetSymbol() for a in mol.GetAtoms()]
        col = np.array([CPK_COLORS.get(s, [0.5, 0.5, 0.5]) for s in sym])

        atoms_data = []
        for index, atom in enumerate(widget.last_rendered_atoms):
            if index >= mol.GetNumAtoms():
                continue
            symbol = atom.element
            base_idx = atom.base_index
            cov = getattr(atom, "u_cart", None)
            if (
                cov is None
                and widget.structure.u_cart is not None
                and base_idx < len(widget.structure.u_cart)
            ):
                cov = widget.structure.u_cart[base_idx]

            has_cov = cov is not None and not np.allclose(cov, 0.0)
            if (
                has_cov
                and symbol == "H"
                and widget is not None
                and hasattr(widget, "fix_h_size")
                and widget.fix_h_size.isChecked()
            ):
                has_cov = False

            pos = list(mol.GetConformer().GetAtomPosition(index))
            atoms_data.append(
                {
                    "element": symbol,
                    "pos": pos,
                    "cov": cov,
                    "has_cov": has_cov,
                }
            )

        show_rings = True
        if widget is not None and hasattr(widget, "show_ellipsoid_rings"):
            show_rings = widget.show_ellipsoid_rings.isChecked()

        circle_color = "black"
        if widget is not None and hasattr(widget, "color_ellipsoid_rings"):
            circle_color = widget.color_ellipsoid_rings.property("color_hex")

        circle_width = 2
        if widget is not None and hasattr(widget, "ellipsoid_ring_width"):
            circle_width = widget.ellipsoid_ring_width.value()

        def render_axes():
            if hasattr(mw, "view_3d_manager") and hasattr(
                mw.view_3d_manager, "apply_3d_settings"
            ):
                try:
                    mw.view_3d_manager.apply_3d_settings(redraw=False)
                except Exception:
                    plotter.render()
            else:
                plotter.render()

        def apply_camera():
            if camera_state is not None:
                try:
                    plotter.camera_position = camera_state
                except Exception as e:
                    logging.debug("Failed to restore camera position: %s", e)
            else:
                try:
                    plotter.reset_camera()
                    if widget is not None:
                        widget._reset_camera_on_next_render = False
                except Exception as e:
                    logging.debug("Failed to reset camera: %s", e)

        def finalize_rendering(
            merged_ellipsoids, fallback_glyphs, h_glyphs, rings_mesh
        ):
            plotter.clear()
            plotter.set_background(bg_color)
            if is_lighting_enabled:
                import pyvista as pv

                light = pv.Light(
                    position=(1, 1, 2),
                    light_type="cameralight",
                    intensity=light_intensity,
                )
                plotter.add_light(light)

            if hasattr(mw, "view_3d_manager"):
                vm = mw.view_3d_manager
                if not hasattr(vm, "_3d_color_map"):
                    vm._3d_color_map = {}
                vm._3d_color_map.clear()
                for i, atom_color in enumerate(col):
                    atom_rgb = [int(c * 255) for c in atom_color]
                    vm._3d_color_map[f"atom_{i}"] = atom_rgb

                conf = mol.GetConformer()
                if hasattr(vm, "_add_3d_bond_cylinders"):
                    vm._add_3d_bond_cylinders(
                        mol, conf, col, "ball_and_stick", mesh_props
                    )

            if merged_ellipsoids is not None:
                plotter.add_mesh(
                    merged_ellipsoids,
                    scalars="colors",
                    rgb=True,
                    style="surface",
                    opacity=1.0,
                    name="cif_viewer_ellipsoids",
                    **mesh_props,
                )

            if fallback_glyphs is not None:
                plotter.add_mesh(
                    fallback_glyphs,
                    scalars="colors",
                    rgb=True,
                    style="surface",
                    opacity=1.0,
                    name="cif_viewer_fallback_atoms",
                    **mesh_props,
                )

            if h_glyphs is not None:
                plotter.add_mesh(
                    h_glyphs,
                    scalars="colors",
                    rgb=True,
                    style="surface",
                    opacity=1.0,
                    name="cif_viewer_h_atoms",
                    **mesh_props,
                )

            if rings_mesh is not None:
                plotter.add_mesh(
                    rings_mesh,
                    color=circle_color,
                    line_width=circle_width,
                    name="cif_viewer_ellipsoid_rings",
                    **mesh_props,
                )

            apply_camera()

            if widget is not None and hasattr(widget, "render_overlays_only"):
                try:
                    widget.render_overlays_only()
                except Exception as e:
                    logging.debug("Failed to call render_overlays_only: %s", e)

            is_testing = "PYTEST_CURRENT_TEST" in os.environ
            if is_testing:
                render_axes()
            else:
                try:
                    from PyQt6.QtCore import QTimer

                    QTimer.singleShot(50, render_axes)
                except Exception:
                    render_axes()

        if not use_threading:
            import pyvista as pv

            fallback_positions = []
            fallback_colors = []
            fallback_radii = []

            h_positions = []
            h_colors = []
            h_radii = []

            ellipsoid_meshes = []
            ellipsoid_circles_to_draw = []

            try:
                from rdkit import Chem

                pt = Chem.GetPeriodicTable()
            except ImportError:
                pt = None

            try:
                from moleditpy.constants import VDW_RADII
            except ImportError:
                try:
                    from moleditpy.utils.constants import VDW_RADII
                except ImportError:
                    VDW_RADII = {}

            for index, atom_info in enumerate(atoms_data):
                color_rgb = col[index] if index < len(col) else [0.5, 0.5, 0.5]
                pos = atom_info["pos"]
                symbol = atom_info["element"]
                cov = atom_info["cov"]
                has_cov = atom_info["has_cov"]

                if has_cov:
                    try:
                        eigenvalues, eigenvectors = np.linalg.eigh(cov)
                        eigenvalues = np.maximum(eigenvalues, 1e-5)
                        radii = np.sqrt(eigenvalues) * scale_factor

                        if np.linalg.det(eigenvectors) < 0:
                            eigenvectors = eigenvectors.copy()
                            eigenvectors[:, 0] = -eigenvectors[:, 0]

                        ellipsoid = pv.Sphere(
                            radius=1.0,
                            theta_resolution=resolution,
                            phi_resolution=resolution,
                        )
                        ellipsoid.scale([radii[0], radii[1], radii[2]], inplace=True)

                        rotation_matrix = np.eye(4)
                        rotation_matrix[:3, :3] = eigenvectors
                        ellipsoid.transform(rotation_matrix, inplace=True)
                        ellipsoid.translate(pos, inplace=True)

                        ellipsoid.compute_normals(
                            auto_orient_normals=True, inplace=True
                        )
                        ellipsoid.point_data["colors"] = np.tile(
                            color_rgb, (ellipsoid.n_points, 1)
                        )
                        ellipsoid_meshes.append(ellipsoid)

                        ellipsoid_circles_to_draw.append(
                            (index, radii, eigenvectors, pos)
                        )
                    except Exception as exc:
                        print(f"Error drawing ellipsoid for {symbol}: {exc}")
                else:
                    if symbol == "H":
                        h_vdw = 1.2
                        if pt is not None:
                            try:
                                h_vdw = pt.GetRvdw("H")
                            except Exception:
                                pass
                        rad = h_vdw * h_scale * atom_scale
                        h_positions.append(pos)
                        h_colors.append(color_rgb)
                        h_radii.append(rad)
                    else:
                        rad = VDW_RADII.get(symbol, 0.4) * atom_scale
                        fallback_positions.append(pos)
                        fallback_colors.append(color_rgb)
                        fallback_radii.append(rad)

            merged_ellipsoids = None
            if ellipsoid_meshes:
                merged_ellipsoids = pv.merge(ellipsoid_meshes)

            fallback_glyphs = None
            if fallback_positions:
                fallback_source = pv.PolyData(np.array(fallback_positions))
                fallback_source["colors"] = np.array(fallback_colors)
                fallback_source["radii"] = np.array(fallback_radii)
                fallback_glyphs = fallback_source.glyph(
                    scale="radii",
                    geom=pv.Sphere(
                        radius=1.0,
                        theta_resolution=resolution,
                        phi_resolution=resolution,
                    ),
                    orient=False,
                )

            h_glyphs = None
            if h_positions:
                h_source = pv.PolyData(np.array(h_positions))
                h_source["colors"] = np.array(h_colors)
                h_source["radii"] = np.array(h_radii)
                h_glyphs = h_source.glyph(
                    scale="radii",
                    geom=pv.Sphere(
                        radius=1.0,
                        theta_resolution=resolution,
                        phi_resolution=resolution,
                    ),
                    orient=False,
                )

            rings_mesh = None
            if show_rings and ellipsoid_circles_to_draw:
                all_ring_points = []
                all_ring_lines = []
                current_pt_idx = 0

                for index, radii, eigenvectors, pos in ellipsoid_circles_to_draw:
                    theta = np.linspace(0, 2 * np.pi, 37)
                    cos_t = np.cos(theta)
                    sin_t = np.sin(theta)
                    zero_t = np.zeros_like(theta)

                    circle_xy = np.column_stack((cos_t, sin_t, zero_t))
                    circle_yz = np.column_stack((zero_t, cos_t, sin_t))
                    circle_zx = np.column_stack((sin_t, zero_t, cos_t))

                    ring_radii = radii * 1.01

                    for base_circle in [circle_xy, circle_yz, circle_zx]:
                        pts = np.dot(base_circle * ring_radii, eigenvectors.T) + pos
                        all_ring_points.append(pts)

                        line_cells = np.empty(38, dtype=np.int32)
                        line_cells[0] = 37
                        line_cells[1:] = np.arange(
                            current_pt_idx, current_pt_idx + 37, dtype=np.int32
                        )
                        all_ring_lines.append(line_cells)

                        current_pt_idx += 37

                points_arr = np.concatenate(all_ring_points, axis=0)
                lines_arr = np.concatenate(all_ring_lines, axis=0)
                rings_mesh = pv.PolyData(points_arr, lines=lines_arr)

            finalize_rendering(merged_ellipsoids, fallback_glyphs, h_glyphs, rings_mesh)

        else:
            progress = None
            if widget is not None:
                try:
                    progress = QProgressDialog(
                        "Drawing thermal ellipsoids, please wait...", None, 0, 0, widget
                    )
                    progress.setWindowTitle("Drawing")
                    progress.setWindowModality(Qt.WindowModality.WindowModal)
                    progress.show()
                    QApplication.processEvents()
                except Exception:
                    progress = None

            def on_ready(
                merged_ellipsoids, fallback_glyphs, h_glyphs, rings_mesh, err_msg
            ):
                if progress is not None:
                    try:
                        progress.close()
                    except Exception:
                        pass
                if err_msg:
                    logging.error(
                        "Failed to generate ellipsoids in thread: %s", err_msg
                    )
                    return
                finalize_rendering(
                    merged_ellipsoids, fallback_glyphs, h_glyphs, rings_mesh
                )
                if hasattr(mw, "_ellipsoid_thread"):
                    mw._ellipsoid_thread = None

            mw._ellipsoid_thread = EllipsoidWorkerThread(
                atoms_data,
                resolution,
                scale_factor,
                h_scale,
                atom_scale,
                col,
                show_rings,
                circle_color,
                circle_width,
            )
            mw._ellipsoid_thread.result_ready.connect(on_ready)
            mw._ellipsoid_thread.start()

    context.add_menu_action("View/CIF Viewer Panel", open_from_menu)

    context.register_file_opener(".cif", open_file, priority=20)
    context.register_drop_handler(handle_drop, priority=20)
    context.register_document_reset_handler(handle_reset)
    context.register_3d_style("Thermal Ellipsoids", draw_ellipsoid_model)
