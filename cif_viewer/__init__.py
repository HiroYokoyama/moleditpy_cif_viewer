PLUGIN_NAME = "CIF Viewer"
PLUGIN_VERSION = "0.2.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "Visualization-only CIF crystal structure viewer with unit-cell and "
    "supercell rendering for MoleditPy."
)
PLUGIN_DEPENDENCIES = ["numpy", "pymatgen", "PyQt6", "pyvista", "rdkit"]

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

    def draw_ellipsoid_model(mw, mol):
        plotter = getattr(mw, "plotter", None)
        if plotter is None and hasattr(mw, "view_3d_manager"):
            plotter = mw.view_3d_manager.plotter
        if plotter is None:
            return

        dock = context.get_window(WINDOW_ID) if hasattr(context, "get_window") else None
        widget = dock.widget() if dock is not None else None
        
        has_adp = False
        if widget is not None and getattr(widget, "structure", None) is not None and widget.structure.u_cart is not None:
            if hasattr(widget, "last_rendered_atoms") and widget.last_rendered_atoms:
                has_adp = True

        if not has_adp:
            context.show_status_message("No ADP data in current CIF. Falling back to Ball & Stick.", 4000)
            if hasattr(mw, "view_3d_manager") and hasattr(mw.view_3d_manager, "set_3d_style"):
                mw.view_3d_manager.set_3d_style("ball_and_stick")
            return

        plotter.clear()
        
        if hasattr(mw, "init_manager") and hasattr(mw.init_manager, "settings"):
            plotter.set_background(mw.init_manager.settings.get("background_color", "#919191"))
            is_lighting_enabled = mw.init_manager.settings.get("lighting_enabled", True)
            if is_lighting_enabled:
                import pyvista as pv
                light = pv.Light(
                    position=(1, 1, 2),
                    light_type="cameralight",
                    intensity=mw.init_manager.settings.get("light_intensity", 1.2),
                )
                plotter.add_light(light)

        try:
            from moleditpy.constants import CPK_COLORS_PV as CPK_COLORS, VDW_RADII, pt
        except ImportError:
            try:
                from moleditpy.utils.constants import CPK_COLORS_PV as CPK_COLORS, VDW_RADII, pt
            except ImportError:
                CPK_COLORS = {}
                VDW_RADII = {}
                try:
                    from rdkit import Chem
                    pt = Chem.GetPeriodicTable()
                except ImportError:
                    pt = None
                
        import numpy as np
        import pyvista as pv
        import re
        import math
        
        sym = [a.GetSymbol() for a in mol.GetAtoms()]
        col = np.array([CPK_COLORS.get(s, [0.5, 0.5, 0.5]) for s in sym])
        
        mesh_props = dict(
            smooth_shading=True,
            specular=mw.init_manager.settings.get("specular", 0.2) if hasattr(mw, "init_manager") else 0.2,
            specular_power=mw.init_manager.settings.get("specular_power", 20) if hasattr(mw, "init_manager") else 20,
            lighting=mw.init_manager.settings.get("lighting_enabled", True) if hasattr(mw, "init_manager") else True,
        )
        
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
                vm._add_3d_bond_cylinders(mol, conf, col, "ball_and_stick", mesh_props)

        def get_scale_factor(p_percent):
            p = p_percent / 100.0
            if p <= 0 or p >= 1: return 1.5382
            def cdf(c): return math.erf(c / math.sqrt(2)) - math.sqrt(2 / math.pi) * c * math.exp(-c**2 / 2)
            low, high = 0.0, 10.0
            for _ in range(50):
                mid = (low + high) / 2
                if cdf(mid) < p: low = mid
                else: high = mid
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

        h_scale = 0.3
        if widget is not None and hasattr(widget, "h_scale_spin"):
            h_scale = widget.h_scale_spin.value() / 100.0
        
        atom_scale = 1.0
        resolution = 16
        if hasattr(mw, "init_manager") and hasattr(mw.init_manager, "settings"):
            atom_scale = mw.init_manager.settings.get("ball_stick_atom_scale", 1.0)
            resolution = mw.init_manager.settings.get("ball_stick_resolution", 16)

        fallback_positions = []
        fallback_colors = []
        fallback_radii = []
        
        h_positions = []
        h_colors = []
        h_radii = []

        for index, atom in enumerate(widget.last_rendered_atoms):
            if index >= mol.GetNumAtoms():
                continue
            
            color_rgb = [float(x) for x in col[index]] if index < len(col) else [0.5, 0.5, 0.5]
            pos = list(mol.GetConformer().GetAtomPosition(index))
            symbol = atom.element
            
            base_idx = atom.base_index
            cov = None
            if base_idx < len(widget.structure.u_cart):
                cov = widget.structure.u_cart[base_idx]
                
            has_cov = cov is not None and not np.allclose(cov, 0.0)
            
            if has_cov:
                try:
                    eigenvalues, eigenvectors = np.linalg.eig(cov)
                    eigenvalues = np.maximum(eigenvalues, 1e-5)
                    radii = np.sqrt(eigenvalues) * scale_factor
                    
                    ellipsoid = pv.Sphere(
                        radius=1.0,
                        theta_resolution=resolution,
                        phi_resolution=resolution
                    )
                    ellipsoid.scale([radii[0], radii[1], radii[2]], inplace=True)
                    
                    rotation_matrix = np.eye(4)
                    rotation_matrix[:3, :3] = eigenvectors
                    ellipsoid.transform(rotation_matrix, inplace=True)
                    ellipsoid.translate(pos, inplace=True)
                    
                    ellipsoid.compute_normals(inplace=True)
                    
                    name = f"cif_viewer_ellipsoid_{index}"
                    plotter.add_mesh(
                        ellipsoid,
                        color=color_rgb,
                        opacity=1.0,
                        name=name,
                        style='surface',
                        **mesh_props
                    )
                    
                    show_rings = True
                    if widget is not None and hasattr(widget, "show_ellipsoid_rings"):
                        show_rings = widget.show_ellipsoid_rings.isChecked()
                    
                    if show_rings:
                        theta = np.linspace(0, 2 * np.pi, 37)
                        cos_t = np.cos(theta)
                        sin_t = np.sin(theta)
                        zero_t = np.zeros_like(theta)
                        
                        circle_xy = np.column_stack((cos_t, sin_t, zero_t))
                        circle_yz = np.column_stack((zero_t, cos_t, sin_t))
                        circle_zx = np.column_stack((sin_t, zero_t, cos_t))
                        
                        ring_radii = radii * 1.01
                        
                        for i, base_circle in enumerate([circle_xy, circle_yz, circle_zx]):
                            pts = np.dot(base_circle * ring_radii, eigenvectors.T) + pos
                            
                            segments = np.empty((2 * (len(pts) - 1), 3))
                            segments[0::2] = pts[:-1]
                            segments[1::2] = pts[1:]
                            
                            line_name = f"cif_viewer_ellipsoid_circle_{index}_{i}"
                            plotter.add_lines(
                                segments,
                                color="black",
                                width=2,
                                name=line_name
                            )
                except Exception as exc:
                    print(f"Error drawing ellipsoid for {atom.element}: {exc}")
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

        if fallback_positions:
            fallback_source = pv.PolyData(np.array(fallback_positions))
            fallback_source["colors"] = np.array(fallback_colors)
            fallback_source["radii"] = np.array(fallback_radii)
            f_glyphs = fallback_source.glyph(
                scale="radii",
                geom=pv.Sphere(radius=1.0, theta_resolution=resolution, phi_resolution=resolution),
                orient=False,
            )
            plotter.add_mesh(f_glyphs, scalars="colors", rgb=True, style='surface', opacity=1.0, name="cif_viewer_fallback_atoms", **mesh_props)

        if h_positions:
            h_source = pv.PolyData(np.array(h_positions))
            h_source["colors"] = np.array(h_colors)
            h_source["radii"] = np.array(h_radii)
            h_glyphs = h_source.glyph(
                scale="radii",
                geom=pv.Sphere(radius=1.0, theta_resolution=resolution, phi_resolution=resolution),
                orient=False,
            )
            plotter.add_mesh(h_glyphs, scalars="colors", rgb=True, style='surface', opacity=1.0, name="cif_viewer_h_atoms", **mesh_props)

        plotter.render()

    context.add_menu_action("View/CIF Viewer Panel", open_from_menu)

    if hasattr(context, "register_file_opener"):
        context.register_file_opener(".cif", open_file, priority=20)
    if hasattr(context, "register_drop_handler"):
        context.register_drop_handler(handle_drop, priority=20)
    if hasattr(context, "register_document_reset_handler"):
        context.register_document_reset_handler(handle_reset)
    if hasattr(context, "register_3d_style"):
        context.register_3d_style("Thermal Ellipsoids", draw_ellipsoid_model)
