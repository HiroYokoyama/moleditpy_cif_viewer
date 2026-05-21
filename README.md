# MoleditPy CIF Viewer (v0.3.0)

A visualization-only, high-performance crystal structure viewer plugin for [MoleditPy](https://github.com/HiroYokoyama/python_molecular_editor). 

It allows researchers and developers to load CIF files, generate supercells, customize rendering styles, view along crystallographic axes, and display anisotropic displacement parameters (Thermal Ellipsoids) with extensive styling options—all while leaving the host application's active session and undo history completely unmodified.

---

## Key Features

### 1. Rendering Optimization (The "Render Once" Strategy)
- **High Performance**: Renders large structures smoothly by avoiding individual actor creation/additions. 
- **Single Mesh Merging**: Compiles individual ellipsoid spheres, scales, rotates, translates, and colors them on the point level, and merges them using `pyvista.merge` into a single `pv.PolyData` mesh for a single draw call.
- **Single Lines Pre-Assembly**: Assembly of all thermal ellipsoid rings/hoops into a single polydata line mesh.
- **Debounced Rendering**: Employs a 50ms `QTimer` to debounce rendering calls when adjusting settings, avoiding redundant redraw cycles.

### 2. Restructured Tabbed Control Panel
The control panel docks in the main window (**View > CIF Viewer Panel**) and is organized into four intuitive tabs:

*   **Structure**: Open CIF files, select structures from multi-structure files, view summary data (atoms, inferred bonds), and export the expanded supercell structure to a new CIF file.
*   **Supercell**: Define unit cell repetitions ($a$, $b$, and $c$), toggle periodic-boundary molecular connectivity, toggle bond drawing, filter hydrogens, reset repetition counts, and apply presets.
*   **Ellipsoids**: Configure thermal ellipsoids probability, hydrogen scaling, circle outlines, custom circle colors, and line widths. Includes a style shortcut to apply the view instantly.
*   **Cell / Axes**: Show or hide the unit cell edges and a/b/c axes. Customize colors for individual axes, cell edges, the origin sphere, and camera view buttons.

### 3. Smart Supercell Reset & Presets
- **Auto-Reset on Load**: When loading a new CIF file, the repetition values are automatically reset to `1 x 1 x 1` to prevent accidental slowdowns from rendering large supercells of complex new structures.
- **Preset Buttons**: Quick-select buttons for `2x2x2` and `3x3x3` supercells.

### 4. Crystallographic Axis Views
- A 2x3 grid of buttons (`a`, `b`, `c`, `-a`, `-b`, `-c`) under the **Cell / Axes** tab.
- Re-orients the 3D camera to look directly along the selected crystallographic axis towards the cell center.
- Automatically adjusts the screen's vertical orientation (`viewup` vector) to match crystallographic standards:
  - $c$-axis is up for $a$- and $b$-views.
  - $b$-axis is up for $c$-views.

### 5. Advanced Thermal Ellipsoids Styling
- **Probability Control**: Specify the ellipsoid displacement probability percentage (e.g. 50% probability boundary).
- **Custom Rings**: Turn circle/ring outlines on or off. Customize ring colors and line widths.
- **Hydrogen Controls**: Toggle the rendering of hydrogen atoms. Opt to keep hydrogen sizes fixed at a constant scale (expressed as a percentage of VDW radius, default 20%) to avoid drawing oversized ADPs.

### 6. Seamless Integration & Style Menu Sync
- **Host Sync**: Pressing "Switch to Ellipsoids Style" in the plugin changes the main app's 3D style to "Thermal Ellipsoids" and automatically checks the corresponding option in the main toolbar's 3D Style dropdown.
- **Rendering Safeguards**: Tagged molecules prevent custom ellipsoid shaders and axis overlays from being drawn on standard non-CIF molecules, falling back gracefully to the Ball & Stick style.

---

## Installation

### Standard Installation
1.  **Download**: Obtain the plugin package from the MoleditPy Plugin Explorer.
2.  **Install**: Place the `cif_viewer` folder into your local MoleditPy plugins directory (typically under `~/.moleditpy/plugins/`).
3.  **Run**: Open MoleditPy and navigate to **View > CIF Viewer Panel**.

### Developer Installation
Install the plugin in editable mode to link it directly within your Python environment:

```bash
pip install -e .
```

---

## Project Structure

```
moleditpy_cif_viewer/
│
├── cif_viewer/
│   ├── __init__.py      # Plugin entry point & style drawing callback
│   ├── parser.py        # CIF parsing (pymatgen & fallback), supercell generator, & exporter
│   ├── rdkit_bridge.py  # Bridges atoms/bonds to RDKit molecules with custom tagging
│   └── viewer.py        # Main dock widget panel layout, camera rotation, & settings persistence
│
├── tests/
│   ├── test_parser.py             # Exporter & parser tests
│   ├── test_plugin_integration.py # Mock GUI, settings, view-from-axis, & checkmark sync tests
│   └── test_pymatgen_parser.py    # Multi-structure pymatgen test coverage
│
├── pyproject.toml       # Project metadata & build configuration
├── run_tests.py         # Test environment initialization runner
└── test_all.py          # Unified test runner with headless fallback
```

---

## Verification & Testing

The plugin contains a comprehensive unit and integration test suite running headlessly.

```bash
# Run the full test suite
python test_all.py

# Run only unit/parser tests (skipping PyQt6 GUI/integration tests)
python test_all.py --unit-only
```

### Dependencies
- `numpy`
- `pymatgen`
- `PyQt6`
- `pyvista`
- `rdkit`
