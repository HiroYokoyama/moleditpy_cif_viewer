# MoleditPy CIF Viewer

Visualization-only CIF crystal structure viewer plugin for MoleditPy.

## Features

- Opens a docked control panel from **View > CIF Viewer Panel**.
- Registers `.cif` as a MoleditPy file opener for **File > Import** and command-line file loading.
- Reads CIF with ASE first, matching CellEditPy's completed unit-cell import path when ASE is available.
- Displays atoms and bonds through MoleditPy's native 3D molecule renderer.
- Overlays CellEditPy-style unit-cell axes: red `a`, green `b`, blue `c`, with remaining cell edges in white.
- Provides right-side supercell controls for `a`, `b`, and `c` directions.
- Keeps the displayed cell axes at the original unit-cell size while supercell atoms are expanded.
- Keeps periodic-boundary molecules connected by default, even when connected atoms need to appear outside the original unit cell.
- Provides axis width, label font, and label size controls plus a one-click supercell reset.
- Pushes the rendered CIF molecule into MoleditPy's current 3D molecule so the host export tools can use it.
- Enters MoleditPy's 3D viewer mode when a CIF is opened, minimizing the 2D editor if the host exposes that UI hook.
- Keeps the host molecule untouched: no editing, no import into the active molecule, and no undo mutation.
- Supports CIF drag/drop and file-open registration when the host exposes those plugin APIs.

## Installation

Copy the `cif_viewer` folder into MoleditPy's plugin directory, or install this repository in editable mode:

```bash
pip install -e .
```

## Development

Run the parser and plugin-contract tests:

```bash
python -m pytest tests/ -v
```

The tests avoid PyQt6/PyVista imports so the CIF parsing and PluginContext contract can be checked in a headless environment.
