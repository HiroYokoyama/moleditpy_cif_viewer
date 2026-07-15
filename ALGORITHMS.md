# CIF Viewer: Core Algorithms & Architecture

This document explains the technical details of the core algorithms used in the CIF viewer, particularly focusing on the difficult-to-understand phases like periodic boundary unwrapping, topological polymer detection, and molecule reconstruction.

## 1. The Challenge of CIF Files

A Crystallographic Information File (CIF) typically only stores the **asymmetric unit**—the smallest unique fragment of the crystal—and a set of symmetry operations. 
To visualize the crystal, the software must:
1. Apply the symmetry operations to generate the full unit cell.
2. Determine which atoms are bonded to each other.
3. For discrete molecules (like benzene), reassemble the pieces that have been cut apart by the periodic boundaries.
4. For infinite frameworks (like MOFs or COFs), realize that the structure extends infinitely and cannot be "reassembled" into a finite molecule.

## 2. Reconstructing Discrete Molecules (`grow_molecules`)

When the user selects **"Whole Molecule"** mode for a molecular crystal, the goal is to show one complete, unbroken molecule, even if the raw coordinates place half of the molecule on the left side of the unit cell and the other half on the right side.

The algorithm proceeds in four steps:

### Step 1: Symmetry Expansion
We apply all crystallographic symmetry operations to the asymmetric unit. Every resulting fractional coordinate is mathematically wrapped into the $[0, 1)$ range using modulo 1.0. This ensures all atoms are mathematically inside a single unit cell box.
*Duplicates are removed using a fast spatial distance check.*

### Step 2: Periodic Adjacency Inference
We use RDKit to determine covalent bonds. Because atoms are wrapped inside the $[0, 1)$ box, two bonded atoms might appear to be on opposite sides of the cell (e.g., $x=0.01$ and $x=0.99$).
To find these bonds, we calculate the **minimum image distance**. If the distance is within the covalent radius threshold, we record a bond along with the periodic **shift** vector required to traverse it (e.g., a shift of `[-1, 0, 0]`).

### Step 3: BFS Topological Unwrapping (The Core Engine)
At this point, we have a graph where nodes are atoms inside the $[0,1)$ box, and edges are bonds that carry a periodic `shift`.
To reconstruct a contiguous molecule in Cartesian space, we use a **Breadth-First Search (BFS)**:
1. Start at an unvisited atom, assign it a Cartesian offset of `[0, 0, 0]`.
2. Follow bonds to neighboring atoms.
3. When reaching a neighbor, assign its offset as: `neighbor_offset = current_offset + bond_shift`.
4. Because we track which atoms have been visited, we only assign an offset to each atom **once**.

This "unwraps" the molecule. When we render the atoms, we add the computed integer offset to their fractional coordinates. The result is a perfectly contiguous molecule in 3D space, completely ignoring the arbitrary cell boundaries.

### Step 4: Filtering
To avoid rendering duplicate molecules, we only keep molecules that physically overlap the original unit cell box (i.e., at least one of their unwrapped atoms has an offset of `[0, 0, 0]`).

## 3. Detecting Frameworks & Polymers (`is_polymer_structure`)

A critical problem arises with infinite frameworks like Covalent Organic Frameworks (COFs), Metal-Organic Frameworks (MOFs), or 1D/2D coordination polymers. 
Because they are infinite, trying to "unwrap" them into a single discrete molecule is mathematically impossible. If we apply the BFS unwrapping to a polymer, the algorithm would technically need to run infinitely. The old code avoided infinite loops by artificially truncating the BFS, which resulted in visually broken, jagged fragments.

To solve this, the plugin detects polymers automatically and falls back to **"Asymmetric Unit"** or **"Packing"** mode.

### The Topological Ring-Closure Check
How do we mathematically distinguish a packed molecular crystal from a polymer? 
* **Wrong approach**: "If any bond crosses the unit cell boundary, it's a polymer." 
  * *Why it fails*: In a densely packed crystal of discrete molecules, atoms from *different* molecules might be close to each other across a cell boundary. The adjacency graph will contain a cross-boundary bond (an intermolecular contact), but the material is not a polymer.
* **Correct approach**: Topological Ring Closure.

A structure is an infinite polymer if and only if you can travel along covalent bonds, form a closed loop (a ring), but end up in a **different unit cell**. 

We implement this in `is_polymer_structure` using BFS:
1. We fully expand the unit cell (Step 1 above).
2. We run the same BFS offset-accumulation algorithm.
3. When the BFS encounters an atom it has **already visited** (a ring-closing bond), we check for consistency:
   `if offsets[neighbor] != offsets[current] + bond_shift:`
   
If the accumulated offset from the new path does not match the previously assigned offset, it means the bond ring wrapped completely around the periodic boundary of the crystal. **The structure is topologically infinite.** We immediately return `True` (Polymer detected).

If the BFS finishes and all rings close perfectly with consistent offsets, the structure consists only of discrete, finite molecules.

## 4. Supercell Expansion (`expand_supercell`)

When the user selects **"Packing"** mode (or when a polymer is rendered), we simply want to tile the unit cell in 3D space.
Unlike `grow_molecules`, `expand_supercell` does not attempt to unwrap or track offsets. 
It simply:
1. Takes the mathematically wrapped unit cell.
2. Clones it across a grid (e.g., $2 \times 2 \times 2$).
3. Renders bonds strictly between atoms that are physically close in that specific expanded grid.
4. It clips bonds at the boundary of the supercell, leaving "dangling" edges, which is standard practice for viewing infinite crystal lattices.

### Decimal (partial) repeats

Repeat counts may be non-integer (e.g. $1.5 \times 1 \times 1$). In that case the grid is tiled $\lceil n \rceil$ times along each axis, and afterwards a **geometric slab** is applied: along every *non-integer* axis, an atom is discarded unless its *drawn* fractional coordinate (i.e. its final position, including any "keep connected" unwrap shift, plus the cell-index offset) lies within $[0, n]$ (boundary-inclusive, $\varepsilon = 10^{-6}$). Cropping the drawn — not the wrapped — position, and clipping **both** faces, is what makes a partial cell render as a literal slab of real space: a connected chain never protrudes behind the origin or past the far face. A repeat below 1 (e.g. $0.5$) therefore renders half of the cell's real-space slab.

Integer axes are never cropped (`crop_axes` is empty), so integer repeats — including the presets — reproduce the exact classic behavior and keep molecules whole. Only decimals slab space, which will slice through any molecule straddling a slab face; this is intentional and matches the "partial structure" semantics. Note that with "keep connected" on, unwrapping can shift an entire discrete molecule out of the $[0,1)$ box, so a sub-unit slab may legitimately contain none of it (turn "keep connected" off to slab the wrapped cell contents instead). The CIF exporter applies the same both-face slab and scales the exported cell lengths by the decimal repeat.
