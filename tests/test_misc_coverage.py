"""Coverage extras for cif_viewer.rdkit_bridge."""

import numpy as np

from cif_viewer.rdkit_bridge import render_atoms_to_rdkit_mol
from cif_viewer.parser import RenderAtom


def test_render_atoms_to_rdkit_mol_dedups_duplicate_bonds():
    atoms = [
        RenderAtom("C1", "C", 0, (0, 0, 0), np.array([0.0, 0.0, 0.0])),
        RenderAtom("C2", "C", 1, (0, 0, 0), np.array([1.5, 0.0, 0.0])),
    ]
    # (0, 1) and (1, 0) are the same bond -- the second occurrence must be
    # skipped rather than raising a "bond already exists" RDKit error.
    bonds = [(0, 1), (1, 0), (0, 1)]
    mol = render_atoms_to_rdkit_mol(atoms, bonds)
    assert mol.GetNumBonds() == 1
