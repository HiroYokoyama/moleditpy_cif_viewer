from __future__ import annotations

from typing import Sequence, Tuple


def render_atoms_to_rdkit_mol(render_atoms, bonds: Sequence[Tuple[int, int]]):
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    rw_mol = Chem.RWMol()
    for atom in render_atoms:
        rw_mol.AddAtom(Chem.Atom(atom.element))

    seen = set()
    for left, right in bonds:
        key = tuple(sorted((int(left), int(right))))
        if key in seen:
            continue
        seen.add(key)
        rw_mol.AddBond(key[0], key[1], Chem.BondType.SINGLE)

    mol = rw_mol.GetMol()
    conformer = Chem.Conformer(len(render_atoms))
    for index, atom in enumerate(render_atoms):
        x, y, z = [float(value) for value in atom.position]
        conformer.SetAtomPosition(index, Point3D(x, y, z))
    mol.AddConformer(conformer, assignId=True)
    return mol

