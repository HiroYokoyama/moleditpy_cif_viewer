from __future__ import annotations

import logging
from typing import Sequence, Tuple


def render_atoms_to_rdkit_mol(
    render_atoms, bonds: Sequence[Tuple[int, int]], determine_bond_order: bool = False
):
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    rw_mol = Chem.RWMol()
    for atom in render_atoms:
        symbol = "H" if atom.element in ("D", "T") else atom.element
        rd_atom = Chem.Atom(symbol)
        if atom.element in ("D", "T"):
            rd_atom.SetIsotope(2 if atom.element == "D" else 3)
        rw_mol.AddAtom(rd_atom)

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

    if determine_bond_order:
        try:
            from rdkit.Chem import rdDetermineBonds

            # Determine bond orders in place (charge=0 by default)
            rdDetermineBonds.DetermineBondOrders(mol, charge=0)
            mol.SetProp("_bond_order_error", "")
        except Exception as exc:
            logging.warning("RDKit DetermineBondOrders failed: %s", exc)
            mol.SetProp("_bond_order_error", str(exc))
    else:
        mol.SetProp("_bond_order_error", "")

    return mol
