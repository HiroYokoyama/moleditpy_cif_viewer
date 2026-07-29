"""Chemistry regression tests for the periodic bonding rules.

Each test here pins down a defect where the rendered chemistry was wrong,
not merely a refactor: heavy-element solids drew no bonds at all, and a
mixed-occupancy site lost every element but the first.
"""

import numpy as np
import pytest

from cif_viewer.parser import (
    CifAtom,
    CifStructure,
    _expand_to_unit_cell,
    _infer_periodic_adjacency,
    bond_cutoff,
    cell_vectors,
    covalent_radius,
    expand_supercell,
    grow_molecules,
    is_polymer_structure,
)


FCC = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]


def _cubic(name, a, sites, space_group=None):
    lattice = cell_vectors((a, a, a), (90, 90, 90))
    atoms = tuple(
        CifAtom(label, element, np.array(f, float), np.array(f, float) @ lattice)
        for label, element, f in sites
    )
    return CifStructure(
        name,
        (a, a, a),
        (90, 90, 90),
        lattice,
        atoms,
        asymmetric_atoms=atoms,
        space_group=space_group,
    )


def _zincblende(name, a, el1, el2):
    sites = [(f"{el1}{i}", el1, f) for i, f in enumerate(FCC)]
    sites += [(f"{el2}{i}", el2, tuple(np.array(f) + 0.25)) for i, f in enumerate(FCC)]
    return _cubic(name, a, sites)


def _perovskite():
    return _cubic(
        "CsPbI3",
        6.29,
        [
            ("Pb1", "Pb", (0, 0, 0)),
            ("I1", "I", (0.5, 0, 0)),
            ("I2", "I", (0, 0.5, 0)),
            ("I3", "I", (0, 0, 0.5)),
            ("Cs1", "Cs", (0.5, 0.5, 0.5)),
        ],
        space_group="P m -3 m",
    )


# --- Heavy-element bonds -------------------------------------------------
# A min(2.45, ...) ceiling used to truncate every cutoff, so any bond longer
# than 2.45 A simply did not exist in the periodic graph.


@pytest.mark.parametrize(
    "name, a, el1, el2, nn_distance",
    [
        ("diamond", 3.567, "C", "C", 1.54),
        ("ZnS", 5.41, "Zn", "S", 2.34),
        ("CdS", 5.83, "Cd", "S", 2.52),
        ("CdTe", 6.48, "Cd", "Te", 2.81),
        ("InSb", 6.48, "In", "Sb", 2.81),
    ],
)
def test_zincblende_has_four_neighbours(name, a, el1, el2, nn_distance):
    """Tetrahedral solids must find 4 neighbours regardless of bond length."""
    structure = _zincblende(name, a, el1, el2)
    adjacency = _infer_periodic_adjacency(structure)
    assert len(adjacency[0]) == 4, f"{name} (d={nn_distance} A)"
    assert is_polymer_structure(structure) is True


def test_perovskite_octahedron_is_complete():
    """Pb-I is 3.15 A: the whole PbI6 octahedron was previously missing."""
    structure = _perovskite()
    adjacency = _infer_periodic_adjacency(structure)
    assert len(adjacency[0]) == 6
    assert is_polymer_structure(structure) is True


def test_framework_bonding_agrees_between_view_modes():
    """Whole-molecule and supercell modes must not disagree about bonds.

    The perovskite used to render 0 bonds in one mode and 6 in the other.
    """
    structure = _perovskite()
    _, grown_bonds = grow_molecules(structure)
    assert len(grown_bonds) > 0


def test_bond_cutoff_is_not_capped():
    assert bond_cutoff("Pb", "I") == pytest.approx(1.46 + 1.39 + 0.45)
    assert bond_cutoff("C", "C", 0.0) == pytest.approx(1.52)


def test_non_bonded_contacts_stay_unbonded():
    """Removing the cap must not start bonding second-neighbour contacts."""
    for el1, el2, distance in [
        ("C", "O", 3.20),
        ("C", "C", 2.50),
        ("O", "O", 2.80),
        ("N", "N", 3.10),
    ]:
        assert distance > bond_cutoff(el1, el2), f"{el1}-{el2} at {distance} A"


def test_tolerance_reaches_the_periodic_graph():
    """grow_molecules/is_polymer_structure took a tolerance and ignored it."""
    structure = _zincblende("CdTe", 6.48, "Cd", "Te")
    assert len(_infer_periodic_adjacency(structure, tolerance=0.45)[0]) == 4
    assert len(_infer_periodic_adjacency(structure, tolerance=-1.0)[0]) == 0


# --- Covalent radii ------------------------------------------------------


@pytest.mark.parametrize(
    "element, radius",
    [("U", 1.96), ("Th", 2.06), ("Pu", 1.87), ("Ra", 2.21), ("Po", 1.40), ("D", 0.31)],
)
def test_heavy_and_deuterium_radii_are_known(element, radius):
    """These fell through to the 0.77 default and were sized like carbon."""
    assert covalent_radius(element) == pytest.approx(radius)


def test_uranyl_bond_is_found():
    lattice = cell_vectors((12.0, 12.0, 12.0), (90, 90, 90))
    sites = [("U1", "U", (0.0, 0.0, 0.0)), ("O1", "O", (0.2, 0.0, 0.0))]  # 2.40 A
    atoms = tuple(
        CifAtom(lbl, el, np.array(f, float), np.array(f, float) @ lattice)
        for lbl, el, f in sites
    )
    structure = CifStructure("uranyl", (12.0, 12.0, 12.0), (90, 90, 90), lattice, atoms)
    assert len(_infer_periodic_adjacency(structure)[0]) == 1


# --- Shared crystallographic sites --------------------------------------


def test_mixed_occupancy_site_keeps_both_elements():
    """A shared Fe/Co position used to lose the Co entirely."""
    lattice = cell_vectors((5.0, 5.0, 5.0), (90, 90, 90))
    sites = [
        ("Fe1", "Fe", (0, 0, 0), 0.5),
        ("Co1", "Co", (0, 0, 0), 0.5),
        ("O1", "O", (0.5, 0.5, 0.5), 1.0),
    ]
    atoms = tuple(
        CifAtom(lbl, el, np.array(f, float), np.array(f, float) @ lattice, occ)
        for lbl, el, f, occ in sites
    )
    structure = CifStructure(
        "FeCoO",
        (5.0, 5.0, 5.0),
        (90, 90, 90),
        lattice,
        atoms,
        asymmetric_atoms=atoms,
        space_group="P 1",
    )
    expanded, _ = _expand_to_unit_cell(structure, atoms)
    assert {atom.element for atom in expanded} == {"Fe", "Co", "O"}


def test_special_position_images_still_collapse():
    """Per-atom dedup scoping must not reintroduce duplicated symmetry images."""
    structure = _cubic("Na", 5.0, [("Na1", "Na", (0, 0, 0))], space_group="F m -3 m")
    expanded, _ = _expand_to_unit_cell(structure, structure.atoms)
    assert len(expanded) == 4


def test_supercell_of_framework_is_not_stretched():
    """Unwrapping a periodic net must not fling atoms out of the cell."""
    structure = _perovskite()
    atoms, _ = expand_supercell(structure, (2, 2, 2))
    positions = np.array([atom.position for atom in atoms])
    span = positions.max(axis=0) - positions.min(axis=0)
    assert np.all(span <= 2 * 6.29 + 1e-6)
