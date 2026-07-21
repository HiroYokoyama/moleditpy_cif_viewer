"""Additional coverage tests for cif_viewer.parser, targeting metadata
extraction edge branches, RenderAtom.disorder_key, and early-return /
disorder-filter branches that tests/test_parser.py and
tests/test_pymatgen_parser.py don't exercise."""

import numpy as np

from cif_viewer.parser import (
    CifAtom,
    CifStructure,
    RenderAtom,
    _expand_to_unit_cell,
    _extract_metadata,
    is_polymer_structure,
)


def test_render_atom_disorder_key_without_assembly():
    atom = RenderAtom(
        "C1", "C", 0, (0, 0, 0), np.array([0.0, 0.0, 0.0]), disorder_group="1"
    )
    assert atom.disorder_key == "1"


def test_render_atom_disorder_key_none_without_group():
    atom = RenderAtom("C1", "C", 0, (0, 0, 0), np.array([0.0, 0.0, 0.0]))
    assert atom.disorder_key is None


def _get_val(data):
    return lambda keys: next((data[k] for k in keys if k in data), None)


def test_extract_metadata_crystal_size_max_only():
    meta = _extract_metadata(_get_val({"_exptl_crystal_size_max": "0.3"}))
    assert meta["crystal_size"] == "0.3"


def test_extract_metadata_theta_range_max_only():
    meta = _extract_metadata(_get_val({"_diffrn_reflns_theta_max": "27.5"}))
    assert meta["theta_range"] == "up to 27.5"


def test_extract_metadata_diff_peak_hole_max_only():
    meta = _extract_metadata(_get_val({"_refine_diff_density_max": "0.45"}))
    assert meta["diff_peak_hole"] == "0.45"


def test_extract_metadata_no_values_all_none():
    meta = _extract_metadata(_get_val({}))
    assert meta["crystal_size"] is None
    assert meta["theta_range"] is None
    assert meta["hkl_ranges"] is None
    assert meta["diff_peak_hole"] is None
    assert meta["z_prime"] is None


def test_extract_metadata_z_prime_calculated_integer():
    meta = _extract_metadata(
        _get_val({"_cell_formula_units_z": "4"}), num_symops=4
    )
    assert meta["z_prime"] == "1"


def test_extract_metadata_z_prime_calculated_fractional():
    meta = _extract_metadata(
        _get_val({"_cell_formula_units_z": "3"}), num_symops=4
    )
    assert meta["z_prime"] == "0.75"


def test_extract_metadata_z_prime_calc_exception_leaves_none():
    # z is not numeric -> parse_cif_number raises, exception is swallowed.
    meta = _extract_metadata(
        _get_val({"_cell_formula_units_z": "not-a-number"}), num_symops=4
    )
    assert meta["z_prime"] is None


def test_extract_metadata_hkl_ranges_present():
    meta = _extract_metadata(
        _get_val(
            {
                "_diffrn_reflns_limit_h_min": "-5",
                "_diffrn_reflns_limit_h_max": "5",
                "_diffrn_reflns_limit_k_min": "-6",
                "_diffrn_reflns_limit_k_max": "6",
                "_diffrn_reflns_limit_l_min": "-7",
                "_diffrn_reflns_limit_l_max": "7",
            }
        )
    )
    assert meta["hkl_ranges"] == "h: -5/5, k: -6/6, l: -7/7"


def _flat_structure(n=2, disorder_group=None):
    lattice = np.eye(3) * 10.0
    atoms = tuple(
        CifAtom(
            f"C{i}",
            "C",
            np.array([i / 10.0, 0.0, 0.0]),
            np.array([float(i), 0.0, 0.0]),
            disorder_group=disorder_group,
        )
        for i in range(n)
    )
    return CifStructure(
        "flat", (10.0, 10.0, 10.0), (90.0, 90.0, 90.0), lattice, atoms
    )


def test_is_polymer_structure_no_core_atoms_returns_false():
    struct = _flat_structure(0)
    assert is_polymer_structure(struct) is False


def test_expand_to_unit_cell_filters_by_disorder_key():
    struct = _flat_structure(2, disorder_group="1")
    other_atom = CifAtom(
        "C2",
        "C",
        np.array([0.5, 0.0, 0.0]),
        np.array([5.0, 0.0, 0.0]),
        disorder_group="2",
    )
    struct = CifStructure(
        struct.name,
        struct.cell_lengths,
        struct.cell_angles,
        struct.lattice,
        struct.atoms + (other_atom,),
    )
    exp_atoms, _ = _expand_to_unit_cell(
        struct, struct.atoms, selected_disorder_key="1"
    )
    assert all(a.disorder_group in (None, "1") for a in exp_atoms)
    assert any(a.label == "C0" for a in exp_atoms)
    assert not any(a.label == "C2" and a.disorder_group == "2" for a in exp_atoms)
