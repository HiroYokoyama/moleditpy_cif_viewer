from __future__ import annotations

from dataclasses import dataclass
import math
import re
import shlex
import logging
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# pymatgen's SpaceGroup (a monty Singleton) emits a UserWarning when a CIF
# uses a non-standard Hermann-Mauguin symbol (e.g. 'I1a1').  The warning is
# cosmetic: pymatgen falls back to the short symbol and parsing continues.
# Suppress it globally so it never reaches the user's console.
warnings.filterwarnings(
    "ignore",
    message="Full symbol not available",
    category=UserWarning,
)


_UNCERTAINTY_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)(?:\(\d+\))?$"
)


@dataclass(frozen=True)
class CifAtom:
    label: str
    element: str
    fract: np.ndarray
    cart: np.ndarray
    occupancy: Optional[float] = None
    disorder_group: Optional[str] = None
    disorder_assembly: Optional[str] = None
    u_cart: Optional[np.ndarray] = None

    @property
    def disorder_key(self) -> Optional[str]:
        if self.disorder_group is None:
            return None
        if self.disorder_assembly is not None:
            return f"{self.disorder_assembly}_{self.disorder_group}"
        return self.disorder_group


@dataclass(frozen=True)
class CifStructure:
    name: str
    cell_lengths: Tuple[float, float, float]
    cell_angles: Tuple[float, float, float]
    lattice: np.ndarray
    atoms: Tuple[CifAtom, ...]
    u_cart: Optional[np.ndarray] = None
    space_group: Optional[str] = None
    space_group_number: Optional[str] = None
    crystal_system: Optional[str] = None
    formula: Optional[str] = None
    r1: Optional[str] = None
    wr2: Optional[str] = None
    goof: Optional[str] = None
    is_asymmetric_unit_only: bool = False
    cell_a_str: Optional[str] = None
    cell_b_str: Optional[str] = None
    cell_c_str: Optional[str] = None
    cell_alpha_str: Optional[str] = None
    cell_beta_str: Optional[str] = None
    cell_gamma_str: Optional[str] = None
    volume: Optional[str] = None
    z: Optional[str] = None
    density: Optional[str] = None
    mu: Optional[str] = None
    f000: Optional[str] = None
    temp: Optional[str] = None
    wavelength: Optional[str] = None
    crystal_size: Optional[str] = None
    theta_range: Optional[str] = None
    hkl_ranges: Optional[str] = None
    reflns_collected: Optional[str] = None
    reflns_unique: Optional[str] = None
    r_int: Optional[str] = None
    completeness: Optional[str] = None
    refinement_method: Optional[str] = None
    num_reflns: Optional[str] = None
    num_params: Optional[str] = None
    num_restraints: Optional[str] = None
    max_shift: Optional[str] = None
    diff_peak_hole: Optional[str] = None
    r1_gt: Optional[str] = None
    wr2_gt: Optional[str] = None
    r1_all: Optional[str] = None
    wr2_all: Optional[str] = None
    z_prime: Optional[str] = None
    flack: Optional[str] = None
    asymmetric_atoms: Optional[Tuple[CifAtom, ...]] = None


@dataclass(frozen=True)
class RenderAtom:
    label: str
    element: str
    base_index: int
    image: Tuple[int, int, int]
    position: np.ndarray
    disorder_group: Optional[str] = None
    disorder_assembly: Optional[str] = None
    is_original_asym: bool = False
    u_cart: Optional[np.ndarray] = None

    @property
    def disorder_key(self) -> Optional[str]:
        if self.disorder_group is None:
            return None
        if self.disorder_assembly is not None:
            return f"{self.disorder_assembly}_{self.disorder_group}"
        return self.disorder_group


def parse_cif_file(path: str) -> CifStructure:
    with open(path, "r", encoding="utf-8") as handle:
        return parse_cif(handle.read(), name=path)


def _get_first_tag_value(block_data, keys: List[str]) -> Optional[str]:
    lower_map = {k.lower(): k for k in block_data.keys()}
    for key in keys:
        key_lower = key.lower()
        if key_lower in lower_map:
            vals = block_data[lower_map[key_lower]]
            if vals is not None:
                val = vals[0] if isinstance(vals, (list, tuple)) else vals
                cleaned = str(val).strip().strip("'\"")
                if cleaned not in {".", "?", ""}:
                    return cleaned
    return None


def _extract_metadata(
    get_val, num_symops: Optional[int] = None
) -> Dict[str, Optional[str]]:
    c_max = get_val(["_exptl_crystal_size_max"])
    c_mid = get_val(["_exptl_crystal_size_mid"])
    c_min = get_val(["_exptl_crystal_size_min"])
    if c_max and c_mid and c_min:
        crystal_size = f"{c_max} x {c_mid} x {c_min}"
    elif c_max:
        crystal_size = c_max
    else:
        crystal_size = None

    t_min = get_val(["_diffrn_reflns_theta_min"])
    t_max = get_val(["_diffrn_reflns_theta_max"])
    if t_min and t_max:
        theta_range = f"{t_min} to {t_max}"
    elif t_max:
        theta_range = f"up to {t_max}"
    else:
        theta_range = None

    h_min = get_val(["_diffrn_reflns_limit_h_min"])
    h_max = get_val(["_diffrn_reflns_limit_h_max"])
    k_min = get_val(["_diffrn_reflns_limit_k_min"])
    k_max = get_val(["_diffrn_reflns_limit_k_max"])
    l_min = get_val(["_diffrn_reflns_limit_l_min"])
    l_max = get_val(["_diffrn_reflns_limit_l_max"])
    if all(v is not None for v in [h_min, h_max, k_min, k_max, l_min, l_max]):
        hkl_ranges = f"h: {h_min}/{h_max}, k: {k_min}/{k_max}, l: {l_min}/{l_max}"
    else:
        hkl_ranges = None

    dp_max = get_val(["_refine_diff_density_max"])
    dp_min = get_val(["_refine_diff_density_min"])
    if dp_max and dp_min:
        diff_peak_hole = f"{dp_max} / {dp_min}"
    elif dp_max:
        diff_peak_hole = dp_max
    else:
        diff_peak_hole = None

    z = get_val(["_cell_formula_units_z"])
    z_prime = get_val(
        [
            "_cell_formula_units_z'",
            "_cell_formula_units_zprime",
            "_cell_formula_units_z_prime",
        ]
    )
    if not z_prime and z and num_symops:
        try:
            z_val = parse_cif_number(z)
            calc_z_prime = z_val / num_symops
            if calc_z_prime.is_integer():
                z_prime = str(int(calc_z_prime))
            else:
                z_prime = f"{calc_z_prime:.4f}".rstrip("0").rstrip(".")
        except Exception as exc:
            logging.debug("Failed to calculate Z' value: %s", exc)

    flack = get_val(
        ["_refine_absolute_configuration_flack", "_refine_ls_abs_structure_Flack"]
    )

    return {
        "cell_a_str": get_val(["_cell_length_a"]),
        "cell_b_str": get_val(["_cell_length_b"]),
        "cell_c_str": get_val(["_cell_length_c"]),
        "cell_alpha_str": get_val(["_cell_angle_alpha"]),
        "cell_beta_str": get_val(["_cell_angle_beta"]),
        "cell_gamma_str": get_val(["_cell_angle_gamma"]),
        "volume": get_val(["_cell_volume"]),
        "z": z,
        "z_prime": z_prime,
        "density": get_val(
            ["_exptl_crystal_density_diffrn", "_exptl_crystal_density_meas"]
        ),
        "mu": get_val(["_exptl_absorpt_coefficient_mu"]),
        "f000": get_val(["_exptl_crystal_f_000"]),
        "temp": get_val(["_diffrn_ambient_temperature", "_diffrn_reflns_temperature"]),
        "wavelength": get_val(["_diffrn_radiation_wavelength"]),
        "crystal_size": crystal_size,
        "theta_range": theta_range,
        "hkl_ranges": hkl_ranges,
        "reflns_collected": get_val(["_diffrn_reflns_number"]),
        "reflns_unique": get_val(["_refine_ls_number_reflns"]),
        "r_int": get_val(
            ["_diffrn_reflns_av_r_equivalents", "_diffrn_reflns_av_uneti/neti"]
        ),
        "completeness": get_val(
            [
                "_diffrn_measured_fraction_theta_max",
                "_diffrn_measured_fraction_theta_full",
            ]
        ),
        "refinement_method": get_val(["_refine_ls_structure_factor_coef"]),
        "num_reflns": get_val(["_refine_ls_number_reflns"]),
        "num_params": get_val(["_refine_ls_number_parameters"]),
        "num_restraints": get_val(["_refine_ls_number_restraints"]),
        "max_shift": get_val(["_refine_ls_shift/su_max", "_refine_ls_shift/esd_max"]),
        "diff_peak_hole": diff_peak_hole,
        "flack": flack,
        "r1_gt": get_val(["_refine_ls_r_factor_gt", "_refine_ls_r_factor_obs"]),
        "wr2_gt": get_val(["_refine_ls_wr_factor_gt"]),
        "r1_all": get_val(["_refine_ls_r_factor_all"]),
        "wr2_all": get_val(["_refine_ls_wr_factor_all", "_refine_ls_wr_factor_ref"]),
    }


def _count_symops_from_block_data(block_data_lower) -> int:
    for key in [
        "_space_group_symop_operation_xyz",
        "_space_group_symop.operation_xyz",
        "_symmetry_equiv_pos_as_xyz",
        "_symmetry.equiv_pos_as_xyz",
    ]:
        if key in block_data_lower:
            vals = block_data_lower[key]
            if isinstance(vals, (list, tuple)):
                return len(vals)
            elif vals is not None:
                return 1
    return 0


def _count_symops_from_loops(loops) -> int:
    for headers, rows in loops:
        for h in headers:
            h_low = h.lower()
            if h_low in {
                "_space_group_symop_operation_xyz",
                "_space_group_symop.operation_xyz",
                "_symmetry_equiv_pos_as_xyz",
                "_symmetry.equiv_pos_as_xyz",
            }:
                return len(rows)
    return 0


def _parse_asymmetric_atoms(
    block_data_lower,
    lattice: np.ndarray,
    label_to_disorder: dict,
    label_to_adp: Optional[dict] = None,
) -> List[CifAtom]:
    atoms = []
    if "_atom_site_label" in block_data_lower:
        labels = block_data_lower["_atom_site_label"]
        fx = block_data_lower.get("_atom_site_fract_x", [])
        fy = block_data_lower.get("_atom_site_fract_y", [])
        fz = block_data_lower.get("_atom_site_fract_z", [])
        elements = block_data_lower.get("_atom_site_type_symbol", [])
        occupancies = block_data_lower.get("_atom_site_occupancy", [])

        # Ensure lists are aligned
        num_labels = len(labels)
        for idx_lbl in range(num_labels):
            try:
                if idx_lbl >= len(fx) or idx_lbl >= len(fy) or idx_lbl >= len(fz):
                    continue
                clean_lbl = str(labels[idx_lbl]).strip().strip("'\"")
                x = parse_cif_number(fx[idx_lbl])
                y = parse_cif_number(fy[idx_lbl])
                z = parse_cif_number(fz[idx_lbl])
                fract = np.array([x, y, z], dtype=float)
                cart = fract @ lattice

                el = "X"
                if idx_lbl < len(elements):
                    el = normalize_element(elements[idx_lbl])
                else:
                    el = normalize_element(clean_lbl)

                occ = None
                if idx_lbl < len(occupancies):
                    try:
                        occ = parse_cif_number(occupancies[idx_lbl])
                    except ValueError as exc:
                        logging.debug("Failed to parse occupancy value: %s", exc)

                g, a = label_to_disorder.get(clean_lbl, (None, None))

                u_cart_atom = None
                if label_to_adp and clean_lbl in label_to_adp:
                    try:
                        u_orig_vals = list(label_to_adp[clean_lbl])
                        u11, u22, u33, u23, u13, u12 = u_orig_vals
                        U_frac_orig = np.array(
                            [[u11, u12, u13], [u12, u22, u23], [u13, u23, u33]],
                            dtype=float,
                        )
                        A = lattice.T
                        A_inv = np.linalg.inv(A)
                        N = np.diag([np.linalg.norm(x) for x in A_inv])
                        mat_ustar = N @ U_frac_orig @ N
                        u_cart_atom = A @ mat_ustar @ A.T
                    except Exception as exc:
                        logging.debug(
                            "Failed to compute asymmetric atom U_cart for %s: %s",
                            clean_lbl,
                            exc,
                        )

                atoms.append(
                    CifAtom(
                        label=clean_lbl,
                        element=el,
                        fract=fract,
                        cart=cart,
                        occupancy=occ,
                        disorder_group=g,
                        disorder_assembly=a,
                        u_cart=u_cart_atom,
                    )
                )
            except Exception as exc:
                logging.debug("Failed to parse atom: %s", exc)
    return atoms


def parse_cif_file_pymatgen(path: str) -> List[CifStructure]:
    from pymatgen.io.cif import CifParser

    # Suppress pymatgen/monty UserWarning emitted when a CIF uses a
    # non-standard Hermann-Mauguin symbol (e.g. 'I1a1') that is not in
    # pymatgen's full-symbol database.  The SpaceGroup singleton is created
    # lazily inside CifParser, _get_structure, get_symops, and
    # SpacegroupAnalyzer — so the filter must cover the entire function.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Full symbol not available",
            category=UserWarning,
            module="monty",
        )

        parser = CifParser(path)
        structures = []

        for name, block in parser._cif.data.items():
            try:
                struct = parser._get_structure(
                    block, primitive=False, symmetrized=False
                )
                if struct is None:
                    continue

                block_data_lower = {k.lower(): v for k, v in block.data.items()}

                # Get spacegroup symmetry operations
                symops = []
                try:
                    symops = parser.get_symops(block)
                except Exception as exc:
                    logging.debug("pymatgen get_symops failed: %s", exc)
                if not symops:
                    try:
                        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

                        sga = SpacegroupAnalyzer(struct)
                        symops = sga.get_space_group_operations()
                    except Exception as exc:
                        logging.debug(
                            "SpacegroupAnalyzer get_space_group_operations failed: %s",
                            exc,
                        )

                label_to_adp = {}
                if "_atom_site_aniso_label" in block_data_lower:
                    aniso_loop = None
                    for loop in block.loops:
                        if any("aniso" in c.lower() for c in loop):
                            aniso_loop = loop
                            break

                    if aniso_loop:
                        label_col = None
                        val_cols = {}
                        is_b = False
                        for col in aniso_loop:
                            c_low = col.lower()
                            if "label" in c_low:
                                label_col = col
                            for suffix in ["11", "22", "33", "12", "13", "23"]:
                                if c_low.endswith("u_" + suffix):
                                    val_cols[suffix] = col
                                elif c_low.endswith("b_" + suffix):
                                    val_cols[suffix] = col
                                    is_b = True

                        if label_col and "11" in val_cols:
                            labels_list = block_data_lower[label_col.lower()]
                            num_entries = len(labels_list)

                            for i in range(num_entries):
                                lbl = labels_list[i]
                                try:
                                    u_vals = {}
                                    for suffix in ["11", "22", "33", "12", "13", "23"]:
                                        col = val_cols.get(suffix)
                                        val = 0.0
                                        if col and i < len(
                                            block_data_lower[col.lower()]
                                        ):
                                            val = parse_cif_number(
                                                block_data_lower[col.lower()][i]
                                            )
                                            if is_b:
                                                val = val / (8.0 * math.pi * math.pi)
                                        u_vals[suffix] = val

                                    phonopy_order = [
                                        u_vals.get("11", 0.0),
                                        u_vals.get("22", 0.0),
                                        u_vals.get("33", 0.0),
                                        u_vals.get("23", 0.0),
                                        u_vals.get("13", 0.0),
                                        u_vals.get("12", 0.0),
                                    ]
                                    label_to_adp[lbl] = phonopy_order
                                except Exception as exc:
                                    logging.debug(
                                        "Failed parsing ADP parameters for label %s: %s",
                                        lbl,
                                        exc,
                                    )

                u_cart_data = None
                if label_to_adp:
                    A = struct.lattice.matrix.T
                    A_inv = np.linalg.inv(A)
                    N = np.diag([np.linalg.norm(x) for x in A_inv])

                    # Build mapping of clean labels to their original asymmetric unit fractional coordinates
                    label_to_frac_orig = {}
                    if "_atom_site_label" in block_data_lower:
                        labels = block_data_lower["_atom_site_label"]
                        fx = block_data_lower.get("_atom_site_fract_x", [])
                        fy = block_data_lower.get("_atom_site_fract_y", [])
                        fz = block_data_lower.get("_atom_site_fract_z", [])
                        for idx_lbl, lbl in enumerate(labels):
                            try:
                                clean_lbl = str(lbl).strip().strip("'\"")
                                x = parse_cif_number(fx[idx_lbl])
                                y = parse_cif_number(fy[idx_lbl])
                                z = parse_cif_number(fz[idx_lbl])
                                label_to_frac_orig[clean_lbl] = np.array([x, y, z])
                            except Exception as exc:
                                logging.debug(
                                    "Failed parsing fractional coordinates for label %s: %s",
                                    clean_lbl,
                                    exc,
                                )

                    u_cart_list = []
                    for site in struct:
                        lbl = str(getattr(site, "label", "")).strip().strip("'\"")
                        if lbl in label_to_adp:
                            u_orig_vals = list(label_to_adp[lbl])
                            frac_orig = label_to_frac_orig.get(lbl)

                            u11, u22, u33, u23, u13, u12 = u_orig_vals
                            U_frac_orig = np.array(
                                [[u11, u12, u13], [u12, u22, u23], [u13, u23, u33]],
                                dtype=float,
                            )

                            # Convert original U_frac to Cartesian U_cart
                            mat_ustar = N @ U_frac_orig @ N
                            U_cart_orig = A @ mat_ustar @ A.T

                            if frac_orig is not None and symops:
                                frac_site = site.frac_coords
                                matched_op = None
                                for op in symops:
                                    res = op.operate(frac_orig)
                                    diff = res - frac_site
                                    diff_mod = diff - np.round(diff)
                                    if np.allclose(diff_mod, 0.0, atol=1e-4):
                                        matched_op = op
                                        break

                                if matched_op is not None:
                                    R = matched_op.rotation_matrix
                                    R_cart = A @ R @ A_inv
                                    U_cart_site = R_cart @ U_cart_orig @ R_cart.T
                                else:
                                    U_cart_site = U_cart_orig
                            else:
                                U_cart_site = U_cart_orig
                            u_cart_list.append(U_cart_site)
                        else:
                            u_cart_list.append(np.zeros((3, 3)))

                    if u_cart_list and any(
                        not np.allclose(m, 0.0) for m in u_cart_list
                    ):
                        u_cart_data = np.array(u_cart_list, dtype=float)

                # Build label_to_disorder map
                label_to_disorder = {}
                label_key = None
                group_key = None
                assembly_key = None
                for k in block_data_lower.keys():
                    k_norm = k.lower().lstrip("_")
                    if k_norm == "atom_site_label":
                        label_key = k
                    elif k_norm == "atom_site_disorder_group":
                        group_key = k
                    elif k_norm == "atom_site_disorder_assembly":
                        assembly_key = k

                if label_key is not None:
                    labels = block_data_lower[label_key]
                    groups = (
                        block_data_lower[group_key] if group_key is not None else []
                    )
                    assemblies = (
                        block_data_lower[assembly_key]
                        if assembly_key is not None
                        else []
                    )
                    for idx_lbl, lbl in enumerate(labels):
                        try:
                            clean_lbl = str(lbl).strip().strip("'\"")
                            g = (
                                str(groups[idx_lbl]).strip().strip("'\"")
                                if idx_lbl < len(groups)
                                else None
                            )
                            a = (
                                str(assemblies[idx_lbl]).strip().strip("'\"")
                                if idx_lbl < len(assemblies)
                                else None
                            )
                            if g in {".", "?", ""}:
                                g = None
                            if a in {".", "?", ""}:
                                a = None
                            label_to_disorder[clean_lbl] = (g, a)
                        except Exception as exc:
                            logging.debug(
                                "Failed to parse disorder info for label %s: %s",
                                lbl,
                                exc,
                            )

                # Extract cell metadata and refinement factors
                space_group = _get_first_tag_value(
                    block_data_lower,
                    [
                        "_space_group_name_h-m_alt",
                        "_symmetry_space_group_name_h-m",
                        "_space_group.symmetry_space_group_name_h-m",
                    ],
                )
                if not space_group:
                    try:
                        space_group = struct.get_space_group_info()[0]
                    except Exception as exc:
                        logging.debug(
                            "Failed to get space group info from structure: %s", exc
                        )

                space_group_number = _get_first_tag_value(
                    block_data_lower,
                    [
                        "_space_group_it_number",
                        "_space_group.it_number",
                        "_symmetry_int_tables_number",
                        "_symmetry.int_tables_number",
                    ],
                )
                if not space_group_number:
                    try:
                        space_group_number = str(struct.get_space_group_info()[1])
                    except Exception as exc:
                        logging.debug(
                            "Failed to get space group number info from structure: %s",
                            exc,
                        )
                if not space_group_number:
                    try:
                        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

                        sga = SpacegroupAnalyzer(struct)
                        space_group_number = str(sga.get_space_group_number())
                    except Exception as exc:
                        logging.debug(
                            "SpacegroupAnalyzer get_space_group_number failed: %s", exc
                        )

                crystal_system = _get_first_tag_value(
                    block_data_lower,
                    ["_space_group_crystal_system", "_symmetry_cell_setting"],
                )
                if not crystal_system:
                    try:
                        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

                        sga = SpacegroupAnalyzer(struct)
                        crystal_system = sga.get_crystal_system()
                    except Exception as exc:
                        logging.debug(
                            "SpacegroupAnalyzer get_crystal_system failed: %s", exc
                        )

                formula = _get_first_tag_value(
                    block_data_lower,
                    [
                        "_chemical_formula_sum",
                        "_chemical_formula_structural",
                        "_chemical_formula_moiety",
                    ],
                )
                if not formula:
                    try:
                        formula = struct.formula
                    except Exception as exc:
                        logging.debug("Failed to get formula from structure: %s", exc)

                r1 = _get_first_tag_value(
                    block_data_lower,
                    [
                        "_refine_ls_r_factor_gt",
                        "_refine_ls_r_factor_obs",
                        "_refine_ls_r_factor_all",
                    ],
                )
                wr2 = _get_first_tag_value(
                    block_data_lower,
                    [
                        "_refine_ls_wr_factor_ref",
                        "_refine_ls_wr_factor_gt",
                        "_refine_ls_wr_factor_all",
                    ],
                )
                goof = _get_first_tag_value(
                    block_data_lower,
                    [
                        "_refine_ls_goodness_of_fit_ref",
                        "_refine_ls_goodness_of_fit_all",
                    ],
                )

                num_symops = (
                    len(symops)
                    if symops
                    else _count_symops_from_block_data(block_data_lower)
                )
                if num_symops == 0:
                    num_symops = None

                metadata = _extract_metadata(
                    lambda keys: _get_first_tag_value(block_data_lower, keys),
                    num_symops=num_symops,
                )

                asym_atoms = _parse_asymmetric_atoms(
                    block_data_lower,
                    struct.lattice.matrix,
                    label_to_disorder,
                    label_to_adp,
                )

                cif_struct = _structure_from_pymatgen(
                    struct,
                    name or "Structure",
                    u_cart_data,
                    label_to_disorder=label_to_disorder,
                    space_group=space_group,
                    space_group_number=space_group_number,
                    crystal_system=crystal_system,
                    formula=formula,
                    r1=r1,
                    wr2=wr2,
                    goof=goof,
                    asymmetric_atoms=tuple(asym_atoms) if asym_atoms else None,
                    **metadata,
                )

                structures.append(cif_struct)
            except Exception as exc:
                logging.error("Failed to parse structure block %s: %s", name, exc)

        return structures


def _structure_from_pymatgen(
    struct,
    name: str,
    u_cart: Optional[np.ndarray] = None,
    label_to_disorder: Optional[Dict[str, Tuple[Optional[str], Optional[str]]]] = None,
    space_group: Optional[str] = None,
    space_group_number: Optional[str] = None,
    crystal_system: Optional[str] = None,
    formula: Optional[str] = None,
    r1: Optional[str] = None,
    wr2: Optional[str] = None,
    goof: Optional[str] = None,
    asymmetric_atoms: Optional[Tuple[CifAtom, ...]] = None,
    **kwargs,
) -> CifStructure:
    cell_lengths = struct.lattice.lengths
    cell_angles = struct.lattice.angles
    lattice = struct.lattice.matrix

    if label_to_disorder is None:
        label_to_disorder = {}

    atoms = []
    for i, site in enumerate(struct):
        label = getattr(site, "label", f"{site.species_string}{i + 1}")
        element = normalize_element(site.species_string)
        fract = site.frac_coords
        cart = site.coords
        occupancy = getattr(site, "occupancy", 1.0)

        clean_lbl = str(label).strip().strip("'\"")
        g, a = label_to_disorder.get(clean_lbl, (None, None))

        u_cart_atom = None
        if u_cart is not None and i < len(u_cart):
            u_cart_atom = u_cart[i]
            if np.allclose(u_cart_atom, 0.0):
                u_cart_atom = None

        atoms.append(
            CifAtom(
                label=label,
                element=element,
                fract=fract,
                cart=cart,
                occupancy=occupancy,
                disorder_group=g,
                disorder_assembly=a,
                u_cart=u_cart_atom,
            )
        )

    return CifStructure(
        name=name,
        cell_lengths=(
            float(cell_lengths[0]),
            float(cell_lengths[1]),
            float(cell_lengths[2]),
        ),
        cell_angles=(
            float(cell_angles[0]),
            float(cell_angles[1]),
            float(cell_angles[2]),
        ),
        lattice=lattice,
        atoms=tuple(atoms),
        u_cart=u_cart,
        space_group=space_group,
        space_group_number=space_group_number,
        crystal_system=crystal_system,
        formula=formula,
        r1=r1,
        wr2=wr2,
        goof=goof,
        is_asymmetric_unit_only=False,
        asymmetric_atoms=asymmetric_atoms,
        **kwargs,
    )


def parse_cif(text: str, name: str = "CIF") -> CifStructure:
    tags, loops, data_name = _read_cif_tokens(text)
    structure_name = data_name or name

    lengths = (
        _required_float(tags, "_cell_length_a"),
        _required_float(tags, "_cell_length_b"),
        _required_float(tags, "_cell_length_c"),
    )
    angles = (
        _required_float(tags, "_cell_angle_alpha"),
        _required_float(tags, "_cell_angle_beta"),
        _required_float(tags, "_cell_angle_gamma"),
    )
    lattice = cell_vectors(lengths, angles)

    atom_loop = _find_atom_loop(loops)
    atoms = _atoms_from_loop(atom_loop, lattice)
    if not atoms:
        raise ValueError("CIF does not contain readable atom positions.")

    def _get_tag_value_dict(tags_dict, keys: List[str]) -> Optional[str]:
        for key in keys:
            if key in tags_dict:
                cleaned = str(tags_dict[key]).strip().strip("'\"")
                if cleaned not in {".", "?", ""}:
                    return cleaned
        return None

    space_group = _get_tag_value_dict(
        tags,
        [
            "_space_group_name_h-m_alt",
            "_symmetry_space_group_name_h-m",
            "_space_group.symmetry_space_group_name_h-m",
        ],
    )
    space_group_number = _get_tag_value_dict(
        tags,
        [
            "_space_group_it_number",
            "_space_group.it_number",
            "_symmetry_int_tables_number",
            "_symmetry.int_tables_number",
        ],
    )
    crystal_system = _get_tag_value_dict(
        tags, ["_space_group_crystal_system", "_symmetry_cell_setting"]
    )
    formula = _get_tag_value_dict(
        tags,
        [
            "_chemical_formula_sum",
            "_chemical_formula_structural",
            "_chemical_formula_moiety",
        ],
    )
    r1 = _get_tag_value_dict(
        tags,
        [
            "_refine_ls_r_factor_gt",
            "_refine_ls_r_factor_obs",
            "_refine_ls_r_factor_all",
        ],
    )
    wr2 = _get_tag_value_dict(
        tags,
        [
            "_refine_ls_wr_factor_ref",
            "_refine_ls_wr_factor_gt",
            "_refine_ls_wr_factor_all",
        ],
    )
    goof = _get_tag_value_dict(
        tags, ["_refine_ls_goodness_of_fit_ref", "_refine_ls_goodness_of_fit_all"]
    )

    num_symops = _count_symops_from_loops(loops)
    if num_symops == 0:
        num_symops = None

    metadata = _extract_metadata(
        lambda keys: _get_tag_value_dict(tags, keys), num_symops=num_symops
    )

    return CifStructure(
        structure_name,
        lengths,
        angles,
        lattice,
        tuple(atoms),
        space_group=space_group,
        space_group_number=space_group_number,
        crystal_system=crystal_system,
        formula=formula,
        r1=r1,
        wr2=wr2,
        goof=goof,
        is_asymmetric_unit_only=True,
        asymmetric_atoms=tuple(atoms),
        **metadata,
    )


def cell_vectors(lengths: Sequence[float], angles_deg: Sequence[float]) -> np.ndarray:
    a_len, b_len, c_len = lengths
    alpha, beta, gamma = [math.radians(angle) for angle in angles_deg]

    a_vec = np.array([a_len, 0.0, 0.0], dtype=float)
    b_vec = np.array([b_len * math.cos(gamma), b_len * math.sin(gamma), 0.0])

    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1e-8:
        raise ValueError("Invalid cell: gamma angle makes the cell singular.")

    c_x = c_len * math.cos(beta)
    c_y = c_len * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    c_z_sq = c_len * c_len - c_x * c_x - c_y * c_y
    if c_z_sq < -1e-6:
        raise ValueError("Invalid cell: angles and lengths are inconsistent.")
    c_vec = np.array([c_x, c_y, math.sqrt(max(c_z_sq, 0.0))])
    return np.vstack([a_vec, b_vec, c_vec])


def fractional_to_cartesian(fract: Sequence[float], lattice: np.ndarray) -> np.ndarray:
    return np.asarray(fract, dtype=float) @ lattice


def expand_supercell(
    structure: CifStructure,
    repeats: Sequence[int],
    keep_connected: bool = True,
    tolerance: float = 0.45,
) -> Tuple[List[RenderAtom], List[Tuple[int, int]]]:
    repeat_a, repeat_b, repeat_c = [max(1, int(value)) for value in repeats]
    atoms: List[RenderAtom] = []
    base_atoms = (
        unwrap_connected_atoms(structure) if keep_connected else list(structure.atoms)
    )

    for ia in range(repeat_a):
        for ib in range(repeat_b):
            for ic in range(repeat_c):
                offset = np.array([ia, ib, ic], dtype=float)
                cart_offset = offset @ structure.lattice
                for base_index, atom in enumerate(base_atoms):
                    u_cart_atom = getattr(atom, "u_cart", None)

                    atoms.append(
                        RenderAtom(
                            label=atom.label,
                            element=atom.element,
                            base_index=base_index,
                            image=(ia, ib, ic),
                            position=atom.cart + cart_offset,
                            disorder_group=atom.disorder_group,
                            disorder_assembly=atom.disorder_assembly,
                            u_cart=u_cart_atom,
                        )
                    )

    return atoms, infer_bonds(atoms, tolerance=tolerance)


def get_space_group_operations(structure: CifStructure) -> list:
    """Helper to get SpaceGroup symmetry operations using pymatgen."""
    symops = []
    if structure.space_group:
        try:
            from pymatgen.symmetry.groups import SpaceGroup

            sg = SpaceGroup(structure.space_group)
            symops = sg.symmetry_ops
        except Exception as exc:
            logging.debug(
                "SpaceGroup instantiation failed for space group %s: %s",
                structure.space_group,
                exc,
            )
    if not symops:
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            from pymatgen.core import Structure

            species = [atom.element for atom in structure.atoms]
            coords = [atom.fract for atom in structure.atoms]
            pm_struct = Structure(structure.lattice, species, coords)
            sga = SpacegroupAnalyzer(pm_struct)
            symops = sga.get_space_group_operations()
        except Exception as exc:
            logging.debug(
                "SpacegroupAnalyzer get_space_group_operations failed: %s", exc
            )
    return symops


def _infer_periodic_adjacency(structure: CifStructure):
    adjacency = {atom_index: [] for atom_index in range(len(structure.atoms))}
    for left_index in range(len(structure.atoms)):
        left_atom = structure.atoms[left_index]
        left_radius = covalent_radius(left_atom.element)

        for right_index in range(left_index, len(structure.atoms)):
            right_atom = structure.atoms[right_index]
            right_radius = covalent_radius(right_atom.element)
            cutoff = min(2.45, left_radius + right_radius + 0.45)

            if (
                left_atom.disorder_key is not None
                and right_atom.disorder_key is not None
            ):
                if left_atom.disorder_key != right_atom.disorder_key:
                    continue

            delta_frac = np.asarray(right_atom.fract) - np.asarray(left_atom.fract)

            # Find the minimum image shift as the baseline to support un-normalized coordinates
            base_shift = -np.rint(delta_frac).astype(int)

            # Sweep a 3x3x3 grid around the minimum image to catch multi-path boundary bonds
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        shift = base_shift + np.array([dx, dy, dz], dtype=int)

                        # Skip self-comparison at zero net shift
                        if left_index == right_index and np.all(shift == 0):
                            continue

                        shifted_delta = delta_frac + shift
                        distance = float(
                            np.linalg.norm(
                                fractional_to_cartesian(
                                    shifted_delta, structure.lattice
                                )
                            )
                        )

                        if 0.25 <= distance <= cutoff:
                            adjacency[left_index].append((right_index, shift))
                            if left_index != right_index:
                                adjacency[right_index].append((left_index, -shift))

    return adjacency


def grow_molecules(
    structure: CifStructure,
    selected_disorder_key: Optional[str] = None,
    tolerance: float = 0.45,
) -> Tuple[List[RenderAtom], List[Tuple[int, int]]]:
    """
    Grow molecules by applying symmetry operations then unwrapping across
    cell boundaries.

    1. Apply all symops to the asymmetric unit, wrap into [0,1) → unique sites.
    2. Call _infer_periodic_adjacency on those sites (reuses existing logic).
    3. BFS with cumulative offset (same as unwrap_connected_atoms) → each
       molecule is always contiguous, never split at a cell boundary.
    4. Keep molecules that contain an original asymmetric-unit atom.
    """
    core_atoms = (
        structure.asymmetric_atoms
        if structure.asymmetric_atoms is not None
        else structure.atoms
    )
    if not core_atoms:
        return [], []

    symops = get_space_group_operations(structure)
    if not symops:
        from pymatgen.core.operations import SymmOp

        symops = [SymmOp.from_xyz_str("x, y, z")]

    # --- Step 1: expand asymmetric unit via symops, wrap to [0,1), deduplicate ---
    exp_atoms: List[CifAtom] = []
    is_asym: List[bool] = []
    seen_fracs: List[np.ndarray] = []

    for atom in core_atoms:
        if selected_disorder_key is not None and atom.disorder_group is not None:
            if (
                atom.disorder_group != selected_disorder_key
                and atom.disorder_key != selected_disorder_key
            ):
                continue

        for op in symops:
            # identity check
            identity = False
            try:
                identity = np.allclose(op.rotation_matrix, np.eye(3)) and np.allclose(
                    op.translation_vector, np.zeros(3)
                )
            except Exception:
                try:
                    t = np.array([0.123, 0.456, 0.789])
                    identity = np.allclose(op.operate(t), t)
                except Exception:
                    pass

            # rotate ADP tensor
            u_cart_site = None
            u0 = getattr(atom, "u_cart", None)
            if u0 is not None and not np.allclose(u0, 0.0):
                try:
                    A = structure.lattice.T
                    R_cart = A @ op.rotation_matrix @ np.linalg.inv(A)
                    u_cart_site = R_cart @ u0 @ R_cart.T
                except Exception:
                    u_cart_site = u0

            frac = op.operate(atom.fract) % 1.0

            # deduplicate
            dup = any(
                np.linalg.norm((frac - p - np.round(frac - p)) @ structure.lattice)
                < 0.05
                for p in seen_fracs
            )
            if dup:
                continue
            seen_fracs.append(frac.copy())

            cart = frac @ structure.lattice
            exp_atoms.append(
                CifAtom(
                    atom.label,
                    atom.element,
                    frac,
                    cart,
                    atom.occupancy,
                    disorder_group=atom.disorder_group,
                    disorder_assembly=atom.disorder_assembly,
                    u_cart=u_cart_site,
                )
            )
            is_asym.append(identity)

    if not exp_atoms:
        return [], []

    # --- Step 2: periodic bond graph (reuse existing function) ---
    tmp = CifStructure(
        structure.name,
        structure.cell_lengths,
        structure.cell_angles,
        structure.lattice,
        tuple(exp_atoms),
    )
    adj = _infer_periodic_adjacency(tmp)

    # --- Step 3: BFS with offset tracking per molecule (= unwrap_connected_atoms) ---
    N = len(exp_atoms)
    visited = [False] * N
    final_atoms: List[RenderAtom] = []
    final_bonds: List[Tuple[int, int]] = []
    atom_base = 0

    for start in range(N):
        if visited[start]:
            continue
        visited[start] = True
        offsets: Dict[int, np.ndarray] = {start: np.zeros(3, dtype=int)}
        queue = [start]
        mol: List[Tuple[int, np.ndarray]] = []

        while queue:
            curr = queue.pop(0)
            mol.append((curr, offsets[curr]))
            for nb, shift in adj[curr]:
                if nb not in offsets:
                    offsets[nb] = offsets[curr] + shift
                    visited[nb] = True
                    queue.append(nb)

        # --- Step 4: keep molecules touching the original asymmetric unit ---
        if not any(is_asym[i] for i, _ in mol):
            continue

        local: Dict[int, int] = {i: li for li, (i, _) in enumerate(mol)}
        for li, (i, off) in enumerate(mol):
            a = exp_atoms[i]
            frac = a.fract + off
            final_atoms.append(
                RenderAtom(
                    label=a.label,
                    element=a.element,
                    base_index=i,
                    image=tuple(int(x) for x in off),
                    position=frac @ structure.lattice,
                    disorder_group=a.disorder_group,
                    disorder_assembly=a.disorder_assembly,
                    is_original_asym=is_asym[i],
                    u_cart=a.u_cart,
                )
            )
        for li, (i, _) in enumerate(mol):
            for nb, _ in adj[i]:
                if nb in local and li < local[nb]:
                    final_bonds.append((atom_base + li, atom_base + local[nb]))
        atom_base += len(mol)

    return final_atoms, final_bonds


def unwrap_connected_atoms(structure: CifStructure) -> List[CifAtom]:
    adjacency = _infer_periodic_adjacency(structure)
    if not adjacency:
        return list(structure.atoms)

    image_offsets: Dict[int, np.ndarray] = {}
    for start_index in range(len(structure.atoms)):
        if start_index in image_offsets:
            continue
        image_offsets[start_index] = np.zeros(3, dtype=int)
        queue = [start_index]
        while queue:
            current_index = queue.pop(0)
            current_offset = image_offsets[current_index]
            for neighbor_index, neighbor_shift in adjacency[current_index]:
                proposed_offset = current_offset + neighbor_shift
                if neighbor_index not in image_offsets:
                    image_offsets[neighbor_index] = proposed_offset
                    queue.append(neighbor_index)

    unwrapped = []
    for atom_index, atom in enumerate(structure.atoms):
        offset = image_offsets.get(atom_index, np.zeros(3, dtype=int))
        fract = np.asarray(atom.fract, dtype=float) + offset
        cart = fractional_to_cartesian(fract, structure.lattice)
        unwrapped.append(
            CifAtom(
                atom.label,
                atom.element,
                fract,
                cart,
                atom.occupancy,
                disorder_group=atom.disorder_group,
                disorder_assembly=atom.disorder_assembly,
                u_cart=getattr(atom, "u_cart", None),
            )
        )
    return unwrapped


def supercell_edges(
    lattice: np.ndarray, repeats: Sequence[int]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    repeat = np.asarray([max(1, int(value)) for value in repeats], dtype=float)
    scaled = lattice * repeat[:, None]
    corners = [
        np.array([ia, ib, ic], dtype=float) @ scaled
        for ia in (0, 1)
        for ib in (0, 1)
        for ic in (0, 1)
    ]
    edges = []
    index = {
        (ia, ib, ic): idx
        for idx, (ia, ib, ic) in enumerate(
            (ia, ib, ic) for ia in (0, 1) for ib in (0, 1) for ic in (0, 1)
        )
    }
    for ia in (0, 1):
        for ib in (0, 1):
            edges.append((corners[index[(ia, ib, 0)]], corners[index[(ia, ib, 1)]]))
    for ia in (0, 1):
        for ic in (0, 1):
            edges.append((corners[index[(ia, 0, ic)]], corners[index[(ia, 1, ic)]]))
    for ib in (0, 1):
        for ic in (0, 1):
            edges.append((corners[index[(0, ib, ic)]], corners[index[(1, ib, ic)]]))
    return edges


def celleditpy_cell_axis_segments(lattice: np.ndarray):
    origin = np.array([0.0, 0.0, 0.0])
    corners = [
        origin,
        origin + lattice[0],
        origin + lattice[1],
        origin + lattice[2],
        origin + lattice[0] + lattice[1],
        origin + lattice[0] + lattice[2],
        origin + lattice[1] + lattice[2],
        origin + lattice[0] + lattice[1] + lattice[2],
    ]
    axis_lines = [
        (corners[0], corners[1], "red", "a"),
        (corners[0], corners[2], "green", "b"),
        (corners[0], corners[3], "blue", "c"),
    ]
    edge_indices = [
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
    edges = [(corners[start], corners[end], "white", "") for start, end in edge_indices]
    return axis_lines + edges


def infer_bonds(
    atoms: Sequence[RenderAtom], tolerance: float = 0.45
) -> List[Tuple[int, int]]:
    if not atoms:
        return []

    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        from rdkit.Geometry import Point3D

        rw_mol = Chem.RWMol()
        for atom in atoms:
            rw_mol.AddAtom(Chem.Atom(atom.element))

        conformer = Chem.Conformer(len(atoms))
        for idx, atom in enumerate(atoms):
            x, y, z = [float(val) for val in atom.position]
            conformer.SetAtomPosition(idx, Point3D(x, y, z))
        rw_mol.AddConformer(conformer, assignId=True)

        cov_factor = 1.0 + (tolerance / 1.5)
        rdDetermineBonds.DetermineConnectivity(
            rw_mol, covFactor=cov_factor, useVdw=True
        )

        rdkit_bonds = []
        for bond in rw_mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            u, v = sorted((u, v))

            left_atom = atoms[u]
            right_atom = atoms[v]
            if (
                left_atom.disorder_key is not None
                and right_atom.disorder_key is not None
            ):
                if left_atom.disorder_key != right_atom.disorder_key:
                    continue

            rdkit_bonds.append((u, v))

        # Sort the bonds to ensure deterministic order matching manual implementation
        rdkit_bonds.sort()
        return rdkit_bonds
    except Exception as exc:
        logging.debug(
            "RDKit DetermineConnectivity failed, falling back to manual bond detection: %s",
            exc,
        )

    bonds: List[Tuple[int, int]] = []
    max_cutoff = max(2.45, 2.0 + tolerance)

    positions = np.array([atom.position for atom in atoms])
    min_coords = np.min(positions, axis=0) - 1.0

    # Voxel grid partitioning with bin size equal to max_cutoff
    bins = {}
    for idx, pos in enumerate(positions):
        bin_idx = tuple(((pos - min_coords) / max_cutoff).astype(int))
        if bin_idx not in bins:
            bins[bin_idx] = []
        bins[bin_idx].append(idx)

    # 26 neighbor offset directions
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]

    for bin_idx, left_indices in bins.items():
        n_left = len(left_indices)
        # 1. Check pairs in the same bin
        for i in range(n_left):
            left = left_indices[i]
            left_atom = atoms[left]
            left_radius = covalent_radius(left_atom.element)
            for j in range(i + 1, n_left):
                right = left_indices[j]
                right_atom = atoms[right]
                if (
                    left_atom.disorder_key is not None
                    and right_atom.disorder_key is not None
                ):
                    if left_atom.disorder_key != right_atom.disorder_key:
                        continue
                right_radius = covalent_radius(right_atom.element)
                cutoff = min(max_cutoff, left_radius + right_radius + tolerance)
                dist_sq = np.sum((positions[left] - positions[right]) ** 2)
                if 0.25 * 0.25 <= dist_sq <= cutoff * cutoff:
                    bonds.append((left, right))

        # 2. Check pairs with adjacent bins
        bx, by, bz = bin_idx
        for dx, dy, dz in neighbor_offsets:
            neigh_bin = (bx + dx, by + dy, bz + dz)
            if neigh_bin in bins:
                right_indices = bins[neigh_bin]
                for left in left_indices:
                    left_atom = atoms[left]
                    left_radius = covalent_radius(left_atom.element)
                    for right in right_indices:
                        if left >= right:
                            continue
                        right_atom = atoms[right]
                        if (
                            left_atom.disorder_key is not None
                            and right_atom.disorder_key is not None
                        ):
                            if left_atom.disorder_key != right_atom.disorder_key:
                                continue
                        right_radius = covalent_radius(right_atom.element)
                        cutoff = min(max_cutoff, left_radius + right_radius + tolerance)
                        dist_sq = np.sum((positions[left] - positions[right]) ** 2)
                        if 0.25 * 0.25 <= dist_sq <= cutoff * cutoff:
                            bonds.append((left, right))

    return bonds


def covalent_radius(element: str) -> float:
    return _COVALENT_RADII.get(normalize_element(element), 0.77)


def normalize_element(value: str) -> str:
    match = re.match(r"([A-Za-z]{1,2})", str(value).strip())
    if not match:
        return "X"
    raw = match.group(1)
    return raw[0].upper() + raw[1:].lower()


def parse_cif_number(value: str) -> float:
    cleaned = str(value).strip().strip("'\"")
    if cleaned in {"?", "."}:
        raise ValueError("Missing numeric CIF value.")
    match = _UNCERTAINTY_RE.match(cleaned)
    if match:
        return float(match.group(1))
    return float(cleaned)


def _read_cif_tokens(text: str):
    lines = list(_logical_lines(text))
    tags: Dict[str, str] = {}
    loops = []
    data_name = ""
    index = 0

    while index < len(lines):
        line = lines[index]
        lower = line.lower()
        if lower.startswith("data_"):
            data_name = line[5:].strip() or data_name
            index += 1
            continue
        if lower == "loop_":
            index += 1
            headers = []
            while index < len(lines) and lines[index].startswith("_"):
                header_tokens = _split_cif_line(lines[index])
                headers.append(_normalize_tag(header_tokens[0]))
                index += 1

            values = []
            while index < len(lines):
                current = lines[index]
                current_lower = current.lower()
                if (
                    current_lower == "loop_"
                    or current.startswith("_")
                    or current_lower.startswith("data_")
                ):
                    break
                values.extend(_split_cif_line(current))
                index += 1

            rows = []
            if headers:
                width = len(headers)
                for start in range(0, len(values), width):
                    row_values = values[start : start + width]
                    if len(row_values) == width:
                        rows.append(dict(zip(headers, row_values)))
            loops.append((headers, rows))
            continue
        if line.startswith("_"):
            tokens = _split_cif_line(line)
            if len(tokens) >= 2:
                tags[_normalize_tag(tokens[0])] = tokens[1]
            elif len(tokens) == 1 and index + 1 < len(lines):
                tags[_normalize_tag(tokens[0])] = lines[index + 1]
                index += 1
        index += 1

    return tags, loops, data_name


def _logical_lines(text: str) -> Iterable[str]:
    multiline_tag = None
    multiline_value: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if multiline_tag is not None:
            if line.startswith(";"):
                yield f"{multiline_tag} {' '.join(multiline_value)}"
                multiline_tag = None
                multiline_value = []
            else:
                multiline_value.append(line)
            continue

        stripped = _strip_comment(line).strip()
        if not stripped:
            continue
        if stripped.startswith("_") and stripped.endswith(" ;"):
            multiline_tag = stripped[:-2].strip()
            multiline_value = []
            continue
        yield stripped


def _strip_comment(line: str) -> str:
    quote = None
    for index, char in enumerate(line):
        if char in {"'", '"'}:
            quote = None if quote == char else char
        elif char == "#" and quote is None:
            return line[:index]
    return line


def _split_cif_line(line: str) -> List[str]:
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _normalize_tag(tag: str) -> str:
    lowered = tag.lower()
    if lowered.startswith("_atom_site_"):
        return "_atom_site." + lowered[len("_atom_site_") :]
    return lowered


def _required_float(tags: Dict[str, str], key: str) -> float:
    try:
        return parse_cif_number(tags[key])
    except KeyError as exc:
        raise ValueError(f"CIF is missing required tag {key}.") from exc


def _find_atom_loop(loops):
    for headers, rows in loops:
        if any(header.startswith("_atom_site.") for header in headers):
            return rows
    return []


def _atoms_from_loop(rows, lattice: np.ndarray) -> List[CifAtom]:
    atoms = []
    for row in rows:
        label = row.get("_atom_site.label") or row.get("_atom_site.id") or "Atom"
        element = normalize_element(
            row.get("_atom_site.type_symbol") or row.get("_atom_site.label") or label
        )

        if all(key in row for key in _FRACT_KEYS):
            fract = np.array(
                [parse_cif_number(row[key]) for key in _FRACT_KEYS], dtype=float
            )
            cart = fractional_to_cartesian(fract, lattice)
        elif all(key in row for key in _CART_KEYS):
            cart = np.array(
                [parse_cif_number(row[key]) for key in _CART_KEYS], dtype=float
            )
            fract = np.linalg.solve(lattice.T, cart)
        else:
            continue

        occupancy = None
        if "_atom_site.occupancy" in row:
            try:
                occupancy = parse_cif_number(row["_atom_site.occupancy"])
            except ValueError:
                occupancy = None

        g = row.get("_atom_site.disorder_group")
        a = row.get("_atom_site.disorder_assembly")

        def _normalize_disorder_string(val) -> Optional[str]:
            if val is None:
                return None
            cleaned = str(val).strip().strip("'\"")
            if cleaned in {".", "?", ""}:
                return None
            return cleaned

        g = _normalize_disorder_string(g)
        a = _normalize_disorder_string(a)

        atoms.append(
            CifAtom(
                label=label,
                element=element,
                fract=fract,
                cart=cart,
                occupancy=occupancy,
                disorder_group=g,
                disorder_assembly=a,
            )
        )
    return atoms


_FRACT_KEYS = (
    "_atom_site.fract_x",
    "_atom_site.fract_y",
    "_atom_site.fract_z",
)
_CART_KEYS = (
    "_atom_site.cartn_x",
    "_atom_site.cartn_y",
    "_atom_site.cartn_z",
)

_COVALENT_RADII = {
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03,
    "Ca": 1.76,
    "Sc": 1.70,
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Ga": 1.22,
    "Ge": 1.20,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "Kr": 1.16,
    "Rb": 2.20,
    "Sr": 1.95,
    "Y": 1.90,
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "Tc": 1.47,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "In": 1.42,
    "Sn": 1.39,
    "Sb": 1.39,
    "Te": 1.38,
    "I": 1.39,
    "Xe": 1.40,
    "Cs": 2.44,
    "Ba": 2.15,
    "La": 2.07,
    "Ce": 2.04,
    "Pr": 2.03,
    "Nd": 2.01,
    "Pm": 1.99,
    "Sm": 1.98,
    "Eu": 1.98,
    "Gd": 1.96,
    "Tb": 1.94,
    "Dy": 1.92,
    "Ho": 1.92,
    "Er": 1.89,
    "Tm": 1.90,
    "Yb": 1.87,
    "Lu": 1.87,
    "Hf": 1.75,
    "Ta": 1.70,
    "W": 1.62,
    "Re": 1.51,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
    "Tl": 1.45,
    "Pb": 1.46,
    "Bi": 1.48,
}


def write_supercell_cif(
    path: str,
    structure: CifStructure,
    repeats: Tuple[int, int, int],
    keep_connected: bool = True,
    selected_disorder_key: Optional[str] = None,
) -> None:
    """Export the expanded supercell crystal structure as a P1 symmetry CIF file."""
    repeat_a, repeat_b, repeat_c = repeats
    base_atoms = (
        unwrap_connected_atoms(structure) if keep_connected else list(structure.atoms)
    )

    if selected_disorder_key is not None:
        base_atoms = [
            atom
            for atom in base_atoms
            if atom.disorder_group is None
            or atom.disorder_group == selected_disorder_key
            or atom.disorder_key == selected_disorder_key
        ]

    new_a = structure.cell_lengths[0] * repeat_a
    new_b = structure.cell_lengths[1] * repeat_b
    new_c = structure.cell_lengths[2] * repeat_c
    alpha, beta, gamma = structure.cell_angles

    lines = [
        "data_supercell",
        "_audit_creation_method 'MoleditPy CIF Viewer Plugin Supercell Export'",
        f"_cell_length_a {new_a:.6f}",
        f"_cell_length_b {new_b:.6f}",
        f"_cell_length_c {new_c:.6f}",
        f"_cell_angle_alpha {alpha:.6f}",
        f"_cell_angle_beta {beta:.6f}",
        f"_cell_angle_gamma {gamma:.6f}",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        "",
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        "'x, y, z'",
        "",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]

    for ia in range(repeat_a):
        for ib in range(repeat_b):
            for ic in range(repeat_c):
                offset = np.array([ia, ib, ic], dtype=float)
                for base_index, atom in enumerate(base_atoms):
                    frac_super = (atom.fract + offset) / np.array(repeats, dtype=float)

                    # Generate a unique clean label
                    clean_label = re.sub(r"[^a-zA-Z0-9]", "", atom.label)
                    label = f"{clean_label}_{ia}_{ib}_{ic}"
                    occ = atom.occupancy if atom.occupancy is not None else 1.0
                    lines.append(
                        f"{label:<12} {atom.element:<3} {frac_super[0]:.6f} {frac_super[1]:.6f} {frac_super[2]:.6f} {occ:.4f}"
                    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
