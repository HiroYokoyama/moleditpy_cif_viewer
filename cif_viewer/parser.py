from __future__ import annotations

from dataclasses import dataclass
import math
import re
import shlex
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_UNCERTAINTY_RE = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)(?:\(\d+\))?$")


@dataclass(frozen=True)
class CifAtom:
    label: str
    element: str
    fract: np.ndarray
    cart: np.ndarray
    occupancy: Optional[float] = None
    disorder_group: Optional[str] = None
    disorder_assembly: Optional[str] = None

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
    crystal_system: Optional[str] = None
    formula: Optional[str] = None
    r1: Optional[str] = None
    wr2: Optional[str] = None
    goof: Optional[str] = None
    is_asymmetric_unit_only: bool = False


@dataclass(frozen=True)
class RenderAtom:
    label: str
    element: str
    base_index: int
    image: Tuple[int, int, int]
    position: np.ndarray
    disorder_group: Optional[str] = None
    disorder_assembly: Optional[str] = None

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
    for key in keys:
        if key in block_data:
            vals = block_data[key]
            if vals is not None:
                val = vals[0] if isinstance(vals, (list, tuple)) else vals
                cleaned = str(val).strip().strip("'\"")
                if cleaned not in {".", "?", ""}:
                    return cleaned
    return None


def parse_cif_file_pymatgen(path: str) -> List[CifStructure]:
    from pymatgen.io.cif import CifParser
    parser = CifParser(path)
    structures = []
    
    for name, block in parser._cif.data.items():
        try:
            struct = parser._get_structure(block, primitive=False, symmetrized=False)
            if struct is None:
                continue
                
            label_to_adp = {}
            if "_atom_site_aniso_label" in block.data:
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
                        labels_list = block.data[label_col]
                        num_entries = len(labels_list)
                        
                        for i in range(num_entries):
                            lbl = labels_list[i]
                            try:
                                u_vals = {}
                                for suffix in ["11", "22", "33", "12", "13", "23"]:
                                    col = val_cols.get(suffix)
                                    val = 0.0
                                    if col and i < len(block.data[col]):
                                        val = parse_cif_number(block.data[col][i])
                                        if is_b:
                                            val = val / (8.0 * math.pi * math.pi)
                                    u_vals[suffix] = val
                                    
                                phonopy_order = [
                                    u_vals.get("11", 0.0), u_vals.get("22", 0.0), u_vals.get("33", 0.0),
                                    u_vals.get("23", 0.0), u_vals.get("13", 0.0), u_vals.get("12", 0.0)
                                ]
                                label_to_adp[lbl] = phonopy_order
                            except Exception:
                                pass
            
            u_cart_data = None
            if label_to_adp:
                A = struct.lattice.matrix.T
                A_inv = np.linalg.inv(A)
                N = np.diag([np.linalg.norm(x) for x in A_inv])

                # Build mapping of clean labels to their original asymmetric unit fractional coordinates
                label_to_frac_orig = {}
                if "_atom_site_label" in block.data:
                    labels = block.data["_atom_site_label"]
                    fx = block.data.get("_atom_site_fract_x", [])
                    fy = block.data.get("_atom_site_fract_y", [])
                    fz = block.data.get("_atom_site_fract_z", [])
                    for idx_lbl, lbl in enumerate(labels):
                        try:
                            clean_lbl = str(lbl).strip().strip("'\"")
                            x = parse_cif_number(fx[idx_lbl])
                            y = parse_cif_number(fy[idx_lbl])
                            z = parse_cif_number(fz[idx_lbl])
                            label_to_frac_orig[clean_lbl] = np.array([x, y, z])
                        except Exception:
                            pass

                # Get spacegroup symmetry operations
                symops = []
                try:
                    symops = parser.get_symops(block)
                except Exception:
                    pass
                if not symops:
                    try:
                        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                        sga = SpacegroupAnalyzer(struct)
                        symops = sga.get_space_group_operations()
                    except Exception:
                        pass

                u_cart_list = []
                for site in struct:
                    lbl = str(getattr(site, "label", "")).strip().strip("'\"")
                    if lbl in label_to_adp:
                        u_orig_vals = list(label_to_adp[lbl])
                        frac_orig = label_to_frac_orig.get(lbl)
                        
                        u11, u22, u33, u23, u13, u12 = u_orig_vals
                        U_frac_orig = np.array([
                            [u11, u12, u13],
                            [u12, u22, u23],
                            [u13, u23, u33]
                        ], dtype=float)
                        
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
                
                if u_cart_list and any(not np.allclose(m, 0.0) for m in u_cart_list):
                    u_cart_data = np.array(u_cart_list, dtype=float)

            # Build label_to_disorder map
            label_to_disorder = {}
            label_key = None
            group_key = None
            assembly_key = None
            for k in block.data.keys():
                k_norm = k.lower().lstrip("_")
                if k_norm == "atom_site_label":
                    label_key = k
                elif k_norm == "atom_site_disorder_group":
                    group_key = k
                elif k_norm == "atom_site_disorder_assembly":
                    assembly_key = k

            if label_key is not None:
                labels = block.data[label_key]
                groups = block.data[group_key] if group_key is not None else []
                assemblies = block.data[assembly_key] if assembly_key is not None else []
                for idx_lbl, lbl in enumerate(labels):
                    try:
                        clean_lbl = str(lbl).strip().strip("'\"")
                        g = str(groups[idx_lbl]).strip().strip("'\"") if idx_lbl < len(groups) else None
                        a = str(assemblies[idx_lbl]).strip().strip("'\"") if idx_lbl < len(assemblies) else None
                        if g in {".", "?", ""}:
                            g = None
                        if a in {".", "?", ""}:
                            a = None
                        label_to_disorder[clean_lbl] = (g, a)
                    except Exception:
                        pass

            # Extract cell metadata and refinement factors
            space_group = _get_first_tag_value(block.data, [
                "_space_group_name_h-m_alt",
                "_symmetry_space_group_name_h-m",
                "_space_group.symmetry_space_group_name_h-m"
            ])
            if not space_group:
                try:
                    space_group = struct.get_space_group_info()[0]
                except Exception:
                    pass
            
            crystal_system = _get_first_tag_value(block.data, [
                "_space_group_crystal_system",
                "_symmetry_cell_setting"
            ])
            if not crystal_system:
                try:
                    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                    sga = SpacegroupAnalyzer(struct)
                    crystal_system = sga.get_crystal_system()
                except Exception:
                    pass
            
            formula = _get_first_tag_value(block.data, [
                "_chemical_formula_sum",
                "_chemical_formula_structural",
                "_chemical_formula_moiety"
            ])
            if not formula:
                try:
                    formula = struct.formula
                except Exception:
                    pass

            r1 = _get_first_tag_value(block.data, [
                "_refine_ls_r_factor_gt",
                "_refine_ls_r_factor_obs",
                "_refine_ls_r_factor_all"
            ])
            wr2 = _get_first_tag_value(block.data, [
                "_refine_ls_wr_factor_ref",
                "_refine_ls_wr_factor_gt",
                "_refine_ls_wr_factor_all"
            ])
            goof = _get_first_tag_value(block.data, [
                "_refine_ls_goodness_of_fit_ref",
                "_refine_ls_goodness_of_fit_all"
            ])

            cif_struct = _structure_from_pymatgen(
                struct,
                name or "Structure",
                u_cart_data,
                label_to_disorder=label_to_disorder,
                space_group=space_group,
                crystal_system=crystal_system,
                formula=formula,
                r1=r1,
                wr2=wr2,
                goof=goof
            )
            structures.append(cif_struct)
        except Exception:
            pass
            
    return structures


def _structure_from_pymatgen(
    struct,
    name: str,
    u_cart: Optional[np.ndarray] = None,
    label_to_disorder: Optional[Dict[str, Tuple[Optional[str], Optional[str]]]] = None,
    space_group: Optional[str] = None,
    crystal_system: Optional[str] = None,
    formula: Optional[str] = None,
    r1: Optional[str] = None,
    wr2: Optional[str] = None,
    goof: Optional[str] = None,
) -> CifStructure:
    cell_lengths = struct.lattice.lengths
    cell_angles = struct.lattice.angles
    lattice = struct.lattice.matrix
    
    if label_to_disorder is None:
        label_to_disorder = {}
        
    atoms = []
    for i, site in enumerate(struct):
        label = getattr(site, "label", f"{site.species_string}{i+1}")
        element = normalize_element(site.species_string)
        fract = site.frac_coords
        cart = site.coords
        occupancy = getattr(site, "occupancy", 1.0)
        
        clean_lbl = str(label).strip().strip("'\"")
        g, a = label_to_disorder.get(clean_lbl, (None, None))
        
        atoms.append(CifAtom(
            label=label,
            element=element,
            fract=fract,
            cart=cart,
            occupancy=occupancy,
            disorder_group=g,
            disorder_assembly=a
        ))
        
    return CifStructure(
        name=name,
        cell_lengths=(float(cell_lengths[0]), float(cell_lengths[1]), float(cell_lengths[2])),
        cell_angles=(float(cell_angles[0]), float(cell_angles[1]), float(cell_angles[2])),
        lattice=lattice,
        atoms=tuple(atoms),
        u_cart=u_cart,
        space_group=space_group,
        crystal_system=crystal_system,
        formula=formula,
        r1=r1,
        wr2=wr2,
        goof=goof,
        is_asymmetric_unit_only=False
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

    space_group = _get_tag_value_dict(tags, [
        "_space_group_name_h-m_alt",
        "_symmetry_space_group_name_h-m",
        "_space_group.symmetry_space_group_name_h-m"
    ])
    crystal_system = _get_tag_value_dict(tags, [
        "_space_group_crystal_system",
        "_symmetry_cell_setting"
    ])
    formula = _get_tag_value_dict(tags, [
        "_chemical_formula_sum",
        "_chemical_formula_structural",
        "_chemical_formula_moiety"
    ])
    r1 = _get_tag_value_dict(tags, [
        "_refine_ls_r_factor_gt",
        "_refine_ls_r_factor_obs",
        "_refine_ls_r_factor_all"
    ])
    wr2 = _get_tag_value_dict(tags, [
        "_refine_ls_wr_factor_ref",
        "_refine_ls_wr_factor_gt",
        "_refine_ls_wr_factor_all"
    ])
    goof = _get_tag_value_dict(tags, [
        "_refine_ls_goodness_of_fit_ref",
        "_refine_ls_goodness_of_fit_all"
    ])

    return CifStructure(
        structure_name, lengths, angles, lattice, tuple(atoms),
        space_group=space_group,
        crystal_system=crystal_system,
        formula=formula,
        r1=r1,
        wr2=wr2,
        goof=goof,
        is_asymmetric_unit_only=True
    )


def cell_vectors(
    lengths: Sequence[float], angles_deg: Sequence[float]
) -> np.ndarray:
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
                    atoms.append(
                        RenderAtom(
                            label=atom.label,
                            element=atom.element,
                            base_index=base_index,
                            image=(ia, ib, ic),
                            position=atom.cart + cart_offset,
                            disorder_group=atom.disorder_group,
                            disorder_assembly=atom.disorder_assembly,
                        )
                    )

    return atoms, infer_bonds(atoms)


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
        unwrapped.append(CifAtom(
            atom.label, atom.element, fract, cart, atom.occupancy,
            disorder_group=atom.disorder_group,
            disorder_assembly=atom.disorder_assembly
        ))
    return unwrapped


def _infer_periodic_adjacency(structure: CifStructure):
    adjacency = {atom_index: [] for atom_index in range(len(structure.atoms))}
    for left_index in range(len(structure.atoms)):
        left_atom = structure.atoms[left_index]
        left_radius = covalent_radius(left_atom.element)
        for right_index in range(left_index + 1, len(structure.atoms)):
            right_atom = structure.atoms[right_index]
            right_radius = covalent_radius(right_atom.element)
            cutoff = min(2.45, left_radius + right_radius + 0.45)
            if left_atom.disorder_key is not None and right_atom.disorder_key is not None:
                if left_atom.disorder_key != right_atom.disorder_key:
                    continue
            delta_frac = np.asarray(right_atom.fract) - np.asarray(left_atom.fract)
            image_shift = -np.rint(delta_frac).astype(int)
            minimum_delta = delta_frac + image_shift
            distance = float(
                np.linalg.norm(fractional_to_cartesian(minimum_delta, structure.lattice))
            )
            if 0.25 <= distance <= cutoff:
                adjacency[left_index].append((right_index, image_shift))
                adjacency[right_index].append((left_index, -image_shift))
    return adjacency


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
    index = {(ia, ib, ic): idx for idx, (ia, ib, ic) in enumerate(
        (ia, ib, ic) for ia in (0, 1) for ib in (0, 1) for ic in (0, 1)
    )}
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
        (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]
    edges = [(corners[start], corners[end], "white", "") for start, end in edge_indices]
    return axis_lines + edges


def infer_bonds(atoms: Sequence[RenderAtom]) -> List[Tuple[int, int]]:
    bonds: List[Tuple[int, int]] = []
    max_cutoff = 2.45
    for left in range(len(atoms)):
        left_radius = covalent_radius(atoms[left].element)
        for right in range(left + 1, len(atoms)):
            if atoms[left].disorder_key is not None and atoms[right].disorder_key is not None:
                if atoms[left].disorder_key != atoms[right].disorder_key:
                    continue
            right_radius = covalent_radius(atoms[right].element)
            cutoff = min(max_cutoff, left_radius + right_radius + 0.45)
            distance = float(np.linalg.norm(atoms[left].position - atoms[right].position))
            if 0.25 <= distance <= cutoff:
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
                if current_lower == "loop_" or current.startswith("_") or current_lower.startswith("data_"):
                    break
                values.extend(_split_cif_line(current))
                index += 1

            rows = []
            if headers:
                width = len(headers)
                for start in range(0, len(values), width):
                    row_values = values[start:start + width]
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
        return "_atom_site." + lowered[len("_atom_site_"):]
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
            fract = np.array([parse_cif_number(row[key]) for key in _FRACT_KEYS], dtype=float)
            cart = fractional_to_cartesian(fract, lattice)
        elif all(key in row for key in _CART_KEYS):
            cart = np.array([parse_cif_number(row[key]) for key in _CART_KEYS], dtype=float)
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

        atoms.append(CifAtom(
            label=label,
            element=element,
            fract=fract,
            cart=cart,
            occupancy=occupancy,
            disorder_group=g,
            disorder_assembly=a
        ))
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
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
    "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
    "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
    "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01,
    "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92,
    "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87, "Lu": 1.87, "Hf": 1.75,
    "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36,
    "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48,
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
            atom for atom in base_atoms
            if atom.disorder_key is None or atom.disorder_key == selected_disorder_key
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

