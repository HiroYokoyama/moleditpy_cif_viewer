"""Finite crystal geometry, isotope and worker lifecycle regressions."""

from dataclasses import replace
from threading import Event
from unittest.mock import MagicMock

import numpy as np
import pytest
from test_parser import NACL_CIF

from cif_viewer import parser as cp
from cif_viewer import viewer
from cif_viewer.rdkit_bridge import render_atoms_to_rdkit_mol


@pytest.mark.parametrize("number", ["nan", "inf", "-inf", "1e999"])
def test_nonfinite_cif_numbers_are_rejected(number):
    with pytest.raises(ValueError, match="finite"):
        cp.parse_cif_number(number)


@pytest.mark.parametrize(
    "lengths,angles",
    [
        ((0, 4, 4), (90, 90, 90)),
        ((-1, 4, 4), (90, 90, 90)),
        ((np.nan, 4, 4), (90, 90, 90)),
        ((4, 4, 4), (60, 60, 120)),
        ((4, 4, 4), (np.inf, 90, 90)),
    ],
)
def test_invalid_cell_geometry_is_rejected(lengths, angles):
    with pytest.raises(ValueError, match="Invalid cell"):
        cp.cell_vectors(lengths, angles)


def test_builtin_parser_does_not_merge_structure_blocks():
    second = NACL_CIF.replace("data_NaCl", "data_other").replace("5.6402(2)", "12")
    parsed = cp.parse_cif(NACL_CIF + second)
    assert parsed.name == "NaCl"
    assert parsed.cell_lengths == pytest.approx((5.6402,) * 3)
    assert len(parsed.atoms) == 2


def test_incomplete_loop_is_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        cp.parse_cif(NACL_CIF + "\nC1 C 0\n")


def test_periodic_connectivity_is_bounded_before_pair_search():
    structure = cp.parse_cif(NACL_CIF)
    structure = replace(structure, atoms=(structure.atoms[0],) * 5001)
    with pytest.raises(ValueError, match="5,000 site limit"):
        cp._infer_periodic_adjacency(structure)


def test_symmetry_expansion_is_bounded_before_generating_atoms(monkeypatch):
    structure = cp.parse_cif(NACL_CIF)
    monkeypatch.setattr(cp, "get_space_group_operations", lambda *_: [None] * 100001)
    with pytest.raises(ValueError, match="atom limit"):
        cp._expand_to_unit_cell(structure, structure.atoms)


def test_oversize_file_is_rejected_before_open(tmp_path, monkeypatch):
    path = tmp_path / "large.cif"
    path.write_text(NACL_CIF)
    monkeypatch.setattr(cp.os.path, "getsize", lambda *_: cp.MAX_CIF_BYTES + 1)
    with pytest.raises(ValueError, match="64 MiB"):
        cp.parse_cif_file(path)
    with pytest.raises(ValueError, match="64 MiB"):
        cp.parse_cif_file_pymatgen(path)


def test_oversize_expansion_is_rejected_before_connectivity(monkeypatch, tmp_path):
    structure = cp.parse_cif(NACL_CIF)
    unwrap = MagicMock(side_effect=AssertionError("must not run"))
    monkeypatch.setattr(cp, "unwrap_connected_atoms", unwrap)
    with pytest.raises(ValueError, match="atom limit"):
        cp.expand_supercell(structure, (50, 50, 50))
    with pytest.raises(ValueError, match="atom limit"):
        cp.write_supercell_cif(
            tmp_path / "too-large.cif", structure, (50, 50, 50), keep_connected=False
        )
    unwrap.assert_not_called()


@pytest.mark.parametrize("repeats", [(np.nan, 1, 1), (1, 1, np.inf), (1, 1)])
def test_invalid_repeats_are_rejected(repeats):
    with pytest.raises(ValueError, match="three finite"):
        cp.normalize_repeats(repeats)


def test_deuterium_and_tritium_reach_rdkit_as_isotopes():
    atoms = [
        cp.RenderAtom(el, el, i, (0, 0, 0), np.array([i, 0, 0]))
        for i, el in enumerate(("D", "T"))
    ]
    mol = render_atoms_to_rdkit_mol(atoms, [])
    assert [(a.GetAtomicNum(), a.GetIsotope()) for a in mol.GetAtoms()] == [
        (1, 2),
        (1, 3),
    ]


def test_hiding_unbonded_hydrogens_keeps_atom_and_bond_indices_aligned(monkeypatch):
    atoms = [
        cp.RenderAtom(el, el, i, (0, 0, 0), np.array([i, 0, 0]))
        for i, el in enumerate(("H", "C", "D", "O", "T"))
    ]
    monkeypatch.setattr(
        cp, "expand_supercell", lambda *_, **kw: (atoms, [(1, 3), (0, 1)])
    )
    displayed, bonds, mol = viewer._run_render_calculation(
        cp.parse_cif(NACL_CIF),
        "Packing",
        None,
        (1, 1, 1),
        False,
        0.45,
        False,
        300,
        False,
        True,
    )
    assert [a.element for a in displayed] == ["C", "O"]
    assert bonds == [(0, 1)]
    assert mol.GetNumAtoms() == 2
    assert mol.GetNumBonds() == 1
    np.testing.assert_allclose(
        mol.GetConformer().GetPositions(), [[1, 0, 0], [3, 0, 0]]
    )


def test_failed_reload_preserves_structure_choices_and_repeats(qtbot, monkeypatch):
    window = viewer.CifViewerWidget()
    qtbot.addWidget(window)
    old = cp.parse_cif(NACL_CIF)
    window.structure = old
    window.all_structures = [old]
    window.repeat_a.blockSignals(True)
    window.repeat_a.setValue(3)
    monkeypatch.setattr(
        viewer, "parse_cif_file_pymatgen", MagicMock(side_effect=ValueError("invalid"))
    )
    monkeypatch.setattr(
        viewer, "parse_cif_file", MagicMock(side_effect=ValueError("invalid"))
    )
    monkeypatch.setattr(viewer.QMessageBox, "critical", lambda *_: None)
    window.load_cif("bad.cif")
    assert window.structure is old
    assert window.all_structures == [old]
    assert window.repeat_a.value() == 3


def test_closing_does_not_terminate_or_wait_for_native_worker(qtbot, monkeypatch):
    window = viewer.CifViewerWidget()
    qtbot.addWidget(window)
    started, finish = Event(), Event()

    def calculation(*_):
        started.set()
        finish.wait(5)
        return [], [], None

    monkeypatch.setattr(viewer, "_run_render_calculation", calculation)
    thread = viewer.RenderThread(
        cp.parse_cif(NACL_CIF),
        "Packing",
        None,
        (1, 1, 1),
        False,
        0.45,
        False,
        300,
        True,
        True,
        parent=window,
    )
    monkeypatch.setattr(
        thread, "terminate", MagicMock(side_effect=AssertionError("unsafe termination"))
    )
    monkeypatch.setattr(
        thread, "wait", MagicMock(side_effect=AssertionError("GUI must not wait"))
    )
    results = []
    thread.result_ready.connect(lambda *args: results.append(args))
    done = []
    thread.finished.connect(lambda: done.append(True))
    window._render_thread = thread
    thread.start()
    try:
        qtbot.waitUntil(started.is_set)
        window.render_timer.start(100)
        window.close()
        assert window._render_thread is None
        assert not window.render_timer.isActive()
        assert thread in viewer._RETIRED_RENDER_THREADS
    finally:
        finish.set()
        qtbot.waitUntil(lambda: bool(done))
    assert not results
