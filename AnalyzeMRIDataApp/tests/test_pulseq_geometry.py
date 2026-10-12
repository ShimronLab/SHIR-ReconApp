"""Pulseq scans placed by the played .seq / trajectory JSON (processing.acquisition)."""

import json

import numpy as np
import pytest

from conftest import install_fake_twix_reader, make_phantom, to_reader_layout
import processing.acquisition as acq
import processing.seq_geometry as sg
import processing.utils as utils
from processing.acquisition import LineCat, Src, assemble_pulseq, image_entropy

N = 64


@pytest.fixture(scope="module")
def ksp():
    return make_phantom(N, N, 4)[2]                    # (read, pe, coils), k = 0 at row 32


def readouts(ksp, pe):
    """Reader layout (lines, coils, read) for readouts acquired at `pe` (k = 0 → 0)."""
    return to_reader_layout(ksp[:, np.asarray(pe) + N // 2, :])


def geo(**kw):
    out = {"readout_os": 1.0, "read_reversed": False, "pe_half_line": False,
           "swapped": False, "fov_phase_mm": 256.0, "fov_read_os_mm": 256.0}
    out.update(kw)
    return out


def assemble(raw, pe, **kw):
    return assemble_pulseq(raw, np.asarray(pe), geo(**kw), src=Src.SEQ, detail="x.seq")


def test_lines_land_on_their_true_rows_whatever_the_order(ksp):
    for order in (np.arange(-32, 32), np.arange(31, -33, -1),
                  np.random.default_rng(0).permutation(np.arange(-32, 32))):
        asm = assemble(readouts(ksp, order), order)
        np.testing.assert_allclose(asm.kspace, ksp)
        assert asm.facts["n_pe_full"] == N and asm.pixel_spacing_mm == (4.0, 4.0)


def test_equispaced_lines_mask_and_calibration_block(ksp):
    pe = np.union1d(np.arange(-32, 32, 4), np.arange(-6, 6))
    asm = assemble(readouts(ksp, pe), pe)
    assert asm.kspace.shape[1] == N and asm.pe_mask.sum() == pe.size
    assert asm.calib_range == (26, 38)
    assert np.all(asm.line_category[26:38] & LineCat.CALIB)
    assert asm.line_category[31] & LineCat.SKIP_ACCEL == 0 and asm.line_category[25] & LineCat.SKIP_ACCEL
    assert asm.r_parts()[1] == 1.0                      # no partial Fourier


def test_espirit_region_stays_inside_an_off_centre_calibration_block(ksp):
    from processing.acquisition import espirit_calib_width
    centred = np.union1d(np.arange(-32, 32, 4), np.arange(-6, 6))          # rows 26..37
    asm = assemble(readouts(ksp, centred), centred)
    assert asm.calib_range == (26, 38) and espirit_calib_width(asm, 24) == 12
    off = np.union1d(np.arange(-32, 32, 4), np.arange(-5, 7))              # rows 27..38
    asm = assemble(readouts(ksp, off), off)
    assert asm.calib_range == (27, 39)
    w = espirit_calib_width(asm, 24)                  # 12 centred rows would be 26..37
    assert w == 10 and (N - w) // 2 == 27             # rows 27..36, all measured


def test_missing_outer_lines_of_an_accelerated_pattern_keep_the_square_grid(ksp):
    pe = np.union1d(np.arange(0, 31, 3), -np.arange(0, 31, 3))   # reaches ±30 of ±32
    asm = assemble(readouts(ksp, pe), pe)
    assert asm.kspace.shape[1] == N and asm.facts["partial_fourier"] == "off"


def test_one_sided_extent_is_partial_fourier(ksp):
    pe = np.arange(-32, 24)                             # 7/8 of 64 lines
    asm = assemble(readouts(ksp, pe), pe)
    assert asm.facts["partial_fourier"] == "7/8" and asm.pi_range == (0, 56)
    assert np.all(asm.line_category[56:] & LineCat.PF_MISSING)


def test_repeated_readouts_keep_the_first_and_all_repeated_refuses(ksp):
    pe = np.r_[np.arange(-32, 32), 0, 0]
    raw = readouts(ksp, pe)
    raw[-2:] *= 0.01                                    # weak filler echoes at k = 0
    asm = assemble(raw, pe)
    np.testing.assert_allclose(asm.kspace, ksp)
    assert any("repeated readout" in w for w in asm.facts.warnings)
    twice = np.repeat(np.arange(-32, 32), 2)
    assert assemble(readouts(ksp, twice), twice) is None


def test_reversed_readout_and_oversampling(ksp):
    order = np.arange(-32, 32)
    raw = readouts(ksp, order)[:, :, ::-1]               # samples from +k to −k
    asm = assemble(raw, order, read_reversed=True)
    np.testing.assert_allclose(asm.kspace, ksp)
    os_ksp = np.fft.fftshift(np.fft.fft(np.pad(np.fft.ifft(np.fft.ifftshift(ksp, axes=0), axis=0),
                                               ((0, N), (0, 0), (0, 0))), axis=0), axes=0)
    asm2 = assemble(readouts(os_ksp, order), order, readout_os=2.0, fov_read_os_mm=512.0)
    assert asm2.kspace.shape[0] == N and asm2.facts["read_os"] == 2
    assert asm2.pixel_spacing_mm == (4.0, 4.0)


def test_lines_shifted_by_one_fit_the_grid_with_a_warning(ksp):
    pe = np.arange(-31, 33)                             # one line above the usual −32…31
    asm = assemble(readouts(np.roll(ksp, -1, axis=1), pe - 1), pe)
    assert asm.kspace.shape[1] == N
    assert any("from the grid centre" in w for w in asm.facts.warnings)


# ── Where the geometry comes from ──────────────────────────────────────────────

@pytest.fixture
def pulseq_file(tmp_path, monkeypatch, ksp):
    dat_dir = tmp_path / "scan" / "dat_files"
    dat_dir.mkdir(parents=True)
    dat = dat_dir / "meas_fully_sampled_se.dat"
    dat.write_bytes(b"")
    order = np.arange(31, -33, -1)                       # top-down, like the lab's SE sequence
    install_fake_twix_reader(monkeypatch, readouts(ksp, order))
    monkeypatch.setattr(utils, "sequence_family", lambda _p: "pulseq")
    monkeypatch.setattr(sg, "played_seq", lambda _p: {"name": "se.seq", "signature": "a" * 32,
                                                      "path": None, "note": "not found"})
    sg.set_seq_request_handler(None)
    return dat, order


def test_played_seq_geometry_is_used(pulseq_file, monkeypatch, ksp):
    dat, order = pulseq_file
    monkeypatch.setattr(sg, "played_seq", lambda _p: {"name": "se.seq", "signature": "a" * 32,
                                                      "path": "se.seq", "note": ""})
    monkeypatch.setattr(sg, "seq_line_geometry",
                        lambda _p: geo(pe_index=order, samples_per_adc=N, signature="a" * 32))
    asm = acq.acquire_cartesian(str(dat))
    np.testing.assert_allclose(asm.kspace, ksp)
    assert asm.facts["geometry_source"] == Src.SEQ.value


def test_trajectory_json_is_used_without_the_seq(pulseq_file, ksp):
    dat, order = pulseq_file
    ky = np.repeat(order.astype(float), N) * 3.90625                    # 1/m, FOV 256 mm
    kx = np.tile(np.arange(-N // 2, N // 2) * 3.90625, order.size)
    (dat.parent.parent / "FullySampled_se.json").write_text(json.dumps({"0": [kx.tolist(), ky.tolist()]}))
    asm = acq.acquire_cartesian(str(dat))
    np.testing.assert_allclose(asm.kspace, ksp)
    assert asm.facts["geometry_source"] == Src.TRAJ_JSON.value
    assert asm.pixel_spacing_mm == (4.0, 4.0)


def test_neither_seq_nor_json_falls_back_to_the_legacy_path(pulseq_file):
    dat, _ = pulseq_file
    assert acq.acquire_cartesian(str(dat)) is None


def test_shared_json_with_se_order_yields_to_the_tse_echo_plan(tmp_path, monkeypatch, ksp):
    from processing.tse_reorder import tse_acq_to_line
    monkeypatch.setattr(utils, "sequence_family", lambda _p: "pulseq")
    plan = tse_acq_to_line(N, 16, 0.012, 0.096) - N // 2            # true TSE echo order
    raw = readouts(ksp, plan)
    sequential = np.arange(-32, 32)                                  # what an SE JSON says
    pe, detail = acq._check_echo_order("x_tse_t2.dat", raw, sequential, "j.json", lambda m: None)
    np.testing.assert_array_equal(pe, plan)
    assert "echo-train plan" in detail
    assert image_entropy(raw, sequential) > image_entropy(raw, plan) + 0.1
    pe2, _ = acq._check_echo_order("x_tse_t2.dat", raw, plan, "j.json", lambda m: None)
    np.testing.assert_array_equal(pe2, plan)          # a correct JSON order is kept


# ── Asking for a missing .seq ──────────────────────────────────────────────────

def test_missing_seq_is_asked_for_once_and_must_match(tmp_path, monkeypatch):
    sig = "b" * 32
    monkeypatch.setattr(sg, "played_seq", lambda _p: {"name": "x.seq", "signature": sig,
                                                      "path": None, "note": "not found"})
    good = tmp_path / "found" / "x.seq"
    good.parent.mkdir()
    good.write_text(f"[SIGNATURE]\nType md5\nHash {sig}\n")
    asked = []
    sg.set_seq_request_handler(lambda info: asked.append(info["name"]) or str(good))
    monkeypatch.setattr(sg, "_ASKED", {})
    monkeypatch.setattr(sg, "_EXTRA_DIRS", [])
    try:
        assert sg.locate_played_seq("a.dat")["path"] == str(good)
        assert sg.locate_played_seq("b.dat")["path"] == str(good)
        assert asked == ["x.seq"] and str(good.parent) in sg._EXTRA_DIRS
        monkeypatch.setattr(sg, "_ASKED", {})
        good.write_text("[SIGNATURE]\nHash " + "c" * 32 + "\n")    # a regenerated version
        assert sg.locate_played_seq("a.dat")["path"] is None
    finally:
        sg.set_seq_request_handler(None)


def test_find_line_json_rules(tmp_path):
    root = tmp_path / "scan"
    (root / "dat_files").mkdir(parents=True)
    for name in ("FullySampled_tse_t2.json", "FullySampled_swapPE_tse_t2.json",
                 "Undersampled2equispaces.json", "Undersampled2equispaces_swapPE.json"):
        (root / name).write_text("{}")
    d = root / "dat_files"
    assert sg.find_line_json(str(d / "meas_fully_sampled_tse_t2.dat")).endswith("FullySampled_tse_t2.json")
    assert sg.find_line_json(str(d / "m_fully_sampled_swapPE_tse_t2.dat")).endswith("swapPE_tse_t2.json")
    assert sg.find_line_json(str(d / "m_equispaced_R2_acs24.dat")).endswith("Undersampled2equispaces.json")
    assert sg.find_line_json(str(d / "m_equispaced_R2_acs24_swapPE.dat")).endswith("_swapPE.json")
    assert sg.find_line_json(str(d / "m_fully_sampled_se_t1.dat")) is None
