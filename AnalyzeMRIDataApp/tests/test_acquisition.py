"""Full-grid assembly of Siemens k-space from header + per-line metadata
(processing.acquisition), on synthetic files shaped like real ones:
a 7/8 partial-Fourier SE (269 lines, 235 stored, centre 134) and an R4 GRE
with integrated reference lines and an asymmetric echo."""

import numpy as np
import pytest

import processing.utils as utils
from processing.acquisition import (AcquisitionConflict, LineCat, UnsupportedAcquisition,
                                    acquire_cartesian, assemble_siemens, finalize_image,
                                    result_extras)
from read_siemens_scans.twix_reading import TwixMeta


def header(n_pe, pf=16, dim=2, read_os=2.0, phase_os=0.0, ref_mode=1, fov=(160.0, 160.0),
           fourier_cols=None):
    meas = {"flReadoutOSFactor": read_os, "flPhaseOS": phase_os}
    if fourier_cols:
        meas["iNoOfFourierColumns"] = float(fourier_cols)
    return {
        "Config": {"SequenceFileName": "%SiemensSeq%\\se"},
        "Meas": meas,
        "MeasYaps": {
            "sKSpace": {"lPhaseEncodingLines": n_pe, "ucPhasePartialFourier": pf,
                        "ucDimension": dim},
            "sPat": {"ucRefScanMode": ref_mode},
            "sSliceArray": {"asSlice": [{"dReadoutFOV": fov[0], "dPhaseFOV": fov[1]}]},
        },
    }


def line_table(lin, centre_lin, centre_col, ncol, sli=None, ref=None):
    lin = np.asarray(lin)
    zeros = np.zeros(lin.size, dtype=int)
    return {"Lin": lin, "Sli": zeros if sli is None else np.asarray(sli), "Par": zeros,
            "Eco": zeros, "Rep": zeros, "Set": zeros, "Ave": zeros,
            "CenterLin": np.full(lin.size, centre_lin), "CenterCol": np.full(lin.size, centre_col),
            "ncol": np.full(lin.size, ncol),
            "ref": np.zeros(lin.size, bool) if ref is None else np.asarray(ref)}


def reader_array(n_rows, n_coils, ncol, acquired_rows, dc_row):
    """(Lin, Cha, Col) like the TWIX reader: zeros except acquired rows; DC strongest."""
    rng = np.random.default_rng(0)
    k = np.zeros((n_rows, n_coils, ncol), dtype=np.complex64)
    k[acquired_rows] = 0.01 * (rng.standard_normal((len(acquired_rows), n_coils, ncol)))
    k[dc_row] += 10.0
    return k


def pf_se(read_os=2.0):
    """7/8 partial Fourier: 269-line grid, lines 0..234 stored, centre at 134."""
    raw = reader_array(235, 2, 64, np.arange(235), dc_row=134)
    meta = TwixMeta(header=header(269, pf=8, read_os=read_os), axes=["Lin", "Cha", "Col"],
                    lines=line_table(np.arange(235), 134, 32, 64))
    return raw, meta


def test_partial_fourier_centre_in_middle_and_missing_lines_marked():
    raw, meta = pf_se()
    asm = assemble_siemens(raw, meta)
    assert asm.kspace.shape == (32, 269, 2)                 # readout OS 2 removed
    cat = asm.line_category
    assert np.all(cat[:235] & LineCat.ACQ)
    assert np.all(cat[235:] == LineCat.PF_MISSING)          # 34 never-acquired lines
    assert np.argmax(np.sum(np.abs(asm.kspace) ** 2, axis=(0, 2))) == 269 // 2
    assert asm.pi_range == (0, 235)
    assert asm.calib_range == (34, 235)                     # symmetric 134 ± 100
    r_accel, r_pf, r_total = asm.r_parts()
    assert r_accel == pytest.approx(1.0) and r_pf == pytest.approx(269 / 235)
    assert asm.facts["partial_fourier"] == "7/8"
    assert asm.pixel_spacing_mm == pytest.approx((160 / 32, 160 / 269), abs=1e-4)
    assert not asm.facts.warnings


def test_integrated_reference_and_asymmetric_echo():
    """R4 GRE: 333-line grid, image lines 2..330 every 4th, reference lines
    154..177 (integrated), 532 readout samples with the echo at 212."""
    n_pe, centre, ncol, ccol = 333, 166, 532, 212
    image_lin = np.arange(2, 331, 4)
    ref_lin = np.arange(154, 178)
    raw = reader_array(331, 2, ncol, image_lin, dc_row=166 if 166 in image_lin else image_lin[41])
    ref_arr = reader_array(178, 2, ncol, ref_lin, dc_row=166)
    meta = TwixMeta(
        header=header(n_pe, read_os=2.0, phase_os=0.300781, ref_mode=2),
        axes=["Lin", "Cha", "Col"],
        lines=line_table(image_lin, centre, ccol, ncol, ref=np.isin(image_lin, ref_lin)),
        refscan=ref_arr, refscan_axes=["Lin", "Cha", "Col"],
        refscan_lines={"Lin": ref_lin, "Sli": np.zeros(ref_lin.size, int)},
    )
    asm = assemble_siemens(raw, meta)
    assert asm.kspace.shape == (320, 333, 2)                # 532 → 640 (echo) → 320 (OS)
    cat = asm.line_category
    assert np.all(cat[154:178] & LineCat.CALIB) and np.all(cat[154:178] & LineCat.ACQ)
    assert asm.calib_range == (154, 178)
    assert np.all(cat[[0, 1, 331, 332]] == LineCat.SKIP_ACCEL)
    assert not np.any(cat & LineCat.PF_MISSING)
    assert asm.n_pe_image == 256                            # 333 / 1.3008
    assert asm.facts["asymmetric_echo"] is True


def test_asymmetric_echo_padding_places_samples_after_the_gap():
    ncol, ccol = 53, 21                                     # grid 64, 11 early samples missing
    raw = reader_array(16, 1, ncol, np.arange(16), dc_row=8)
    meta = TwixMeta(header=header(16, read_os=1.0), axes=["Lin", "Cha", "Col"],
                    lines=line_table(np.arange(16), 8, ccol, ncol))
    asm = assemble_siemens(raw, meta)
    assert asm.kspace.shape == (64, 16, 1)
    assert np.all(asm.kspace[:11] == 0)
    assert np.array_equal(asm.kspace[11:, :, 0], raw[:, 0, :].T.astype(np.complex128))


def test_fourier_matrix_from_header_drops_samples_beyond_it():
    """Like the lab's T1 SE: 512 samples, echo at 188, 512 Fourier columns — the
    matrix stays 512 (base resolution × OS), centred on the echo; the 68
    samples past it are not used."""
    ncol, ccol = 64, 23
    raw = reader_array(16, 1, ncol, np.arange(16), dc_row=8)
    meta = TwixMeta(header=header(16, read_os=1.0, fourier_cols=64), axes=["Lin", "Cha", "Col"],
                    lines=line_table(np.arange(16), 8, ccol, ncol))
    asm = assemble_siemens(raw, meta)
    assert asm.kspace.shape == (64, 16, 1)                  # not 2·max(23, 41) = 82
    assert np.all(asm.kspace[:9] == 0)                      # 32 − 23 early samples missing
    np.testing.assert_array_equal(asm.kspace[9:, :, 0], raw[:, 0, :55].T.astype(np.complex128))
    assert any("9 readout samples" in w for w in asm.facts.warnings)


def test_finalize_image_crops_phase_oversampling():
    raw, meta = pf_se()
    asm = assemble_siemens(raw, meta)
    asm.n_pe_image = 200
    img = np.arange(32 * 269, dtype=float).reshape(32, 269)
    out = finalize_image(img, asm)
    assert out.shape == (32, 200)
    assert out[0, 100] == img[0, 269 // 2]                  # centre stays the centre
    assert finalize_image(img, None) is img


def test_multi_slice_uses_selected_slice():
    raw1, meta1 = pf_se()
    raw = np.stack([raw1 * 0, raw1, raw1 * 0])              # slice 1 holds the data
    lines = line_table(np.tile(np.arange(235), 3), 134, 32, 64,
                       sli=np.repeat([0, 1, 2], 235))
    meta = TwixMeta(header=meta1.header, axes=["Sli", "Lin", "Cha", "Col"], lines=lines)
    asm = assemble_siemens(raw, meta, slice_selector_cb=lambda n: 1)
    assert asm.slice_idx == 1 and asm.facts["n_slices"] == 3
    assert np.abs(asm.kspace).max() > 0


@pytest.mark.parametrize("change,error", [
    (lambda m: m.header["MeasYaps"]["sKSpace"].update(ucDimension=4), UnsupportedAcquisition),
    (lambda m: setattr(m, "axes", ["Eco", "Lin", "Cha", "Col"]), UnsupportedAcquisition),
    (lambda m: m.lines["CenterLin"].__setitem__(0, 120), AcquisitionConflict),
])
def test_refuses_instead_of_guessing(change, error):
    raw, meta = pf_se()
    change(meta)
    with pytest.raises(error):
        assemble_siemens(raw, meta)


def test_result_extras():
    raw, meta = pf_se()
    extras = result_extras(assemble_siemens(raw, meta))
    assert extras["geometry"]["n_pe_full"] == 269 and extras["geometry"]["n_pe_image"] == 269
    assert extras["acq_facts"]["n_pe_full"]["source"] == "protocol header"
    assert "PF 7/8" in extras["acq_summary"] and "235/269 lines" in extras["acq_summary"]
    assert result_extras(None) == {}


@pytest.fixture
def siemens_pf_file(monkeypatch):
    """Make 'pf_se.dat' read as the synthetic 7/8 partial-Fourier Siemens scan."""
    import read_siemens_scans.twix_reading as twix_reading
    raw, meta = pf_se()
    monkeypatch.setattr(utils, "read_twix_header", lambda _p: meta.header)

    def fake_reader(path, fix_oversampling=False, *, with_meta=False):
        return (raw.copy(), None, meta) if with_meta else (raw.copy(), None)
    monkeypatch.setattr(twix_reading, "read_raw_twix_data", fake_reader)
    return "pf_se.dat"


def test_fully_sampled_pipeline_uses_full_grid(siemens_pf_file):
    from processing.fullysamp_recon import reconstruct_fullysamp
    res = reconstruct_fullysamp(siemens_pf_file)
    assert res["image_rss"].shape == (32, 269)
    assert res["kspace_rss"].shape == (32, 269)
    assert np.all(res["kspace_rss"][:, 235:] == 0)          # never-acquired lines stay visible
    assert res["acq_summary"].startswith("PF 7/8")
    assert res["pf_method"] == "zerofill (34 lines zero)"
    filled = reconstruct_fullysamp(siemens_pf_file, pf_method="homodyne")
    assert filled["pf_method"] == "homodyne (34 lines filled)"
    assert not np.allclose(filled["image_rss"], res["image_rss"])


@pytest.mark.gui
def test_result_tab_shows_facts_strip_and_true_aspect(qtbot, siemens_pf_file):
    """The tab states the acquisition, draws a sampling strip beside the
    k-space (never on top), and shows the image with its real pixel shape."""
    from processing.constants import SCAN_FULLY_SAMPLED
    from processing.fullysamp_recon import reconstruct_fullysamp
    from ui.results_viewer import ReconTab

    res = reconstruct_fullysamp(siemens_pf_file, pf_method="homodyne")
    res.update(scan_type=SCAN_FULLY_SAMPLED, dat_path=siemens_pf_file)
    tab = ReconTab(res, "T")
    qtbot.addWidget(tab)
    assert tab._lbl_facts.text().startswith("Acquisition: PF 7/8 · 235/269 lines")
    assert not tab._btn_kfilled.isHidden()
    assert len(tab._strip_requests) == 1
    kspace_ax = tab._strip_requests[0][0]
    image_ax = next(ax for ax, _im, _d, main in tab._axes_images if main)
    # Image drawn as rot90(read × phase): rows are phase → aspect = Δphase / Δread.
    d_read, d_phase = res["pixel_spacing_mm"]
    assert image_ax.get_aspect() == pytest.approx(d_phase / d_read)
    # k-space keeps its physical proportions too: Δk = 1 / (N · pixel) per axis
    n_read, n_phase = np.shape(res["kspace_rss"])[:2]
    assert kspace_ax.get_aspect() == pytest.approx((n_read * d_read) / (n_phase * d_phase))
    # child insets: the sampling strip left of the data, the colorbar right of it —
    # beside the image, never on top of it
    tab._canvas.draw()
    box = kspace_ax.get_position()
    kids = sorted(kspace_ax.child_axes, key=lambda c: c.get_position().x0)
    assert len(kids) == 2
    assert kids[0].get_position().x1 <= box.x0 + 1e-6        # strip, left of the data
    assert kids[1].get_position().x0 >= box.x1 - 1e-6        # colorbar, right of it

    tab._on_rotate()                                        # odd → even total rotation
    assert next(ax for ax, _im, _d, main in tab._axes_images if main).get_aspect() == \
        pytest.approx(d_read / d_phase)


def test_reference_loader_uses_full_grid(siemens_pf_file):
    from processing.reference import load_reference_rss
    assert load_reference_rss(siemens_pf_file).shape == (32, 269)


def test_noise_whitener_decorrelates_coils():
    from processing.acquisition import noise_whitener
    rng = np.random.default_rng(1)
    mix = np.array([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.3, 0.5, 0.9]])
    white = rng.standard_normal((50000, 3)) + 1j * rng.standard_normal((50000, 3))
    noise = white @ mix.T                                  # correlated coils
    W, max_corr = noise_whitener(noise)
    assert max_corr > 0.5
    y = noise @ W.T                                        # coil c ← Σ_d W[c, d] n_d
    cov = (y.conj().T @ y / len(y)).T
    var = np.real(np.diag(noise.conj().T @ noise / len(noise))).mean()
    assert np.allclose(cov, var * np.eye(3), atol=0.03 * var)


def test_prewhitening_applied_with_noise_and_switchable(monkeypatch):
    import processing.acquisition as acq
    raw, meta = pf_se()
    rng = np.random.default_rng(2)
    meta.noise = (rng.standard_normal((4000, 2)) + 1j * rng.standard_normal((4000, 2))) \
        @ np.array([[1.0, 0.0], [0.9, 0.4]]).T
    on = assemble_siemens(raw, meta)
    assert on.facts["prewhitened"] is True and "prewhitened" in on.facts.summary()
    monkeypatch.setattr(acq, "PREWHITEN", False)
    off = assemble_siemens(raw, meta)
    assert off.facts["prewhitened"] is False
    assert not np.allclose(on.kspace, off.kspace)
    assert np.array_equal(on.line_category, off.line_category)   # empty lines stay empty


def test_pulseq_and_unreadable_files_keep_the_old_path(monkeypatch):
    assert acquire_cartesian("/no/such/file.dat") is None
    monkeypatch.setattr(utils, "read_twix_header",
                        lambda _p: {"Config": {"SequenceFileName": "%CustomerSeq%\\pulseq"}})
    assert acquire_cartesian("scan.dat") is None


def test_trajectory_facts_describe_a_band():
    """Band / Poisson results get the same facts table as Cartesian ones, read
    from the trajectory: readouts, k-step and FOV, and the band's angle."""
    from processing.acquisition import Src, band_angle_deg, trajectory_facts
    n_lines, n_samples, dk = 6, 32, 4.0                       # 1/m → FOV 250 mm
    th = np.radians(30.0)
    along, across = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    u = (np.arange(n_samples) - n_samples // 2) * dk
    v = (np.arange(n_lines) - n_lines // 2) * dk
    traj = (u[None, :, None] * along + v[:, None, None] * across).reshape(-1, 2)
    assert band_angle_deg(traj) == pytest.approx(30.0)
    out = trajectory_facts("missing.dat", traj, n_lines, n_samples, 8,
                           json_path="/x/Bands.json", json_key="2",
                           rows=[("band", 2, Src.TRAJ_JSON, "")], summary="Band 2",
                           pixel_mm=(1.0, 1.0))
    f = out["acq_facts"]
    assert f["geometry source"]["detail"] == "Bands.json [2]"
    assert (f["readouts"]["value"], f["samples per readout"]["value"], f["coils"]["value"]) == (6, 32, 8)
    assert f["readout k-step (1/m)"]["value"] == pytest.approx(dk)
    assert f["readout FOV (mm)"]["value"] == pytest.approx(250.0)
    assert f["band"]["value"] == 2 and out["acq_summary"] == "Band 2"
    assert out["pixel_spacing_mm"] == (1.0, 1.0)
