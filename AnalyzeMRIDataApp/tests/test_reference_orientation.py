"""Reference auto-orientation (processing.reference) and the viewer's
screen↔array orientation mapping (ui.results_viewer)."""

import numpy as np
import pytest

from processing.reference import (
    ORIENTATIONS,
    auto_orient_reference,
    compare_to_reference,
    orient_image,
)


def _phantom(shape=(96, 80)):
    """Asymmetric 'head': off-centre ellipse + a bright blob in one quadrant, so
    every rotation/flip of it is distinguishable."""
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(float)
    cy, cx = shape[0] * 0.45, shape[1] * 0.55
    img = (((yy - cy) / (shape[0] * 0.38)) ** 2
           + ((xx - cx) / (shape[1] * 0.30)) ** 2 < 1).astype(float)
    img += 1.5 * (((yy - shape[0] * 0.30) ** 2 + (xx - shape[1] * 0.40) ** 2) < 36)
    img += 0.7 * (np.abs(xx - shape[1] * 0.65) < 3) * (np.abs(yy - cy) < shape[0] * 0.2)
    return img


@pytest.mark.parametrize("key", list(ORIENTATIONS))
def test_auto_orient_recovers_each_orientation(key):
    """A recon that is the reference seen in orientation `key` (plus noise) is
    matched by orienting the reference with `key`."""
    rng = np.random.default_rng(0)
    ref = _phantom()
    recon = orient_image(ref, key) + 0.05 * rng.standard_normal(
        orient_image(ref, key).shape)
    assert auto_orient_reference(recon, ref)["orientation"] == key


def test_auto_keeps_identity_on_symmetric_image():
    """Ties (a symmetric image) stay 'none' rather than flipping on noise."""
    yy, xx = np.mgrid[:64, :64] - 31.5
    ref = ((yy / 25) ** 2 + (xx / 25) ** 2 < 1).astype(float)
    assert auto_orient_reference(ref, ref)["orientation"] == "none"


def test_compare_auto_fixes_swapped_axes_and_metrics():
    """Readout/PE-swapped (non-square) reference: auto orients it, so there is
    no spurious matrix-mismatch warning and SSIM is high; forcing 'none'
    reproduces the bad, mismatched comparison."""
    ref = _phantom((96, 80))
    recon = orient_image(ref, "rot270")                       # 80×96
    good = compare_to_reference(recon, ref, recon_name="a.dat",
                                reference_name="b.dat")
    assert good["reference_orientation"] == "rot270"
    assert good["reference_orientation_auto"] is True
    assert good["reference_warning"] is None
    assert good["ssim"] > 0.99
    assert good["reference_image"].shape == recon.shape
    assert good["reference_image_raw"].shape == ref.shape    # untouched copy

    bad = compare_to_reference(recon, ref, recon_name="a.dat",
                               reference_name="b.dat", orientation="none")
    assert bad["reference_orientation_auto"] is False
    assert "matrix mismatch" in bad["reference_warning"]
    assert bad["ssim"] < good["ssim"]


def test_unknown_orientation_rejected():
    with pytest.raises(ValueError):
        orient_image(np.zeros((4, 4)), "sideways")


@pytest.mark.parametrize("k", range(4))
def test_viewer_screen_array_mapping(k):
    """The viewer labels ops in screen space: for display rotation k, the array
    op chosen must look like the screen op once drawn, and the mapping must be
    a bijection that inverts cleanly."""
    from ui.results_viewer import _array_to_screen_orient, _screen_to_array_orient

    x = np.arange(15.0).reshape(3, 5)
    seen = set()
    for screen in ORIENTATIONS:
        a = _screen_to_array_orient(screen, k)
        np.testing.assert_array_equal(
            np.rot90(orient_image(x, a), k), orient_image(np.rot90(x, k), screen))
        assert _array_to_screen_orient(a, k) == screen
        seen.add(a)
    assert seen == set(ORIENTATIONS)


def test_ambiguous_flag_on_mirror_pair():
    """Left-right-symmetric content rotated 90°: rot90 and its mirror
    (transpose) score the same, so the pick is flagged as a close call."""
    yy, xx = np.mgrid[:96, :80].astype(float)
    ref = (((yy - 40) / 35) ** 2 + ((xx - 39.5) / 25) ** 2 < 1).astype(float)
    res = auto_orient_reference(np.rot90(ref), ref)
    assert res["orientation"] != "none"
    assert res["ambiguous"] is True
    assert auto_orient_reference(ref, ref)["ambiguous"] is False


def test_reference_pairing_matches_swappe(tmp_path):
    """Each scan gets the fully-sampled reference with the same swapPE-ness,
    whichever sorts first on disk."""
    from processing.constants import SCAN_EQUISPACED, SCAN_POISSON
    from ui.main_window import _attach_references

    names = ["m516_fully_sampled_swapPE_tse_t2.dat", "m517_fully_sampled_tse_t2.dat",
             "m510_equispaced_R2_acs24_swapPE_tse_t2.dat",
             "m511_equispaced_R2_acs24_tse_t2.dat", "m518_PoissonDisc_tse_t2.dat"]
    for n in names:
        (tmp_path / n).write_bytes(b"")
    p = {n: str(tmp_path / n) for n in names}
    jobs = [
        {"dat_path": p[names[2]], "scan_type": SCAN_EQUISPACED},
        {"dat_path": p[names[3]], "scan_type": SCAN_EQUISPACED},
        {"dat_path": p[names[4]], "scan_type": SCAN_POISSON},
    ]
    jobs = _attach_references(jobs)
    assert jobs[0]["reference_dat_path"] == p[names[0]]      # swapPE → swapPE
    assert jobs[1]["reference_dat_path"] == p[names[1]]      # plain → plain
    assert jobs[2]["reference_dat_path"] == p[names[1]]      # Poisson → plain


def test_pulseq_scan_is_not_scored_against_a_product_fully_sampled(tmp_path, monkeypatch):
    """A Siemens product scan named '…fully_sampled…' sorts before the Pulseq
    one ('S' < 'm'); the Pulseq scan must still get the Pulseq reference."""
    import processing.utils as utils
    from processing.constants import SCAN_EQUISPACED, SCAN_FULLY_SAMPLED
    from ui.main_window import _attach_references

    names = ["Siemens_meas_t1_se_tra_fully_sampled.dat", "meas_band_se_fully_sampled.dat",
             "meas_equispaced_R2_acs24.dat"]
    p = {n: str(tmp_path / n) for n in names}
    for path in p.values():
        open(path, "wb").close()
    monkeypatch.setattr(utils, "sequence_family",
                        lambda path: "siemens" if "Siemens_" in str(path) else "pulseq")
    jobs = [{"dat_path": p[names[0]], "scan_type": SCAN_FULLY_SAMPLED},
            {"dat_path": p[names[1]], "scan_type": SCAN_FULLY_SAMPLED},
            {"dat_path": p[names[2]], "scan_type": SCAN_EQUISPACED}]
    jobs = _attach_references(jobs)
    assert jobs[2]["reference_dat_path"] == p[names[1]]


@pytest.mark.gui
def test_viewer_ref_orient_override(qtbot):
    """Poisson tab with a rotated reference: auto fixes it; a manual pick
    re-scores without a re-run; rotating the tab relabels but keeps it."""
    from processing.constants import SCAN_POISSON
    from ui.results_viewer import ReconTab

    ref = _phantom((96, 80))
    recon = orient_image(ref, "rot270")
    result = {"scan_type": SCAN_POISSON, "combined_img": recon,
              **compare_to_reference(recon, ref, recon_name="p.dat",
                                     reference_name="fs.dat")}
    tab = ReconTab(result, "T")
    qtbot.addWidget(tab)
    combo = tab._combo_ref_orient
    assert not combo.isHidden()
    assert combo.currentData() == "auto"
    good_ssim = result["ssim"]
    assert good_ssim > 0.99

    # Force "As loaded" → the bad (unrotated) comparison, no re-run.
    combo.setCurrentIndex(combo.findData("none"))
    tab._on_ref_orient_changed()
    assert result["reference_orientation"] == "none"
    assert result["ssim"] < good_ssim
    assert np.array_equal(tab._eq_base["ref"], result["reference_image"])

    # Back to Auto restores the good match.
    combo.setCurrentIndex(0)
    tab._on_ref_orient_changed()
    assert result["reference_orientation"] == "rot270"
    assert result["ssim"] == pytest.approx(good_ssim)

    # Manual rot270 (Poisson draws arrays as-is → screen "Rotate 90° CW"),
    # then rotate the tab: a rotation keeps its label, the applied op is unchanged.
    combo.setCurrentIndex(combo.findData("rot270"))
    tab._on_ref_orient_changed()
    tab._on_rotate()
    assert combo.currentData() == "rot270"
    assert result["reference_orientation"] == "rot270"
    assert "rotate 90° cw (manual)" in tab._ref_orient_note()
