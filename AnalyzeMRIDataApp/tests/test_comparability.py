"""Comparability gate: metrics against a reference are withheld when the
Siemens headers prove it is a different acquisition (sequence, TE/TR, slice,
FOV or image grid). Pulseq headers are placeholders and cannot be judged."""

import numpy as np
import pytest

import processing.utils as utils
from processing.acquisition import acquisition_mismatch
from processing.reference import compare_to_reference


def header(seq="se", te=(15000,), tr=666000, pos=(0.72, -69.39, -17.73),
           fov=(160.0, 160.0), thickness=5.0):
    return {
        "Config": {"SequenceFileName": f"%SiemensSeq%\\{seq}"},
        "MeasYaps": {
            "alTE": list(te) + [2000, 3000], "alTR": [tr], "lContrasts": len(te),
            "sSliceArray": {"asSlice": [{
                "sPosition": {"dSag": pos[0], "dCor": pos[1], "dTra": pos[2]},
                "sNormal": {"dCor": 1.0}, "dThickness": thickness,
                "dReadoutFOV": fov[0], "dPhaseFOV": fov[1]}]},
        },
    }


@pytest.fixture
def headers(monkeypatch):
    """headers[path] = header dict, served by read_twix_header."""
    table = {}
    monkeypatch.setattr(utils, "read_twix_header", lambda path: table.get(path))
    return table


def image(shape=(64, 48)):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (((yy - 30) / 22.0) ** 2 + ((xx - 22) / 16.0) ** 2 < 1).astype(float) + 0.1


def test_same_protocol_matches(headers):
    headers["a.dat"] = headers["b.dat"] = header()
    assert acquisition_mismatch("a.dat", "b.dat") == []


@pytest.mark.parametrize("change,reason", [
    (dict(seq="tse", te=(11000,)), "different sequence (tse vs se)"),
    (dict(te=(20000,)), "different TE (20 vs 15 ms)"),
    (dict(tr=500000), "different TR (500 vs 666 ms)"),
    (dict(pos=(0.72, -60.0, -17.73)), "different slice position"),
    (dict(fov=(220.0, 220.0)), "different FOV (220/220 vs 160/160 mm)"),
])
def test_each_difference_is_named(headers, change, reason):
    headers["recon.dat"], headers["ref.dat"] = header(**change), header()
    issues = acquisition_mismatch("recon.dat", "ref.dat")
    assert any(reason in issue for issue in issues), issues


def test_pulseq_cannot_be_judged(headers):
    headers["siemens.dat"] = header()
    headers["pulseq.dat"] = {"Config": {"SequenceFileName": "%CustomerSeq%\\pulseq"}}
    assert acquisition_mismatch("siemens.dat", "pulseq.dat") is None
    assert acquisition_mismatch("missing.dat", "siemens.dat") is None


def test_different_acquisition_withholds_metrics_unless_forced(headers):
    headers["tse.dat"], headers["se.dat"] = header(seq="tse", te=(11000,)), header()
    img = image()
    kw = dict(recon_name="tse.dat", reference_name="se.dat",
              recon_dat_path="tse.dat", reference_dat_path="se.dat")
    out = compare_to_reference(img, img, **kw)
    assert out["ssim"] is None and out["rmse"] is None and out["nmse"] is None
    assert "different sequence (tse vs se)" in out["reference_blocked_reason"]
    assert out["reference_image"].shape == img.shape          # still shown

    forced = compare_to_reference(img, img, score_anyway=True, **kw)
    assert forced["ssim"] == pytest.approx(1.0)
    assert forced["reference_blocked_reason"] is None
    assert forced["reference_warning"].startswith("Scored anyway — NOT a valid comparison")


def test_matching_reference_is_scored_without_warning(headers):
    headers["r4.dat"] = headers["fs.dat"] = header()
    img = image()
    out = compare_to_reference(img, img, recon_name="r4.dat", reference_name="fs.dat",
                               recon_dat_path="r4.dat", reference_dat_path="fs.dat")
    assert out["ssim"] == pytest.approx(1.0)
    assert out["reference_warning"] is None and out["reference_blocked_reason"] is None


def test_different_image_grid_is_blocked(headers):
    headers["a.dat"] = headers["b.dat"] = header()
    out = compare_to_reference(image((64, 48)), image((60, 48)),
                               recon_name="a.dat", reference_name="b.dat",
                               recon_dat_path="a.dat", reference_dat_path="b.dat")
    assert "different image grid (64×48 vs 60×48)" in out["reference_blocked_reason"]


@pytest.mark.gui
def test_viewer_score_anyway_toggle(qtbot, headers):
    from processing.constants import SCAN_POISSON
    from ui.results_viewer import ReconTab

    headers["tse.dat"], headers["se.dat"] = header(seq="tse", te=(11000,)), header()
    img = image()
    result = {"scan_type": SCAN_POISSON, "combined_img": img,
              "dat_path": "tse.dat", "reference_dat_path": "se.dat",
              **compare_to_reference(img, img, recon_name="tse.dat", reference_name="se.dat",
                                     recon_dat_path="tse.dat", reference_dat_path="se.dat")}
    tab = ReconTab(result, "T")
    qtbot.addWidget(tab)
    assert not tab._btn_score_anyway.isHidden()
    assert result["ssim"] is None

    tab._btn_score_anyway.setChecked(True)
    tab._toggle_score_anyway()
    assert result["ssim"] == pytest.approx(1.0)
    assert result["reference_warning"].startswith("Scored anyway")

    tab._btn_score_anyway.setChecked(False)
    tab._toggle_score_anyway()
    assert result["ssim"] is None and result["reference_blocked_reason"]
