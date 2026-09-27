"""Numerics-equivalence tests for processing.imaging (the consolidated helpers).

These lock the historical formulas: any change to imaging.py that alters the
values below changes every metric/display in the app and must be deliberate.
"""

import numpy as np
from skimage.metrics import structural_similarity as _ssim

from processing.imaging import fft2c, ifft2c, metrics_vs_reference, norm01, rss


def test_ifft2c_fft2c_roundtrip():
    rng = np.random.default_rng(0)
    ks = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))
    assert np.allclose(fft2c(ifft2c(ks)), ks, atol=1e-12)


def test_rss():
    x = np.ones((4, 4, 9))
    assert np.allclose(rss(x), 3.0)


def test_norm01_range():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((16, 16)) * 7 + 3
    n = norm01(x)
    assert n.min() == 0.0 and abs(n.max() - 1.0) < 1e-9


def test_metrics_match_historical_formula():
    """metrics_vs_reference must reproduce the exact historical p95 arithmetic
    (formerly duplicated in reference.py / retro_undersample.py /
    results_viewer.py / batch_report.py)."""
    rng = np.random.default_rng(2)
    ref = np.abs(rng.standard_normal((48, 48))) + 1.0
    recon = ref + 0.05 * rng.standard_normal((48, 48))

    # Historical inline formula (verbatim):
    scale = float(np.percentile(np.abs(ref).ravel(), 95)) + 1e-12
    g, r = np.abs(recon) / scale, np.abs(ref) / scale
    diff = g - r
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    nmse = float(np.sum(diff ** 2) / (np.sum(r ** 2) + 1e-12))
    psnr = float(20.0 * np.log10(1.0 / (rmse + 1e-12)))
    ssim = float(_ssim(g, r, data_range=1.0))

    m = metrics_vs_reference(recon, ref, with_diff=True)
    assert abs(m["rmse"] - rmse) < 1e-12
    assert abs(m["nmse"] - nmse) < 1e-12
    assert abs(m["psnr"] - psnr) < 1e-12
    assert abs(m["ssim"] - ssim) < 1e-12
    assert np.allclose(m["diff"], np.abs(diff), atol=1e-6)


def test_metrics_normalized_by_reference_p95_only():
    """Scaling the RECON must change the metrics; scaling the REFERENCE must
    rescale both onto the same relative scale (p95-of-reference convention)."""
    rng = np.random.default_rng(3)
    ref = np.abs(rng.standard_normal((32, 32))) + 1.0
    recon = ref * 1.1
    m1 = metrics_vs_reference(recon, ref)
    m2 = metrics_vs_reference(recon * 2, ref)         # recon scale matters
    assert m2["rmse"] > m1["rmse"]
    m3 = metrics_vs_reference(recon * 5, ref * 5)     # joint scale cancels
    assert abs(m3["rmse"] - m1["rmse"]) < 1e-9
