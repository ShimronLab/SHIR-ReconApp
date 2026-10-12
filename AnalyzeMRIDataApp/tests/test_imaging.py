"""Numerics-equivalence tests for processing.imaging (the consolidated helpers).

These lock the historical formulas: any change to imaging.py that alters the
values below changes every metric/display in the app and must be deliberate.
"""

import numpy as np
from skimage.metrics import structural_similarity as _ssim

import pytest

from processing.imaging import (centre_block, espirit_normalise, fft2c, ifft2c,
                                metrics_vs_reference, norm01, rss)


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


def _k4d(dtype):
    rng = np.random.default_rng(4)
    k = rng.standard_normal((24, 20, 1, 4)) + 1j * rng.standard_normal((24, 20, 1, 4))
    return k.astype(dtype)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_espirit_normalise_clamp_matches_inline_formula(dtype):
    """Byte-identical to the inline code in equispaced/retro/poisson, dtype included."""
    import sigpy as sp
    k4d = _k4d(dtype)
    scale = float(np.percentile(np.abs(sp.rss(sp.ifft(k4d, axes=(0, 1)), axes=-1)).ravel(), 95))
    expected = k4d / max(scale, 1e-12)
    got, got_scale = espirit_normalise(k4d, style="clamp")
    assert got.dtype == expected.dtype == dtype
    assert np.array_equal(got, expected) and got_scale == max(scale, 1e-12)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_espirit_normalise_add_matches_inline_formula(dtype):
    """Byte-identical to the inline code in fullysamp/generic/band — including the
    numpy-scalar promotion of complex64 input to complex128."""
    import sigpy as sp
    k4d = _k4d(dtype)
    scale = np.percentile(np.abs(sp.rss(sp.ifft(k4d, axes=(0, 1)), axes=-1)).ravel(), 95)
    expected = k4d / (scale + 1e-12)
    got, _ = espirit_normalise(k4d, style="add")
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("n,w,clamp,expected", [
    (256, 24, False, (116, 140)),
    (255, 24, False, (115, 139)),
    (256, 25, False, (116, 141)),
    (10, 24, False, (-7, 17)),
    (10, 24, True, (0, 10)),
])
def test_centre_block(n, w, clamp, expected):
    assert centre_block(n, w, clamp=clamp) == expected
