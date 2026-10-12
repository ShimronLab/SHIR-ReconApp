"""Partial-Fourier fill (processing.partial_fourier) on a multi-coil phantom with
7/8 coverage: homodyne and POCS must beat zero-fill, POCS must keep every
measured line, and fully covered data must pass through untouched."""

import numpy as np
import pytest

from conftest import make_phantom
from processing.imaging import ifft2c, rss
from processing.partial_fourier import pf_fill, symmetric_band

PF_MISSING = 8


def seven_eighths(n_pe=64, high_side=True):
    _image, _maps, ksp = make_phantom(64, n_pe, 4)
    n_keep = n_pe * 7 // 8
    cat = np.ones(n_pe, dtype=np.uint8)                       # ACQ
    missing = slice(n_keep, n_pe) if high_side else slice(0, n_pe - n_keep)
    cat[missing] = PF_MISSING
    measured = ksp.copy()
    measured[:, missing, :] = 0
    return ksp, measured, cat


def error(k, truth):
    a, b = rss(ifft2c(k)), rss(ifft2c(truth))
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.mark.parametrize("high_side", [True, False])
@pytest.mark.parametrize("method", ["homodyne", "pocs"])
def test_fill_beats_zero_fill(method, high_side):
    truth, measured, cat = seven_eighths(high_side=high_side)
    zf = error(measured, truth)
    filled = error(pf_fill(measured, cat, method), truth)
    assert filled < zf


def test_pocs_keeps_measured_lines_exactly():
    _truth, measured, cat = seven_eighths()
    out = pf_fill(measured, cat, "pocs")
    keep = (cat & PF_MISSING) == 0
    assert np.array_equal(out[:, keep, :], measured[:, keep, :])
    assert np.abs(out[:, ~keep, :]).max() > 0                 # missing lines were filled


@pytest.mark.parametrize("method", ["zerofill", "homodyne", "pocs"])
def test_full_coverage_is_untouched(method):
    truth, _measured, _cat = seven_eighths()
    cat = np.ones(truth.shape[1], dtype=np.uint8)
    assert pf_fill(truth, cat, method) is truth


def test_zero_fill_is_identity_and_unknown_method_rejected():
    _truth, measured, cat = seven_eighths()
    assert pf_fill(measured, cat, "zerofill") is measured
    with pytest.raises(ValueError):
        pf_fill(measured, cat, "magic")


def test_symmetric_band_matches_demo_scan_geometry():
    """269-line grid, lines 0..234 measured, centre 134 → band 34..234."""
    missing = np.zeros(269, bool)
    missing[235:] = True
    assert symmetric_band(missing) == (34, 235)
