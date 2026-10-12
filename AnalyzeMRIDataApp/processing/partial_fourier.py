"""
Partial-Fourier fill along the phase-encode axis.

Input is multi-coil k-space on the full grid, (n_read, n_pe, n_coils), with the
k-space centre at row n_pe // 2 and the never-acquired rows marked
PF_MISSING in the per-line categories (processing.acquisition). Acceleration
gaps must already be filled (GRAPPA runs first); iterative SENSE/CS instead
treats partial-Fourier lines as unsampled and needs no separate step.

Methods (each coil independently):
  zerofill  leave the missing rows at zero (default; never invents data).
  homodyne  Noll et al. 1991: ramp weighting across the symmetric band plus a
            low-resolution phase estimate from that band; the result is the
            phase-corrected real image, returned as k-space.
  pocs      Haacke et al. 1991: impose the low-resolution phase, keep every
            measured line, repeat.

With no partial-Fourier rows the input array itself is returned unchanged, so
fully covered data (all current Pulseq scans) are untouched.
"""

from typing import Optional, Tuple

import numpy as np

from .imaging import fft2c, ifft2c

PF_METHODS = ("zerofill", "homodyne", "pocs")
_PF_MISSING = 8   # processing.acquisition.LineCat.PF_MISSING


def pf_fill(kspace: np.ndarray, line_category: np.ndarray, method: str = "zerofill",
            *, pocs_iters: int = 15) -> np.ndarray:
    """Fill partial-Fourier rows of ``kspace`` (see module doc)."""
    if method not in PF_METHODS:
        raise ValueError(f"unknown partial-Fourier method {method!r}; use one of {PF_METHODS}")
    missing = (np.asarray(line_category) & _PF_MISSING) != 0
    if not missing.any() or method == "zerofill":
        return kspace
    band = symmetric_band(missing)
    phase = _low_res_phase(kspace, band)
    if method == "homodyne":
        return _homodyne(kspace, missing, band, phase)
    return _pocs(kspace, missing, phase, pocs_iters)


def symmetric_band(missing: np.ndarray) -> Tuple[int, int]:
    """[start, end) rows measured on both sides of the centre."""
    n = missing.size
    centre = n // 2
    measured = np.flatnonzero(~missing)
    half = min(centre - measured.min(), measured.max() - centre)
    return centre - half, centre + half + 1


def _low_res_phase(kspace: np.ndarray, band: Tuple[int, int]) -> np.ndarray:
    """Per-coil image phase from the symmetric band, Hann-tapered to limit ringing."""
    lo, hi = band
    window = np.zeros(kspace.shape[1])
    window[lo:hi] = np.hanning(hi - lo + 2)[1:-1]
    return np.angle(ifft2c(kspace * window[None, :, None]))


def _homodyne_weights(missing: np.ndarray, band: Tuple[int, int]) -> np.ndarray:
    """2 on the measured-only side, a linear ramp 2→0 across the band, 0 where missing."""
    lo, hi = band
    w = np.where(missing, 0.0, 2.0)
    ramp = np.linspace(2.0, 0.0, hi - lo)
    missing_high = missing[hi:].any()
    w[lo:hi] = ramp if missing_high else ramp[::-1]
    return w


def _homodyne(kspace, missing, band, phase) -> np.ndarray:
    w = _homodyne_weights(missing, band)
    img = ifft2c(kspace * w[None, :, None])
    real_img = np.real(img * np.exp(-1j * phase))
    return fft2c(real_img * np.exp(1j * phase))


def _pocs(kspace, missing, phase, iters) -> np.ndarray:
    measured = ~missing
    estimate = kspace.copy()
    for _ in range(iters):
        img = ifft2c(estimate)
        estimate = fft2c(np.abs(img) * np.exp(1j * phase))
        estimate[:, measured, :] = kspace[:, measured, :]
    return estimate


def describe(method: str, line_category: Optional[np.ndarray]) -> str:
    """Short label for logs/reports, e.g. 'homodyne (34 lines filled)'."""
    if line_category is None:
        return "none"
    n = int(((np.asarray(line_category) & _PF_MISSING) != 0).sum())
    if n == 0:
        return "none (full coverage)"
    return f"{method} ({n} lines {'zero' if method == 'zerofill' else 'filled'})"
