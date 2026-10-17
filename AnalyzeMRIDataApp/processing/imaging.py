"""
Shared image-domain helpers: centred FFTs, coil combination, normalisation and
the reference-comparison metrics.

These were previously re-implemented (identically) in results_viewer,
report_generator, batch_report, reference and retro_undersample; this module is
the single definition. The numerics are byte-for-byte the historical ones —
any change here changes every reconstruction display/metric in the app.
"""

from typing import Any, Dict

import numpy as np


def ifft2c(ks: np.ndarray, axes=(0, 1)) -> np.ndarray:
    """Centred 2-D IFFT (readout, phase) — the app-wide convention."""
    return np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(ks, axes=axes), axes=axes), axes=axes
    )


def fft2c(img: np.ndarray, axes=(0, 1)) -> np.ndarray:
    """Centred 2-D FFT — inverse of :func:`ifft2c`."""
    return np.fft.ifftshift(
        np.fft.fft2(np.fft.fftshift(img, axes=axes), axes=axes), axes=axes
    )


def rss(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Root-sum-of-squares combination along ``axis`` (default: coils last)."""
    return np.sqrt(np.sum(np.abs(x) ** 2, axis=axis))


def espirit_normalise(k4d: np.ndarray, *, style: str = "clamp"):
    """
    Scale multi-coil k-space so its RSS image's 95th percentile is ~1, which
    keeps ESPIRiT / PICS numerically stable. Returns ``(k_norm, scale)``.

    ``style`` reproduces the two historical variants exactly (dtype included —
    numpy 2 promotes complex64 / numpy-float64 to complex128, but not
    complex64 / Python float):

    * ``"clamp"`` — Python-float scale floored at 1e-12; falls back to
      ``max|k|`` on any error. Keeps complex64 input complex64.
    * ``"add"``   — numpy-scalar scale plus 1e-12; falls back to ``max|k|``
      only when sigpy is missing (``scale`` is then None).
    """
    if style == "clamp":
        try:
            import sigpy as sp
            img = sp.rss(sp.ifft(k4d, axes=(0, 1)), axes=-1)
            scale = float(np.percentile(np.abs(img).ravel(), 95))
        except Exception:
            scale = float(np.max(np.abs(k4d)))
        scale = max(scale, 1e-12)
        return k4d / scale, scale
    if style == "add":
        try:
            import sigpy as sp
            img = sp.rss(sp.ifft(k4d, axes=(0, 1)), axes=-1)
            scale = np.percentile(np.abs(img).ravel(), 95)
            return k4d / (scale + 1e-12), scale
        except ImportError:
            return k4d / (np.max(np.abs(k4d)) + 1e-12), None
    raise ValueError(f"unknown style {style!r}")


def centre_block(n: int, width: int, *, clamp: bool = False):
    """
    ``(start, end)`` of a ``width``-line block centred on line ``n // 2`` —
    the calibration (ACS) region convention used across the app. ``clamp``
    keeps the block inside ``[0, n)``.
    """
    start = n // 2 - width // 2
    end = n // 2 + (width - width // 2)
    if clamp:
        start, end = max(0, start), min(n, end)
    return start, end


def crop_center(arr: np.ndarray, n: int, axis: int) -> np.ndarray:
    """
    Keep the central ``n`` samples along ``axis`` (centre convention: index
    ``size // 2`` stays the centre, as for the centred FFTs above).
    """
    size = arr.shape[axis]
    if n >= size:
        return arr
    start = size // 2 - n // 2
    index = [slice(None)] * arr.ndim
    index[axis] = slice(start, start + n)
    return arr[tuple(index)]


def display_aspect(shown_shape, spacing_mm, turns: int, kspace: bool = False) -> float:
    """
    Matplotlib ``aspect`` (height / width of one drawn pixel) for a panel that
    shows ``np.rot90(a, turns)`` of a (read, phase) array ``a``, so it keeps its
    physical proportions instead of stretching to the panel.

    ``shown_shape`` is the drawn array's shape; ``spacing_mm`` the image pixel
    size (read, phase), or None for square pixels. Image panels use the pixel
    size; k-space panels the k-space step 1 / (N · pixel) along each axis, so a
    square acquisition is square in both domains and angles are true.
    """
    d_read, d_phase = spacing_mm if spacing_mm else (1.0, 1.0)
    odd = turns % 2 == 1                     # odd: phase runs down the rows
    h, w = shown_shape[:2]
    n_read, n_phase = (w, h) if odd else (h, w)
    if kspace:
        s_read, s_phase = 1.0 / (n_read * d_read), 1.0 / (n_phase * d_phase)
    else:
        s_read, s_phase = d_read, d_phase
    return s_phase / s_read if odd else s_read / s_phase


def norm01(img: np.ndarray) -> np.ndarray:
    """Min-max normalise the magnitude of ``img`` to [0, 1]."""
    arr = np.abs(np.squeeze(img)).astype(float)
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)


def metrics_vs_reference(
    recon: np.ndarray,
    ref: np.ndarray,
    *,
    with_diff: bool = False,
) -> Dict[str, Any]:
    """
    RMSE / NMSE / PSNR / SSIM of ``recon`` against a same-shape reference.

    Both images are normalised by the *reference's* 95th percentile — one
    shared, outlier-robust intensity scale — before scoring, matching the
    difference maps shown in the viewer/PDF. ``ssim`` is None when it cannot
    be computed. ``with_diff=True`` adds the normalised absolute-difference
    image under ``"diff"``.
    """
    from skimage.metrics import structural_similarity as _ssim

    g = np.abs(np.squeeze(recon)).astype(float)
    r = np.abs(np.squeeze(ref)).astype(float)
    scale = float(np.percentile(r.ravel(), 95)) + 1e-12
    g = g / scale
    r = r / scale
    diff = g - r
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    nmse = float(np.sum(diff ** 2) / (np.sum(r ** 2) + 1e-12))
    psnr = float(20.0 * np.log10(1.0 / (rmse + 1e-12)))
    try:
        ssim = float(_ssim(g, r, data_range=1.0))
    except Exception:
        ssim = None
    out: Dict[str, Any] = {"rmse": rmse, "nmse": nmse, "psnr": psnr, "ssim": ssim}
    if with_diff:
        out["diff"] = np.abs(diff).astype(np.float32)
    return out
