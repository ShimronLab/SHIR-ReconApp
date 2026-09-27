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
