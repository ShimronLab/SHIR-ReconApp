"""
Fully-sampled reference reconstruction.

Two combination modes (selected at call time via *use_sense*):

  RSS  (default):
    read_raw_twix_data → permute (Nread, Nphase, Ncoils) → centred IFFT2 → RSS

  ESPIRiT + SENSE:
    … same read/permute … → normalise k-space → BART ecalib → BART pics -S
    Returns the SENSE-combined image plus the sensitivity maps.
"""

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np


from .utils import ensure_twix_importable as _add_twix_path


def _ifft2c(ks: np.ndarray) -> np.ndarray:
    """Centred 2-D IFFT over axes 0 (readout) and 1 (phase)."""
    return np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(ks, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )


def reconstruct_fullysamp(
    dat_path: str,
    *,
    use_sense: bool = False,
    calib_width: int = 24,
    espirit_crop: float = 0.8,      # ESPIRiT eigenvalue mask threshold (both engines)
    expected_nx: Optional[int] = None,
    tse: Optional[bool] = None,
    necho: Optional[int] = None,
    te1: Optional[float] = None,
    teeff: Optional[float] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    slice_selector_cb: Optional[Callable[[int], int]] = None,
) -> Dict[str, Any]:
    """
    Reconstruct a fully-sampled 2-D Cartesian scan.

    Parameters
    ----------
    dat_path : str
        Path to the .dat TWIX file.
    use_sense : bool, optional
        If True, compute ESPIRiT sensitivity maps with BART and combine
        using PICS (SENSE).  If False (default), use Root-Sum-of-Squares.
    calib_width : int, optional
        ESPIRiT calibration region width in pixels (used only when
        ``use_sense=True``).  Default 24.
    progress_cb : callable(str), optional
        Status callback.

    Returns
    -------
    dict with keys:
        kspace_rss     : (Nreadout, Nphase) float32 RSS k-space magnitude (for display)
        image_rss      : (Nreadout, Nphase)         RSS magnitude image
        image_combined : (Nreadout, Nphase)         SENSE combined image
                         (same as image_rss when use_sense=False)
        sens_maps      : (Nreadout, Nphase, 1, Ncoils) or None
        combination    : "rss" | "sense"
        params         : dict of parameters used

    Note: kspace_3d and per-coil images are not retained to reduce memory.
    """

    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    _add_twix_path()
    from read_siemens_scans.twix_reading import read_raw_twix_data
    from .utils import crop_readout_oversampling

    log(f"Reading TWIX data: {Path(dat_path).name}")
    kspace_raw, _ = read_raw_twix_data(str(dat_path), fix_oversampling=False)
    log(f"  Raw shape: {kspace_raw.shape}")

    # Multi-slice Siemens scans return (Nslices, Nacq_PE, Ncoils, Nreadout).
    # Select a single slice before the 3-axis transpose below, otherwise the
    # transpose fails with "axes don't match array".
    if kspace_raw.ndim == 4:
        n_slices = kspace_raw.shape[0]
        if slice_selector_cb is not None:
            sl_idx = int(slice_selector_cb(n_slices))
        else:
            sl_idx = n_slices // 2
        log(f"  Multi-slice scan ({n_slices} slices) — using slice {sl_idx}")
        kspace_raw = kspace_raw[sl_idx]

    # Twix reader returns (Nacq_PE, Ncoils, Nreadout)
    log(f"  Slice shape (Nacq, Ncoils, Nread): {kspace_raw.shape}")

    # Crop readout oversampling if the sequence was acquired with 2× OS.
    # The factor comes from the TWIX header; when absent, it is guessed by
    # comparing the readout size to expected_nx (falls back to n_phase).
    _exp_nx = expected_nx if expected_nx is not None else kspace_raw.shape[0]
    kspace_raw = crop_readout_oversampling(
        kspace_raw, expected_nx=_exp_nx, readout_axis=2, log_fn=log,
        dat_path=dat_path,
    )

    # TSE un-scramble: our Pulseq TSE sequences set no LIN label, so the reader
    # returns lines in echo-train (acquisition) order, not ascending PE order.
    # Reorder each acquired line to its true PE position before the transpose,
    # otherwise the fully-sampled TSE image is scrambled. SE / non-TSE → skipped.
    from .tse_reorder import detect_tse_params, unscramble_rows
    _tse = detect_tse_params(dat_path, tse=tse, necho=necho, te1=te1, teeff=teeff)
    if _tse is not None:
        log(f"  TSE detected — un-scrambling echo order (necho={_tse['necho']}).")
        kspace_raw = unscramble_rows(
            kspace_raw, _tse["necho"], _tse["TE1"], _tse["TEeff"], log_fn=log
        )

    kspace_3d = np.transpose(kspace_raw, (2, 0, 1)).astype(np.complex128)
    del kspace_raw   # free raw TWIX buffer immediately
    # (Nreadout, Nphase, Ncoils)
    log(f"  Rearranged shape: {kspace_3d.shape}")

    log("Reconstructing image via centred IFFT2 + RSS…")
    images    = _ifft2c(kspace_3d)                                # (Nr, Np, Nc)
    image_rss = np.sqrt(np.sum(np.abs(images) ** 2, axis=2))     # (Nr, Np)
    del images   # per-coil images not needed beyond RSS
    log(f"  Image shape: {image_rss.shape}")

    # Pre-compute 2-D RSS k-space magnitude for display (saves storing 3-D complex array)
    kspace_rss = np.sqrt(np.sum(np.abs(kspace_3d) ** 2, axis=2)).astype(np.float32)

    sens_maps     = None
    image_combined = image_rss   # default fallback

    if use_sense:
        from .recon_backend import active_backend_label, espirit_maps, pics

        log(f"Running ESPIRiT + SENSE combination ({active_backend_label()})…")
        try:
            # BART layout: (Nread, Nphase, 1, Ncoils)
            k4d = kspace_3d[:, :, np.newaxis, :].astype(np.complex64)
            del kspace_3d   # free the complex128 array now that k4d is built

            # Normalise k-space so ecalib is numerically stable
            try:
                import sigpy as sp
                img_tmp = sp.rss(sp.ifft(k4d, axes=(0, 1)), axes=-1)
                scale   = np.percentile(np.abs(img_tmp).ravel(), 95)
                k4d_norm = k4d / (scale + 1e-12)
            except ImportError:
                k4d_norm = k4d / (np.max(np.abs(k4d)) + 1e-12)

            log(f"  Computing sensitivity maps (ESPIRiT, calib={calib_width})…")
            sens_maps = espirit_maps(k4d_norm, calib_width, num_maps=1,
                                     crop=espirit_crop, log=log)

            log("  Running PICS (SENSE coil combination)…")
            combined_raw   = pics(k4d_norm, sens_maps, soft_sense=True, log=log)
            image_combined = np.abs(np.squeeze(combined_raw))   # (Nread, Nphase)
        except (RuntimeError, ImportError) as exc:
            log(f"  SENSE failed ({exc}); falling back to RSS")
            sens_maps      = None
            image_combined = image_rss

    else:
        del kspace_3d   # not needed in RSS-only path

    log("Fully-sampled reconstruction complete.")
    # NOTE: kspace_3d and images omitted from result to reduce memory;
    # kspace_rss (2-D float32) is stored instead for the k-space display panel.
    return {
        "kspace_rss":     kspace_rss,
        "image_rss":      image_rss,
        "image_combined": image_combined,
        "sens_maps":      sens_maps,
        "combination":    "sense" if use_sense else "rss",
        "params": {
            "use_sense":   use_sense,
            "calib_width": calib_width,
            "tse":         _tse,
        },
    }
