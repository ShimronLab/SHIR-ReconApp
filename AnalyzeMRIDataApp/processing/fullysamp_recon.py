"""
Fully-sampled reference reconstruction.

Two combination modes (selected at call time via *use_sense*):

  RSS  (default):
    read k-space (Nread, Nphase, Ncoils) → centred IFFT2 → RSS

  ESPIRiT + SENSE:
    … same read … → normalise k-space → BART ecalib → BART pics -S
    Returns the SENSE-combined image plus the sensitivity maps.

Siemens files are read through processing.acquisition: lines sit on the
protocol's full phase-encode grid (partial-Fourier lines stay zero or are
filled by `pf_method`), and images are cropped to the prescribed field of
view. Pulseq files keep the original read path.
"""

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np


from .imaging import espirit_normalise, ifft2c as _ifft2c
from .utils import ensure_twix_importable as _add_twix_path


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
    pf_method: str = "zerofill",
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
    pf_method : str, optional
        Partial-Fourier fill for RSS ("zerofill", "homodyne", "pocs"); SENSE
        treats partial-Fourier lines as unsampled instead.

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

    from .acquisition import acquire_cartesian, finalize_image, result_extras
    from .partial_fourier import describe, pf_fill

    asm = acquire_cartesian(dat_path, slice_selector_cb=slice_selector_cb, log=log)
    if asm is None:                                     # Pulseq / unreadable header
        kspace_3d, _tse = _read_legacy(dat_path, expected_nx, tse, necho, te1, teeff,
                                       slice_selector_cb, log)
        measured = kspace_3d
    else:
        _tse = None
        measured = asm.kspace
        kspace_3d = pf_fill(measured, asm.line_category, pf_method)
        log(f"  Full-grid k-space (read, phase, coils): {kspace_3d.shape}")

    log("Reconstructing image via centred IFFT2 + RSS…")
    images    = _ifft2c(kspace_3d)                                # (Nr, Np, Nc)
    image_rss = finalize_image(np.sqrt(np.sum(np.abs(images) ** 2, axis=2)), asm)
    del images   # per-coil images not needed beyond RSS
    log(f"  Image shape: {image_rss.shape}")

    # Pre-compute 2-D RSS k-space magnitude for display (saves storing 3-D
    # complex array); it shows what was measured, never-acquired lines as zeros.
    kspace_rss = np.sqrt(np.sum(np.abs(measured) ** 2, axis=2)).astype(np.float32)
    # When homodyne / POCS filled never-acquired lines, keep that k-space too so
    # the viewer can show "measured" vs "after reconstruction".
    kspace_filled_rss = (np.sqrt(np.sum(np.abs(kspace_3d) ** 2, axis=2)).astype(np.float32)
                         if kspace_3d is not measured else None)
    del kspace_3d

    sens_maps     = None
    image_combined = image_rss   # default fallback

    if use_sense:
        from .recon_backend import active_backend_label, espirit_maps, pics

        log(f"Running ESPIRiT + SENSE combination ({active_backend_label()})…")
        try:
            # BART layout: (Nread, Nphase, 1, Ncoils). SENSE sees the measured
            # k-space, so partial-Fourier lines count as unsampled.
            k4d = measured[:, :, np.newaxis, :].astype(np.complex64)
            del measured   # free the complex128 array now that k4d is built

            # Normalise k-space so ecalib is numerically stable
            k4d_norm, _ = espirit_normalise(k4d, style="add")

            calib = calib_width
            if asm is not None:   # calibrate only inside the symmetric measured region
                from .acquisition import espirit_calib_width
                calib = espirit_calib_width(asm, calib_width)
            log(f"  Computing sensitivity maps (ESPIRiT, calib={calib})…")
            sens_maps = espirit_maps(k4d_norm, calib, num_maps=1,
                                     crop=espirit_crop, log=log)

            log("  Running PICS (SENSE coil combination)…")
            combined_raw   = pics(k4d_norm, sens_maps, soft_sense=True, log=log)
            image_combined = finalize_image(np.abs(np.squeeze(combined_raw)), asm)
            sens_maps      = finalize_image(sens_maps, asm)
        except (RuntimeError, ImportError) as exc:
            log(f"  SENSE failed ({exc}); falling back to RSS")
            sens_maps      = None
            image_combined = image_rss

    log("Fully-sampled reconstruction complete.")
    # NOTE: kspace_3d and images omitted from result to reduce memory;
    # kspace_rss (2-D float32) is stored instead for the k-space display panel.
    result = {
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
    if asm is not None:
        result.update(result_extras(asm))
        result["pf_method"] = (describe("model (PICS)", asm.line_category) if use_sense
                               else describe(pf_method, asm.line_category))
        if kspace_filled_rss is not None and not use_sense:
            result["kspace_filled_rss"] = kspace_filled_rss
    return result


def _read_legacy(dat_path, expected_nx, tse, necho, te1, teeff, slice_selector_cb, log):
    """
    Original read path (Pulseq and unreadable headers): the stored lines are
    the phase-encode grid. Returns ((Nread, Nphase, Ncoils) complex128, tse params).
    """
    _add_twix_path()
    from read_siemens_scans.twix_reading import read_raw_twix_data
    from .utils import crop_readout_oversampling, select_slice

    log(f"Reading TWIX data: {Path(dat_path).name}")
    kspace_raw, _ = read_raw_twix_data(str(dat_path), fix_oversampling=False)
    log(f"  Raw shape: {kspace_raw.shape}")

    # Multi-slice Siemens scans return (Nslices, Nacq_PE, Ncoils, Nreadout).
    # Select a single slice before the 3-axis transpose below, otherwise the
    # transpose fails with "axes don't match array".
    kspace_raw, _ = select_slice(kspace_raw, slice_selector_cb, log)

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
    from .tse_reorder import detect_tse_params, tse_acq_to_line, unscramble_rows
    n_acq = kspace_raw.shape[0]
    rows = np.arange(n_acq)                           # line placement, for the .seq check
    _tse = detect_tse_params(dat_path, tse=tse, necho=necho, te1=te1, teeff=teeff)
    if _tse is not None:
        log(f"  TSE detected — un-scrambling echo order (necho={_tse['necho']}).")
        kspace_raw = unscramble_rows(
            kspace_raw, _tse["necho"], _tse["TE1"], _tse["TEeff"], log_fn=log
        )
        if n_acq % _tse["necho"] == 0:
            rows = tse_acq_to_line(n_acq, _tse["necho"], _tse["TE1"], _tse["TEeff"])
    from .seq_geometry import shadow_check
    shadow_check(dat_path, rows, n_acq, log)          # logs only (shadow mode)

    kspace_3d = np.transpose(kspace_raw, (2, 0, 1)).astype(np.complex128)
    log(f"  Rearranged shape: {kspace_3d.shape}")   # (Nreadout, Nphase, Ncoils)
    return kspace_3d, _tse
