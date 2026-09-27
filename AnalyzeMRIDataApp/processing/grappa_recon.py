"""
GRAPPA reconstruction for 4x-equispaced under-sampled data.

Pipeline (mirrors ReadDatPrintSizeLinux.ipynb):
  read_raw_twix_data  →  reconstruct PE mask  →  place acquired lines in full k-space
  →  pygrappa cgrappa  →  IFFT2 + RSS
  Optionally compares vs. fully-sampled reference.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np


def _add_twix_path() -> None:
    here = Path(__file__).resolve()
    root = here.parent.parent.parent
    twix_dir = root / "read_siemens_scans-Alon"
    twix_str = str(twix_dir)
    if twix_str not in sys.path:
        sys.path.insert(0, twix_str)


def _ifft2c(ks: np.ndarray) -> np.ndarray:
    """Centred 2-D IFFT over axes 0 and 1."""
    return np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(ks, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )


def _build_grappa_mask(
    n_phase: int,
    r_factor: int,
    acs_lines: int,
) -> tuple:
    """
    Reconstruct the equispaced+ACS PE mask used by the Siemens sequence.
    Returns (pe_mask, acquired_pe_idx, acs_start, acs_end).
    """
    center_idx = n_phase // 2
    idx_right  = np.arange(center_idx, n_phase,  r_factor)
    idx_left   = np.arange(center_idx, -1,       -r_factor)

    pe_mask = np.zeros(n_phase, dtype=bool)
    pe_mask[idx_right] = True
    pe_mask[idx_left]  = True

    acs_start = center_idx - acs_lines // 2
    acs_end   = center_idx + (acs_lines - acs_lines // 2)
    pe_mask[acs_start:acs_end] = True

    return pe_mask, np.where(pe_mask)[0], acs_start, acs_end


def reconstruct_grappa(
    dat_path: str,
    *,
    acs_lines: int = 24,
    r_factor: int = 4,
    kernel_size: tuple = (5, 5),
    lamda: float = 0.01,
    reference_dat_path: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    GRAPPA reconstruction of 4x-equispaced under-sampled data.

    Parameters
    ----------
    dat_path : str
        Path to the GRAPPA .dat file.
    acs_lines : int
        Number of ACS (auto-calibration signal) lines (default 24).
    r_factor : int
        Under-sampling factor (default 4).
    kernel_size : tuple
        GRAPPA kernel size, rows × cols (default (5, 5)).
    lamda : float
        Tikhonov regularisation weight passed to pygrappa (default 0.1).
    reference_dat_path : str, optional
        If provided, loads a fully-sampled reference and computes RMSE/PSNR.
    progress_cb : callable(str), optional
        Status callback.

    Returns
    -------
    dict with keys:
        undersampled_kspace : (Nreadout, Nphase, Ncoils)
        grappa_kspace       : (Nreadout, Nphase, Ncoils)
        zf_image            : (Nreadout, Nphase) zero-filled RSS image
        grappa_image        : (Nreadout, Nphase) GRAPPA RSS image
        pe_mask             : (Nphase,) bool
        acs_start, acs_end  : int
        actual_r            : float
        reference_image     : (Nr, Np) or None
        rmse                : float or None
        psnr                : float or None
        params              : dict
    """

    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    _add_twix_path()
    from read_siemens_scans.twix_reading import read_raw_twix_data
    from .utils import crop_readout_oversampling

    # ── 1. Read k-space ───────────────────────────────────────────────────────
    log(f"Reading TWIX data: {Path(dat_path).name}")
    kspace_raw, _ = read_raw_twix_data(str(dat_path), fix_oversampling=False)
    # Shape: (Nacq_PE, Ncoils, Nreadout)
    log(f"  Raw shape: {kspace_raw.shape}")
    n_acq, n_coils, _n_readout_raw = kspace_raw.shape

    # Determine the true PE matrix size from the fully-sampled reference scan.
    # On Siemens scanners, PE matrix size is independent of n_readout — never assume equality.
    # The reference is fully sampled so ref_kspace.shape[0] == n_phase exactly.
    _ref_raw_cache = None
    if reference_dat_path is not None:
        log(f"Probing reference for PE matrix size: {Path(reference_dat_path).name}")
        _ref_raw_cache, _ = read_raw_twix_data(str(reference_dat_path), fix_oversampling=False)
        if _ref_raw_cache.ndim == 4:
            _ref_sl = n_acq  # dummy; use middle slice
            _ref_raw_cache = _ref_raw_cache[_ref_raw_cache.shape[0] // 2]
        n_phase = _ref_raw_cache.shape[0]
        # Crop readout OS from reference so sizes stay consistent with main data.
        _ref_raw_cache = crop_readout_oversampling(
            _ref_raw_cache, expected_nx=n_phase, readout_axis=2, log_fn=log
        )
        log(f"  n_phase={n_phase} from reference (raw readout={_n_readout_raw})")
    else:
        n_phase = _n_readout_raw  # fallback: assume square acquisition

    # Crop readout oversampling from main k-space.
    kspace_raw = crop_readout_oversampling(
        kspace_raw, expected_nx=n_phase, readout_axis=2, log_fn=log
    )
    n_acq, n_coils, n_readout = kspace_raw.shape

    # Verify/correct n_phase: the analytical equispaced+ACS mask must match n_acq.
    # If the reference had a different matrix size, search common candidate values.
    _mask_check, _idx_check, _, _ = _build_grappa_mask(n_phase, r_factor, acs_lines)
    if len(_idx_check) != n_acq:
        _CANDIDATES = [64, 96, 128, 160, 192, 224, 240, 256, 288, 320, 384, 400, 512]
        log(
            f"  n_phase={n_phase} → mask.sum()={len(_idx_check)} ≠ n_acq={n_acq}; "
            f"searching candidate n_phase values…"
        )
        for _np in _CANDIDATES:
            _, _idx_c, _, _ = _build_grappa_mask(_np, r_factor, acs_lines)
            if len(_idx_c) == n_acq:
                n_phase = _np
                log(f"  Corrected n_phase={n_phase} (mask.sum()={len(_idx_c)} matches n_acq={n_acq})")
                break
        else:
            log(f"  No candidate n_phase matched n_acq={n_acq}; keeping n_phase={n_phase}")

    # ── 2. Reconstruct PE mask ────────────────────────────────────────────────
    log("Reconstructing GRAPPA sampling mask…")
    pe_mask, acquired_pe_idx, acs_start, acs_end = _build_grappa_mask(
        n_phase, r_factor, acs_lines
    )
    actual_r = n_phase / len(acquired_pe_idx)
    log(f"  Mask: {len(acquired_pe_idx)} / {n_phase} lines  (R = {actual_r:.2f}×)")

    if len(acquired_pe_idx) != n_acq:
        log(
            f"  WARNING: mask has {len(acquired_pe_idx)} True entries but reader "
            f"returned {n_acq} lines — using available lines in order."
        )
        acquired_pe_idx = acquired_pe_idx[: n_acq]

    # ── 3. Place acquired lines into full k-space ─────────────────────────────
    full_kspace = np.zeros((n_readout, n_phase, n_coils), dtype=np.complex128)
    # kspace_raw: (Nacq, Ncoils, Nreadout) → rearrange to (Nreadout, Nacq, Ncoils)
    kspace_rearranged = np.transpose(kspace_raw, (2, 0, 1)).astype(np.complex128)
    full_kspace[:, acquired_pe_idx, :] = kspace_rearranged
    log(f"  Full k-space shape: {full_kspace.shape}")

    calib = full_kspace[:, acs_start:acs_end, :]
    log(f"  ACS calibration block: {calib.shape}")

    # ── 4. GRAPPA ─────────────────────────────────────────────────────────────
    log(f"Running GRAPPA (pygrappa / cgrappa), kernel {kernel_size}…")
    try:
        # from pygrappa import cgrappa
        # from pygrappa import grappa
        # grappa_kspace = cgrappa(full_kspace, calib, kernel_size=kernel_size, tikhonov=0.01)
        from pygrappa import mdgrappa
        # grappa_kspace = grappa(
        #     full_kspace,
        #     calib,
        #     kernel_size=kernel_size,
        #     lamda=lamda,
        # )
        grappa_kspace = mdgrappa(
            kspace=full_kspace, 
            calib=calib, 
            kernel_size=kernel_size, 
            lamda=lamda
        )
    except ImportError:
        log("  pygrappa not found – falling back to zero-fill")
        grappa_kspace = full_kspace.copy()
    log(f"  Filled k-space shape: {grappa_kspace.shape}")

    # ── 5. Reconstruct images ─────────────────────────────────────────────────
    log("Reconstructing images (IFFT + RSS)…")
    grappa_image = np.sqrt(np.sum(np.abs(_ifft2c(grappa_kspace)) ** 2, axis=2))
    zf_image     = np.sqrt(np.sum(np.abs(_ifft2c(full_kspace))   ** 2, axis=2))
    log(f"  Image shape: {grappa_image.shape}")
    # Pre-compute 2-D RSS k-space magnitude for display, then free 3-D arrays
    undersampled_kspace_rss = np.sqrt(np.sum(np.abs(full_kspace)   ** 2, axis=2)).astype(np.float32)
    grappa_kspace_rss       = np.sqrt(np.sum(np.abs(grappa_kspace) ** 2, axis=2)).astype(np.float32)
    del full_kspace, grappa_kspace
    # ── 6. Optional comparison with fully-sampled reference ───────────────────
    reference_image = None
    rmse = psnr = None

    if reference_dat_path is not None:
        log(f"Loading reference: {Path(reference_dat_path).name}")
        # Reuse the already-read reference (probed early for n_phase); re-read only if needed.
        if _ref_raw_cache is not None:
            ref_raw = _ref_raw_cache
            _ref_raw_cache = None
        else:
            from read_siemens_scans.twix_reading import read_raw_twix_data as _rrtd
            from .utils import crop_readout_oversampling
            ref_raw, _ = _rrtd(str(reference_dat_path), fix_oversampling=False)
            ref_raw = crop_readout_oversampling(ref_raw, expected_nx=n_phase, readout_axis=2, log_fn=log)
        ref_3d   = np.transpose(ref_raw, (2, 0, 1)).astype(np.complex128)
        del ref_raw
        ref_imgs = _ifft2c(ref_3d)
        del ref_3d
        reference_image = np.sqrt(np.sum(np.abs(ref_imgs) ** 2, axis=2))
        del ref_imgs

        from scipy.ndimage import zoom
        if reference_image.shape != grappa_image.shape:
            zf = (
                grappa_image.shape[0] / reference_image.shape[0],
                grappa_image.shape[1] / reference_image.shape[1],
            )
            reference_image = zoom(reference_image, zf, order=1)

        # Normalize BOTH images by the reference's 95th percentile, so the
        # metrics reflect true intensity error on one shared, outlier-robust
        # scale (matches the difference map shown in the viewer).
        scale = float(np.percentile(np.abs(reference_image).ravel(), 95)) + 1e-12
        g_n = grappa_image / scale
        r_n = reference_image / scale
        diff = g_n - r_n
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        psnr = float(20.0 * np.log10(1.0 / (rmse + 1e-12)))
        log(f"  RMSE={rmse:.4f}  PSNR={psnr:.2f} dB")

    log("GRAPPA reconstruction complete.")
    # NOTE: 3-D k-space arrays (full_kspace / grappa_kspace) are not stored;
    # pre-computed 2-D RSS magnitudes are used for the k-space display panels.
    return {
        "undersampled_kspace_rss": undersampled_kspace_rss,
        "grappa_kspace_rss":       grappa_kspace_rss,
        "zf_image":                zf_image,
        "grappa_image":            grappa_image,
        "pe_mask":                 pe_mask,
        "acs_start":               acs_start,
        "acs_end":                 acs_end,
        "actual_r":                actual_r,
        "reference_image":         reference_image,
        "rmse":                    rmse,
        "psnr":                    psnr,
        "params": {
            "acs_lines":   acs_lines,
            "r_factor":    r_factor,
            "kernel_size": kernel_size,
            "lamda":       lamda,
        },
    }
