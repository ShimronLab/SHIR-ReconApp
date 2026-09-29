"""
Poisson-disc compressed-sensing reconstruction.

The "PoissonDisc" acquisition is really **Cartesian variable-density phase-encode
undersampling**: every acquired line sits on an integer ky grid position, the
readout is fully sampled, and the k-space centre is a small fully-sampled ACS
block. It is therefore reconstructed as Cartesian CS — NOT with a NUFFT.

Pipeline:
  read_raw_twix_data  →  map acquired lines to their integer PE grid index
  →  place them in a zero-filled Cartesian k-space (keeps the sampling mask)
  →  ESPIRiT (ecalib)  →  CS PICS (ℓ1-wavelet, mask-aware)  →  IFFT2/RSS

Why not the old NUFFT path: gridding the (already Cartesian) data onto an
oversampled grid and FFT-ing back produced a *fully-populated* k-space, which
LOSES the sampling mask. PICS then degenerates into a plain SENSE combination
and cannot undo the R≈4 undersampling, leaving heavy variable-density aliasing
(concentric rings) in the image. Keeping the mask (this Cartesian path) lets
PICS actually remove the aliasing. See Reconstruct_PoissonDisc.ipynb for the
full diagnosis, and equispaced_recon.py for the same ESPIRiT+PICS machinery.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .utils import (
    crop_readout_oversampling,
    load_trajectory,
    scale_trajectory,
)


from .utils import ensure_twix_importable as _add_twix_path


def _ifft2c(ks: np.ndarray) -> np.ndarray:
    """Centred 2-D IFFT over axes 0 (readout) and 1 (phase)."""
    return np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(ks, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )


def _centered_fft2(img: np.ndarray) -> np.ndarray:
    """Centred 2-D FFT of an image (sigpy when available, else numpy)."""
    try:
        import sigpy as sp
        return sp.fft(img, axes=(0, 1), center=True)
    except ImportError:
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))


def reconstruct_poisson(
    dat_path: str,
    poisson_json_path: str,
    *,
    grid_size: int = 512,
    calib_width: int = 24,
    scan_fov_m: float = 256e-3,
    crop_size: Optional[int] = None,
    traj_margin: float = 1.0,
    lambda_wav: float = 0.01,
    max_iter: int = 100,
    use_zerofill: bool = False,
    tse: Optional[bool] = None,
    reference_dat_path: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Compressed-sensing reconstruction of Poisson-disc (Cartesian variable-density)
    sampled data.

    Parameters
    ----------
    dat_path : str
        Path to the PoissonDisc .dat TWIX file.
    poisson_json_path : str
        Path to PoissonDisc.json for this scan folder (used to map each acquired
        line to its integer phase-encode grid index).
    calib_width : int
        ESPIRiT calibration region width (default 24). Clamped to the width of
        the fully-sampled ACS block when that is smaller.
    lambda_wav : float
        ℓ1-wavelet regularisation weight for PICS (default 0.01).
    max_iter : int
        Number of PICS iterations (default 100).
    use_zerofill : bool
        If True, skip ESPIRiT + PICS and return a Root-Sum-of-Squares image from
        the zero-filled Cartesian k-space. Fast but unregularised. Default False.
    tse : bool or None
        TSE hint (unused for the undersampled data — the trajectory maps each
        line to its true PE index, so no echo un-scramble is needed here; the
        reference IS un-scrambled). Kept for signature compatibility.
    reference_dat_path : str or None
        Optional fully-sampled reference for SSIM/NMSE/PSNR + difference.
    progress_cb : callable(str), optional
        Status callback.

    ``grid_size`` / ``scan_fov_m`` / ``crop_size`` / ``traj_margin`` are accepted
    for signature compatibility with callers but are unused by the Cartesian path
    (they only governed the old NUFFT gridding).

    Returns
    -------
    dict with keys:
        traj_scaled     : (N, 2) trajectory, normalised to [-0.5, 0.5)
        imgs_per_coil   : (Nread, Nphase, Ncoils) per-coil images
        sens_maps       : ESPIRiT maps (None in zero-fill mode)
        combined_img    : (Nread, Nphase) reconstructed image
        kspace_combined : (Nread, Nphase) k-space of the combined image
        recon_mode      : "pics" | "zerofill"
        params          : dict of parameters used
      plus, when a reference is given: reference_image, ssim, nmse, psnr, rmse,
        reference_warning.
    """

    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    def _reference_extra(combined: np.ndarray) -> Dict[str, Any]:
        """Compare the final image to a fully-sampled reference (if provided)."""
        if not reference_dat_path:
            return {}
        try:
            from .reference import load_reference_rss, compare_to_reference
            # The CS (PICS) recon is coil-sensitivity-combined, so compare it
            # against a matched SENSE-combined reference; zero-fill is RSS.
            _combine = "rss" if use_zerofill else "sense"
            ref_img = load_reference_rss(
                reference_dat_path, combine=_combine, slice_idx=None, tse=tse,
                calib_width=calib_width, log=log
            )
            # The recon and the IFFT+RSS reference can sit on different intensity
            # scales, so rescale the recon so its 95th percentile matches the
            # reference before comparing, else the shared-scale metrics are
            # meaningless.
            rec = np.abs(np.squeeze(combined)).astype(float)
            r_scale = np.percentile(rec.ravel(), 95) + 1e-12
            f_scale = float(np.percentile(np.abs(ref_img).ravel(), 95)) + 1e-12
            rec = rec * (f_scale / r_scale)
            return compare_to_reference(
                rec, ref_img,
                recon_name=Path(dat_path).name,
                reference_name=Path(reference_dat_path).name, log=log,
            )
        except Exception as exc:
            log(f"  [WARN] reference comparison failed ({exc}).")
            return {}

    _add_twix_path()
    from read_siemens_scans.twix_reading import read_raw_twix_data

    # ── 1. Read raw k-space ──────────────────────────────────────────────────
    log(f"Reading TWIX data: {Path(dat_path).name}")
    kspace_orig, _ = read_raw_twix_data(str(dat_path), fix_oversampling=False)
    if kspace_orig.ndim == 4:                       # (Nslices, Nlines, Ncoils, Nread)
        sl = kspace_orig.shape[0] // 2
        log(f"  Multi-slice scan ({kspace_orig.shape[0]} slices) — using slice {sl}")
        kspace_orig = kspace_orig[sl]
    if kspace_orig.ndim != 3:
        raise ValueError(
            f"Unexpected raw k-space ndim {kspace_orig.ndim} (shape "
            f"{kspace_orig.shape}); expected 3-D (lines, coils, read) or 4-D."
        )
    n_lines, n_coils, n_read0 = kspace_orig.shape
    log(f"  raw k-space (lines, coils, read): {kspace_orig.shape}")

    # ── 2. Trajectory → integer Cartesian PE grid indices ────────────────────
    log("Loading Poisson-disc trajectory (idx=0)…")
    traj_raw    = load_trajectory(poisson_json_path, "0")        # (N, 2) kx, ky
    traj_scaled = scale_trajectory(traj_raw)
    expected_len = n_lines * n_read0
    if len(traj_raw) != expected_len:
        raise ValueError(
            f"[OS] Trajectory length ({len(traj_raw)}) ≠ lines*read "
            f"({n_lines}*{n_read0}={expected_len}). The Poisson JSON and the .dat "
            f"file were not acquired together. Re-run the MATLAB sequence builder "
            f"to regenerate the JSON alongside the .seq."
        )

    ky_line = traj_raw[:, 1].reshape(n_lines, n_read0)[:, 0]     # one ky per line
    uy = np.sort(np.unique(np.round(ky_line, 6)))
    dky = float(np.min(np.diff(uy))) if uy.size > 1 else 1.0     # PE grid step (1/FOV)
    n_phase = int(round((uy[-1] - uy[0]) / dky)) + 1             # PE matrix size
    pe_idx = np.clip(np.round((ky_line - uy[0]) / dky).astype(int), 0, n_phase - 1)
    pe_mask = np.zeros(n_phase, dtype=bool)
    pe_mask[pe_idx] = True
    n_sampled = int(pe_mask.sum())
    actual_r  = n_phase / max(n_sampled, 1)
    log(f"  Cartesian VD undersampling: {n_sampled}/{n_phase} PE lines "
        f"(R≈{actual_r:.2f}), readout {n_read0}, {n_coils} coils")

    # ── 3. Crop readout oversampling, then place acquired lines ──────────────
    kspace_orig = crop_readout_oversampling(
        kspace_orig, expected_nx=n_phase, readout_axis=2, log_fn=log,
    )
    n_read = kspace_orig.shape[2]
    full_k = np.zeros((n_read, n_phase, n_coils), dtype=np.complex64)
    k_lcr  = np.transpose(kspace_orig, (2, 0, 1))               # (read, lines, coils)
    del kspace_orig
    for l, pi in enumerate(pe_idx):
        full_k[:, pi, :] = k_lcr[:, l, :]
    del k_lcr

    # ACS = longest run of consecutive sampled PE indices (the fully-sampled centre)
    sidx = np.flatnonzero(pe_mask)
    acs_width = 0
    if sidx.size:
        run = 1; best = 1
        for i in range(1, len(sidx)):
            run = run + 1 if sidx[i] == sidx[i - 1] + 1 else 1
            best = max(best, run)
        acs_width = best
    calib = min(int(calib_width), acs_width) if acs_width > 0 else int(calib_width)
    log(f"  fully-sampled ACS block ≈ {acs_width} lines → ESPIRiT calib {calib}")

    # Per-coil images (kept so the coil panels / All-coils viewer keep working).
    imgs_per_coil = _ifft2c(full_k)                             # (read, phase, coils)
    zf_image = np.sqrt(np.sum(np.abs(imgs_per_coil) ** 2, axis=2))

    # ── 4. Zero-fill early exit ──────────────────────────────────────────────
    if use_zerofill:
        log("Zero-fill mode: RSS of the zero-filled Cartesian k-space…")
        combined = zf_image
        kspace_combined = _centered_fft2(combined)
        log("Zero-fill reconstruction complete.")
        return {
            "traj_scaled":     traj_scaled,
            "imgs_per_coil":   imgs_per_coil,
            "sens_maps":       None,
            "combined_img":    combined,
            "kspace_combined": kspace_combined,
            "recon_mode":      "zerofill",
            "params": {
                "n_phase":     n_phase,
                "calib_width": calib,
                "actual_r":    actual_r,
                "lambda_wav":  lambda_wav,
                "max_iter":    max_iter,
            },
            **_reference_extra(combined),
        }

    # ── 5. Normalise → ESPIRiT → mask-aware CS PICS (ℓ1-wavelet) ─────────────
    from .recon_backend import active_backend_label, espirit_maps, pics

    k4d = full_k[:, :, np.newaxis, :]                          # (read, phase, 1, coils)
    del full_k
    try:
        import sigpy as sp
        img_tmp = sp.rss(sp.ifft(k4d, axes=(0, 1)), axes=-1)
        scale = float(np.percentile(np.abs(img_tmp).ravel(), 95))
    except Exception:
        scale = float(np.max(np.abs(k4d)))
    scale = max(scale, 1e-12)
    k4d_norm = k4d / scale

    sens_maps = None
    try:
        log(f"Computing ESPIRiT sensitivity maps (calib={calib}, {active_backend_label()})…")
        sens_maps = espirit_maps(k4d_norm, calib, num_maps=1, log=log)
        log(f"Running CS PICS (ℓ1-wavelet, λ={lambda_wav}, iters={max_iter}, "
            f"{active_backend_label()})…")
        recon = pics(k4d_norm, sens_maps, reg="W", lam=lambda_wav,
                     iters=max_iter, log=log)
        combined = np.abs(np.squeeze(recon))
    except (RuntimeError, ImportError) as exc:
        log(f"  CS PICS failed ({exc}); falling back to zero-fill RSS.")
        sens_maps = None
        combined = zf_image

    # PICS runs on k-space divided by `scale` and does not rescale its output
    # back, so it lands on an arbitrary ~O(1) intensity. Rescale to the
    # zero-filled image's 95th percentile so it shares the reference's native
    # scale (same fix as equispaced_recon.py) — otherwise the difference map
    # saturates and the metrics are meaningless.
    if sens_maps is not None:
        c95 = float(np.percentile(np.abs(combined).ravel(), 95))
        z95 = float(np.percentile(np.abs(zf_image).ravel(), 95))
        if c95 > 0:
            combined = combined * (z95 / c95)
            log(f"  Rescaled PICS output to data scale (×{z95 / c95:.3e}).")

    kspace_combined = _centered_fft2(combined)
    log("Poisson-disc CS reconstruction complete.")
    return {
        "traj_scaled":     traj_scaled,
        "imgs_per_coil":   imgs_per_coil,
        "sens_maps":       sens_maps,
        "combined_img":    combined,
        "kspace_combined": kspace_combined,
        "recon_mode":      "pics",
        "params": {
            "n_phase":     n_phase,
            "calib_width": calib,
            "actual_r":    actual_r,
            "lambda_wav":  lambda_wav,
            "max_iter":    max_iter,
        },
        **_reference_extra(combined),
    }
