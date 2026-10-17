"""
Band (non-Cartesian trajectory) reconstruction using BART.

Pipeline (mirrors RestorebandBart.ipynb):
  read_raw_twix_data  →  reshape k-space & trajectory  →  BART nufft -a
  →  crop  →  BART fft  →  normalise  →  BART ecalib  →  BART pics -S
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .utils import (
    load_trajectory,
    make_bart_trajectory,
    scale_trajectory,
)


from .imaging import espirit_normalise
from .utils import ensure_twix_importable as _add_twix_path


def reconstruct_band(
    dat_path: str,
    bands_json_path: str,
    band_idx: int,
    *,
    grid_size: int = 512,
    calib_width: int = 24,
    scan_fov_m: float = 256e-3,
    crop_size: Optional[int] = None,
    traj_margin: float = 1.0,
    espirit_crop: float = 0.8,      # ESPIRiT eigenvalue mask threshold (both engines)
    tse: Optional[bool] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Full band reconstruction pipeline.

    Parameters
    ----------
    dat_path : str
        Path to the .dat TWIX file.
    bands_json_path : str
        Path to Bands.json for this scan folder.
    band_idx : int
        Which band index to reconstruct — the key looked up in Bands.json.
        Any index present in the JSON is valid (not limited to 0–15).
    grid_size : int
        BART NUFFT oversampled grid size (default 512).
    calib_width : int
        ESPIRiT calibration region width in pixels (default 24).
    scan_fov_m : float
        Physical scan FOV in metres, used to compute the image-domain crop
        size automatically when ``crop_size`` is None (default 256e-3 m).
    crop_size : int or None
        Image-domain output size in pixels.  None (default) → auto-compute
        from ``scan_fov_m`` and the trajectory scaling.  An explicit value
        forces a fixed output size.
    traj_margin : float
        Safety margin in BART grid pixels to keep trajectory samples strictly
        inside [-N/2, N/2) (default 1.0).
    progress_cb : callable(str), optional
        Called with status messages during processing.

    Returns
    -------
    dict with keys:
        traj_scaled     : (N, 2) scaled trajectory
        imgs_per_coil   : (crop_size, crop_size, Ncoils) cropped adjoint NUFFT images
        sens_maps       : (crop_size, crop_size, 1, Ncoils) sensitivity maps
        combined_img    : (crop_size, crop_size) complex combined image
        kspace_combined : (crop_size, crop_size) k-space of combined image
        band_idx        : int
        params          : dict of parameters used (includes computed crop_size, pixel_size)

    Note: kspace_orig, traj_raw, and cart_kspace are deliberately not included
    in the result to minimise per-scan memory when processing many bands.
    """

    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    _add_twix_path()
    from read_siemens_scans.twix_reading import read_raw_twix_data

    # TSE needs NO echo un-scramble here: BART NUFFT pairs each k-space sample
    # with its trajectory-JSON coordinate, and Bands.json is written by
    # writeTSE_kband (ktraj_adc) in the SAME acquisition order as the reader, so
    # the echo-train scrambling cancels out. (The Cartesian fullysamp/equispaced
    # recons DO need the un-scramble; NUFFT does not.)
    from .tse_reorder import is_tse
    if is_tse(dat_path) if tse is None else tse:
        log("  TSE detected — NUFFT trajectory pairing preserves echo order; no un-scramble.")

    # ── 1. Read raw k-space ──────────────────────────────────────────────────
    log(f"Reading TWIX data: {Path(dat_path).name}")
    kspace_orig, _ = read_raw_twix_data(str(dat_path), fix_oversampling=False)
    # Output: (dim1, Ncoils, dim2)  →  move coil axis to front → (Ncoils, dim1, dim2)
    kspace_data = np.moveaxis(kspace_orig, 1, 0)
    del kspace_orig          # free raw TWIX buffer immediately
    num_coils, dim1, dim2 = kspace_data.shape
    log(f"  k-space shape (coils, dim1, dim2): {kspace_data.shape}")

    # ── 2. Load & prepare trajectory ─────────────────────────────────────────
    log(f"Loading trajectory: band {band_idx}")
    traj_raw = load_trajectory(bands_json_path, str(band_idx))
    traj_scaled = scale_trajectory(traj_raw)
    # Sanity-check: the trajectory holds one (kx, ky) sample per acquired k-space
    # point, so its length must equal dim1 * dim2 — the number of samples that
    # make_bart_trajectory reshapes into a (3, dim1, dim2) array.  A mismatch
    # means an old JSON is paired with a new .dat — e.g. a non-oversampled
    # trajectory with 2× oversampled data — or vice versa.
    expected_len = dim1 * dim2
    if len(traj_raw) != expected_len:
        raise ValueError(
            f"[OS] Trajectory length ({len(traj_raw)}) ≠ dim1*dim2 "
            f"({dim1}*{dim2}={expected_len}). "
            f"The bands JSON and the .dat file were not acquired together. "
            f"Re-run the MATLAB sequence builder to regenerate the JSON alongside the .seq."
        )
    bart_traj, pixel_size = make_bart_trajectory(
        traj_raw, dim1, dim2, N_grid=grid_size, margin=traj_margin
    )
    # crop_size: use the explicit UI override when provided, otherwise compute
    # it from the scan FOV and trajectory scaling so it adapts to the acquired
    # matrix (e.g. larger for a 320 acquisition than a 256 one).
    if crop_size is None:
        crop_size = int(round(scan_fov_m / pixel_size))
        crop_how = f"scan FOV {scan_fov_m * 1e3:.0f} mm / pixel"
        log(f"  pixel_size={pixel_size * 1e3:.3f} mm, crop_size={crop_size} px (auto from FOV)")
    else:
        crop_size = int(crop_size)
        crop_how = "user override"
        log(f"  pixel_size={pixel_size * 1e3:.3f} mm, crop_size={crop_size} px (user override)")

    from .recon_backend import (
        active_backend_label, espirit_maps, nufft_adjoint_to_cartesian, pics,
    )

    # ── 3-6. Adjoint NUFFT → crop → Cartesian k-space ────────────────────────
    log(f"Gridding non-cartesian data ({active_backend_label()})…")
    imgs_per_coil, cart_kspace = nufft_adjoint_to_cartesian(
        bart_traj, kspace_data, grid_size, crop_size, log=log,
    )   # imgs_per_coil (crop, crop, Ncoils); cart_kspace (crop, crop, 1, Ncoils)
    # Raw samples (readouts, samples, coils), each paired with its (kx, ky);
    # make_bart_trajectory reshapes the trajectory the same (C) way.
    from .acquisition import kspace_export
    exported = kspace_export(np.moveaxis(kspace_data, 0, -1),
                             trajectory=traj_raw.reshape(dim1, dim2, 2))
    del kspace_data

    # ── 7. Normalise k-space ──────────────────────────────────────────────────
    log("Normalising k-space for ESPIRiT…")
    cart_kspace_norm, _ = espirit_normalise(cart_kspace, style="add")

    # ── 8-9. ESPIRiT maps + PICS (SENSE combination) ─────────────────────────
    sens_maps = None
    combination = "ESPIRiT + PICS (SENSE)"
    try:
        log(f"Computing sensitivity maps (ESPIRiT, calib={calib_width})…")
        sens_maps = espirit_maps(cart_kspace_norm, calib_width, num_maps=1,
                                 crop=espirit_crop, log=log)
        log("Running PICS (SENSE coil combination)…")
        combined_img = pics(cart_kspace_norm, sens_maps, soft_sense=True, iters=1, log=log)
        combined_sq  = np.squeeze(combined_img)   # (crop, crop) complex
    except (RuntimeError, ImportError) as exc:
        log(f"  SENSE failed ({exc}); falling back to RSS of coil images")
        combination = f"RSS of coil images (SENSE failed: {exc})"
        # RSS of the cropped adjoint-NUFFT images (same frame as the PICS path).
        combined_sq = np.sqrt(np.sum(np.abs(imgs_per_coil) ** 2, axis=-1))

    # ── 10. k-space of combined image ─────────────────────────────────────────
    log("Computing k-space of combined image…")
    try:
        import sigpy as sp
        kspace_combined = sp.fft(combined_sq, axes=(0, 1), center=True)
    except ImportError:
        kspace_combined = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(combined_sq))
        )

    # ── 11. Acquisition facts (Acquisition… table + facts line) ──────────────
    from .acquisition import Src, band_angle_deg, trajectory_facts
    angle = band_angle_deg(traj_raw)
    pixel_mm = pixel_size * 1e3
    facts = trajectory_facts(
        dat_path, traj_raw, dim1, dim2, num_coils,
        json_path=bands_json_path, json_key=str(band_idx),
        rows=[
            ("band", int(band_idx), Src.TRAJ_JSON, f"{Path(bands_json_path).name} key {band_idx}"),
            ("band angle (°)", round(angle, 2), Src.DERIVED,
             "long axis of the trajectory, from +kx toward +ky"),
            ("NUFFT grid", f"{grid_size}×{grid_size}", Src.SETTING, f"trajectory margin {traj_margin} px"),
            ("image matrix", f"{crop_size}×{crop_size}", Src.DERIVED, crop_how),
            ("pixel (mm)", round(pixel_mm, 3), Src.DERIVED, "from the trajectory's k max"),
            ("coil combination", combination, Src.SETTING,
             f"{active_backend_label()}; ESPIRiT calib {calib_width}, crop {espirit_crop}"),
        ],
        summary=(f"Band {band_idx} · {angle:.1f}° · {dim1} readouts × {dim2} samples · "
                 f"{crop_size}×{crop_size} image"),
        pixel_mm=(pixel_mm, pixel_mm),
    )

    log("Band reconstruction complete.")
    # NOTE: kspace_orig, traj_raw, cart_kspace are intentionally omitted to
    # keep peak memory low when processing many bands in sequence.
    return {
        **facts,
        "traj_scaled":      traj_scaled,
        "imgs_per_coil":    imgs_per_coil,    # (crop_size, crop_size, Ncoils)
        "sens_maps":        sens_maps,
        "combined_img":     combined_sq,
        "kspace_combined":  kspace_combined,
        "band_idx":         band_idx,
        **exported,
        "params": {
            "grid_size":   grid_size,
            "calib_width": calib_width,
            "scan_fov_m":  scan_fov_m,
            "traj_margin": traj_margin,
            "pixel_size":  pixel_size,
            "crop_size":   crop_size,
        },
    }
