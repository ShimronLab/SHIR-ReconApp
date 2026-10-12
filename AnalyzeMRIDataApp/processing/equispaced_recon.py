"""
Equispaced under-sampled reconstruction.

Supports two reconstruction methods selectable at call time via *method*:

  "grappa"       — pygrappa mdgrappa (parallel imaging, Cartesian), RSS coil combination
  "grappa_sense" — pygrappa mdgrappa + ESPIRiT sensitivity maps for optimal coil combination
  "sense"        — BART ecalib + BART pics -S  (pure SENSE, no regularisation)
  "sense_cs"     — BART ecalib + BART pics -l1 -r{lambda} (compressed sensing SENSE)

PE mask is derived from the Undersampled4equispaces.json trajectory file when
*json_path* is provided; falls back to the analytically-computed equispaced
pattern otherwise.

Pipeline:
  read_raw_twix_data  →  build PE mask (JSON or computed)
  →  place acquired lines in full k-space
  →  GRAPPA (mdgrappa)  OR  SENSE (BART ecalib + pics)
  →  IFFT2 + RSS
  Optionally compares vs. fully-sampled reference.
"""

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
from skimage.metrics import structural_similarity as _ssim


def _parse_r_from_filename(dat_path: str) -> Optional[int]:
    """Extract R factor from filenames like equispaced_R2_acs24.dat."""
    m = re.search(r'_[Rr](\d+)', Path(dat_path).stem)
    return int(m.group(1)) if m else None


from .utils import ensure_twix_importable as _add_twix_path


def _ifft2c(ks: np.ndarray) -> np.ndarray:
    """Centred 2-D IFFT over axes 0 and 1."""
    return np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(ks, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PE mask builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_pe_mask_from_json(
    json_path: str,
    n_phase: int,
    log: Callable[[str], None],
) -> Optional[np.ndarray]:
    """
    Derive a boolean PE mask from Undersampled4equispaces.json that exactly
    matches the MATLAB generate_equispaced_1D arithmetic mask.

    Strategy
    --------
    The MATLAB script applies a k-space centering shift before writing the JSON,
    so raw ky → PE-index conversion carries a small integer offset relative to
    the analytical MATLAB mask.  To remove this offset:

      1. Compute delta_ky = minimum positive spacing between sorted unique ky
         values  (= 1/FOV, the single-line step inside the ACS region).
      2. Do a first-pass ky → PE-index conversion to get a raw index array.
      3. Detect R from the median large step (equispaced step = R indices).
      4. Detect ACS_lines = length of the longest consecutive step-1 run.
      5. Rebuild the mask ANALYTICALLY using MATLAB-equivalent arithmetic:
             center = n_phase // 2
             equispaced : center ± k*R  (k = 0, 1, 2, …)
             ACS        : center ± acs_lines//2  (fully sampled block)
         This strips the centering-shift offset and gives an exact match.

    Returns a bool array of shape (n_phase,), or None on failure.
    """
    try:
        from .utils import load_trajectory
        traj = load_trajectory(json_path, "0")      # (N_samples, 2)  kx, ky

        # Auto-detect which physical axis carries the phase-encode (under-sampled)
        # direction. In a standard acquisition readout = kx (dense, ~Nx unique
        # values) and PE = ky (sparse, = number of sampled lines). A readout/PE
        # swap (90-deg rotation, the 'swapPE' sequences) flips this so the PE
        # pattern lives in kx. The PE axis is simply the one with FEWER unique
        # sample positions, so reading it makes R/ACS detection work for both
        # orientations from a physically-truthful trajectory.
        n_uniq_kx = np.unique(traj[:, 0]).size
        n_uniq_ky = np.unique(traj[:, 1]).size
        pe_col    = 0 if n_uniq_kx < n_uniq_ky else 1
        axis_name = "kx" if pe_col == 0 else "ky"
        log(f"  PE axis detected: {axis_name} (unique kx={n_uniq_kx}, ky={n_uniq_ky})")
        ky_vals = traj[:, pe_col]

        unique_ky = np.unique(ky_vals)
        ky_diffs  = np.diff(unique_ky)
        pos_diffs = ky_diffs[ky_diffs > 1e-9]
        if pos_diffs.size == 0:
            raise ValueError("Could not determine delta_ky: no positive ky steps found.")
        delta_ky = float(np.min(pos_diffs))

        # First-pass ky → PE index (may carry a centering offset)
        pe_float = unique_ky / delta_ky + (n_phase / 2)
        pe_raw   = np.unique(np.clip(np.round(pe_float).astype(int), 0, n_phase - 1))

        # Detect R from large steps (equispaced lines are R indices apart)
        pe_steps    = np.diff(pe_raw)
        large_steps = pe_steps[pe_steps > 1]
        if large_steps.size == 0:
            raise ValueError("Could not detect R: no step > 1 found in PE indices.")
        R = int(np.round(np.median(large_steps)))

        # Detect ACS lines: longest run of consecutive step-1 transitions
        step1_idx = np.where(pe_steps == 1)[0]
        if step1_idx.size > 0:
            groups = np.split(step1_idx, np.where(np.diff(step1_idx) > 1)[0] + 1)
            acs_lines_detected = len(max(groups, key=len))
        else:
            acs_lines_detected = 0

        log(f"  JSON: detected R={R}, ACS={acs_lines_detected}, delta_ky={delta_ky:.4f}")

        # Rebuild mask analytically (MATLAB-equivalent) to eliminate offset
        center_idx = n_phase // 2
        mask = np.zeros(n_phase, dtype=bool)

        mask[np.arange(center_idx, n_phase, R)]  = True
        mask[np.arange(center_idx, -1,      -R)] = True

        acs_h_lo = acs_lines_detected // 2
        acs_h_hi = acs_lines_detected - acs_h_lo
        acs_start = max(0, center_idx - acs_h_lo)
        acs_end   = min(n_phase, center_idx + acs_h_hi)
        mask[acs_start:acs_end] = True

        log(
            f"  JSON mask: {mask.sum()} / {n_phase} lines sampled  "
            f"(R={R}, ACS={acs_lines_detected}, analytical, from {Path(json_path).name})"
        )
        log(f"  First-5 PE indices: {np.where(mask)[0][:5].tolist()}")
        return mask
    except Exception as exc:
        log(f"  WARNING: could not load JSON trajectory ({exc}); falling back to computed mask.")
        return None


def _build_grappa_mask_computed(
    n_phase: int,
    r_factor: int,
    acs_lines: int,
    log: Callable[[str], None],
) -> tuple:
    """
    Analytically reconstruct the equispaced+ACS PE mask used by the Siemens sequence.
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

    log(f"  Computed mask: {pe_mask.sum()} / {n_phase} lines  (R={r_factor}, ACS={acs_lines})")
    log(f"  First-5 PE indices (computed): {np.where(pe_mask)[0][:5].tolist()}")
    return pe_mask, np.where(pe_mask)[0], acs_start, acs_end


# ─────────────────────────────────────────────────────────────────────────────
# Main reconstruction function
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_equispaced(
    dat_path: str,
    *,
    method: str = "grappa",
    json_path: Optional[str] = None,
    mask_method: str = "analytical",
    # GRAPPA params
    acs_lines: int = 24,
    r_factor: int = 4,
    kernel_size: tuple = (5, 5),
    lamda: float = 0.01,
    # SENSE params
    calib_width: int = 24,
    sense_lambda: float = 0.01,
    sense_max_iter: int = 100,
    sense_l2: bool = False,
    # ESPIRiT params
    espirit_num_maps: int = 2,
    espirit_crop: float = 0.8,      # ESPIRiT eigenvalue mask threshold (both engines)
    # TSE un-scramble (echo-train order → true PE order)
    tse: Optional[bool] = None,
    necho: Optional[int] = None,
    te1: Optional[float] = None,
    teeff: Optional[float] = None,
    # Misc
    reference_dat_path: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    slice_selector_cb: Optional[Callable[[int], int]] = None,
    mask_fallback_cb: Optional[Callable[[], str]] = None,
) -> Dict[str, Any]:
    """
    Equispaced under-sampled reconstruction.

    Parameters
    ----------
    dat_path : str
        Path to the equispaced .dat file.
    method : str
        Reconstruction method: "grappa" | "grappa_sense" | "sense" | "sense_cs" | "espirit" | "espirit_l1".
    json_path : str, optional
        Path to Undersampled4equispaces.json for PE mask derivation.
        Falls back to computed equispaced mask if None or loading fails.
    acs_lines : int
        Number of ACS lines (GRAPPA only, default 24).
    r_factor : int
        Under-sampling factor (GRAPPA fallback mask, default 4).
    kernel_size : tuple
        GRAPPA kernel size rows×cols (default (5, 5)).
    lamda : float
        Tikhonov regularisation weight for pygrappa (default 0.01).
    calib_width : int
        ESPIRiT calibration region width in pixels (SENSE only, default 24).
    sense_lambda : float
        L1 regularisation weight for CS-SENSE (default 0.01).
    sense_max_iter : int
        PICS iteration count for CS-SENSE (default 100).
    sense_l2 : bool
        When True and method="sense", add L2 regularisation (BART PICS -S -l2).
        Has no effect for "sense_cs" or "grappa" (default False).
    reference_dat_path : str, optional
        Fully-sampled reference for RMSE/PSNR computation.
    progress_cb : callable(str), optional
        Status callback.

    Returns
    -------
    dict with keys:
        undersampled_kspace_rss : (Nreadout, Nphase) float32
        recon_kspace_rss        : (Nreadout, Nphase) float32   (GRAPPA k-space or zero-filled)
        zf_image                : (Nreadout, Nphase)
        grappa_image            : (Nreadout, Nphase)  primary reconstructed image (any method)
        sense_image             : (Nreadout, Nphase) or None   (SENSE methods only)
        sens_maps               : ndarray or None               (SENSE methods only)
        pe_mask                 : (Nphase,) bool
        acs_start, acs_end      : int
        actual_r                : float
        reference_image         : (Nr, Np) or None
        rmse, psnr              : float or None
        method                  : str
        params                  : dict
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
    # Shape: (Nacq_PE, Ncoils, Nreadout) — or (Nslices, Nacq_PE, Ncoils, Nreadout) for multi-slice
    log(f"  Raw shape: {kspace_raw.shape}")
    _refscan_sl = None   # saved slice index for refscan loading below
    if kspace_raw.ndim == 4:
        n_slices = kspace_raw.shape[0]
        if slice_selector_cb is not None:
            sl_idx = int(slice_selector_cb(n_slices))
        else:
            sl_idx = n_slices // 2
        log(f"  Multi-slice scan ({n_slices} slices) — using slice {sl_idx}")
        kspace_raw = kspace_raw[sl_idx]
        _refscan_sl = sl_idx
    n_acq, n_coils, _n_readout_raw = kspace_raw.shape

    # Determine the true PE matrix size from the fully-sampled reference scan.
    # On Siemens scanners, PE matrix size is independent of n_readout — never assume equality.
    # The reference is fully sampled so ref_kspace.shape[0] == n_phase exactly.
    _ref_raw_cache = None
    if reference_dat_path is not None:
        log(f"Probing reference for PE matrix size: {Path(reference_dat_path).name}")
        _ref_raw_cache, _ = read_raw_twix_data(str(reference_dat_path), fix_oversampling=False)
        if _ref_raw_cache.ndim == 4:
            _ref_sl = _refscan_sl if _refscan_sl is not None else _ref_raw_cache.shape[0] // 2
            _ref_raw_cache = _ref_raw_cache[_ref_sl]
        n_phase = _ref_raw_cache.shape[0]
        # Crop readout OS from reference so sizes stay consistent with main data.
        _ref_raw_cache = crop_readout_oversampling(
            _ref_raw_cache, expected_nx=n_phase, readout_axis=2, log_fn=log,
            dat_path=reference_dat_path,
        )
        log(f"  n_phase={n_phase} from reference (raw readout={_n_readout_raw})")
    else:
        n_phase = _n_readout_raw  # fallback: assume square acquisition

    # Crop readout oversampling from main k-space.
    kspace_raw = crop_readout_oversampling(
        kspace_raw, expected_nx=n_phase, readout_axis=2, log_fn=log,
        dat_path=dat_path,
    )
    n_acq, n_coils, n_readout = kspace_raw.shape

    # Auto-detect R factor from filename (e.g. equispaced_R2_acs24 → 2)
    r_auto = _parse_r_from_filename(dat_path)
    if r_auto is not None and r_auto != r_factor:
        log(f"  Auto-detected R={r_auto} from filename (overriding params R={r_factor})")
        r_factor = r_auto

    # Verify/correct n_phase: the analytical equispaced+ACS mask must match n_acq.
    # If the reference had a different matrix size (e.g. Siemens SE with 240 PE lines
    # when the custom scan uses 256), search common candidate values until one fits.
    _mask_check, _idx_check, _, _ = _build_grappa_mask_computed(
        n_phase, r_factor, acs_lines, lambda _: None
    )
    if len(_idx_check) != n_acq:
        _CANDIDATES = [64, 96, 128, 160, 192, 224, 240, 256, 288, 320, 384, 400, 512]
        log(
            f"  n_phase={n_phase} → mask.sum()={len(_idx_check)} ≠ n_acq={n_acq}; "
            f"searching candidate n_phase values…"
        )
        for _np in _CANDIDATES:
            _, _idx_c, _, _ = _build_grappa_mask_computed(_np, r_factor, acs_lines, lambda _: None)
            if len(_idx_c) == n_acq:
                n_phase = _np
                log(f"  Corrected n_phase={n_phase} (mask.sum()={len(_idx_c)} matches n_acq={n_acq})")
                break
        else:
            log(f"  No candidate n_phase matched n_acq={n_acq}; keeping n_phase={n_phase}")

    # ── 2. Build PE mask ──────────────────────────────────────────────────────
    log("Building sampling mask…")
    acs_start = n_phase // 2 - acs_lines // 2
    acs_end   = n_phase // 2 + (acs_lines - acs_lines // 2)

    # The JSON trajectory files were generated for the custom sequences only.
    # Siemens scans (filename prefixed with 'siemens') have no matching JSON, so
    # ignore any JSON path for them and always ask how to build the mask.
    is_siemens = Path(dat_path).name.lower().startswith("siemens")

    json_mask = None
    if json_path is not None and not is_siemens:
        json_mask = _build_pe_mask_from_json(json_path, n_phase, log)
    elif is_siemens and json_path is not None:
        log("  Siemens scan — ignoring JSON trajectory (created for custom sequences).")

    # No usable JSON trajectory: let the caller choose how to build the mask
    # (analytical computed pattern vs. BART pattern extraction). Always ask for
    # Siemens scans; for custom scans only ask when the JSON failed. Skip the
    # prompt when the user already pre-selected the BART method.
    if json_mask is None and mask_method != "bart_pattern" and mask_fallback_cb is not None:
        try:
            choice = mask_fallback_cb()
        except Exception as exc:
            log(f"  Mask fallback prompt failed ({exc}); using analytical mask.")
            choice = "analytical"
        _reason = "Siemens scan" if is_siemens else "No JSON trajectory found"
        if choice == "bart_pattern":
            mask_method = "bart_pattern"
            log(f"  {_reason} — user chose BART pattern mask.")
        else:
            log(f"  {_reason} — user chose analytical mask.")

    if json_mask is not None:
        pe_mask = json_mask
        acquired_pe_idx = np.where(pe_mask)[0]
        # Detect ACS bounds from the dense block (step-1 consecutive lines)
        # near the centre of the mask.
        diffs_between = np.diff(acquired_pe_idx)
        dense_positions = np.where(diffs_between == 1)[0]
        if dense_positions.size > 0:
            acs_start = int(acquired_pe_idx[dense_positions[0]])
            acs_end   = int(acquired_pe_idx[dense_positions[-1]]) + 2  # inclusive → exclusive
        else:
            # Fallback: use hardcoded ACS bounds from params
            centre    = n_phase // 2
            acs_start = centre - acs_lines // 2
            acs_end   = centre + (acs_lines - acs_lines // 2)
        log(f"  ACS bounds from JSON mask: [{acs_start}, {acs_end})")
    else:
        pe_mask, acquired_pe_idx, acs_start, acs_end = _build_grappa_mask_computed(
            n_phase, r_factor, acs_lines, log
        )

    actual_r = n_phase / len(acquired_pe_idx)
    log(f"  Mask: {len(acquired_pe_idx)} / {n_phase} lines  (R = {actual_r:.2f}×)")

    if len(acquired_pe_idx) != n_acq:
        log(
            f"  WARNING: mask has {len(acquired_pe_idx)} True entries but reader "
            f"returned {n_acq} lines — rebuilding mask with R={r_factor}."
        )
        pe_mask, acquired_pe_idx, acs_start, acs_end = _build_grappa_mask_computed(
            n_phase, r_factor, acs_lines, log
        )
        if len(acquired_pe_idx) != n_acq:
            log(
                f"  WARNING: rebuilt mask has {len(acquired_pe_idx)} entries (expected {n_acq}); "
                f"falling back to sequential PE indices."
            )
            acquired_pe_idx = np.arange(n_acq)
            # Real Siemens data: the reader pre-allocates ALL PE positions
            # (including zeros for unacquired lines), so n_acq IS the full PE
            # matrix dimension, not just the acquired subset.
            n_phase   = n_acq
            acs_start = max(0, n_phase // 2 - acs_lines // 2)
            acs_end   = min(n_phase, n_phase // 2 + (acs_lines - acs_lines // 2))
            pe_mask   = np.ones(n_phase, dtype=bool)   # refined below from data energy

    # ── 3. Place acquired lines into full k-space ─────────────────────────────────────────────
    full_kspace = np.zeros((n_readout, n_phase, n_coils), dtype=np.complex128)
    kspace_rearranged = np.transpose(kspace_raw, (2, 0, 1)).astype(np.complex128)

    # TSE un-scramble: our Pulseq TSE sequences set no LIN label, so the reader
    # returns the acquired lines in echo-train (acquisition) order — NOT ascending
    # PE order. Map each acquired line to its true PE position with the same masked
    # TSE plan; otherwise every echo lands on the wrong PE line. SE/non-TSE and any
    # mismatch fall back to the ascending `acquired_pe_idx` placement.
    from .tse_reorder import detect_tse_params, tse_acq_to_line
    _tse = detect_tse_params(dat_path, tse=tse, necho=necho, te1=te1, teeff=teeff)
    placed = False
    if _tse is not None and n_phase % _tse["necho"] == 0:
        acq2line = tse_acq_to_line(
            n_phase, _tse["necho"], _tse["TE1"], _tse["TEeff"], pe_mask=pe_mask
        )
        if len(acq2line) == n_acq:
            full_kspace[:, acq2line, :] = kspace_rearranged
            placed = True
            log(f"  TSE echo order un-scrambled (necho={_tse['necho']}, {n_acq} lines).")
        else:
            log(
                f"  WARNING: TSE map has {len(acq2line)} lines but reader returned "
                f"{n_acq}; falling back to ascending PE placement."
            )
    elif _tse is not None:
        log(
            f"  WARNING: necho={_tse['necho']} does not divide n_phase={n_phase}; "
            f"falling back to ascending PE placement."
        )
    if not placed:
        full_kspace[:, acquired_pe_idx, :] = kspace_rearranged
    log(f"  Full k-space shape: {full_kspace.shape}")

    # ── 3a. Auto-refine pe_mask from non-zero row energy (sequential fallback case) ───────
    # When pe_mask is all-True it means the sequential fallback was triggered and
    # the pre-allocated data may already encode the undersampling as zero rows.
    if np.all(pe_mask):
        row_energy = np.sum(np.abs(full_kspace) ** 2, axis=(0, 2))  # shape: (n_phase,)
        thresh = row_energy.max() * 1e-6
        energy_mask = row_energy > thresh
        if 0 < energy_mask.sum() < n_phase:   # actual undersampling detected in data
            pe_mask         = energy_mask
            acquired_pe_idx = np.where(pe_mask)[0]
            actual_r        = n_phase / max(len(acquired_pe_idx), 1)
            log(
                f"  Detected undersampling from row energy: "
                f"{pe_mask.sum()}/{n_phase} non-zero lines  (R={actual_r:.2f}×)"
            )
            # Recompute ACS bounds from the detected dense block
            diffs = np.diff(acquired_pe_idx)
            step1 = np.where(diffs == 1)[0]
            if step1.size > 0:
                acs_start = int(acquired_pe_idx[step1[0]])
                acs_end   = int(acquired_pe_idx[step1[-1]]) + 2
            log(f"  Refined ACS bounds: [{acs_start}, {acs_end})")

    calib = full_kspace[:, acs_start:acs_end, :]
    log(f"  ACS calibration block: {calib.shape}")

    # ── 3c. Siemens refscan overlay ──────────────────────────────────────────────────
    # For native Siemens GRAPPA sequences the ACS block is stored in a separate
    # 'refscan' measurement (not in the image data).  Detect this situation:
    # if ANY line inside the ACS range is missing (zero), load the refscan.
    # Custom sequences always fill all ACS lines, so they never trigger this.
    _n_acs = acs_end - acs_start
    _acs_row_e = np.sum(np.abs(full_kspace[:, acs_start:acs_end, :]) ** 2, axis=(0, 2))
    _acs_nz    = int(np.sum(_acs_row_e > 1e-6 * (_acs_row_e.max() + 1e-30)))
    if _acs_nz < _n_acs:
        log(
            f"  ACS region sparse ({_acs_nz}/{_n_acs} non-zero lines) — "
            f"checking TWIX file for a Siemens refscan…"
        )
        try:
            _add_twix_path()
            import twixtools as _twx
            _twix_obj = _twx.read_twix(
                str(dat_path), keep_syncdata=True, keep_acqend=True
            )
            _mapped = _twx.map_twix(_twix_obj)
            if "refscan" in _mapped[-1]:
                _ref_mmap = _mapped[-1]["refscan"]
                _ref_mmap.flags["remove_os"] = False
                _ref_all  = np.asarray(_ref_mmap[:]).squeeze()
                # Extract the same slice used for the image measurement
                if _ref_all.ndim == 4:
                    _rs = _refscan_sl if _refscan_sl is not None else _ref_all.shape[0] // 2
                    _ref_slc = _ref_all[_rs]          # (N_pe, N_coils, N_read)
                elif _ref_all.ndim == 3:
                    _ref_slc = _ref_all               # (N_pe, N_coils, N_read)
                else:
                    raise ValueError(f"Unexpected refscan ndim={_ref_all.ndim}")

                # The refscan carries the SAME readout oversampling as the main
                # acquisition (remove_os=False above). The main k-space has
                # already had its OS cropped to n_readout, so crop the refscan
                # readout to match — otherwise the ACS block (N_read=532) cannot
                # be placed into full_kspace (N_read=266) and GRAPPA loses its
                # calibration data, leaving the undersampling un-resolved.
                if _ref_slc.shape[2] != n_readout:
                    _ref_slc = crop_readout_oversampling(
                        _ref_slc, expected_nx=n_readout, readout_axis=2, log_fn=log,
                        dat_path=dat_path,
                    )

                _ref_e = np.sum(np.abs(_ref_slc) ** 2, axis=(1, 2))
                if _ref_e.max() > 0:
                    _nz = np.where(_ref_e > 1e-8 * _ref_e.max())[0]
                    _nz = _nz[_nz < n_phase]           # clamp to current n_phase
                    if _nz.size > 0:
                        acs_start = int(_nz[0])
                        acs_end   = int(_nz[-1]) + 1
                        _ref_acs  = _ref_slc[acs_start:acs_end]  # (N_acs, N_coils, N_read)
                        full_kspace[:, acs_start:acs_end, :] = (
                            np.transpose(_ref_acs, (2, 0, 1)).astype(np.complex128)
                        )
                        calib = full_kspace[:, acs_start:acs_end, :]
                        log(
                            f"  Siemens refscan ACS overlaid at "
                            f"[{acs_start}:{acs_end}] ({acs_end - acs_start} lines)"
                        )
                    else:
                        log("  Refscan ACS indices all exceed n_phase — skipping overlay")
                else:
                    log("  Refscan has no non-zero energy — skipping overlay")
            else:
                log("  No 'refscan' key in TWIX file")
        except Exception as _exc:
            log(f"  Refscan overlay failed ({_exc}) — continuing without it")

    # ── 3b. Optional: re-extract PE mask via bart pattern ─────────────────────
    if mask_method == "bart_pattern":
        from .recon_backend import use_bart
        log("Re-extracting PE mask via sampling pattern…")
        try:
            if use_bart():
                from .utils import BartTempDir, readcfl, run_bart, writecfl
                with BartTempDir("bart_pattern_") as tmp:
                    ksp_cfl  = os.path.join(tmp, "ksp")
                    mask_cfl = os.path.join(tmp, "mask")
                    k4d = full_kspace[:, :, np.newaxis, :].astype(np.complex64)
                    writecfl(ksp_cfl, k4d)
                    run_bart("pattern", ksp_cfl, mask_cfl)
                    mask_raw = readcfl(mask_cfl)
                    mask_sq  = np.squeeze(mask_raw).real
            else:
                # sigpy/no-BART: the sampling pattern is simply where k-space is
                # nonzero (BART `pattern` computes exactly this).
                mask_sq = (np.abs(full_kspace) != 0).any(axis=-1).astype(float)
            # Collapse readout / coil dims → 1-D PE array
            if mask_sq.ndim == 2:
                pe_mask_bart = mask_sq.any(axis=0).astype(bool)
            elif mask_sq.ndim >= 3:
                pe_mask_bart = mask_sq.reshape(mask_sq.shape[0], mask_sq.shape[1], -1).any(axis=0).any(axis=-1).astype(bool)
            else:
                pe_mask_bart = mask_sq.astype(bool)
            if pe_mask_bart.shape == (n_phase,):
                pe_mask         = pe_mask_bart
                acquired_pe_idx = np.where(pe_mask)[0]
                actual_r        = n_phase / max(len(acquired_pe_idx), 1)
                log(f"  sampling-pattern mask: {pe_mask.sum()}/{n_phase} lines  (R={actual_r:.2f}×)")
            else:
                log(f"  pattern: unexpected shape {pe_mask_bart.shape}; keeping analytical mask")
        except Exception as exc:
            log(f"  pattern extraction failed ({exc}); keeping analytical mask")

    # ── 4a. GRAPPA reconstruction ─────────────────────────────────────────────
    recon_kspace  = None
    sense_image   = None
    sens_maps_out = None

    if method in ("grappa", "grappa_sense"):
        log(f"Running GRAPPA (pygrappa / mdgrappa), kernel {kernel_size}…")
        try:
            from pygrappa import mdgrappa
            recon_kspace = mdgrappa(
                kspace=full_kspace,
                calib=calib,
                kernel_size=kernel_size,
                lamda=lamda,
            )
        except ImportError:
            log("  pygrappa not found – falling back to zero-fill")
            recon_kspace = full_kspace.copy()
        log(f"  Filled k-space shape: {recon_kspace.shape}")

        # ── 4a-ii. GRAPPA + sensitivity-map coil combination ──────────────────
        if method == "grappa_sense":
            from .recon_backend import espirit_maps
            log("Computing ESPIRiT sensitivity maps for optimal coil combination…")
            try:
                k4d = full_kspace[:, :, np.newaxis, :].astype(np.complex64)
                try:
                    import sigpy as sp
                    img_tmp = sp.rss(sp.ifft(k4d, axes=(0, 1)), axes=-1)
                    scale   = float(np.percentile(np.abs(img_tmp).ravel(), 95))
                except Exception:
                    scale = float(np.max(np.abs(k4d)))
                scale = max(scale, 1e-12)
                k4d_norm = (k4d / scale).astype(np.complex64)

                actual_calib = acs_end - acs_start
                log(f"  ESPIRiT calibration (calib={actual_calib})…")
                sens_maps_out = espirit_maps(k4d_norm, actual_calib, num_maps=1,
                                         crop=espirit_crop, log=log)
                log(f"  Sensitivity maps shape: {sens_maps_out.shape}")

                # Optimal coil combination: image = sum(conj(S) * C_i) / sqrt(sum(|S|^2))
                # recon_kspace: (Nread, Nphase, Ncoils)
                coil_imgs = _ifft2c(recon_kspace)           # (Nread, Nphase, Ncoils)
                S = np.squeeze(sens_maps_out)               # (Nread, Nphase, Ncoils)
                numerator   = np.sum(np.conj(S) * coil_imgs, axis=2)
                denominator = np.sqrt(np.sum(np.abs(S) ** 2, axis=2)) + 1e-12
                sense_image = np.abs(numerator / denominator).astype(np.float32)
                log(f"  GRAPPA+SENSE combined image shape: {sense_image.shape}")

            except (RuntimeError, ImportError) as exc:
                log(f"  BART failed ({exc}); falling back to GRAPPA RSS")
                sens_maps_out = None

    # ── 4b. SENSE / CS-SENSE reconstruction via BART ──────────────────────────
    elif method in ("sense", "sense_cs"):
        from .recon_backend import active_backend_label, espirit_maps, pics
        log(f"Running {'Pure SENSE' if method == 'sense' else 'CS-SENSE'} ({active_backend_label()})…")
        try:
            # BART layout: (Nread, Nphase, 1, Ncoils)
            k4d = full_kspace[:, :, np.newaxis, :].astype(np.complex64)

            # Normalise k-space for numerical stability
            try:
                import sigpy as sp
                img_tmp = sp.rss(sp.ifft(k4d, axes=(0, 1)), axes=-1)
                scale   = float(np.percentile(np.abs(img_tmp).ravel(), 95))
            except Exception:
                scale = float(np.max(np.abs(k4d)))
            scale = max(scale, 1e-12)
            k4d_norm = k4d / scale

            actual_calib = acs_end - acs_start   # ACS width from JSON/mask
            log(f"  Computing sensitivity maps (ESPIRiT, calib={actual_calib})…")
            sens_maps_out = espirit_maps(k4d_norm, actual_calib, num_maps=1,
                                         crop=espirit_crop, log=log)

            if method == "sense":
                log(f"  Running PICS {'SENSE + L2' if sense_l2 else 'pure SENSE'} (iters={sense_max_iter})…")
                recon_raw = pics(k4d_norm, sens_maps_out, soft_sense=True, l2=sense_l2,
                                 iters=sense_max_iter, log=log)
            else:
                log(f"  Running PICS CS-SENSE (λ={sense_lambda}, iters={sense_max_iter})…")
                recon_raw = pics(k4d_norm, sens_maps_out, l1=True, lam=sense_lambda,
                                 iters=sense_max_iter, log=log)

            sense_image = np.abs(np.squeeze(recon_raw))   # (Nread, Nphase)
            log(f"  SENSE image shape: {sense_image.shape}")

        except RuntimeError as exc:
            log(f"  Recon failed ({exc}); falling back to zero-fill RSS")
            sense_image = None
        except ImportError as exc:
            log(f"  Missing dependency ({exc}); falling back to zero-fill RSS")
            sense_image = None

    # ── 4c. True ESPIRiT reconstruction via BART ──────────────────────────────
    elif method in ("espirit", "espirit_l1", "espirit_l2", "espirit_tv", "espirit_l1wav"):
        _espirit_label = {
            "espirit":     "ESPIRiT",
            "espirit_l1":  "ESPIRiT+L1",
            "espirit_l2":  "ESPIRiT+L2",
            "espirit_tv":  "ESPIRiT+TV",
            "espirit_l1wav": "ESPIRiT+L1-Wav",
        }.get(method, method)
        from .recon_backend import active_backend_label, espirit_maps, pics
        log(f"Running {_espirit_label} ({active_backend_label()}, {espirit_num_maps} map sets)…")
        try:
            # BART layout: (Nread, Nphase, 1, Ncoils)
            k4d = full_kspace[:, :, np.newaxis, :].astype(np.complex64)

            # Normalise k-space for numerical stability
            try:
                import sigpy as sp
                img_tmp = sp.rss(sp.ifft(k4d, axes=(0, 1)), axes=-1)
                scale   = float(np.percentile(np.abs(img_tmp).ravel(), 95))
            except Exception:
                scale = float(np.max(np.abs(k4d)))
            scale = max(scale, 1e-12)
            k4d_norm = k4d / scale

            actual_calib = acs_end - acs_start
            log(f"  ESPIRiT calibration (calib={actual_calib}, maps={espirit_num_maps})…")
            sens_maps_out = espirit_maps(k4d_norm, actual_calib,
                                         num_maps=espirit_num_maps,
                                         crop=espirit_crop, log=log)

            _reg = {"espirit_l2": "Q", "espirit_tv": "T",
                    "espirit_l1": "W", "espirit_l1wav": "W"}.get(method)
            log(f"  Running PICS {_espirit_label} (iters={sense_max_iter})…")
            recon_raw = pics(k4d_norm, sens_maps_out, reg=_reg, lam=sense_lambda,
                             iters=sense_max_iter, log=log)

            recon_sq = np.squeeze(recon_raw)
            # BART pics with the ESPIRiT operator outputs one image per map
            # set — combine via RSS across the last dimension.
            if recon_sq.ndim == 3:
                sense_image = np.sqrt(
                    np.sum(np.abs(recon_sq) ** 2, axis=-1)
                ).astype(np.float32)
            else:
                sense_image = np.abs(recon_sq).astype(np.float32)
            log(f"  ESPIRiT image shape: {sense_image.shape}")

        except RuntimeError as exc:
            log(f"  Recon failed ({exc}); falling back to zero-fill RSS")
            sense_image = None
        except ImportError as exc:
            log(f"  Missing dependency ({exc}); falling back to zero-fill RSS")
            sense_image = None

    # ── 5. Reconstruct images ─────────────────────────────────────────────────
    log("Reconstructing images (IFFT + RSS)…")
    zf_image = np.sqrt(np.sum(np.abs(_ifft2c(full_kspace)) ** 2, axis=2))
    log(f"  Zero-filled image shape: {zf_image.shape}")

    # BART PICS (SENSE / CS-SENSE / ESPIRiT*) runs on k-space that was divided by
    # `scale` for numerical stability, and its output image is NOT rescaled back,
    # so it lands on an arbitrary normalised intensity scale (~O(1)). zf_image,
    # the GRAPPA image and the reference all stay in raw IFFT units (~1e-6 here).
    # Downstream, both recon and reference are normalised by the reference's 95th
    # percentile — so an un-rescaled BART image blows up by ~1e6, the absolute-
    # difference map saturates to uniform white ("blank") and PSNR/SSIM are
    # meaningless. Rescale BART outputs to the zero-filled image's 95th
    # percentile so they share the reference's native scale. (grappa_sense is
    # excluded: its sense_image is _ifft2c of GRAPPA-filled k-space, already raw.)
    if sense_image is not None and method in (
        "sense", "sense_cs", "espirit", "espirit_l1",
        "espirit_l2", "espirit_tv", "espirit_l1wav",
    ):
        _se_p95 = float(np.percentile(np.abs(sense_image).ravel(), 95))
        _zf_p95 = float(np.percentile(np.abs(zf_image).ravel(), 95))
        if _se_p95 > 0:
            sense_image = (sense_image * (_zf_p95 / _se_p95)).astype(np.float32)
            log(f"  Rescaled BART output to data scale (×{_zf_p95 / _se_p95:.3e}).")

    if recon_kspace is not None and sense_image is not None and method == "grappa_sense":
        # grappa_sense: primary image is the sensitivity-weighted combination
        grappa_image = sense_image
    elif recon_kspace is not None:
        grappa_image = np.sqrt(np.sum(np.abs(_ifft2c(recon_kspace)) ** 2, axis=2))
    elif sense_image is not None:
        grappa_image = sense_image
    else:
        grappa_image = zf_image

    # Pre-compute 2-D RSS k-space magnitudes for display
    undersampled_kspace_rss = np.sqrt(np.sum(np.abs(full_kspace) ** 2, axis=2)).astype(np.float32)
    if recon_kspace is not None:
        recon_kspace_rss = np.sqrt(np.sum(np.abs(recon_kspace) ** 2, axis=2)).astype(np.float32)
    else:
        recon_kspace_rss = undersampled_kspace_rss.copy()

    del full_kspace
    if recon_kspace is not None:
        del recon_kspace

    # ── 6. Optional comparison with fully-sampled reference ───────────────────
    reference_image = None
    rmse = psnr = ssim = None
    reference_warning = None
    _ref_orient_keys: Dict[str, Any] = {}

    if reference_dat_path is not None:
        log(f"Loading reference: {Path(reference_dat_path).name}")
        # Reuse the already-read reference (probed early for n_phase); re-read only if needed.
        if _ref_raw_cache is not None:
            ref_raw = _ref_raw_cache
            _ref_raw_cache = None
        else:
            from read_siemens_scans.twix_reading import read_raw_twix_data as _rrtd
            ref_raw, _ = _rrtd(str(reference_dat_path), fix_oversampling=False)
            if ref_raw.ndim == 4:
                _ref_sl = _refscan_sl if _refscan_sl is not None else ref_raw.shape[0] // 2
                ref_raw = ref_raw[_ref_sl]
            ref_raw = crop_readout_oversampling(
                ref_raw, expected_nx=n_phase, readout_axis=2, log_fn=log,
                dat_path=reference_dat_path,
            )
        # Un-scramble the reference too when it is a fully-sampled TSE scan,
        # otherwise the comparison target is echo-scrambled and RMSE/PSNR/SSIM
        # are meaningless. Detected from the reference filename (full mask).
        from .tse_reorder import detect_tse_params as _dtp, unscramble_rows as _usr
        _ref_tse = _dtp(reference_dat_path, tse=tse, necho=necho, te1=te1, teeff=teeff)
        if _ref_tse is not None:
            log(f"  Reference is TSE — un-scrambling (necho={_ref_tse['necho']}).")
            ref_raw = _usr(ref_raw, _ref_tse["necho"], _ref_tse["TE1"], _ref_tse["TEeff"], log_fn=log)
        ref_3d    = np.transpose(ref_raw, (2, 0, 1)).astype(np.complex128)
        del ref_raw
        ref_imgs  = _ifft2c(ref_3d)
        del ref_3d
        reference_image = np.sqrt(np.sum(np.abs(ref_imgs) ** 2, axis=2))
        del ref_imgs

        # Coil-sensitivity-combined recons (SENSE / CS-SENSE / ESPIRiT* /
        # GRAPPA+SENSE) are scored against a MATCHED (SENSE-combined) reference;
        # an RSS reference would otherwise penalise the coil-combination
        # difference rather than the reconstruction. Pure GRAPPA outputs RSS, so
        # it keeps the RSS reference above.
        if method != "grappa":
            try:
                from .reference import load_reference_rss as _load_ref_sense
                reference_image = _load_ref_sense(
                    reference_dat_path, combine="sense", slice_idx=_refscan_sl,
                    expected_nx=n_phase, tse=tse, necho=necho, te1=te1, teeff=teeff,
                    calib_width=calib_width, log=log,
                )
                log("  Using a matched SENSE-combined reference for metrics.")
            except Exception as _e:
                log(f"  [WARN] matched SENSE reference failed ({_e}); using RSS.")

        # Resize the reference to the recon, flag a sequence/matrix mismatch,
        # and compute the metrics — shared with the Poisson recon.
        from .reference import compare_to_reference
        _cmp = compare_to_reference(
            grappa_image, reference_image,
            recon_name=Path(dat_path).name,
            reference_name=Path(reference_dat_path).name,
            log=log,
        )
        reference_image   = _cmp["reference_image"]
        rmse, psnr, ssim  = _cmp["rmse"], _cmp["psnr"], _cmp["ssim"]
        reference_warning = _cmp["reference_warning"]
        _ref_orient_keys = {k: _cmp[k] for k in (
            "reference_image_raw", "reference_orientation",
            "reference_orientation_auto", "reference_orientation_ambiguous",
            "recon_name", "reference_name")}

    from .constants import method_label as _mlabel
    log(f"{_mlabel(method)} reconstruction complete.")

    return {
        "undersampled_kspace_rss": undersampled_kspace_rss,
        "recon_kspace_rss":        recon_kspace_rss,
        # Keep legacy key name so existing display / PDF code works unchanged
        "grappa_kspace_rss":       recon_kspace_rss,
        "zf_image":                zf_image,
        "grappa_image":            grappa_image,
        "sense_image":             sense_image,
        "sens_maps":               sens_maps_out,
        "pe_mask":                 pe_mask,
        "acs_start":               acs_start,
        "acs_end":                 acs_end,
        "actual_r":                actual_r,
        "reference_image":         reference_image,
        "rmse":                    rmse,
        "psnr":                    psnr,
        "ssim":                    ssim,
        "reference_warning":       reference_warning,
        **_ref_orient_keys,
        "method":                  method,
        "mask_method": mask_method,
        "params": {
            "method":           method,
            "mask_method":      mask_method,
            "acs_lines":        acs_lines,
            "r_factor":         r_factor,
            "kernel_size":      kernel_size,
            "lamda":            lamda,
            "calib_width":      calib_width,
            "sense_lambda":     sense_lambda,
            "sense_max_iter":   sense_max_iter,
            "espirit_num_maps": espirit_num_maps,
            "tse":              _tse,
        },
    }
