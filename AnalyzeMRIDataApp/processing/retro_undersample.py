"""
Retrospective Undersampling Simulator.

Takes a fully-sampled .dat file, builds a 1-D PE undersampling mask
(equispaced or variable-density), applies it to the loaded k-space, and
optionally runs a reconstruction.

Mask algorithms are exact Python ports of the MATLAB functions
generate_equispaced_1D() and generate_variable_density_1D() in
CreateSeqFiles/kband_se_t1.m.

Reconstruction reuses the GRAPPA / SENSE / CS-SENSE pipeline from
processing/equispaced_recon.py.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
from skimage.metrics import structural_similarity as _ssim


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

from .imaging import centre_block, espirit_normalise, ifft2c as _ifft2c
from .utils import ensure_twix_importable as _add_twix_path


def _grappa_pad(acs_shape, af, ks):
    """Replicate ggrappa's (commented-out) internal input padding.

    ``GRAPPA_Recon`` still crops the output at the end of the function but no
    longer pads the input, so the result comes back smaller/misaligned unless
    ``sig`` is pre-padded by the matching amount. Mirrors the sbl/tbl/pos
    formula from ggrappa's estimation.py.

    Parameters
    ----------
    acs_shape : tuple (nc, acsny, acsnz, acsnx)
    af        : [afy, afz]
    ks        : (ky, kz, kx) kernel size

    Returns
    -------
    pad tuple for F.pad: (kx_l, kx_r, kz_l, kz_r, ky_l, ky_r)
    """
    _, acsny, acsnz, acsnx = acs_shape
    k0, k1, k2 = ks
    af0, af1 = af
    sbly = min((k0 - 1) * af0 + 1, acsny)
    sblz = min((k1 - 1) * af1 + 1, acsnz)
    sblx = min(k2, acsnx)
    tbly, tblz, tblx = af0, af1, 1
    xpos = (sblx - 1) // 2
    ypos = (sbly - tbly) // 2
    zpos = (sblz - tblz) // 2
    return (xpos, sblx - xpos - tblx,
            (af1 - zpos) % tblz + zpos, sblz - zpos - tblz,
            (af0 - ypos) % tbly + ypos, sbly - ypos - tbly)


# ─────────────────────────────────────────────────────────────────────────────
# Mask builders — exact Python ports of the MATLAB functions
# ─────────────────────────────────────────────────────────────────────────────

def build_equispaced_mask(n: int, R: float, acs_lines: int) -> np.ndarray:
    """
    Build a 1-D equispaced undersampling mask.

    Exact port of MATLAB generate_equispaced_1D():
      center_idx = floor(n/2) + 1  (1-based) → n//2 (0-based)
      idx_right  = center_idx : R_factor : n   (1-based)
      idx_left   = center_idx : -R_factor : 1  (1-based)
      ACS block  = center_idx - floor(acs/2) .. center_idx + ceil(acs/2) - 1

    Parameters
    ----------
    n         : PE matrix size (e.g. 256)
    R         : acceleration factor (need not be integer)
    acs_lines : number of fully-sampled ACS lines at the centre

    Returns
    -------
    bool ndarray of shape (n,)
    """
    r = int(round(R))                  # integer step (matches MATLAB integer indexing)
    # MATLAB: center_idx = floor(n/2) + 1   (1-based)
    # Python 0-based equivalent:
    center = n // 2

    mask = np.zeros(n, dtype=bool)

    # Step outward from centre (mirrors MATLAB idx_right / idx_left)
    mask[np.arange(center, n,  r)] = True
    mask[np.arange(center, -1, -r)] = True

    # ACS block (mirrors MATLAB lines acs_start:acs_end)
    acs_start, acs_end = centre_block(n, acs_lines, clamp=True)
    mask[acs_start:acs_end] = True

    return mask


def build_vd_mask(
    n: int,
    R: float,
    power_decay: float,
    acs_lines: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Build a 1-D variable-density undersampling mask.

    Port of MATLAB generate_variable_density_1D():
      - PDF = (1 - |x|)^power_decay + 0.05 floor, x ∈ [-1, 1]
      - Fixed indices: first & last sample + ACS block
      - Remaining budget sampled from PDF using numpy weighted random choice
        (replaces MATLAB Poisson-disc step; distribution matches faithfully)

    Parameters
    ----------
    n           : PE matrix size
    R           : target acceleration factor
    power_decay : PDF exponent (1=linear, 2=parabolic)
    acs_lines   : fully-sampled ACS lines at centre
    rng         : optional numpy Generator for reproducibility

    Returns
    -------
    bool ndarray of shape (n,)
    """
    if rng is None:
        rng = np.random.default_rng()

    # MATLAB 1-based → Python 0-based
    center = n // 2                                         # MATLAB floor(n/2)+1 → n//2
    acs_start = max(0, center - acs_lines // 2)            # MATLAB acs_start-1
    acs_end   = min(n - 1, center + (acs_lines - acs_lines // 2) - 1)  # MATLAB acs_end-1

    # Fixed indices: ACS + very outer edges (index 0 and n-1)
    fixed_set = set(range(acs_start, acs_end + 1))
    fixed_set.add(0)
    fixed_set.add(n - 1)
    fixed_indices = np.array(sorted(fixed_set), dtype=int)

    # Budget
    target_total = int(round(n / R))
    lines_to_sample = target_total - len(fixed_indices)

    # PDF — (1 - |x|)^power_decay + floor
    x   = np.linspace(-1.0, 1.0, n)
    pdf = (1.0 - np.abs(x)) ** power_decay + 0.05

    # Zero out fixed regions so they don't compete for the random budget
    pdf[fixed_indices] = 0.0

    if lines_to_sample <= 0:
        # ACS alone already meets or exceeds the budget
        mask = np.zeros(n, dtype=bool)
        mask[fixed_indices] = True
        return mask

    # Normalise to probability
    pdf_sum = pdf.sum()
    if pdf_sum < 1e-12:
        # Degenerate: fill uniformly
        probs = np.ones(n, dtype=float)
        probs[fixed_indices] = 0.0
        probs /= probs.sum()
    else:
        probs = pdf / pdf_sum

    # Candidate non-fixed indices
    candidate_idx = np.where(probs > 0)[0]
    candidate_probs = probs[candidate_idx]
    candidate_probs /= candidate_probs.sum()

    n_choose = min(lines_to_sample, len(candidate_idx))
    chosen = rng.choice(candidate_idx, size=n_choose, replace=False, p=candidate_probs)

    # Combine
    all_idx = np.unique(np.concatenate([fixed_indices, chosen]))
    mask = np.zeros(n, dtype=bool)
    mask[all_idx] = True
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Main reconstruction function
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_retro(
    dat_path: str,
    *,
    mask_type: str = "equispaced",      # "equispaced" | "vd"
    R: float = 4.0,
    power_decay: float = 1.0,           # VD only
    acs_lines: int = 24,
    recon_method: str = "none",         # "none" | "grappa" | "sense" | "sense_cs" | "espirit" | "espirit_l1"
    # Recon params (forwarded to equispaced pipeline)
    kernel_size: tuple = (5, 5),
    grappa_lambda: float = 0.1,
    calib_width: int = 24,
    sense_lambda: float = 0.01,
    sense_max_iter: int = 100,
    sense_l2: bool = False,
    espirit_num_maps: int = 2,
    espirit_crop: float = 0.8,      # ESPIRiT eigenvalue mask threshold (both engines)
    swap_pe_read: bool = False,     # True → undersample Readout (L-R) instead of Phase (A-P)
    rng_seed: Optional[int] = None,
    tse: Optional[bool] = None,
    necho: Optional[int] = None,
    te1: Optional[float] = None,
    teeff: Optional[float] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    slice_selector_cb: Optional[Callable[[int], int]] = None,
) -> Dict[str, Any]:
    """
    Retrospectively undersample fully-sampled k-space and optionally reconstruct.

    Parameters
    ----------
    dat_path     : path to a fully-sampled TWIX .dat file
    mask_type    : "equispaced" or "vd"
    R            : target acceleration factor
    power_decay  : PDF exponent for VD mask
    acs_lines    : ACS lines at centre
    recon_method : "none" | "grappa" | "sense" | "sense_cs" | "espirit" | "espirit_l1"
    swap_pe_read : False (default) → undersample Phase axis (A-P, axis 1);
                   True            → undersample Readout axis (L-R, axis 0)
    rng_seed     : optional seed for VD mask randomness

    Returns
    -------
    dict with keys:
        pe_mask              : (n_phase,) bool
        acs_start, acs_end   : int
        actual_r             : float
        mask_type            : str
        fully_sampled_image  : (Nr, Np) float — RSS of fully-sampled data
        zf_image             : (Nr, Np) float — zero-filled image
        recon_image          : (Nr, Np) float or None
        diff_image           : (Nr, Np) float or None (|recon - fs|, normed)
        undersampled_kspace_rss : (Nr, Np) float32
        rmse, psnr           : float or None
        method               : str ("none" | "grappa" | "sense" | "sense_cs")
        params               : dict
    """

    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    _add_twix_path()
    from read_siemens_scans.twix_reading import read_raw_twix_data
    from .utils import crop_readout_oversampling, select_slice

    # ── 1. Read k-space ───────────────────────────────────────────────────────
    log(f"[Retro Sim] Reading TWIX: {Path(dat_path).name}")
    kspace_raw, _ = read_raw_twix_data(str(dat_path), fix_oversampling=False)
    # Shape: (Nacq_PE, Ncoils, Nreadout) — or (Nslices, Nacq_PE, Ncoils, Nreadout) for multi-slice
    log(f"  Raw shape: {kspace_raw.shape}")
    kspace_raw, _ = select_slice(kspace_raw, slice_selector_cb, log)
    # Shape: (Nacq_PE, Ncoils, Nreadout) — fully sampled, so Nacq_PE == n_phase.
    n_acq, n_coils, _n_readout_raw = kspace_raw.shape
    n_phase = n_acq   # n_acq IS the full PE dimension (all lines present)

    # Crop readout oversampling before transpose; uses n_phase as expected_nx
    # (valid for square fully-sampled scans; auto-detect handles non-OS data gracefully).
    kspace_raw = crop_readout_oversampling(
        kspace_raw, expected_nx=n_phase, readout_axis=2, log_fn=log,
        dat_path=dat_path,
    )
    n_acq, n_coils, n_readout = kspace_raw.shape

    # TSE un-scramble: a fully-sampled TSE input is stored in echo-train order,
    # so un-scramble to true PE order before retrospective masking — otherwise
    # both the reference and the simulated undersampling operate on scrambled
    # k-space. SE/non-TSE → skipped.
    from .tse_reorder import detect_tse_params, unscramble_rows
    _tse = detect_tse_params(dat_path, tse=tse, necho=necho, te1=te1, teeff=teeff)
    if _tse is not None:
        log(f"  TSE detected — un-scrambling echo order (necho={_tse['necho']}).")
        kspace_raw = unscramble_rows(
            kspace_raw, _tse["necho"], _tse["TE1"], _tse["TEeff"], log_fn=log
        )

    # Rearrange to (Nreadout, Nphase, Ncoils)
    full_kspace = np.transpose(kspace_raw, (2, 0, 1)).astype(np.complex128)

    # ── 2. Fully-sampled reference image ──────────────────────────────────────
    log("  Computing fully-sampled RSS reference…")
    fs_imgs = _ifft2c(full_kspace)
    fully_sampled_image = np.sqrt(np.sum(np.abs(fs_imgs) ** 2, axis=2)).astype(np.float32)

    # Pre-compute fully-sampled k-space RSS for display
    fs_kspace_rss = np.sqrt(np.sum(np.abs(full_kspace) ** 2, axis=2)).astype(np.float32)

    # Coil-combined recon methods (SENSE / CS-SENSE / ESPIRiT*) produce a
    # sensitivity-combined image, so compare them against a MATCHED
    # (SENSE-combined) fully-sampled reference — an RSS reference would penalise
    # the coil-combination difference, not the reconstruction. Built from the
    # SAME full k-space so the sizes match; falls back to RSS on any failure.
    if recon_method not in ("none", "grappa", "ggrappa"):
        try:
            from .recon_backend import espirit_maps as _emaps, pics as _pics
            _k4d = full_kspace[:, :, np.newaxis, :].astype(np.complex64)
            _k4d_norm, _ = espirit_normalise(_k4d)
            _maps = _emaps(_k4d_norm, calib_width, num_maps=1, log=log)
            _ref = _pics(_k4d_norm, _maps, soft_sense=True, iters=sense_max_iter, log=log)
            fully_sampled_image = np.abs(np.squeeze(_ref)).astype(np.float32)
            log("  Using a matched SENSE-combined fully-sampled reference.")
        except Exception as _e:
            log(f"  [WARN] matched SENSE reference failed ({_e}); using RSS.")

    # ── 3. Build undersampling mask ───────────────────────────────────────────
    axis_label = "RO (L-R)" if swap_pe_read else "PE (A-P)"
    log(f"  Building {mask_type} mask (R={R:.1f}, ACS={acs_lines}, axis={axis_label})…")
    mask_size = n_readout if swap_pe_read else n_phase
    if mask_type == "vd":
        rng = np.random.default_rng(rng_seed)
        pe_mask = build_vd_mask(mask_size, R, power_decay, acs_lines, rng=rng)
    else:
        pe_mask = build_equispaced_mask(mask_size, R, acs_lines)

    acquired_pe_idx = np.where(pe_mask)[0]
    actual_r = mask_size / max(len(acquired_pe_idx), 1)
    log(f"  Mask ({axis_label}): {len(acquired_pe_idx)}/{mask_size} lines  (actual R = {actual_r:.2f}×)")

    # ACS bounds — always use the same formula as the mask builders
    acs_start, acs_end = centre_block(mask_size, acs_lines, clamp=True)

    # ── 4. Apply mask (zero out non-sampled lines) ────────────────────────────
    undersampled_kspace = np.zeros_like(full_kspace)
    if swap_pe_read:
        # Simulate Swap Phase/Read: undersample the Readout axis (axis 0, L-R)
        undersampled_kspace[acquired_pe_idx, :, :] = full_kspace[acquired_pe_idx, :, :]
    else:
        # Standard: undersample the Phase Encode axis (axis 1, A-P)
        undersampled_kspace[:, acquired_pe_idx, :] = full_kspace[:, acquired_pe_idx, :]

    undersampled_kspace_rss = np.sqrt(
        np.sum(np.abs(undersampled_kspace) ** 2, axis=2)
    ).astype(np.float32)

    # ── 5. Zero-filled image ──────────────────────────────────────────────────
    zf_image = np.sqrt(np.sum(np.abs(_ifft2c(undersampled_kspace)) ** 2, axis=2)).astype(np.float32)

    # ── 6. Optional reconstruction ────────────────────────────────────────────
    recon_image = None
    recon_kspace_rss = None   # only set when GRAPPA produces a reconstructed k-space
    rmse = psnr = ssim = None
    # Initialised here (not inside the if-block): the return dict references
    # sens_maps_out unconditionally, including on the recon_method == "none" path.
    recon_kspace  = None
    sense_img     = None
    sens_maps_out = None

    if recon_method != "none":
        log(f"  Running reconstruction: {recon_method}…")
        # Extract calibration region from the correct axis
        if swap_pe_read:
            calib = undersampled_kspace[acs_start:acs_end, :, :]
        else:
            calib = undersampled_kspace[:, acs_start:acs_end, :]

        if recon_method == "grappa":
            try:
                from pygrappa import mdgrappa
                recon_kspace = mdgrappa(
                    kspace=undersampled_kspace,
                    calib=calib,
                    kernel_size=kernel_size,
                    lamda=grappa_lambda,
                )
                log("  GRAPPA done.")
            except ImportError:
                log("  pygrappa not found — falling back to zero-fill")
                recon_kspace = undersampled_kspace.copy()

        elif recon_method == "ggrappa":
            try:
                import torch
                import torch.nn.functional as F
                from ggrappa import GRAPPA_Recon

                if mask_type == "vd":
                    log("  WARNING: ggrappa assumes equispaced acceleration; "
                        "VD mask results may be poor.")

                af = [max(1, int(round(R))), 1]
                ks = (int(kernel_size[0]), 1, int(kernel_size[1]))

                # Build ggrappa layout (nc, ky, kz=1, kx) so the undersampled
                # axis maps to ky. ACS block is contiguous along that same axis.
                if swap_pe_read:
                    # RO undersampled (axis 0) → ky = Nread, kx = Nphase
                    sig_np = np.transpose(undersampled_kspace, (2, 0, 1))[:, :, None, :]
                else:
                    # PE undersampled (axis 1) → ky = Nphase, kx = Nread
                    sig_np = np.transpose(undersampled_kspace, (2, 1, 0))[:, :, None, :]

                sig = torch.from_numpy(sig_np).to(torch.complex64)
                acs = sig[:, acs_start:acs_end, :, :].clone()
                pad = _grappa_pad(tuple(acs.shape), af, ks)
                sig_p = F.pad(sig, pad)
                log(f"  ggrappa: sig{tuple(sig.shape)} acs{tuple(acs.shape)} "
                    f"af={af} kernel={ks} pad->{tuple(sig_p.shape)}")

                out = GRAPPA_Recon(sig_p, acs, af=af, kernel_size=ks,
                                   lambda_=grappa_lambda, cuda=False, quiet=True)
                rec = (out[0] if isinstance(out, (tuple, list)) else out)
                rec = rec.detach().cpu().numpy()

                # Restore (Nread, Nphase, Ncoils) layout
                if swap_pe_read:
                    recon_kspace = np.transpose(rec[:, :, 0, :], (1, 2, 0))
                else:
                    recon_kspace = np.transpose(rec[:, :, 0, :], (2, 1, 0))
                recon_kspace = recon_kspace.astype(np.complex128)

                # Data consistency — re-insert truly acquired samples
                samp = np.abs(undersampled_kspace).sum(axis=2) > 0
                recon_kspace[samp, :] = undersampled_kspace[samp, :]
                log("  ggrappa done.")
            except ImportError:
                log("  ggrappa not found — falling back to zero-fill")
                recon_kspace = undersampled_kspace.copy()

        elif recon_method in ("sense", "sense_cs"):
            from .recon_backend import active_backend_label, espirit_maps, pics
            try:
                k4d = undersampled_kspace[:, :, np.newaxis, :].astype(np.complex64)
                k4d_norm, _ = espirit_normalise(k4d)

                actual_calib = acs_end - acs_start
                log(f"  ESPIRiT (calib={actual_calib}, {active_backend_label()})…")
                sens_maps_out = espirit_maps(k4d_norm, actual_calib, num_maps=1,
                                             crop=espirit_crop, log=log)

                if recon_method == "sense":
                    sense_img_raw = pics(k4d_norm, sens_maps_out, soft_sense=True,
                                         l2=sense_l2, iters=sense_max_iter, log=log)
                else:
                    sense_img_raw = pics(k4d_norm, sens_maps_out, l1=True,
                                         lam=sense_lambda, iters=sense_max_iter, log=log)

                sense_img = np.abs(np.squeeze(sense_img_raw)).astype(np.float32)
                log(f"  SENSE image shape: {sense_img.shape}")

            except (RuntimeError, ImportError) as exc:
                log(f"  Recon failed ({exc}); falling back to zero-fill")

        elif recon_method in ("espirit", "espirit_l1", "espirit_l2", "espirit_tv", "espirit_l1wav"):
            from .recon_backend import active_backend_label, espirit_maps, pics
            try:
                k4d = undersampled_kspace[:, :, np.newaxis, :].astype(np.complex64)
                k4d_norm, _ = espirit_normalise(k4d)

                actual_calib = acs_end - acs_start
                log(f"  ESPIRiT calibration (calib={actual_calib}, maps={espirit_num_maps}, "
                    f"{active_backend_label()})…")
                sens_maps_out = espirit_maps(k4d_norm, actual_calib,
                                             num_maps=espirit_num_maps,
                                             crop=espirit_crop, log=log)

                _reg = {"espirit_l2": "Q", "espirit_tv": "T",
                        "espirit_l1": "W", "espirit_l1wav": "W"}.get(recon_method)
                log(f"  Running PICS {recon_method} (iters={sense_max_iter})…")
                recon_raw = pics(k4d_norm, sens_maps_out, reg=_reg, lam=sense_lambda,
                                 iters=sense_max_iter, log=log)

                recon_sq = np.squeeze(recon_raw)
                # RSS-combine across map sets if multiple sets were used.
                if recon_sq.ndim == 3:
                    sense_img = np.sqrt(
                        np.sum(np.abs(recon_sq) ** 2, axis=-1)
                    ).astype(np.float32)
                else:
                    sense_img = np.abs(recon_sq).astype(np.float32)
                log(f"  ESPIRiT image shape: {sense_img.shape}")

            except (RuntimeError, ImportError) as exc:
                log(f"  Recon failed ({exc}); falling back to zero-fill")

        # Final reconstructed image
        if recon_kspace is not None:
            recon_image = np.sqrt(
                np.sum(np.abs(_ifft2c(recon_kspace)) ** 2, axis=2)
            ).astype(np.float32)
            recon_kspace_rss = np.sqrt(
                np.sum(np.abs(recon_kspace) ** 2, axis=2)
            ).astype(np.float32)
        elif sense_img is not None:
            recon_image = sense_img
            # Pseudo k-space for display: FFT of reconstructed magnitude image
            recon_kspace_rss = np.abs(
                np.fft.fftshift(
                    np.fft.fft2(
                        np.fft.ifftshift(sense_img.astype(np.complex64))
                    )
                )
            ).astype(np.float32)
        else:
            recon_image = zf_image.copy()

    # ── 7. Metrics & difference image ─────────────────────────────────────────
    diff_image = None
    if recon_image is not None:
        from .imaging import metrics_vs_reference
        # SENSE / CS-SENSE / ESPIRiT (BART PICS) images sit on an arbitrary
        # intensity scale; rescale so the recon's 95th percentile matches the
        # reference before scoring, else the shared-scale metrics and the
        # difference image are meaningless (mirrors the rescale in poisson_recon).
        _rec = np.abs(recon_image).astype(np.float32)
        _r95 = float(np.percentile(_rec.ravel(), 95)) + 1e-12
        _f95 = float(np.percentile(np.abs(fully_sampled_image).ravel(), 95)) + 1e-12
        recon_image = (_rec * (_f95 / _r95)).astype(np.float32)
        m = metrics_vs_reference(recon_image, fully_sampled_image, with_diff=True)
        diff_image = m["diff"]
        rmse, psnr, ssim = m["rmse"], m["psnr"], m["ssim"]
        _ssim_txt = f"{ssim:.4f}" if ssim is not None else "n/a"
        log(f"  RMSE={rmse:.4f}  PSNR={psnr:.2f} dB  SSIM={_ssim_txt}")

    from .constants import method_label as _mlabel
    log(f"[Retro Sim] Done — {mask_type.upper()}, axis={axis_label}, "
        f"R={actual_r:.2f}×, method={_mlabel(recon_method)}")

    return {
        "pe_mask":                  pe_mask,
        "acs_start":                acs_start,
        "acs_end":                  acs_end,
        "actual_r":                 actual_r,
        "mask_type":                mask_type,
        "swap_pe_read":             swap_pe_read,
        "fully_sampled_image":      fully_sampled_image,
        "zf_image":                 zf_image,
        "recon_image":              recon_image,
        "diff_image":               diff_image,
        "undersampled_kspace_rss":  undersampled_kspace_rss,
        "fs_kspace_rss":            fs_kspace_rss,
        "recon_kspace_rss":         recon_kspace_rss,
        "sens_maps":                sens_maps_out,
        "rmse":                     rmse,
        "psnr":                     psnr,
        "ssim":                     ssim,
        "method":                   recon_method,
        "params": {
            "mask_type":        mask_type,
            "R":                R,
            "acs_lines":        acs_lines,
            "power_decay":      power_decay,
            "recon_method":     recon_method,
            "kernel_size":      kernel_size,
            "grappa_lambda":    grappa_lambda,
            "calib_width":      calib_width,
            "sense_lambda":     sense_lambda,
            "sense_max_iter":   sense_max_iter,
            "espirit_num_maps": espirit_num_maps,
            "swap_pe_read":     swap_pe_read,
            "rng_seed":         rng_seed,
        },
    }
