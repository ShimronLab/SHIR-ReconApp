"""
Shared fully-sampled reference comparison for undersampled reconstructions.

Both the Equispaced and the Poisson-disc recons compare their result against a
fully-sampled reference (image + absolute difference + SSIM/NMSE/PSNR). The
metric/resize/mismatch-warning logic lives here so both use identical behavior.

- ``load_reference_rss``   : read a reference .dat → RSS magnitude image.
- ``compare_to_reference`` : orient the reference to the recon (auto or forced),
  resize it, flag a sequence/matrix mismatch, and compute
  95th-percentile-normalised metrics.
- ``orient_image`` / ``auto_orient_reference`` : the 8 rotate/flip orientations
  and the correlation-based picker. Needed because a scan whose readout/PE axes
  are swapped relative to the reference (e.g. a Pulseq sequence vs the scanner's
  own fully-sampled one) comes out rotated/mirrored in the (read, phase) array.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np


def _noop(_msg: str) -> None:
    pass


from .utils import ensure_twix_importable as _add_twix_path


def _ifft2c(ks: np.ndarray) -> np.ndarray:
    """Centred 2-D IFFT over axes 0 (readout) and 1 (phase)."""
    return np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(ks, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )


def _seq_kind(name: str) -> Optional[str]:
    """Classify a filename's pulse sequence (for mismatch warnings)."""
    n = name.lower()
    if "tse" in n:
        return "TSE"
    if "_se_" in n or "se_tra" in n or "_se_t" in n or "t1_se" in n or "t2_se" in n:
        return "SE"
    if "fl2d" in n or "gre" in n or "flash" in n:
        return "GRE"
    return None


def load_reference_rss(
    reference_dat_path: str,
    *,
    slice_idx: Optional[int] = None,
    expected_nx: Optional[int] = None,
    tse: Optional[bool] = None,
    necho: Optional[int] = None,
    te1: Optional[float] = None,
    teeff: Optional[float] = None,
    combine: str = "rss",
    calib_width: int = 24,
    log: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """
    Read a fully-sampled reference TWIX file and return its magnitude image.

    ``combine="rss"`` (default) returns the root-sum-of-squares image — the right
    target for RSS-combined recons (e.g. GRAPPA). ``combine="sense"`` returns an
    ESPIRiT+SENSE coil-combined image (matched to SENSE/CS/ESPIRiT recons); an
    RSS reference would otherwise penalise the coil-combination difference rather
    than the reconstruction itself. The SENSE path falls back to RSS if BART/
    ESPIRiT is unavailable.

    Multi-slice (4-D) references pick ``slice_idx`` (or the middle slice).
    Readout oversampling is cropped (best-effort — skipped, not fatal, on an
    exotic factor). A fully-sampled TSE reference is echo-un-scrambled so the
    comparison target is not scrambled.
    """
    log = log or _noop

    if combine == "sense":
        from .fullysamp_recon import reconstruct_fullysamp
        _res = reconstruct_fullysamp(
            str(reference_dat_path), use_sense=True, calib_width=calib_width,
            expected_nx=expected_nx, tse=tse, necho=necho, te1=te1, teeff=teeff,
            slice_selector_cb=(None if slice_idx is None
                               else (lambda _n, _i=slice_idx: int(_i))),
            progress_cb=log,
        )
        _img = _res.get("image_combined")
        if _img is None:
            _img = _res.get("image_rss")
        return np.abs(np.squeeze(_img)).astype(np.float64)

    _add_twix_path()
    from read_siemens_scans.twix_reading import read_raw_twix_data
    from .utils import crop_readout_oversampling

    log(f"Loading reference: {Path(reference_dat_path).name}")
    ref_raw, _ = read_raw_twix_data(str(reference_dat_path), fix_oversampling=False)
    if ref_raw.ndim == 4:
        sl = ref_raw.shape[0] // 2 if slice_idx is None else int(slice_idx)
        log(f"  Reference is multi-slice ({ref_raw.shape[0]}) — using slice {sl}")
        ref_raw = ref_raw[sl]
    exp = int(expected_nx) if expected_nx is not None else ref_raw.shape[0]
    try:
        ref_raw = crop_readout_oversampling(
            ref_raw, expected_nx=exp, readout_axis=2, log_fn=log
        )
    except Exception as exc:                       # exotic OS factor → no crop
        log(f"  [WARN] reference OS crop skipped ({exc}).")

    from .tse_reorder import detect_tse_params, unscramble_rows
    _tse = detect_tse_params(reference_dat_path, tse=tse, necho=necho,
                             te1=te1, teeff=teeff)
    if _tse is not None:
        log(f"  Reference is TSE — un-scrambling (necho={_tse['necho']}).")
        ref_raw = unscramble_rows(
            ref_raw, _tse["necho"], _tse["TE1"], _tse["TEeff"], log_fn=log
        )

    ref_3d = np.transpose(ref_raw, (2, 0, 1)).astype(np.complex128)
    del ref_raw
    imgs = _ifft2c(ref_3d)
    return np.sqrt(np.sum(np.abs(imgs) ** 2, axis=2))


# The 8 orientations of a 2-D array (the dihedral group), as array operations.
# "none" first: auto-orientation prefers it on (near-)ties.
ORIENTATIONS = {
    "none":          lambda a: a,
    "rot90":         lambda a: np.rot90(a, 1),
    "rot180":        lambda a: np.rot90(a, 2),
    "rot270":        lambda a: np.rot90(a, 3),
    "flipud":        lambda a: a[::-1, :],
    "fliplr":        lambda a: a[:, ::-1],
    "transpose":     lambda a: a.T,
    "antitranspose": lambda a: np.rot90(a, 2).T,
}

# Auto keeps the reference as-is unless another orientation correlates better
# by at least this much (guards against flipping a near-symmetric head on noise).
_AUTO_ORIENT_MARGIN = 0.02


def orient_image(img: np.ndarray, orientation: str) -> np.ndarray:
    """Apply one of :data:`ORIENTATIONS` (array-level) to a 2-D image."""
    if orientation not in ORIENTATIONS:
        raise ValueError(f"Unknown orientation '{orientation}' "
                         f"(expected one of {list(ORIENTATIONS)}).")
    return np.ascontiguousarray(ORIENTATIONS[orientation](img))


def _resize_to(img: np.ndarray, shape) -> np.ndarray:
    if img.shape == tuple(shape):
        return img
    from scipy.ndimage import zoom
    return zoom(img, (shape[0] / img.shape[0], shape[1] / img.shape[1]), order=1)


def auto_orient_reference(recon: np.ndarray, ref: np.ndarray) -> Dict[str, Any]:
    """
    Pick the orientation of ``ref`` that best matches ``recon``.

    Scores every orientation by the Pearson correlation of the magnitude images
    (reference resized to the recon's shape), keeps ``"none"`` unless another
    orientation wins by :data:`_AUTO_ORIENT_MARGIN`. ``ambiguous`` is True when
    the pick is not "none" and the runner-up is within that margin (typically a
    mirror pair on a near-symmetric head) — the user should check it. Returns
    ``{"orientation": key, "scores": {key: corr}, "ambiguous": bool}``.
    """
    rec = np.abs(np.squeeze(recon)).astype(float).ravel()
    rec = (rec - rec.mean()) / (rec.std() + 1e-12)
    ref = np.abs(np.squeeze(ref)).astype(float)

    scores: Dict[str, float] = {}
    for key in ORIENTATIONS:
        o = _resize_to(orient_image(ref, key), recon.shape).ravel()
        o = (o - o.mean()) / (o.std() + 1e-12)
        scores[key] = float(np.mean(rec * o))

    ranked = sorted(scores, key=scores.get, reverse=True)
    best = ranked[0]
    if scores[best] - scores["none"] < _AUTO_ORIENT_MARGIN:
        best = "none"
    ambiguous = (best != "none"
                 and scores[best] - scores[ranked[1]] < _AUTO_ORIENT_MARGIN)
    return {"orientation": best, "scores": scores, "ambiguous": ambiguous}


def compare_to_reference(
    recon_image: np.ndarray,
    reference_image: np.ndarray,
    *,
    recon_name: str,
    reference_name: str,
    orientation: str = "auto",
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Compare a recon image to a fully-sampled reference.

    Orients the reference to the recon (``orientation="auto"`` picks the best of
    the 8 rotations/flips by correlation; any :data:`ORIENTATIONS` key forces
    one), resizes it to the recon's shape, warns on a sequence/matrix mismatch,
    and computes RMSE / NMSE / PSNR / SSIM on a shared 95th-percentile scale.
    Returns the (oriented, resized) ``reference_image``, the untouched
    ``reference_image_raw`` plus ``recon_name`` / ``reference_name`` (so a viewer
    can re-run this without re-reading), the applied ``reference_orientation`` /
    ``reference_orientation_auto``, the metrics and an optional
    ``reference_warning``.
    """
    log = log or _noop

    from .imaging import metrics_vs_reference

    recon = np.abs(np.squeeze(recon_image)).astype(float)
    ref_raw = np.abs(np.squeeze(reference_image)).astype(np.float32)

    is_auto = orientation == "auto"
    ambiguous = False
    if is_auto:
        _ao = auto_orient_reference(recon, ref_raw)
        orientation = _ao["orientation"]
        ambiguous = _ao["ambiguous"]
        _sc = "  ".join(f"{k}={v:+.3f}" for k, v in _ao["scores"].items())
        log(f"  Reference orientation scores (corr): {_sc}")
        if orientation != "none":
            log(f"  Reference auto-oriented to the recon: {orientation} "
                f"(readout/PE axes likely differ between the two scans).")
        if ambiguous:
            log("  [WARN] Reference orientation is a close call (mirror "
                "ambiguity) — check it visually and override in the viewer.")
    elif orientation != "none":
        log(f"  Reference orientation forced: {orientation}")
    ref = orient_image(ref_raw, orientation).astype(float)
    ref_orig_shape = ref.shape

    recon_seq, ref_seq = _seq_kind(recon_name), _seq_kind(reference_name)
    seq_mismatch = bool(recon_seq and ref_seq and recon_seq != ref_seq)
    shape_mismatch = ref_orig_shape != recon.shape
    warning = None
    if seq_mismatch or shape_mismatch:
        parts = []
        if seq_mismatch:
            parts.append(f"sequence mismatch (recon={recon_seq} vs reference={ref_seq})")
        if shape_mismatch:
            parts.append(
                f"matrix mismatch (recon={recon.shape[0]}×{recon.shape[1]} "
                f"vs reference={ref_orig_shape[0]}×{ref_orig_shape[1]})"
            )
        warning = (
            "Reference does NOT match the reconstructed scan: " + "; ".join(parts)
            + ". The reference was resized to compare, so SSIM/PSNR reflect "
            "sequence/contrast/FOV differences — not just reconstruction error. "
            "Use a fully-sampled scan of the SAME sequence for a valid comparison."
        )
        log(f"  WARNING: {warning}")

    ref = _resize_to(ref, recon.shape)

    # Match intensity scales: a coil-combined (BART PICS / SENSE) recon and the
    # reference can sit on different arbitrary scales, so rescale the recon so its
    # 95th percentile matches the reference before scoring — otherwise the
    # shared-scale metrics are meaningless.
    _r95 = float(np.percentile(recon.ravel(), 95)) + 1e-12
    _f95 = float(np.percentile(ref.ravel(), 95)) + 1e-12
    recon = recon * (_f95 / _r95)

    m = metrics_vs_reference(recon, ref)
    _ssim_txt = f"{m['ssim']:.4f}" if m["ssim"] is not None else "n/a"
    log(f"  RMSE={m['rmse']:.4f}  NMSE={m['nmse']:.4f}  "
        f"PSNR={m['psnr']:.2f} dB  SSIM={_ssim_txt}")
    return {
        "reference_image":   ref,
        "reference_image_raw":        ref_raw,
        "reference_orientation":      orientation,
        "reference_orientation_auto": is_auto,
        "reference_orientation_ambiguous": ambiguous,
        "recon_name":        recon_name,
        "reference_name":    reference_name,
        "ssim":              m["ssim"],
        "nmse":              m["nmse"],
        "psnr":              m["psnr"],
        "rmse":              m["rmse"],
        "reference_warning": warning,
    }
