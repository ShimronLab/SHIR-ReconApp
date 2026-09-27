"""
Shared fully-sampled reference comparison for undersampled reconstructions.

Both the Equispaced and the Poisson-disc recons compare their result against a
fully-sampled reference (image + absolute difference + SSIM/NMSE/PSNR). The
metric/resize/mismatch-warning logic lives here so both use identical behavior.

- ``load_reference_rss``   : read a reference .dat → RSS magnitude image.
- ``compare_to_reference`` : resize the reference to the recon, flag a
  sequence/matrix mismatch, and compute 95th-percentile-normalised metrics.
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
    log: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """
    Read a fully-sampled reference TWIX file and return its RSS magnitude image.

    Multi-slice (4-D) references pick ``slice_idx`` (or the middle slice).
    Readout oversampling is cropped (best-effort — skipped, not fatal, on an
    exotic factor). A fully-sampled TSE reference is echo-un-scrambled so the
    comparison target is not scrambled.
    """
    log = log or _noop
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


def compare_to_reference(
    recon_image: np.ndarray,
    reference_image: np.ndarray,
    *,
    recon_name: str,
    reference_name: str,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Compare a recon image to a fully-sampled reference.

    Resizes the reference to the recon's shape, warns on a sequence/matrix
    mismatch, and computes RMSE / NMSE / PSNR / SSIM on a shared 95th-percentile
    scale. Returns the (resized) ``reference_image`` plus the metrics and an
    optional ``reference_warning``.
    """
    log = log or _noop
    from scipy.ndimage import zoom

    from .imaging import metrics_vs_reference

    recon = np.abs(np.squeeze(recon_image)).astype(float)
    ref = np.abs(np.squeeze(reference_image)).astype(float)
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

    if ref.shape != recon.shape:
        zf = (recon.shape[0] / ref.shape[0], recon.shape[1] / ref.shape[1])
        ref = zoom(ref, zf, order=1)

    m = metrics_vs_reference(recon, ref)
    _ssim_txt = f"{m['ssim']:.4f}" if m["ssim"] is not None else "n/a"
    log(f"  RMSE={m['rmse']:.4f}  NMSE={m['nmse']:.4f}  "
        f"PSNR={m['psnr']:.2f} dB  SSIM={_ssim_txt}")
    return {
        "reference_image":   ref,
        "ssim":              m["ssim"],
        "nmse":              m["nmse"],
        "psnr":              m["psnr"],
        "rmse":              m["rmse"],
        "reference_warning": warning,
    }
