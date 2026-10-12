"""
Batch reconstruction → PDF report.

Reads a JSON file that lists a base directory and a set of TWIX (.dat) scans,
reconstructs every scan and writes a single PDF report.

Two scan categories are handled automatically (by filename, see processing.utils):

  * FULLY_SAMPLED  → reconstructed once with ESPIRiT (SENSE coil combination).
                     Used as the reference for the matching undersampled scans.
  * EQUISPACED     → reconstructed with every method offered by the app:
                       GRAPPA, GRAPPA+SENSE, Pure SENSE,
                       ESPIRiT, ESPIRiT+L1, ESPIRiT+L2, ESPIRiT+TV, ESPIRiT+L1-Wav

Two acquisition structures are handled:

  * "My" data    → a single slice (3-D raw array) → reconstructed as usual.
  * Siemens data → multiple slices (4-D raw array) → EVERY slice is reconstructed.

Each reconstruction produces one PDF page showing the reconstructed image, the
fully-sampled reference, the absolute difference and the evaluation metrics
SSIM, NMSE and PSNR.

Usage
-----
    python batch_report.py paths.json
    python batch_report.py paths.json --output report.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend for PDF generation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# ─────────────────────────────────────────────────────────────────────────────
# Make local packages importable regardless of the current working directory
# ─────────────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_APP_DIR = _HERE.parent                       # AnalyzeMRIDataApp/
_ROOT = _APP_DIR.parent                       # repository root
for _p in (str(_APP_DIR), str(_ROOT / "read_siemens_scans")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from processing.constants import METHOD_LABELS                   # noqa: E402
from processing.equispaced_recon import reconstruct_equispaced  # noqa: E402
from processing.imaging import ifft2c as _ifft2c                 # noqa: E402
from processing.imaging import metrics_vs_reference              # noqa: E402
from processing.imaging import norm01 as _norm01                 # noqa: E402
from processing.utils import ScanType, auto_detect_scan_type     # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Methods exposed by the app for undersampled reconstruction
# ─────────────────────────────────────────────────────────────────────────────
UNDERSAMPLED_METHODS: List[str] = [
    "grappa",
    "grappa_sense",
    "sense",
    "espirit",
    "espirit_l1",
    "espirit_l2",
    "espirit_tv",
    "espirit_l1wav",
]


def _log(msg: str) -> None:
    print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# TWIX helpers
# ─────────────────────────────────────────────────────────────────────────────

def _probe_num_slices(dat_path: str) -> int:
    """Return the number of slices in a TWIX file (1 for single-slice 'my' data).

    Siemens headers state the slice count, so only the header is read. Pulseq
    headers carry a placeholder, so for those the data itself is checked."""
    from processing.utils import probe_num_slices, sequence_family
    if sequence_family(dat_path) == "siemens":
        return probe_num_slices(dat_path)

    from read_siemens_scans.twix_reading import read_raw_twix_data
    kspace_raw, _ = read_raw_twix_data(str(dat_path), fix_oversampling=False)
    if kspace_raw.ndim == 4:
        return int(kspace_raw.shape[0])
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(recon: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    """SSIM / NMSE / PSNR / RMSE on the shared reference-p95 scale
    (single definition in processing.imaging, same as the live viewer)."""
    m = metrics_vs_reference(recon, reference)
    return {"ssim": m["ssim"], "nmse": m["nmse"], "psnr": m["psnr"], "rmse": m["rmse"]}


# ─────────────────────────────────────────────────────────────────────────────
# Fully-sampled ESPIRiT reconstruction (per slice, multi-slice aware)
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_fullysamp_espirit(
    dat_path: str,
    *,
    slice_idx: Optional[int] = None,
    calib_width: int = 24,
) -> Dict[str, object]:
    """ESPIRiT (SENSE) reconstruction of a fully-sampled scan.

    Supports single-slice ('my' data) and multi-slice (Siemens) acquisitions.
    Falls back to RSS if BART is unavailable.

    Returns a dict with keys: ``espirit_image``, ``rss_image``, ``kspace_rss``.
    """
    from read_siemens_scans.twix_reading import read_raw_twix_data

    kspace_raw, _ = read_raw_twix_data(str(dat_path), fix_oversampling=False)
    if kspace_raw.ndim == 4:
        n_slices = kspace_raw.shape[0]
        sl = n_slices // 2 if slice_idx is None else int(slice_idx)
        kspace_raw = kspace_raw[sl]

    # (Nacq_PE, Ncoils, Nreadout) → (Nreadout, Nphase, Ncoils)
    kspace_3d = np.transpose(kspace_raw, (2, 0, 1)).astype(np.complex128)

    coil_imgs = _ifft2c(kspace_3d)
    rss_image = np.sqrt(np.sum(np.abs(coil_imgs) ** 2, axis=2)).astype(np.float32)
    kspace_rss = np.sqrt(np.sum(np.abs(kspace_3d) ** 2, axis=2)).astype(np.float32)

    espirit_image = rss_image
    try:
        from processing.utils import BartTempDir, readcfl, run_bart, writecfl

        with BartTempDir("bart_fs_espirit_") as tmp:
            ksp_cfl = os.path.join(tmp, "ksp")
            sens_cfl = os.path.join(tmp, "sens")
            recon_cfl = os.path.join(tmp, "recon")

            k4d = kspace_3d[:, :, np.newaxis, :].astype(np.complex64)
            scale = float(np.percentile(np.abs(rss_image).ravel(), 95)) or 1.0
            writecfl(ksp_cfl, (k4d / max(scale, 1e-12)).astype(np.complex64))

            run_bart("ecalib", f"-r{calib_width}:{calib_width}:1", "-m1", ksp_cfl, sens_cfl)
            run_bart("pics", "-S", ksp_cfl, sens_cfl, recon_cfl)
            espirit_image = np.abs(np.squeeze(readcfl(recon_cfl))).astype(np.float32)
    except Exception as exc:  # noqa: BLE001 - BART optional, RSS fallback
        _log(f"    BART ESPIRiT failed ({exc}); using RSS image.")

    return {
        "espirit_image": espirit_image,
        "rss_image": rss_image,
        "kspace_rss": kspace_rss,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PDF pages
# ─────────────────────────────────────────────────────────────────────────────

def _disp(img: np.ndarray) -> np.ndarray:
    """Orient an image consistently for display."""
    return np.rot90(_norm01(img))


def _imshow(ax, data: np.ndarray, title: str, cmap: str = "gray") -> None:
    vmin = float(np.percentile(data, 1))
    vmax = max(float(np.percentile(data, 99)), vmin + 1e-9)
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _add_recon_page(
    pdf: PdfPages,
    *,
    recon: np.ndarray,
    reference: Optional[np.ndarray],
    metrics: Optional[Dict[str, float]],
    scan_name: str,
    method_label: str,
    slice_label: str,
    extra: str = "",
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    suptitle = f"{method_label}  —  {scan_name}"
    if slice_label:
        suptitle += f"  —  {slice_label}"
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    if reference is not None:
        # Shared normalisation: divide BOTH images by the reference's 95th
        # percentile so the difference map shows true intensity error on one
        # outlier-robust scale (matches compute_metrics and the live viewer).
        scale = float(np.percentile(np.abs(reference).ravel(), 95)) + 1e-12
        g = np.rot90(np.abs(recon) / scale)
        r = np.rot90(np.abs(reference) / scale)
        _imshow(axes[0], g, "Reconstruction")
        _imshow(axes[1], r, "Fully-Sampled Reference")
        # Fixed 0–0.1 window: a 10% intensity error saturates to white.
        axes[2].imshow(np.abs(g - r), cmap="gray", vmin=0.0, vmax=0.1,
                       aspect="auto", rasterized=True)
        axes[2].set_title("Absolute Difference", fontsize=10)
        axes[2].axis("off")
    else:
        _imshow(axes[0], _disp(recon), "Reconstruction")
        axes[1].axis("off")
        axes[2].axis("off")

    if metrics is not None:
        txt = (
            f"SSIM = {metrics['ssim']:.4f}      "
            f"NMSE = {metrics['nmse']:.4f}      "
            f"PSNR = {metrics['psnr']:.2f} dB"
        )
        if extra:
            txt = extra + "        " + txt
        fig.text(0.5, 0.02, txt, ha="center", fontsize=11)
    elif extra:
        fig.text(0.5, 0.02, extra, ha="center", fontsize=11)

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    pdf.savefig(fig, dpi=100)
    plt.close(fig)


def _add_title_page(pdf: PdfPages, json_path: str, base_dir: str, n_scans: int) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.70, "MRI Reconstruction Report", ha="center", fontsize=22, fontweight="bold")
    lines = [
        f"Generated : {datetime.now():%Y-%m-%d %H:%M}",
        f"Input JSON: {Path(json_path).name}",
        f"Base dir  : {base_dir}",
        f"Scans     : {n_scans}",
        "",
        "Undersampled methods: " + ", ".join(METHOD_LABELS[m] for m in UNDERSAMPLED_METHODS),
        "Fully-sampled method: ESPIRiT (SENSE)",
    ]
    fig.text(0.5, 0.50, "\n".join(lines), ha="center", va="top", fontsize=11, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Scan processing
# ─────────────────────────────────────────────────────────────────────────────

def _slice_label(slice_idx: Optional[int], n_slices: int) -> str:
    if n_slices <= 1:
        return ""
    return f"slice {slice_idx + 1}/{n_slices}"


def _process_undersampled(
    pdf: PdfPages,
    dat_path: str,
    reference_path: Optional[str],
    scan_name: str,
) -> None:
    n_slices = _probe_num_slices(dat_path)
    _log(f"  Undersampled scan with {n_slices} slice(s).")

    for sl in range(n_slices):
        slice_idx = sl if n_slices > 1 else None
        slabel = _slice_label(sl, n_slices)
        for method in UNDERSAMPLED_METHODS:
            label = METHOD_LABELS[method]
            _log(f"    [{label}] {slabel or 'single slice'} …")
            try:
                result = reconstruct_equispaced(
                    dat_path,
                    method=method,
                    reference_dat_path=reference_path,
                    progress_cb=lambda m: _log(f"      {m}"),
                    slice_selector_cb=(None if slice_idx is None else (lambda _n, _i=slice_idx: _i)),
                )
            except Exception as exc:  # noqa: BLE001 - keep report going
                _log(f"      ERROR: {label} failed ({exc}); skipping.")
                continue

            recon = result.get("grappa_image")
            reference = result.get("reference_image")
            if recon is None:
                _log(f"      ERROR: no image produced for {label}; skipping.")
                continue

            metrics = compute_metrics(recon, reference) if reference is not None else None
            actual_r = result.get("actual_r")
            extra = f"R = {actual_r:.2f}x" if actual_r else ""
            if result.get("reference_warning"):
                extra = (extra + "   (reference mismatch)").strip()

            _add_recon_page(
                pdf,
                recon=recon,
                reference=reference,
                metrics=metrics,
                scan_name=scan_name,
                method_label=label,
                slice_label=slabel,
                extra=extra,
            )


def _process_generic(pdf: PdfPages, dat_path: str, scan_name: str) -> None:
    """Generic RSS/NUFFT fallback page for scans of unrecognised type."""
    from processing.generic_recon import reconstruct_generic
    from processing.utils import find_generic_traj_json

    json_path = find_generic_traj_json(dat_path)
    _log(f"  Generic fallback (trajectory JSON: "
         f"{Path(json_path).name if json_path else 'none — Cartesian RSS'}).")
    result = reconstruct_generic(
        dat_path,
        json_path=json_path,
        progress_cb=lambda m: _log(f"    {m}"),
    )
    recon = result.get("combined_img")
    if recon is None:
        recon = result.get("image_rss")
    if recon is None:
        _log("  ERROR: generic fallback produced no image; skipping.")
        return
    label = ("Generic NUFFT RSS (zero-fill)"
             if result.get("recon_path") == "nufft" else "Generic RSS")
    _add_recon_page(
        pdf,
        recon=np.abs(np.squeeze(recon)),
        reference=None,
        metrics=None,
        scan_name=scan_name,
        method_label=label,
        slice_label="",
        extra="Unrecognised scan type — reconstructed with the generic fallback.",
    )


def _process_fullysampled(pdf: PdfPages, dat_path: str, scan_name: str) -> None:
    n_slices = _probe_num_slices(dat_path)
    _log(f"  Fully-sampled scan with {n_slices} slice(s) → ESPIRiT.")

    for sl in range(n_slices):
        slice_idx = sl if n_slices > 1 else None
        slabel = _slice_label(sl, n_slices)
        _log(f"    [ESPIRiT] {slabel or 'single slice'} …")
        try:
            res = reconstruct_fullysamp_espirit(dat_path, slice_idx=slice_idx)
        except Exception as exc:  # noqa: BLE001
            _log(f"      ERROR: ESPIRiT failed ({exc}); skipping.")
            continue

        recon = res["espirit_image"]
        reference = res["rss_image"]
        metrics = compute_metrics(recon, reference)
        _add_recon_page(
            pdf,
            recon=recon,
            reference=reference,
            metrics=metrics,
            scan_name=scan_name,
            method_label="ESPIRiT (fully sampled)",
            slice_label=slabel,
            extra="Reference: RSS of the same fully-sampled scan",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Reference matching
# ─────────────────────────────────────────────────────────────────────────────

def _group_key(filename: str) -> str:
    """Group scans so undersampled files are matched to the right reference.

    Siemens scans are prefixed with 'Siemens_'; everything else is 'my' data.
    """
    return "siemens" if Path(filename).name.lower().startswith("siemens") else "mine"


def _build_reference_map(files: List[str], base_dir: Path) -> Dict[str, str]:
    """Map each group key to its fully-sampled reference .dat path (if any)."""
    refs: Dict[str, str] = {}
    for fname in files:
        if auto_detect_scan_type(fname) == ScanType.FULLY_SAMPLED:
            refs[_group_key(fname)] = str(base_dir / fname)
    return refs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(json_path: str, output_pdf: Optional[str] = None) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    base_dir = Path(spec["base_directory"])
    files: List[str] = list(spec["files"])

    if output_pdf is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_pdf = str(_APP_DIR / "results" / f"recon_report_{stamp}.pdf")
    Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)

    ref_map = _build_reference_map(files, base_dir)
    _log(f"References: {ref_map or 'none found'}")

    with PdfPages(output_pdf) as pdf:
        _add_title_page(pdf, json_path, str(base_dir), len(files))

        for fname in files:
            dat_path = base_dir / fname
            if not dat_path.exists():
                _log(f"SKIP (missing): {fname}")
                continue

            stype = auto_detect_scan_type(fname)
            _log(f"\n=== {fname}  [{stype.name}] ===")

            if stype == ScanType.FULLY_SAMPLED:
                _process_fullysampled(pdf, str(dat_path), fname)
            elif stype == ScanType.EQUISPACED:
                reference_path = ref_map.get(_group_key(fname))
                if reference_path is None:
                    _log("  WARNING: no fully-sampled reference for this group; "
                         "metrics will be unavailable.")
                _process_undersampled(pdf, str(dat_path), reference_path, fname)
            else:
                _log(f"  Unrecognised scan type {stype.name} — generic RSS fallback.")
                try:
                    _process_generic(pdf, str(dat_path), fname)
                except Exception as exc:  # noqa: BLE001 - keep report going
                    _log(f"  SKIP: generic fallback failed "
                         f"({type(exc).__name__}: {exc}).")

    _log(f"\nReport written to: {output_pdf}")
    return output_pdf


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Batch reconstruct scans listed in a JSON file into a PDF report.")
    parser.add_argument("json_path", help="Path to the JSON file listing base_directory and files.")
    parser.add_argument("-o", "--output", default=None, help="Output PDF path (default: results/recon_report_<timestamp>.pdf).")
    args = parser.parse_args(argv)

    if not Path(args.json_path).exists():
        parser.error(f"JSON file not found: {args.json_path}")

    generate_report(args.json_path, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
