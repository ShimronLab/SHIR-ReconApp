"""
PDF report generator using matplotlib PdfPages.
One PDF per scan folder; one page per reconstruction.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from processing.constants import method_display_label as _method_display_label
from processing.constants import (
    SCAN_BAND,
    SCAN_EQUISPACED,
    SCAN_FULLY_SAMPLED,
    SCAN_POISSON,
    SCAN_RETRO_UNDERSAMPLE,
    SCAN_UNKNOWN,
)


def _norm(img: np.ndarray) -> np.ndarray:
    """Normalise array to [0, 1] for display."""
    arr = np.abs(np.squeeze(img)).astype(float)
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)


# Per-page display rotation (number of 90° turns), set from result["_rotation"]
# so the PDF matches the on-screen orientation the user chose in the viewer.
# The viewer applies the same np.rot90 to the final displayed arrays.
_PAGE_ROT = 0


def _set_page_rotation(result: Dict[str, Any]) -> None:
    global _PAGE_ROT
    try:
        _PAGE_ROT = int(result.get("_rotation", 0)) % 4
    except Exception:
        _PAGE_ROT = 0


def _rot(a):
    """Apply the current page rotation to a 2-D image array (no-op if 0)."""
    if _PAGE_ROT and a is not None and np.asarray(a).ndim >= 2:
        return np.rot90(np.asarray(a), _PAGE_ROT)
    return a


def _imshow_adjusted(
    ax,
    data: np.ndarray,
    cmap: str,
    wl_min: int,
    wl_max: int,
    brightness: float,
    is_main: bool = False,
    **kwargs,
) -> None:
    """imshow with percentile W/L and optional brightness scaling.

    Brightness is applied only to *main* reconstructed images (not k-space,
    sensitivity maps, or the GRAPPA difference image).
    """
    data = _rot(data)                      # match the viewer's chosen rotation
    vmin = np.percentile(data, wl_min)
    vmax = np.percentile(data, wl_max)
    if is_main and brightness > 0:
        vmax = vmax / brightness
    vmax = max(float(vmax), float(vmin) + 1e-9)
    # ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax, **kwargs)
    ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax, rasterized=True, **kwargs)


def _add_all_coil_sens_page(
    pdf: PdfPages,
    sens_maps: np.ndarray,
    title: str,
    wl_min: int,
    wl_max: int,
    brightness: float,
    orient=None,
) -> None:
    """Add a separate page showing the ESPIRiT sensitivity maps for *all* coils.

    ``orient`` is an optional callable applied to each 2-D coil map before
    display, so the grid matches the orientation the rest of the page uses
    (transpose / rot90). Defaults to ``np.abs`` (no reorientation).
    """
    if sens_maps is None:
        return
    arr = np.squeeze(sens_maps)
    if arr.ndim < 3:
        return
    if orient is None:
        orient = lambda a: np.abs(a)

    nc_coils = arr.shape[-1]
    ncols_g  = int(np.ceil(np.sqrt(nc_coils)))
    nrows_g  = int(np.ceil(nc_coils / ncols_g))
    fig, axes = plt.subplots(nrows_g, ncols_g, figsize=(3 * ncols_g, 3 * nrows_g))
    fig.suptitle(f"{title}\n— ESPIRiT Sensitivity Maps (all coils)", fontsize=10)
    axes = np.array(axes).ravel()
    for c in range(nc_coils):
        _imshow_adjusted(axes[c], orient(arr[..., c]), "viridis",
                         wl_min, wl_max, brightness, is_main=False)
        axes[c].set_title(f"Coil {c}", fontsize=7)
        axes[c].axis("off")
    for c in range(nc_coils, len(axes)):
        axes[c].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=80)
    plt.close(fig)


def _add_recon_page(pdf: PdfPages, result: Dict[str, Any], title: str) -> None:
    """Add one figure page for a reconstruction result."""
    _set_page_rotation(result)   # applies to every image drawn for this page
    stype = result.get("scan_type", "Unknown")

    if stype == SCAN_BAND or stype == SCAN_POISSON:
        _page_nufft(pdf, result, title)
    elif stype == SCAN_FULLY_SAMPLED:
        _page_fullysamp(pdf, result, title)
    elif stype == SCAN_EQUISPACED:
        _page_equispaced(pdf, result, title)
    elif stype == SCAN_RETRO_UNDERSAMPLE:
        _page_retro(pdf, result, title)
    elif result.get("combined_img") is not None:
        _page_nufft(pdf, result, title)          # generic NUFFT-RSS / CS result
    elif result.get("image_rss") is not None:
        _page_fullysamp(pdf, result, title)      # generic Cartesian-RSS result
    else:
        _page_generic(pdf, result, title)


def _page_generic(pdf: PdfPages, result: Dict[str, Any], title: str) -> None:
    """Fallback page for a result with no recognised image keys."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.5, 0.5, f"{title}\n\n(no displayable image in this result)",
            ha="center", va="center", fontsize=12)
    pdf.savefig(fig)
    plt.close(fig)


def _page_nufft(pdf: PdfPages, result: Dict[str, Any], title: str) -> None:
    """Page for NUFFT-based reconstructions (band / Poisson)."""
    combined = result.get("combined_img")
    kspace_c = result.get("kspace_combined")
    sens     = result.get("sens_maps")
    coil_imgs = result.get("imgs_per_coil")
    traj     = result.get("traj_scaled")

    wl_min     = result.get("_wl_min", 0)
    wl_max     = result.get("_wl_max", 100)
    brightness = result.get("_brightness", 1.0)

    ncols  = 3
    nrows  = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # (0,0) combined image
    if combined is not None:
        _imshow_adjusted(axes[0, 0], _norm(combined), "gray", wl_min, wl_max, brightness, is_main=True)
    axes[0, 0].set_title("Combined Image")
    axes[0, 0].axis("off")

    # (0,1) k-space of combined image
    if kspace_c is not None:
        kd = np.log(np.abs(np.squeeze(kspace_c)) + 1e-9)
        _imshow_adjusted(axes[0, 1], _norm(kd), "gray", wl_min, wl_max, brightness, is_main=False)
        axes[0, 1].set_title("K-space (log)")
        axes[0, 1].axis("off")

    # (0,2) trajectory scatter
    if traj is not None:
        kx = traj[:, 0] if traj.ndim == 2 else traj[0]
        ky = traj[:, 1] if traj.ndim == 2 else traj[1]
        # axes[0, 2].scatter(kx, -ky, s=0.5, alpha=0.4, c="royalblue") *** flatten
        axes[0, 2].scatter(kx, -ky, s=0.5, alpha=0.4, c="royalblue", rasterized=True)
        axes[0, 2].set_title("Trajectory")
        axes[0, 2].set_xlabel("kx"); axes[0, 2].set_ylabel("ky  (display: −ky)")
        axes[0, 2].set_aspect("equal")
    else:
        axes[0, 2].axis("off")

    # (1, 0-2) first 3 coil images
    if coil_imgs is not None:
        arr = np.squeeze(coil_imgs)
        n   = min(arr.shape[-1], 3) if arr.ndim == 3 else 1
        for c in range(n):
            img = arr[:, :, c] if arr.ndim == 3 else arr
            _imshow_adjusted(axes[1, c], np.rot90(_norm(img), k=3), "gray", wl_min, wl_max, brightness, is_main=False)
            axes[1, c].set_title(f"Coil {c} (adjoint NUFFT)")
            axes[1, c].axis("off")
        for c in range(n, 3):
            axes[1, c].axis("off")
    elif sens is not None:
        arr = np.squeeze(sens)
        n   = min(arr.shape[-1], 3) if arr.ndim == 3 else 1
        for c in range(n):
            img = arr[:, :, c] if arr.ndim == 3 else arr
            _imshow_adjusted(axes[1, c], _norm(img), "viridis", wl_min, wl_max, brightness, is_main=False)
            axes[1, c].set_title(f"Sensitivity Map — Coil {c}")
            axes[1, c].axis("off")
        for c in range(n, 3):
            axes[1, c].axis("off")
    else:
        for c in range(3):
            axes[1, c].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=100)
    plt.close(fig)

    # Separate page: all coil images grid
    if coil_imgs is not None:
        arr  = np.squeeze(coil_imgs)
        if arr.ndim == 3:
            nc_coils = arr.shape[-1]
            ncols_g  = int(np.ceil(np.sqrt(nc_coils)))
            nrows_g  = int(np.ceil(nc_coils / ncols_g))
            fig2, axes2 = plt.subplots(nrows_g, ncols_g, figsize=(3 * ncols_g, 3 * nrows_g))
            fig2.suptitle(f"{title}\n— Per-coil adjoint NUFFT images", fontsize=10)
            axes2 = np.array(axes2).ravel()
            for c in range(nc_coils):
                _imshow_adjusted(axes2[c], np.rot90(_norm(arr[:, :, c]),  k=3), "gray",
                                 wl_min, wl_max, brightness, is_main=False)
                axes2[c].set_title(f"Coil {c}", fontsize=7)
                axes2[c].axis("off")
            for c in range(nc_coils, len(axes2)):
                axes2[c].axis("off")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig2, dpi=80)
            plt.close(fig2)

    # Separate page: all ESPIRiT sensitivity maps (matches the combined-image
    # orientation used above, i.e. _norm with no transpose).
    _add_all_coil_sens_page(pdf, sens, title, wl_min, wl_max, brightness,
                            orient=_norm)


def _page_fullysamp(pdf: PdfPages, result: Dict[str, Any], title: str) -> None:
    """Page for fully-sampled reconstruction."""
    image_rss      = result.get("image_rss")
    image_combined = result.get("image_combined")
    ksp_rss        = result.get("kspace_rss")   # pre-computed 2-D float32 RSS magnitude
    sens_maps      = result.get("sens_maps")
    combination    = result.get("combination", "rss")

    wl_min     = result.get("_wl_min", 0)
    wl_max     = result.get("_wl_max", 100)
    brightness = result.get("_brightness", 1.0)

    has_sense = (combination == "sense" and image_combined is not None
                 and sens_maps is not None)

    if has_sense:
        fig, axes_all = plt.subplots(2, 3, figsize=(15, 10))
        axes_row0 = axes_all[0]
        axes_row1 = axes_all[1]
    else:
        fig, axes_row0 = plt.subplots(1, 3, figsize=(15, 5))
        axes_row1 = None

    fig.suptitle(title, fontsize=12, fontweight="bold")

    # ── Row 0: main images ────────────────────────────────────────────────────
    if has_sense and image_combined is not None:
        _imshow_adjusted(axes_row0[0], np.rot90(_norm(image_combined)), "gray",
                         wl_min, wl_max, brightness, is_main=True)
        axes_row0[0].set_title("SENSE Combined (ESPIRiT)")
    elif image_rss is not None:
        _imshow_adjusted(axes_row0[0], np.rot90(_norm(image_rss)), "gray",
                         wl_min, wl_max, brightness, is_main=True)
        axes_row0[0].set_title("RSS Image")
    axes_row0[0].axis("off")

    if has_sense and image_rss is not None:
        _imshow_adjusted(axes_row0[1], np.rot90(_norm(image_rss)), "gray",
                         wl_min, wl_max, brightness, is_main=True)
        axes_row0[1].set_title("RSS Image (reference)")
        axes_row0[1].axis("off")
    else:
        axes_row0[1].axis("off")

    if ksp_rss is not None:
        kd = np.rot90(np.log(ksp_rss + 1e-9))
        _imshow_adjusted(axes_row0[2], _norm(kd), "gray",
                         wl_min, wl_max, brightness, is_main=False)
        axes_row0[2].set_title("K-space (log RSS)")
        axes_row0[2].axis("off")
    else:
        axes_row0[2].axis("off")

    # ── Row 1: first 3 sensitivity maps (SENSE only) ──────────────────────────
    if has_sense and axes_row1 is not None:
        arr = np.squeeze(sens_maps)
        n = min(arr.shape[-1], 3) if arr.ndim == 3 else 1
        for c in range(n):
            img_s = np.rot90(_norm(arr[:, :, c] if arr.ndim == 3 else arr))
            _imshow_adjusted(axes_row1[c], img_s, "viridis",
                             wl_min, wl_max, brightness, is_main=False)
            axes_row1[c].set_title(f"Sensitivity Map — Coil {c}")
            axes_row1[c].axis("off")
        for c in range(n, 3):
            axes_row1[c].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=100)
    plt.close(fig)

    # Separate page: all sensitivity maps (SENSE only), matching the row-1
    # orientation above (rot90 of the normalised map).
    if has_sense:
        _add_all_coil_sens_page(pdf, sens_maps, title, wl_min, wl_max, brightness,
                                orient=lambda a: np.rot90(_norm(a)))


def _page_equispaced(pdf: PdfPages, result: Dict[str, Any], title: str) -> None:
    """Page for equispaced under-sampled reconstruction (GRAPPA / SENSE / CS-SENSE)."""
    method      = result.get("method", "grappa")
    grappa_img  = result.get("grappa_image")
    zf_img      = result.get("zf_image")
    ref_img     = result.get("reference_image")
    grappa_ksp  = result.get("grappa_kspace_rss")        # pre-computed 2-D RSS
    under_ksp   = result.get("undersampled_kspace_rss")  # pre-computed 2-D RSS
    sens_maps   = result.get("sens_maps")
    rmse        = result.get("rmse")
    psnr        = result.get("psnr")
    ssim_val    = result.get("ssim")
    actual_r    = result.get("actual_r", 4)

    wl_min     = result.get("_wl_min", 0)
    wl_max     = result.get("_wl_max", 100)
    brightness = result.get("_brightness", 1.0)

    method_label = _method_display_label(result, method)
    ksp_title    = f"{method_label} K-space (log RSS)" if method in ("grappa", "grappa_sense") else "Recon K-space (log RSS)"

    has_ref  = ref_img is not None
    has_sens = sens_maps is not None

    def norm01(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-12)

    # Shared normalisation: divide BOTH recon and reference by the reference's
    # 95th percentile (see the other report page) so the difference map reflects
    # true intensity error. Min-max only when there is no reference.
    if has_ref:
        _ref_p95 = float(np.percentile(np.abs(ref_img).ravel(), 95)) + 1e-12
        def norm_img(x):
            return x / _ref_p95
    else:
        norm_img = norm01

    nrows, ncols = (3 if has_sens else 2, 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 12 if not has_sens else 16))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    if grappa_img is not None:
        g = norm_img(grappa_img)
        rm = f"  RMSE={rmse:.4f}  PSNR={psnr:.1f}dB" if rmse else ""
        if rmse and ssim_val is not None:
            rm += f"  SSIM={ssim_val:.4f}"
        _imshow_adjusted(axes[0, 0], g.T, "gray", wl_min, wl_max, brightness, is_main=True)
        axes[0, 0].set_title(f"{method_label} (R={actual_r:.3f}×){rm}")

    if has_ref:
        r = norm_img(ref_img)
        _imshow_adjusted(axes[0, 1], r.T, "gray", wl_min, wl_max, brightness, is_main=True)
        axes[0, 1].set_title("Fully-Sampled Reference")

        abs_diff = np.abs(g - r).T if grappa_img is not None else None
        if abs_diff is not None:
            _imshow_adjusted(axes[0, 2], abs_diff, "gray", wl_min, wl_max, brightness, is_main=False)
            axes[0, 2].set_title("Absolute Difference")
    else:
        if zf_img is not None:
            _imshow_adjusted(axes[0, 1], norm01(zf_img).T, "gray", wl_min, wl_max, brightness, is_main=True)
            axes[0, 1].set_title("Zero-Fill Reconstruction")
        axes[0, 2].axis("off")

    if under_ksp is not None:
        _imshow_adjusted(axes[1, 0], np.log(under_ksp.T + 1e-9), "gray",
                         wl_min, wl_max, brightness, is_main=False)
        axes[1, 0].set_title("Undersampled K-space (log RSS)")

    if grappa_ksp is not None:
        _imshow_adjusted(axes[1, 1], np.log(grappa_ksp.T + 1e-9), "gray",
                         wl_min, wl_max, brightness, is_main=False)
        axes[1, 1].set_title(ksp_title)

    axes[1, 2].axis("off")
    params = result.get("params", {})
    kr = params.get("kernel_size", (None, None))
    grappa_lamda  = params.get("lamda")
    sense_lambda  = params.get("sense_lambda")
    sense_iters   = params.get("sense_max_iter")
    acs   = params.get("acs_lines")
    r_fac = params.get("r_factor")

    info_lines = ["Parameters\n"]
    info_lines.append(f"Method     : {method_label}")
    if r_fac is not None:
        info_lines.append(f"R-factor   : {r_fac}")
    if method in ("grappa", "grappa_sense"):
        if acs is not None:
            info_lines.append(f"ACS lines  : {acs}")
        if kr[0] is not None:
            info_lines.append(f"Kernel size: {kr[0]} × {kr[1]}")
        if grappa_lamda is not None:
            info_lines.append(f"λ GRAPPA   : {grappa_lamda}")
    elif method in ("sense", "sense_cs", "espirit", "espirit_l1",
                    "espirit_l2", "espirit_tv", "espirit_l1wav"):
        if sense_iters is not None:
            info_lines.append(f"SENSE iters: {sense_iters}")
        if method in ("sense_cs", "espirit_l1", "espirit_l2", "espirit_tv", "espirit_l1wav"):
            if sense_lambda is not None:
                info_lines.append(f"λ          : {sense_lambda}")
        if method in ("espirit", "espirit_l1", "espirit_l2", "espirit_tv", "espirit_l1wav"):
            n_maps = params.get("espirit_num_maps")
            if n_maps is not None:
                info_lines.append(f"Map sets   : {n_maps}")
    if rmse is not None:
        info_lines.append(f"\nMetrics\n")
        info_lines.append(f"RMSE : {rmse:.4f}")
        info_lines.append(f"PSNR : {psnr:.2f} dB")
        if ssim_val is not None:
            info_lines.append(f"SSIM : {ssim_val:.4f}")

    axes[1, 2].text(
        0.1, 0.6,
        "\n".join(info_lines),
        transform=axes[1, 2].transAxes,
        fontsize=13, va="top",
    )

    # ── Row 2: sensitivity maps (first 3 coils) ──────────────────────────
    if has_sens:
        arr = np.squeeze(sens_maps)
        n   = min(arr.shape[-1], 3) if arr.ndim >= 3 else 1
        for c in range(n):
            img_s = arr[..., c] if arr.ndim >= 3 else arr
            _imshow_adjusted(axes[2, c], np.abs(img_s).T, "viridis",
                             wl_min, wl_max, brightness, is_main=False)
            axes[2, c].set_title(f"Sensitivity Map — Coil {c}")
            axes[2, c].axis("off")
        for c in range(n, 3):
            axes[2, c].axis("off")

    for ax in axes.ravel():
        ax.set_xlabel(""); ax.set_ylabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=100)
    plt.close(fig)

    # ── Separate page: all-coil sensitivity maps grid ─────────────────────
    if has_sens:
        _add_all_coil_sens_page(pdf, sens_maps, title, wl_min, wl_max, brightness,
                                orient=lambda a: np.abs(a).T)


def _page_retro(pdf: PdfPages, result: Dict[str, Any], title: str) -> None:
    """Page for retrospective undersampling simulator results."""
    pe_mask      = result.get("pe_mask")
    fs_img       = result.get("fully_sampled_image")
    zf_img       = result.get("zf_image")
    recon_img    = result.get("recon_image")
    diff_img     = result.get("diff_image")
    fs_ksp       = result.get("fs_kspace_rss")
    under_ksp    = result.get("undersampled_kspace_rss")
    recon_ksp    = result.get("recon_kspace_rss")
    actual_r     = result.get("actual_r", 0.0)
    mask_type    = result.get("mask_type", "equispaced")
    swap_pe_read = result.get("swap_pe_read", False)
    acs_start    = result.get("acs_start")
    acs_end      = result.get("acs_end")
    rmse         = result.get("rmse")
    psnr         = result.get("psnr")
    ssim_val     = result.get("ssim")
    method       = result.get("method", "none")
    params_dict  = result.get("params", {})

    axis_label  = "RO (L-R)" if swap_pe_read else "PE (A-P)"
    axis_ylabel = "RO index" if swap_pe_read else "PE index"

    wl_min     = result.get("_wl_min",    0)
    wl_max     = result.get("_wl_max",    100)
    brightness = result.get("_brightness", 1.0)

    method_label = _method_display_label(result, method)

    def norm01(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-12)

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Row 0: Fully-Sampled | Zero-Filled | PE Mask (2-D image)
    if fs_img is not None:
        _imshow_adjusted(axes[0, 0], np.rot90(norm01(fs_img)), "gray",
                         wl_min, wl_max, brightness, is_main=True)
        axes[0, 0].set_title("Fully-Sampled (Reference)")
    axes[0, 0].axis("off")

    if zf_img is not None:
        _imshow_adjusted(axes[0, 1], np.rot90(norm01(zf_img)), "gray",
                         wl_min, wl_max, brightness, is_main=True)
        axes[0, 1].set_title("Zero-Filled")
    axes[0, 1].axis("off")

    if pe_mask is not None:
        n = len(pe_mask)
        width = max(1, n // 4)
        mask_2d = np.tile(pe_mask.reshape(-1, 1).astype(float), (1, width))
        axes[0, 2].imshow(_rot(mask_2d), cmap="gray", aspect="auto", origin="upper",
                          vmin=0, vmax=1)
        if acs_start is not None and acs_end is not None:
            axes[0, 2].axhspan(acs_start - 0.5, acs_end - 0.5, alpha=0.35,
                               color="orange", label="ACS")
            axes[0, 2].legend(fontsize=7)
        axes[0, 2].set_title(
            f"{axis_label} Mask — {mask_type.upper()}  R={actual_r:.2f}×  ({pe_mask.sum()}/{n})"
        )
        axes[0, 2].set_ylabel(axis_ylabel)
    else:
        axes[0, 2].axis("off")

    # Row 1: Reconstructed | Difference | Metrics
    if recon_img is not None and method != "none":
        rm = f"  RMSE={rmse:.4f}  PSNR={psnr:.1f}dB" if rmse is not None else ""
        if rmse is not None and ssim_val is not None:
            rm += f"  SSIM={ssim_val:.4f}"
        _imshow_adjusted(axes[1, 0], np.rot90(norm01(recon_img)), "gray",
                         wl_min, wl_max, brightness, is_main=True)
        axes[1, 0].set_title(f"{method_label} (R={actual_r:.3f}×){rm}")
    axes[1, 0].axis("off")

    if diff_img is not None and method != "none":
        axes[1, 1].imshow(_rot(np.rot90(diff_img)), cmap="gray", aspect="auto", rasterized=True)
        axes[1, 1].set_title("Absolute Difference")
    axes[1, 1].axis("off")

    axes[1, 2].axis("off")
    info_lines = ["Simulation Parameters\n"]
    info_lines.append(f"Mask type  : {mask_type.upper()}")
    info_lines.append(f"Undersamp. : {axis_label}")
    info_lines.append(f"Target R   : {params_dict.get('R', '?')}")
    info_lines.append(f"Actual R   : {actual_r:.3f}×")
    info_lines.append(f"ACS lines  : {params_dict.get('acs_lines', '?')}")
    sampled = pe_mask.sum() if pe_mask is not None else "?"
    total   = len(pe_mask)  if pe_mask is not None else "?"
    info_lines.append(f"Sampled    : {sampled}/{total}")
    if mask_type == "vd":
        info_lines.append(f"Power decay: {params_dict.get('power_decay', '?')}")
    info_lines.append(f"Recon      : {method_label}")
    if method == "grappa":
        ks = params_dict.get('kernel_size', '?')
        ks_str = f"{ks[0]}×{ks[1]}" if isinstance(ks, (list, tuple)) else str(ks)
        info_lines.append(f"  Kernel    : {ks_str}")
        info_lines.append(f"  λ GRAPPA  : {params_dict.get('grappa_lambda', '?')}")
    elif method in ("sense", "sense_cs", "espirit", "espirit_l1",
                    "espirit_l2", "espirit_tv", "espirit_l1wav"):
        info_lines.append(f"  Iterations: {params_dict.get('sense_max_iter', '?')}")
        if method in ("sense_cs", "espirit_l1", "espirit_l2", "espirit_tv", "espirit_l1wav"):
            info_lines.append(f"  λ         : {params_dict.get('sense_lambda', '?')}")
        if method in ("espirit", "espirit_l1", "espirit_l2", "espirit_tv", "espirit_l1wav"):
            info_lines.append(f"  Map sets  : {params_dict.get('espirit_num_maps', '?')}")
    if rmse is not None:
        info_lines.append(f"\nMetrics\n")
        info_lines.append(f"RMSE : {rmse:.4f}")
        info_lines.append(f"PSNR : {psnr:.2f} dB")
        if ssim_val is not None:
            info_lines.append(f"SSIM : {ssim_val:.4f}")
    axes[1, 2].text(
        0.05, 0.9,
        "\n".join(info_lines),
        transform=axes[1, 2].transAxes,
        fontsize=12, va="top",
    )

    # Row 2: FS k-space | Undersampled k-space | Reconstructed k-space
    if fs_ksp is not None:
        axes[2, 0].imshow(_rot(np.rot90(np.log(fs_ksp + 1e-9))), cmap="gray",
                          aspect="auto", rasterized=True)
        axes[2, 0].set_title("FS K-space (log)")
    else:
        axes[2, 0].axis("off")

    if under_ksp is not None:
        axes[2, 1].imshow(_rot(np.rot90(np.log(under_ksp + 1e-9))), cmap="gray",
                          aspect="auto", rasterized=True)
        axes[2, 1].set_title("Undersampled K-space (log)")
    else:
        axes[2, 1].axis("off")

    if recon_ksp is not None:
        axes[2, 2].imshow(_rot(np.rot90(np.log(recon_ksp + 1e-9))), cmap="gray",
                          aspect="auto", rasterized=True)
        axes[2, 2].set_title(f"{method_label} K-space (log)")
    else:
        axes[2, 2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=100)
    plt.close(fig)

    # Separate page: all ESPIRiT sensitivity maps (SENSE/ESPIRiT recons only;
    # None for ZF-only and GRAPPA). Match the main-image orientation (rot90).
    _add_all_coil_sens_page(pdf, result.get("sens_maps"), title,
                            wl_min, wl_max, brightness,
                            orient=lambda a: np.rot90(np.abs(a)))


def generate_pdf(
    results: List[Dict[str, Any]],
    output_path: str,
    folder_name: str = "",
    params_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Generate a multi-page PDF report from a list of reconstruction results.

    Parameters
    ----------
    results : list of dicts
        Each dict must have a 'scan_type' key (str) and 'label' key (str),
        plus the reconstruction-specific keys.
    output_path : str
        Destination .pdf path.
    folder_name : str
        Displayed on the cover page.
    params_summary : dict, optional
        Parameter snapshot printed on the cover page.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        # ── Cover page ────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            "MRI Reconstruction Analysis Report",
            "",
            f"Folder : {folder_name}",
            f"Date   : {now}",
            f"Scans  : {len(results)}",
        ]
        if params_summary:
            lines.append("")
            lines.append("Parameters:")
            for k, v in params_summary.items():
                lines.append(f"  {k} : {v}")

        ax.text(0.5, 0.55, "\n".join(lines), transform=ax.transAxes,
                fontsize=14, va="center", ha="center",
                bbox=dict(boxstyle="round", facecolor="#eef3fa", alpha=0.8))
        plt.tight_layout()
        pdf.savefig(fig, dpi=100)
        plt.close(fig)

        # ── One page (or more) per result ─────────────────────────────────────
        for res in results:
            label = res.get("label", "Unknown")
            _add_recon_page(pdf, res, label)

        # ── Metadata page ─────────────────────────────────────────────────────
        d = pdf.infodict()
        d["Title"]   = f"MRI Report - {folder_name}"
        d["Author"]  = "SHIR — System for Hybrid Image Reconstruction"
        d["Subject"] = "MRI Reconstruction Results"
        d["CreationDate"] = datetime.now()
