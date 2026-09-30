"""
Reconstruction backend abstraction: BART (default) with a sigpy fallback.

The app historically performed all ESPIRiT / SENSE / CS / NUFFT reconstruction
through BART (an external Linux/WSL binary). This module lets the same recon
pipelines run through **sigpy** (a pure-Python dependency that is already
bundled) when BART is not available — so the standalone app works with no WSL/
BART install.

Backend selection
-----------------
A process-global mode chooses the engine:
  "auto"  → BART if `bart_available()` else sigpy   (default)
  "bart"  → always BART (errors surface if it is missing)
  "sigpy" → always sigpy

Set it once per job from the UI via `set_backend_mode(...)`.

Dispatching operations
----------------------
Each op has a BART implementation (writecfl → run_bart → readcfl, identical to
the historical inline code) and a sigpy implementation. They consume/return the
same NumPy arrays and layouts the call sites already use:

  espirit_maps(k4d, calib, num_maps)          → sens maps  (Nread,Nphase,1,Ncoils[,Nmaps])
  pics(k4d, sens, **spec)                      → complex recon image (squeezed)
  nufft_adjoint_to_cartesian(bart_traj, ksp,…) → (imgs_per_coil, cart_kspace)

BART numerics are unchanged when BART is the active backend; sigpy results
differ (different solvers/normalisation) but the callers' normalise→rescale
bracketing keeps intensities comparable.
"""

import functools
import os
from typing import Callable, Optional, Tuple

import numpy as np

from .utils import BartTempDir, readcfl, run_bart, writecfl


# ─────────────────────────────────────────────────────────────────────────────
# Backend mode + availability
# ─────────────────────────────────────────────────────────────────────────────

_VALID_MODES = ("auto", "bart", "sigpy")
_backend_mode = "auto"

# ESPIRiT eigenvalue crop threshold for the SigPy path. SigPy's EspiritCalib
# defaults to 0.95 (much harsher than BART's ecalib 0.8), which masks out real
# tissue on lower-SNR data and leaves signal-void holes in the recon. Match BART.
ESPIRIT_CROP = 0.8


def set_backend_mode(mode: str) -> None:
    """Set the process-global recon backend mode ('auto' | 'bart' | 'sigpy')."""
    global _backend_mode
    mode = (mode or "auto").lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown backend mode {mode!r}; expected one of {_VALID_MODES}")
    _backend_mode = mode


def get_backend_mode() -> str:
    """Return the current backend mode."""
    return _backend_mode


@functools.lru_cache(maxsize=1)
def bart_available() -> bool:
    """
    True if BART can be launched (cached for the process lifetime).

    Runs `bart version` once through the normal runner (which bridges into WSL
    on Windows). Any failure — missing binary, missing WSL, non-zero exit — is
    treated as "not available".
    """
    try:
        run_bart("version", check=True)
        return True
    except Exception:
        return False


def use_bart() -> bool:
    """Return True if the BART backend should be used for the current mode."""
    if _backend_mode == "bart":
        return True
    if _backend_mode == "sigpy":
        return False
    return bart_available()   # "auto"


def active_backend_label() -> str:
    """Human-readable label of the engine that will run ('BART' or 'sigpy')."""
    return "BART" if use_bart() else "sigpy"


def sigpy_available() -> bool:
    """True if sigpy can be imported (the fallback engine)."""
    try:
        import sigpy  # noqa: F401
        return True
    except Exception:
        return False


def _noop(_msg: str) -> None:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers (BART Cartesian layout is (Nread, Nphase, 1, Ncoils[, Nmaps]))
# ─────────────────────────────────────────────────────────────────────────────

def _k4d_to_coilfirst(k4d: np.ndarray) -> np.ndarray:
    """(Nread, Nphase, 1, Ncoils) → (Ncoils, Nread, Nphase) for sigpy."""
    k = k4d
    if k.ndim == 4:            # (Nread, Nphase, 1, Ncoils)
        k = k[:, :, 0, :]
    elif k.ndim == 3:          # (Nread, Nphase, Ncoils)
        pass
    else:
        raise ValueError(f"Unexpected k-space ndim {k.ndim} (shape {k.shape})")
    return np.moveaxis(k, -1, 0)                     # (Ncoils, Nread, Nphase)


def _maps_to_coilfirst(sens: np.ndarray) -> np.ndarray:
    """
    BART sens (Nread,Nphase,1,Ncoils[,Nmaps]) → (Ncoils, Nread, Nphase),
    taking the first map set (sigpy SenseRecon uses a single map set).
    """
    s = sens
    if s.ndim == 5:            # (Nread, Nphase, 1, Ncoils, Nmaps)
        s = s[:, :, 0, :, 0]
    elif s.ndim == 4:          # (Nread, Nphase, 1, Ncoils)
        s = s[:, :, 0, :]
    elif s.ndim == 3:          # (Nread, Nphase, Ncoils)
        pass
    else:
        raise ValueError(f"Unexpected sens ndim {s.ndim} (shape {s.shape})")
    return np.moveaxis(s, -1, 0)                     # (Ncoils, Nread, Nphase)


def _coilfirst_maps_to_bart(mps: np.ndarray) -> np.ndarray:
    """(Ncoils, Nread, Nphase) → (Nread, Nphase, 1, Ncoils) BART layout."""
    return np.moveaxis(mps, 0, -1)[:, :, np.newaxis, :]


def _sampling_mask_2d(k4d: np.ndarray) -> np.ndarray:
    """Binary (Nread, Nphase) mask of acquired k-space (nonzero along coils)."""
    k = k4d[:, :, 0, :] if k4d.ndim == 4 else k4d
    return (np.abs(k).sum(axis=-1) > 0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ESPIRiT sensitivity maps
# ─────────────────────────────────────────────────────────────────────────────

def espirit_maps(
    k4d_norm: np.ndarray,
    calib: int,
    num_maps: int = 1,
    *,
    crop: Optional[float] = None,
    log: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """
    ESPIRiT calibration. Returns sens maps in BART layout
    (Nread, Nphase, 1, Ncoils) — or (…, Ncoils, Nmaps) for BART multi-map.

    BART:  ecalib -r{calib}:{calib}:1 -c{crop} -m{num_maps}
    sigpy: sigpy.mri.app.EspiritCalib (single map set)

    `crop` is the eigenvalue mask threshold, applied identically to BOTH engines
    so they stay a fair, apples-to-apples pair (default ESPIRIT_CROP = 0.8).
    SigPy's EspiritCalib otherwise defaults to 0.95 — much harsher than BART's
    ecalib (0.8) — which on lower-SNR data zeroes out genuine tissue voxels and
    leaves signal-void "holes" in the reconstruction. Exposed so users can tune
    the calibration aggressiveness for both engines from one control.
    """
    log = log or _noop
    if crop is None:
        crop = ESPIRIT_CROP
    if use_bart():
        with BartTempDir("bart_ecalib_") as tmp:
            ksp_cfl = os.path.join(tmp, "ksp")
            sens_cfl = os.path.join(tmp, "sens")
            writecfl(ksp_cfl, k4d_norm)
            run_bart("ecalib", f"-r{calib}:{calib}:1", f"-c{crop}", f"-m{num_maps}",
                     ksp_cfl, sens_cfl)
            return readcfl(sens_cfl)

    # sigpy fallback
    import sigpy.mri as mr
    if num_maps > 1:
        log(f"  sigpy ESPIRiT: using a single map set (BART -m{num_maps} not reproduced).")
    ksp_c = _k4d_to_coilfirst(k4d_norm)              # (Ncoils, Nread, Nphase)
    log(f"  sigpy EspiritCalib (calib={calib}, crop={crop})…")
    mps = mr.app.EspiritCalib(ksp_c, calib_width=calib, crop=crop,
                              show_pbar=False).run()
    mps = np.asarray(mps)
    return _coilfirst_maps_to_bart(mps)              # (Nread, Nphase, 1, Ncoils)


# ─────────────────────────────────────────────────────────────────────────────
# PICS (SENSE / CS-SENSE / ESPIRiT-regularised) reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def pics(
    k4d_norm: np.ndarray,
    sens: np.ndarray,
    *,
    soft_sense: bool = False,
    l1: bool = False,
    l2: bool = False,
    reg: Optional[str] = None,       # None | 'Q' (L2) | 'T' (TV) | 'W' (L1-wavelet)
    lam: float = 0.0,
    iters: Optional[int] = None,     # None → BART default iterations (no -i flag)
    weights: Optional[np.ndarray] = None,
    log: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """
    Parallel-imaging / CS reconstruction. Returns the complex recon image
    (squeezed) — callers take abs / RSS-combine as before.

    Flag → BART pics mapping (identical to the historical inline calls):
      soft_sense  -> -S
      l2          -> -l2            (built-in Tikhonov data term)
      l1          -> -l1 -r{lam}    (CS-SENSE, ℓ1 data term)
      reg='Q'     -> -R Q:{lam}     (ESPIRiT + L2)
      reg='T'     -> -R T:7:0:{lam} (ESPIRiT + total variation)
      reg='W'     -> -R W:7:0:{lam} (ESPIRiT + L1-wavelet)
      iters       -> -i{iters}

    sigpy fallback:
      reg='W' or l1 -> L1WaveletRecon(lamda=lam)
      reg='T'       -> TotalVariationRecon(lamda=lam)
      else          -> SenseRecon(lamda = lam if (reg=='Q' or l2) else 0)
    """
    log = log or _noop

    if use_bart():
        with BartTempDir("bart_pics_") as tmp:
            ksp_cfl = os.path.join(tmp, "ksp")
            sens_cfl = os.path.join(tmp, "sens")
            recon_cfl = os.path.join(tmp, "recon")
            writecfl(ksp_cfl, k4d_norm)
            writecfl(sens_cfl, sens)
            args = ["pics"]
            if soft_sense:
                args.append("-S")
            if l2:
                args.append("-l2")
            if l1:
                args += ["-l1", f"-r{lam}"]
            if reg == "Q":
                args += ["-R", f"Q:{lam}"]
            elif reg == "T":
                args += ["-R", f"T:7:0:{lam}"]
            elif reg == "W":
                args += ["-R", f"W:7:0:{lam}"]
            if iters is not None:
                args.append(f"-i{iters}")
            run_bart(*args, ksp_cfl, sens_cfl, recon_cfl)
            return np.squeeze(readcfl(recon_cfl))

    # sigpy fallback
    import sigpy.mri as mr
    ksp_c = _k4d_to_coilfirst(k4d_norm)              # (Ncoils, Nread, Nphase)
    mps_c = _maps_to_coilfirst(sens)                 # (Ncoils, Nread, Nphase)
    if weights is None:
        weights = _sampling_mask_2d(k4d_norm)        # restrict data consistency
    max_iter = int(iters) if iters else 30           # sigpy default when BART used its own

    if reg == "W" or l1:
        lamda = lam if lam > 0 else 1e-3
        log(f"  sigpy L1WaveletRecon (λ={lamda}, iters={max_iter})…")
        img = mr.app.L1WaveletRecon(ksp_c, mps_c, lamda=lamda, weights=weights,
                                    max_iter=max_iter, show_pbar=False).run()
    elif reg == "T":
        lamda = lam if lam > 0 else 1e-3
        log(f"  sigpy TotalVariationRecon (λ={lamda}, iters={max_iter})…")
        img = mr.app.TotalVariationRecon(ksp_c, mps_c, lamda=lamda, weights=weights,
                                         max_iter=max_iter, show_pbar=False).run()
    else:
        lamda = lam if (reg == "Q" or l2) else 0.0
        if l2 and lamda == 0.0:
            lamda = 1e-2                             # approximate BART's -l2 default
        log(f"  sigpy SenseRecon (λ={lamda}, iters={max_iter})…")
        img = mr.app.SenseRecon(ksp_c, mps_c, lamda=lamda, weights=weights,
                                max_iter=max_iter, show_pbar=False).run()
    return np.asarray(img)                            # (Nread, Nphase) complex


# ─────────────────────────────────────────────────────────────────────────────
# Non-cartesian front-end: adjoint NUFFT → crop → Cartesian k-space
# ─────────────────────────────────────────────────────────────────────────────

def nufft_adjoint_to_cartesian(
    bart_traj: np.ndarray,
    kspace_coilfirst: np.ndarray,
    grid_size: int,
    crop_size: int,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Grid non-cartesian data and return:
      imgs_per_coil : (crop, crop, Ncoils)   cropped adjoint-NUFFT coil images
      cart_kspace   : (crop, crop, 1, Ncoils) Cartesian k-space (orientation-
                      corrected via the historical (1,0,2,3) transpose)

    `bart_traj` is (3, dim1, dim2) in grid-index units ([-grid/2, grid/2]);
    `kspace_coilfirst` is (Ncoils, dim1, dim2).

    BART:  nufft -a  → crop → fft         (identical to the inline pipelines)
    sigpy: sigpy.nufft.nufft_adjoint → numpy crop → centered fft
    """
    log = log or _noop
    num_coils, dim1, dim2 = kspace_coilfirst.shape

    if use_bart():
        with BartTempDir("bart_nufft_") as tmp:
            traj_cfl = os.path.join(tmp, "traj")
            ksp_cfl = os.path.join(tmp, "ksp")
            imgs_cfl = os.path.join(tmp, "imgs")
            cx_cfl = os.path.join(tmp, "cx")
            cxy_cfl = os.path.join(tmp, "cxy")
            cart_cfl = os.path.join(tmp, "cart")

            writecfl(traj_cfl, bart_traj)
            # BART ksp layout: (1, dim1, dim2, Ncoils)
            k = np.transpose(kspace_coilfirst, (1, 2, 0))[np.newaxis, ...]
            writecfl(ksp_cfl, k)

            log(f"  BART adjoint NUFFT (grid {grid_size}×{grid_size})…")
            run_bart("nufft", "-a", f"-x{grid_size}:{grid_size}:1", "-t",
                     traj_cfl, ksp_cfl, imgs_cfl)

            imgs_full = readcfl(imgs_cfl)             # (grid, grid, Ncoils)
            imgs_per_coil = _center_crop2d(imgs_full, crop_size)
            del imgs_full

            log(f"  Cropping to {crop_size}×{crop_size} and FFT…")
            run_bart("crop", "1", str(crop_size), imgs_cfl, cx_cfl)
            run_bart("crop", "0", str(crop_size), cx_cfl, cxy_cfl)
            bitmask = run_bart("bitmask", "0", "1").strip()
            run_bart("fft", bitmask, cxy_cfl, cart_cfl)
            cart_kspace = readcfl(cart_cfl)           # (crop, crop, 1, Ncoils)
            cart_kspace = np.transpose(cart_kspace, (1, 0, 2, 3))
            return imgs_per_coil, cart_kspace

    # sigpy fallback
    import sigpy
    log(f"  sigpy adjoint NUFFT (grid {grid_size}×{grid_size})…")
    coord = np.stack([bart_traj[0].reshape(-1), bart_traj[1].reshape(-1)], axis=-1)
    ksp_flat = kspace_coilfirst.reshape(num_coils, dim1 * dim2)
    imgs = sigpy.nufft_adjoint(ksp_flat, coord, oshape=(num_coils, grid_size, grid_size))
    imgs = np.moveaxis(np.asarray(imgs), 0, -1)       # (grid, grid, Ncoils)
    imgs_per_coil = _center_crop2d(imgs, crop_size)

    # Cartesian k-space of the cropped coil images (centered 2-D FFT), then the
    # same orientation transpose the BART path applies.
    cart = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(imgs_per_coil, axes=(0, 1)),
                                       axes=(0, 1)), axes=(0, 1))
    cart_kspace = cart[:, :, np.newaxis, :]           # (crop, crop, 1, Ncoils)
    cart_kspace = np.transpose(cart_kspace, (1, 0, 2, 3))
    return imgs_per_coil, cart_kspace


def _center_crop2d(arr: np.ndarray, crop_size: int) -> np.ndarray:
    """Center-crop the first two axes of `arr` to crop_size×crop_size."""
    h, w = arr.shape[:2]
    ch, cw = h // 2, w // 2
    hr, wr = crop_size // 2, crop_size // 2
    return arr[ch - hr:ch + hr, cw - wr:cw + wr, ...].copy()
