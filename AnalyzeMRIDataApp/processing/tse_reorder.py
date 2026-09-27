"""
TSE (Turbo Spin-Echo) phase-encode un-scrambling.

Our custom Pulseq TSE sequences (writeTSE_kband) set NO `LIN` label on the ADC,
so the Siemens interpreter numbers the phase-encode counter sequentially in
ACQUISITION order.  `read_raw_twix_data` therefore returns k-space lines in TSE
echo-train (acquisition) order — NOT ascending PE order.  Any *Cartesian*
reconstruction that places lines by a computed PE index (fully-sampled IFFT,
equispaced GRAPPA) must first re-map each acquired line to its true PE position,
or every echo lands on the wrong k-space line and the image is corrupted.

The core mapping `tse_acq_to_line` is a verbatim port of make_pe_plan.m
(CreateSeqFiles/kband_tse/make_pe_plan.m), validated in ProcessTSE.ipynb.

SE data (necho == 1) is acquired in ascending PE order, so it needs no
un-scramble — `detect_tse_params` returns None for non-TSE filenames and the
callers skip the reorder entirely.

BART NUFFT recons (band, poisson) do NOT use this module: they pair each k-space
sample with its trajectory-JSON coordinate, and the JSON is written in the same
acquisition order as the reader, so the echo scrambling cancels out.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

_logger = logging.getLogger(__name__)

# Defaults matching the generated sequences (build_base_params.m):
#   TSE turbo factor = 16, echo spacing TE1 = 12 ms.
#   t1 -> TEeff = 12 ms (central echo 1);  t2 -> TEeff = 96 ms (central echo 8).
_DEFAULT_NECHO = 16
_DEFAULT_TE1 = 12e-3
_TEEFF_BY_CONTRAST = {"t1": 12e-3, "t2": 96e-3}
_DEFAULT_TEEFF = 96e-3  # assume T2 when contrast is not encoded in the filename


def is_tse(filename: str) -> bool:
    """True if the filename stem denotes a TSE sequence (contains 'tse')."""
    return bool(re.search(r"tse", Path(filename).stem, re.IGNORECASE))


def _contrast_from_name(stem: str) -> Optional[str]:
    """Return 't1' or 't2' if the filename encodes a contrast, else None."""
    if re.search(r"t1", stem, re.IGNORECASE):
        return "t1"
    if re.search(r"t2", stem, re.IGNORECASE):
        return "t2"
    return None


def detect_tse_params(
    filename: str,
    *,
    tse: Optional[bool] = None,
    necho: Optional[int] = None,
    te1: Optional[float] = None,
    teeff: Optional[float] = None,
) -> Optional[dict]:
    """
    Resolve the TSE un-scramble parameters for a .dat file.

    Parameters
    ----------
    filename : str
        The .dat path (only the stem is inspected).
    tse : bool or None
        Force TSE on/off.  None (default) → auto-detect from the filename
        ('tse' substring).
    necho, te1, teeff : optional overrides
        Explicit values win over filename inference.  `te1`/`teeff` are in
        seconds.  Only `round(teeff/te1)` (the central-echo shift) matters.

    Returns
    -------
    dict {"necho", "TE1", "TEeff"}  when the scan is TSE (necho > 1),
    otherwise None (SE / non-TSE → caller skips the reorder).
    """
    stem = Path(filename).stem
    use_tse = is_tse(filename) if tse is None else bool(tse)

    _necho = int(necho) if necho else (_DEFAULT_NECHO if use_tse else 1)
    if _necho <= 1:
        return None  # SE — acquired in ascending PE order, no un-scramble

    _te1 = float(te1) if te1 else _DEFAULT_TE1
    if teeff:
        _teeff = float(teeff)
    else:
        contrast = _contrast_from_name(stem)
        _teeff = _TEEFF_BY_CONTRAST.get(contrast, _DEFAULT_TEEFF)

    return {"necho": _necho, "TE1": _te1, "TEeff": _teeff}


def tse_acq_to_line(
    Ny: int,
    necho: int,
    TE1: float,
    TEeff: float,
    pe_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    True 0-based PE line for each ADC, in Lin/acquisition order.

    Verbatim port of make_pe_plan.m (validated in ProcessTSE.ipynb).  Returns an
    array whose i-th entry is the true PE line index of the i-th acquired ADC.
    With `pe_mask` given, only the sampled lines are returned, still in
    echo-train order.

    Requires necho to divide Ny (the sequence uses nex = Ny // necho and reshapes
    the full echo-train onto (necho, nex)); callers must guard this.
    """
    pe_mask = np.ones(Ny, bool) if pe_mask is None else np.asarray(pe_mask, bool).ravel()
    nex = Ny // necho
    pe_steps = np.arange(1, necho * nex + 1) - 0.5 * necho * nex - 1.0
    if necho % 2 == 0:
        pe_steps = np.roll(pe_steps, -round(nex / 2))          # even-echo contrast shift
    k0curr = int(np.argmin(np.abs(pe_steps))) // nex + 1
    k0prescr = max(round(TEeff / TE1), 1)
    PEorder = pe_steps.reshape(nex, necho, order="F").T.astype(float)   # (necho, nex)
    PEorder = np.roll(PEorder, k0prescr - k0curr, axis=0)      # TEeff → k-centre
    sampled_steps = (np.where(pe_mask)[0] + 1) - 0.5 * Ny - 1.0
    PEorder[~np.isin(PEorder, sampled_steps)] = np.nan
    PEorder = PEorder[:, ~np.all(np.isnan(PEorder), axis=0)]   # drop empty shots
    steps = PEorder.flatten(order="F")                         # column-major = acq order
    steps = steps[~np.isnan(steps)]
    return np.round(steps + 0.5 * Ny + 1).astype(int) - 1      # 0-based true line


def unscramble_rows(
    kspace_raw: np.ndarray,
    necho: int,
    TE1: float,
    TEeff: float,
    log_fn=None,
) -> np.ndarray:
    """
    Reorder a fully-sampled TSE k-space array so row = true PE line.

    Parameters
    ----------
    kspace_raw : (Nacq, Ncoils, Nread) — reader output, echo-train order.
    necho, TE1, TEeff : TSE plan params (see `detect_tse_params`).

    Returns
    -------
    (Ny=Nacq, Ncoils, Nread) with each acquired line moved to its true PE index.
    Returns the input unchanged (with a warning) if the un-scramble guards fail
    (necho does not divide Nacq, or the mapping is not a permutation of all rows)
    — a safe fallback to the pre-TSE ascending behaviour.
    """
    def _emit(msg: str) -> None:
        _logger.info(msg)
        if log_fn is not None:
            log_fn(msg)

    n_acq = kspace_raw.shape[0]
    if necho <= 1:
        return kspace_raw
    if n_acq % necho != 0:
        _emit(
            f"  [TSE] necho={necho} does not divide Nacq={n_acq}; "
            f"skipping un-scramble (ascending order kept)."
        )
        return kspace_raw

    acq2line = tse_acq_to_line(n_acq, necho, TE1, TEeff)   # full mask
    if len(acq2line) != n_acq or np.unique(acq2line).size != n_acq:
        _emit(
            f"  [TSE] un-scramble map invalid "
            f"(len={len(acq2line)}, unique={np.unique(acq2line).size}, Nacq={n_acq}); "
            f"skipping un-scramble."
        )
        return kspace_raw

    out = np.zeros_like(kspace_raw)
    out[acq2line] = kspace_raw
    _emit(
        f"  [TSE] un-scrambled {n_acq} lines → true PE order "
        f"(necho={necho}, central echo={max(round(TEeff / TE1), 1)})."
    )
    return out
