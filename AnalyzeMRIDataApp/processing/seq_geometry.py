"""
Pulseq geometry from the .seq file that was actually played.

A Pulseq TWIX header carries interpreter placeholders for matrix, FOV and
oversampling, but it reliably names the played file (``Meas.tFree``) and its
signature (``Dicom.tSequenceVariant`` == the ``[SIGNATURE] Hash`` line of the
.seq). When that exact .seq is found, its gradients give the k-space position
of every recorded sample — and with it the phase-encode index and order of
every line (TSE echo order included), the readout oversampling and the FOV.

Search order for the .seq (each must match the signature):
  the scan folder, its parent, ``seq_files/`` and ``*_seq/`` next to them or
  one level up, then the user's .seq library folder (set_library_dir).
A same-name file with a different signature is NOT used for geometry: it was
regenerated after the scan and may differ.

pypulseq is imported lazily; without it (or without the file) this module
reports that the geometry is unavailable and callers keep their own path.
"""

import functools
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HASH_RE = re.compile(rb"^Hash\s+([0-9a-fA-F]{32})\s*$", re.MULTILINE)
_LIBRARY_DIR: Optional[str] = None


def set_library_dir(path: Optional[str]) -> None:
    """Folder (searched recursively) holding the lab's .seq files."""
    global _LIBRARY_DIR
    _LIBRARY_DIR = path or None


def seq_signature(seq_path: str) -> Optional[str]:
    """The ``[SIGNATURE] Hash`` of a .seq file (read from its tail), or None."""
    try:
        with open(seq_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 4096))
            match = _HASH_RE.search(fh.read())
    except OSError:
        return None
    return match.group(1).decode().lower() if match else None


@functools.lru_cache(maxsize=32)
def _signature_index(folder: str, recursive: bool, _mtime: float) -> Dict[str, str]:
    """{signature: path} for the .seq files in a folder (cached per folder state)."""
    root = Path(folder)
    files = root.rglob("*.seq") if recursive else root.glob("*.seq")
    index = {}
    for f in files:
        sig = seq_signature(str(f))
        if sig:
            index.setdefault(sig, str(f))
    return index


def _search_dirs(dat_path: str) -> List[Tuple[Path, bool]]:
    here = Path(dat_path).resolve().parent
    dirs = [(here, False), (here.parent, False)]
    # seq_files/ or *_seq/ next to the scan folder, or one level up (a session
    # folder holding several subjects, e.g. 18_5_26/SE_seq).
    for base in (here, here.parent, here.parent.parent):
        dirs += [(d, True) for d in sorted(base.glob("seq_files")) + sorted(base.glob("*_seq"))
                 if d.is_dir()]
    if _LIBRARY_DIR and Path(_LIBRARY_DIR).is_dir():
        dirs.append((Path(_LIBRARY_DIR), True))
    return dirs


def played_seq(dat_path: str) -> Dict[str, Any]:
    """
    What the header says was played, and where that exact .seq is:
    ``{"name", "signature", "path" (or None), "note"}``. Only Pulseq files have
    a name/signature.
    """
    from .utils import read_twix_header, sequence_family
    out = {"name": None, "signature": None, "path": None, "note": ""}
    if sequence_family(dat_path) != "pulseq":
        out["note"] = "not a Pulseq scan"
        return out
    hdr = read_twix_header(dat_path) or {}
    out["name"] = str((hdr.get("Meas", {}) or {}).get("tFree", "") or "") or None
    sig = str((hdr.get("Dicom", {}) or {}).get("tSequenceVariant", "") or "").lower()
    out["signature"] = sig if re.fullmatch(r"[0-9a-f]{32}", sig) else None
    if not out["signature"]:
        out["note"] = "header has no .seq signature"
        return out
    for folder, recursive in _search_dirs(dat_path):
        try:
            mtime = folder.stat().st_mtime
        except OSError:
            continue
        path = _signature_index(str(folder), recursive, mtime).get(out["signature"])
        if path:
            out["path"] = path
            return out
    out["note"] = "the played .seq (matching signature) was not found"
    return out


@functools.lru_cache(maxsize=16)
def seq_line_geometry(seq_path: str) -> Dict[str, Any]:
    """
    Line geometry from a .seq: for every ADC in acquisition order its
    phase-encode index (k-space centre = 0), plus samples per ADC, readout
    oversampling (Δk_pe / Δk_read), the PE grid size and the FOVs in mm.
    Raises ImportError without pypulseq, ValueError for non-Cartesian data.
    """
    import pypulseq as pp
    seq = pp.Sequence()
    seq.read(seq_path)
    k_adc = np.asarray(seq.calculate_kspace()[0])           # (3, n_samples_total) in 1/m
    samples = [seq.get_block(b).adc.num_samples for b in seq.block_events
               if seq.get_block(b).adc is not None]
    if not samples or len(set(samples)) != 1:
        raise ValueError("ADC sample counts vary — not a Cartesian line acquisition.")
    n_adc, n_samp = len(samples), samples[0]
    kx = k_adc[0].reshape(n_adc, n_samp)
    ky = k_adc[1].reshape(n_adc, n_samp)
    scale = max(np.abs(k_adc[:2]).max(), 1e-12)
    if np.allclose(ky.std(axis=1), 0, atol=1e-6 * scale):
        read_k, pe_k, swapped = kx, ky[:, 0], False
    elif np.allclose(kx.std(axis=1), 0, atol=1e-6 * scale):
        read_k, pe_k, swapped = ky, kx[:, 0], True
    else:
        raise ValueError("Neither k axis is constant within an ADC — not Cartesian.")
    pe_values = np.unique(np.round(pe_k, 6))
    d_pe = float(np.min(np.diff(pe_values))) if pe_values.size > 1 else 1.0
    d_read = float(np.median(np.abs(np.diff(read_k[0]))))
    pe_index = np.round(pe_k / d_pe).astype(int)
    return {
        "pe_index": pe_index,                          # per ADC, acquisition order
        "n_adc": n_adc,
        "samples_per_adc": n_samp,
        "readout_os": d_pe / d_read if d_read else None,
        "n_pe_full": int(2 * max(-pe_index.min(), pe_index.max() + 1)),
        "fov_phase_mm": 1000.0 / d_pe,
        "fov_read_os_mm": 1000.0 / d_read if d_read else None,
        "swapped": swapped,
        "signature": seq_signature(seq_path),
    }


def shadow_check(dat_path: str, legacy_rows, n_phase: int, log=None) -> None:
    """compare_with_legacy for use inside reconstruction code: logging only,
    any failure is swallowed so the reconstruction is never affected."""
    try:
        compare_with_legacy(dat_path, np.asarray(legacy_rows), int(n_phase), log)
    except Exception as exc:                          # noqa: BLE001 — shadow mode
        if log:
            log(f"  [SEQ] cross-check skipped ({type(exc).__name__}: {exc}).")


def compare_with_legacy(dat_path: str, legacy_rows: np.ndarray, n_phase: int,
                        log=None) -> Optional[str]:
    """
    Shadow mode: check the legacy line placement (row of each acquired line in
    acquisition order, on an n_phase grid with the centre at n_phase // 2)
    against the played .seq. Logs and returns a one-line verdict; never
    changes the reconstruction. None when no .seq geometry is available.
    """
    info = played_seq(dat_path)
    if not info["path"]:
        if log and info["signature"]:
            log(f"  [SEQ] {info['note']} ({info['name']}) — geometry not cross-checked.")
        return None
    try:
        geo = seq_line_geometry(info["path"])
    except ImportError:
        return None
    except ValueError as exc:
        if log:
            log(f"  [SEQ] {Path(info['path']).name}: {exc}")
        return None
    seq_rows = geo["pe_index"] + n_phase // 2
    legacy_rows = np.asarray(legacy_rows)
    if seq_rows.size != legacy_rows.size:
        verdict = (f"line count differs (.seq {seq_rows.size} vs placed {legacy_rows.size})")
    elif np.array_equal(seq_rows, legacy_rows):
        verdict = "matches the played .seq exactly"
    elif np.unique(seq_rows - (n_phase - 1 - legacy_rows)).size == 1:
        verdict = ("REVERSED phase-encode direction vs the played .seq "
                   "(image mirrored along phase encode)")
    elif np.array_equal(np.sort(seq_rows), np.sort(legacy_rows)):
        verdict = "same lines, different ORDER than the played .seq"
    else:
        n_diff = int(np.sum(seq_rows != legacy_rows))
        verdict = f"{n_diff}/{seq_rows.size} lines placed differently than the played .seq"
    if log:
        log(f"  [SEQ] {Path(info['path']).name}: line placement {verdict} "
            f"(readout OS {geo['readout_os']:.2f}, FOV {geo['fov_phase_mm']:.0f} mm).")
    return verdict
