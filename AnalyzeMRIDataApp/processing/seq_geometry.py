"""
Pulseq line geometry: where every recorded readout sits in k-space.

A Pulseq TWIX header carries interpreter placeholders for matrix, FOV and
oversampling, but it reliably names the played file (``Meas.tFree``) and its
signature (``Dicom.tSequenceVariant`` == the ``[SIGNATURE] Hash`` line of the
.seq). When that exact .seq is found, its gradients give the k-space position
of every recorded sample — and with it the phase-encode index and order of
every line (TSE echo order included), the readout oversampling and the FOV.

Search order for the .seq (each must match the signature):
  the scan folder, its parent, ``seq_files/`` and ``*_seq/`` next to them or
  one level up, the user's .seq library folder (set_library_dir), and folders
  the user pointed to this session (locate_played_seq asks, when a handler is
  set). A same-name file with a different signature is NOT used: it was
  regenerated after the scan and may differ.

Without the .seq, the trajectory JSON written with the scan (find_line_json)
carries the same per-line k positions (json_line_geometry).

pypulseq is imported lazily; without it (or without either file) callers fall
back to their own path.
"""

import functools
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_HASH_RE = re.compile(rb"^Hash\s+([0-9a-fA-F]{32})\s*$", re.MULTILINE)
_LIBRARY_DIR: Optional[str] = None
_EXTRA_DIRS: List[str] = []                    # folders the user pointed to this session
_REQUEST_HANDLER: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None
_ASKED: Dict[str, Optional[str]] = {}          # signature → chosen path (None = skipped)


def set_library_dir(path: Optional[str]) -> None:
    """Folder (searched recursively) holding the lab's .seq files."""
    global _LIBRARY_DIR
    _LIBRARY_DIR = path or None


def set_seq_request_handler(handler: Optional[Callable[[Dict[str, Any]], Optional[str]]]) -> None:
    """
    ``handler(info) -> path or None`` is called when a Pulseq scan's played
    .seq cannot be found (info as returned by played_seq). The GUI shows a
    file dialog; None means "continue without it". Each signature is asked
    about once per session.
    """
    global _REQUEST_HANDLER
    _REQUEST_HANDLER = handler


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
    dirs += [(Path(d), False) for d in _EXTRA_DIRS if Path(d).is_dir()]
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


def locate_played_seq(dat_path: str, log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    played_seq, and when the exact file is not found, ask the user once per
    signature (if a request handler is set). A chosen file is used only if its
    signature matches; its folder is then searched for the rest of the session.
    """
    info = played_seq(dat_path)
    sig = info["signature"]
    if info["path"] or not sig or _REQUEST_HANDLER is None:
        return info
    if sig not in _ASKED:
        chosen = _REQUEST_HANDLER(dict(info))
        if chosen and seq_signature(chosen) == sig:
            _ASKED[sig] = chosen
            folder = str(Path(chosen).parent)
            if folder not in _EXTRA_DIRS:
                _EXTRA_DIRS.append(folder)
        else:
            if chosen and log:
                log(f"  [SEQ] {Path(chosen).name} is a different version of the sequence "
                    f"(signature {seq_signature(chosen) or 'none'}, scan played {sig}) — not used.")
            _ASKED[sig] = None
    if _ASKED[sig]:
        info = dict(info, path=_ASKED[sig], note="")
    return info


@functools.lru_cache(maxsize=16)
def seq_line_geometry(seq_path: str) -> Dict[str, Any]:
    """
    Line geometry from a .seq: for every ADC in acquisition order its
    phase-encode index (k-space centre = 0), plus samples per ADC, readout
    oversampling (Δk_pe / Δk_read), readout direction, the PE grid size and
    the FOVs in mm. Raises ImportError without pypulseq, ValueError for
    non-Cartesian data.
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
    geo = _lines_from_k(k_adc[0].reshape(n_adc, n_samp), k_adc[1].reshape(n_adc, n_samp))
    geo["signature"] = seq_signature(seq_path)
    return geo


def json_line_geometry(json_path: str, n_read: int) -> Dict[str, Any]:
    """
    The same line geometry from a trajectory JSON written with the scan
    ({"0": [[kx…], [ky…]]}, samples in acquisition order, n_read per line).
    Raises ValueError when the JSON does not describe n_read-sample lines.
    """
    st = os.stat(json_path)
    return dict(_json_line_geometry(str(json_path), int(n_read), st.st_mtime_ns, st.st_size))


@functools.lru_cache(maxsize=8)
def _json_line_geometry(json_path: str, n_read: int, _mtime: int, _size: int) -> Dict[str, Any]:
    with open(json_path, "r") as fh:
        data = json.load(fh)
    traj = np.asarray(data["0"] if isinstance(data, dict) else data, dtype=float)
    if traj.ndim != 2 or traj.shape[0] < 2:
        raise ValueError("not a [[kx…], [ky…]] trajectory")
    kx, ky = traj[0], traj[1]
    if kx.size % n_read:
        raise ValueError(f"{kx.size} samples are not whole {n_read}-sample lines")
    n_lines = kx.size // n_read
    geo = _lines_from_k(kx.reshape(n_lines, n_read), ky.reshape(n_lines, n_read))
    fov = geo["fov_phase_mm"]
    if not 20.0 <= fov <= 1000.0:                      # JSON in other units: no FOV
        geo["fov_phase_mm"] = geo["fov_read_os_mm"] = None
    return geo


def _lines_from_k(kx: np.ndarray, ky: np.ndarray) -> Dict[str, Any]:
    """Per-line geometry from k positions shaped (n_lines, n_samples)."""
    n_adc, n_samp = kx.shape
    scale = max(np.abs(kx).max(), np.abs(ky).max(), 1e-12)
    if np.allclose(ky.std(axis=1), 0, atol=1e-6 * scale):
        read_k, pe_k, swapped = kx, ky[:, 0], False
    elif np.allclose(kx.std(axis=1), 0, atol=1e-6 * scale):
        read_k, pe_k, swapped = ky, kx[:, 0], True
    else:
        raise ValueError("Neither k axis is constant within a line — not Cartesian.")
    direction = np.sign(read_k[:, -1] - read_k[:, 0])
    if np.unique(direction).size != 1 or direction[0] == 0:
        raise ValueError("Readout direction alternates between lines (EPI-like) — not supported.")
    pe_values = np.unique(np.round(pe_k, 6))
    steps = np.diff(pe_values)
    d_pe = float(steps.min()) if steps.size else 1.0
    if steps.size:                                     # average the one-line steps
        d_pe = float(steps[steps < 1.1 * d_pe].mean())  # (gradient rounding jitters them)
    d_read = float(np.median(np.abs(np.diff(read_k[0]))))
    # Lines sit either on k = j·Δk (k = 0 sampled) or on a half-line grid
    # k = (j + ½)·Δk, symmetric about k = 0 (no k = 0 line); on the half-line
    # grid, line j is the one just above k = 0's row, i.e. floor(k / Δk).
    pos = pe_k / d_pe
    half_line = bool(pos.size and np.all(np.abs(pos - np.floor(pos) - 0.5) < 0.1))
    pe_index = (np.floor(pos) if half_line else np.round(pos)).astype(int)
    return {
        "pe_index": pe_index,                          # per ADC, acquisition order
        "pe_half_line": half_line,                     # k grid offset by ½ line
        "n_adc": n_adc,
        "samples_per_adc": n_samp,
        "readout_os": d_pe / d_read if d_read else None,
        "n_pe_full": int(2 * max(-pe_index.min(), pe_index.max() + 1)),
        "fov_phase_mm": 1000.0 / d_pe,
        "fov_read_os_mm": 1000.0 / d_read if d_read else None,
        "swapped": swapped,
        "read_reversed": bool(direction[0] < 0),       # samples run from +k to −k
    }


def find_line_json(dat_path: str) -> Optional[str]:
    """
    The trajectory JSON written with a Cartesian scan, in the scan folder (the
    parent of dat_files/) or next to the .dat:
      * ``Undersampled<R>equispace(d|s)[_swapPE].json`` for an ``_R<n>`` scan;
      * ``FullySampled[_<tokens>].json`` whose tokens all appear in the .dat
        name (e.g. FullySampled_tse_t2.json ↔ …fully_sampled_tse_t2.dat).
    """
    dat = Path(dat_path)
    stem = dat.stem.lower()
    folders = [dat.parent.parent, dat.parent]
    swap = "swappe" in stem
    m = re.search(r"_r(\d+)", stem)
    if m:
        r = m.group(1)
        names = [f"{p}ndersampled{r}equispace{e}.json" for p in ("U", "u") for e in ("d", "s")]
        if swap:
            names = [n.replace(".json", "_swapPE.json") for n in names] + names
        for folder in folders:
            for name in names:
                if (folder / name).is_file():
                    return str(folder / name)
        return None
    if "fully" not in stem:
        return None
    flat = stem.replace("_", "")
    for folder in folders:
        for cand in sorted(folder.glob("[Ff]ully[Ss]ampled*.json")):
            tokens = [t for t in cand.stem.lower().split("_")[1:] if t]
            if ("swappe" in tokens) == swap and all(t in flat for t in tokens):
                return str(cand)
    return None


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
    elif np.unique(seq_rows - legacy_rows).size == 1:
        verdict = (f"shifted by {int((seq_rows - legacy_rows)[0]):+d} line(s) vs the played .seq "
                   "(a linear phase; magnitude unaffected)")
    elif np.unique(seq_rows - (n_phase - 1 - legacy_rows)).size == 1:
        verdict = ("REVERSED phase-encode direction vs the played .seq "
                   "(image mirrored along phase encode)")
    elif np.array_equal(np.sort(seq_rows), np.sort(legacy_rows)):
        verdict = "same lines, different ORDER than the played .seq"
    else:
        n_diff = int(np.sum(seq_rows != legacy_rows))
        verdict = f"{n_diff}/{seq_rows.size} lines placed differently than the played .seq"
    if geo.get("pe_half_line"):
        verdict += "; the .seq samples ky half a line off k = 0 (magnitude unaffected)"
    if log:
        log(f"  [SEQ] {Path(info['path']).name}: line placement {verdict} "
            f"(readout OS {geo['readout_os']:.2f}, FOV {geo['fov_phase_mm']:.0f} mm).")
    return verdict
