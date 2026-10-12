"""
Acquisition description and full-grid k-space assembly (one slice).

For Siemens sequences every geometry decision comes from the file itself: the
protocol header gives the phase-encode matrix, partial Fourier, oversampling
and acceleration, and every readout's MDH gives its line number and the
k-space centre. Lines are placed on the protocol's full phase-encode grid with
the centre in the middle, so partial-Fourier and acceleration gaps stay zero,
calibration is centred on the true k-space centre, and the image can be
cropped to the prescribed field of view.

Each value is recorded with its source (``AcqFacts``) so the UI and report can
show where it came from.

Pulseq headers carry interpreter placeholders (matrix, oversampling, even
"3D"), so Pulseq geometry comes from the sequence itself instead: the played
.seq (found by the signature in the header), else the trajectory JSON written
with the scan. Each gives the k-space position of every readout, so lines are
placed by their true phase-encode index in the same way. Without either (or
for unreadable headers) ``acquire_cartesian`` returns None and callers keep
their existing path, which infers the placement from the file name and line
count.
"""

from dataclasses import dataclass
from enum import Enum, IntFlag
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .imaging import crop_center


class Src(str, Enum):
    """Where an acquisition fact came from."""
    HEADER = "protocol header"
    MDH = "per-line header"
    NOISE = "noise-adjust scan"
    SEQ = "played .seq"
    TRAJ_JSON = "trajectory JSON"
    DATA = "inferred from data"
    DERIVED = "derived"
    SETTING = "app setting"


# Coil noise prewhitening for files assembled here (Siemens). Set per job from
# the parameters panel (like the recon engine); Pulseq files keep their path.
PREWHITEN = True


def set_prewhitening(on: bool) -> None:
    global PREWHITEN
    PREWHITEN = bool(on)


def noise_whitener(noise: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Coil whitening matrix from noise samples (n_samples, n_coils): W = L⁻¹ with
    Ψ = L Lᴴ the noise covariance, scaled by √mean(diag Ψ) so image intensity
    keeps its scale. Returns (W, largest coil correlation coefficient).
    """
    noise = np.asarray(noise, dtype=np.complex128)
    cov = noise.conj().T @ noise / max(noise.shape[0] - 1, 1)   # Ψ_ij = E[n_i* n_j]
    cov = cov.T                                                 # Ψ_ij = E[n_i n_j*]
    var = np.real(np.diag(cov))
    corr = np.abs(cov) / np.sqrt(np.outer(var, var))
    np.fill_diagonal(corr, 0.0)
    L = np.linalg.cholesky(cov)
    W = np.linalg.inv(L) * np.sqrt(var.mean())
    return W, float(corr.max()) if corr.size else 0.0


@dataclass(frozen=True)
class Fact:
    value: Any
    src: Src
    detail: str = ""


class AcqFacts:
    """Named acquisition facts, each with its source and a short detail."""

    def __init__(self) -> None:
        self._facts: Dict[str, Fact] = {}
        self.warnings: List[str] = []

    def set(self, name: str, value: Any, src: Src, detail: str = "") -> Any:
        self._facts[name] = Fact(value, src, detail)
        return value

    def __getitem__(self, name: str) -> Any:
        return self._facts[name].value

    def get(self, name: str, default: Any = None) -> Any:
        fact = self._facts.get(name)
        return default if fact is None else fact.value

    def fact(self, name: str) -> Optional[Fact]:
        return self._facts.get(name)

    def table(self) -> List[Tuple[str, Any, str, str]]:
        """(name, value, source, detail) rows for display."""
        return [(k, f.value, f.src.value, f.detail) for k, f in self._facts.items()]

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """JSON-safe form stored on results."""
        def plain(v):
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, tuple):
                return [plain(x) for x in v]
            return v
        out = {k: {"value": plain(f.value), "source": f.src.value, "detail": f.detail}
               for k, f in self._facts.items()}
        if self.warnings:
            out["warnings"] = {"value": list(self.warnings), "source": Src.DERIVED.value,
                               "detail": ""}
        return out

    def summary(self) -> str:
        """One line for the result tab and report, e.g.
        'PF 7/8 · 235/269 lines · R 1.0 · phase OS 0% · readout OS 2×'."""
        parts = []
        pf = self.get("partial_fourier")
        if pf and pf != "off":
            parts.append(f"PF {pf}")
        n_acq, n_full = self.get("n_pe_acquired"), self.get("n_pe_full")
        if n_acq is not None and n_full is not None:
            parts.append(f"{n_acq}/{n_full} lines")
        if self.get("accel_r") is not None:
            parts.append(f"R {self['accel_r']:g}")
        if self.get("phase_os") is not None:
            parts.append(f"phase OS {round(100 * self['phase_os'])}%")
        if self.get("read_os") is not None:
            parts.append(f"readout OS {self['read_os']}×")
        if self.get("asymmetric_echo"):
            parts.append("asymmetric echo")
        if self.get("prewhitened"):
            parts.append("prewhitened")
        return " · ".join(parts)


class LineCat(IntFlag):
    """Per phase-encode line of the full grid."""
    ACQ = 1          # measured
    CALIB = 2        # calibration (ACS) line
    SKIP_ACCEL = 4   # skipped by parallel-imaging acceleration
    PF_MISSING = 8   # never acquired: partial Fourier


class UnsupportedAcquisition(ValueError):
    """The file describes an acquisition this app cannot reconstruct."""


class AcquisitionConflict(ValueError):
    """The file contradicts itself; refuse instead of guessing."""


@dataclass
class Assembled:
    """One slice of k-space on the protocol's full phase-encode grid."""
    kspace: np.ndarray                    # (n_read, n_pe_full, n_coils) complex128
    line_category: np.ndarray             # (n_pe_full,) uint8 LineCat
    calib_range: Tuple[int, int]          # [start, end) symmetric about the centre
    pi_range: Tuple[int, int]             # [start, end) rows parallel imaging may fill
    n_pe_image: int                       # PE lines kept after phase-oversampling crop
    pixel_spacing_mm: Optional[Tuple[float, float]]   # (read, phase) of the final image
    facts: AcqFacts
    slice_idx: Optional[int] = None

    @property
    def pe_mask(self) -> np.ndarray:
        return (self.line_category & LineCat.ACQ) != 0

    def r_parts(self) -> Tuple[float, float, float]:
        """(R from acceleration, R from partial Fourier, total R)."""
        n_full = len(self.line_category)
        n_acq = int(self.pe_mask.sum())
        n_pf = int(((self.line_category & LineCat.PF_MISSING) != 0).sum())
        r_pf = n_full / max(n_full - n_pf, 1)
        r_total = n_full / max(n_acq, 1)
        return r_total / r_pf, r_pf, r_total


_PF_FRACTION = {1: "4/8", 2: "5/8", 4: "6/8", 8: "7/8", 16: "off"}
_SUPPORTED_AXES = {"Sli", "Lin", "Cha", "Col"}


def acquire_cartesian(
    dat_path: str,
    *,
    slice_selector_cb: Optional[Callable[[int], int]] = None,
    slice_idx: Optional[int] = None,
    json_path: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[Assembled]:
    """
    Read a Cartesian 2-D file and assemble one slice on the full grid.

    Siemens files from their header; Pulseq files from the played .seq or the
    scan's trajectory JSON (``json_path``, else found next to the scan).
    Returns None when neither applies (unreadable header, Pulseq without .seq
    or JSON) — the caller then keeps its existing path. Raises
    UnsupportedAcquisition / AcquisitionConflict when the file cannot be
    reconstructed correctly.
    """
    from .utils import ensure_twix_importable, sequence_family
    family = sequence_family(dat_path)
    if family == "pulseq":
        return acquire_pulseq(dat_path, slice_selector_cb=slice_selector_cb,
                              slice_idx=slice_idx, json_path=json_path, log=log)
    if family != "siemens":
        return None
    if log:
        log(f"Reading TWIX data: {Path(dat_path).name}")
    ensure_twix_importable()
    from read_siemens_scans.twix_reading import read_raw_twix_data
    out = read_raw_twix_data(str(dat_path), fix_oversampling=False, with_meta=True)
    if len(out) < 3 or out[2] is None:          # e.g. a test double without metadata
        return None
    raw, _dims, meta = out
    return assemble_siemens(raw, meta, slice_selector_cb=slice_selector_cb,
                            slice_idx=slice_idx, log=log)


def assemble_siemens(
    raw: np.ndarray,
    meta,
    *,
    slice_selector_cb: Optional[Callable[[int], int]] = None,
    slice_idx: Optional[int] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Assembled:
    """Place one slice of a Siemens acquisition on its full grid (see module doc)."""
    from .utils import crop_readout_oversampling

    def emit(msg: str) -> None:
        if log:
            log(msg)

    hdr = meta.header
    yaps = hdr.get("MeasYaps", {}) or {}
    kspace_hdr = yaps.get("sKSpace", {}) or {}
    facts = AcqFacts()
    facts.set("family", "siemens", Src.HEADER, "Config.SequenceFileName")

    # ── Refuse what we cannot reconstruct correctly ───────────────────────────
    if kspace_hdr.get("ucDimension") == 4:
        raise UnsupportedAcquisition("3-D acquisitions are not supported yet.")
    extra = [a for a in meta.axes if a not in _SUPPORTED_AXES]
    if extra:
        raise UnsupportedAcquisition(
            f"Unsupported data dimensions {extra} (only slices, lines, coils and "
            f"readout are handled; multi-echo / repetitions / 3-D are not).")

    # ── Slice ─────────────────────────────────────────────────────────────────
    axes = list(meta.axes)
    lines = meta.lines
    sel = np.ones(lines["Lin"].size, dtype=bool)
    if "Sli" in axes:
        n_sli = raw.shape[axes.index("Sli")]
        if slice_idx is None:
            slice_idx = int(slice_selector_cb(n_sli)) if slice_selector_cb else n_sli // 2
        raw = np.take(raw, slice_idx, axis=axes.index("Sli"))
        axes.remove("Sli")
        sel = lines["Sli"] == slice_idx
        facts.set("n_slices", n_sli, Src.MDH, f"slice {slice_idx} reconstructed")
        emit(f"  [ACQ] slice {slice_idx} of {n_sli}")
    if "Cha" not in axes:                       # single-coil data lost its coil axis
        raw = np.expand_dims(raw, axis=len(axes))
        axes.append("Cha")
    arr = np.transpose(raw, (axes.index("Col"), axes.index("Lin"), axes.index("Cha")))
    n_coils = arr.shape[2]
    facts.set("n_coils", n_coils, Src.MDH)

    lin = lines["Lin"][sel]
    if lin.size == 0:
        raise AcquisitionConflict("No image readouts found for the selected slice.")

    def single(name: str) -> int:
        values = np.unique(lines[name][sel])
        if values.size != 1:
            raise AcquisitionConflict(f"Readouts disagree on {name}: {values.tolist()[:6]}.")
        return int(values[0])

    centre_lin, centre_col, ncol = single("CenterLin"), single("CenterCol"), single("ncol")

    # ── Phase-encode grid: centre line in the middle ──────────────────────────
    n_full = int(facts.set("n_pe_full", int(kspace_hdr["lPhaseEncodingLines"]), Src.HEADER,
                           "MeasYaps.sKSpace.lPhaseEncodingLines"))
    facts.set("pe_centre_line", centre_lin, Src.MDH, "CenterLin")
    offset = n_full // 2 - centre_lin
    rows = np.unique(lin) + offset
    if rows.min() < 0 or rows.max() >= n_full:
        raise AcquisitionConflict(
            f"Line numbers {lin.min()}..{lin.max()} with centre {centre_lin} do not fit "
            f"the {n_full}-line grid.")

    # ── Readout: pad an asymmetric echo, then remove oversampling ─────────────
    if arr.shape[0] != ncol:
        raise AcquisitionConflict(f"Readout length {arr.shape[0]} ≠ samples per readout {ncol}.")
    # The protocol's Fourier matrix (base resolution × readout OS) with the echo
    # centre in its middle; an asymmetric echo leaves the short side zero, and
    # samples beyond the matrix on the long side are not used (as on the scanner).
    n_fourier = _header_float(hdr, ("Meas", "iNoOfFourierColumns"))
    n_col_full = 2 * max(centre_col, ncol - centre_col)
    how = "symmetric about CenterCol"
    if n_fourier and int(n_fourier) % 2 == 0 and int(n_fourier) // 2 >= min(centre_col, ncol - centre_col):
        n_col_full, how = int(n_fourier), "Meas.iNoOfFourierColumns"
    col_off = n_col_full // 2 - centre_col
    src_lo, src_hi = max(0, -col_off), min(ncol, n_col_full - col_off)
    facts.set("n_adc_samples", ncol, Src.MDH, "SamplesInScan")
    facts.set("n_read_full", n_col_full, Src.HEADER if how.startswith("Meas") else Src.DERIVED, how)
    facts.set("asymmetric_echo", n_col_full != ncol or centre_col * 2 != ncol, Src.MDH,
              f"CenterCol {centre_col} of {ncol} samples")
    if src_hi - src_lo < ncol:
        facts.warnings.append(f"{ncol - (src_hi - src_lo)} readout samples beyond the "
                              f"{n_col_full}-column Fourier matrix not used.")
    # The reader's line axis is indexed by the Lin counter, so array row
    # (row - offset) holds full-grid row `row`.
    kspace = np.zeros((n_col_full, n_full, n_coils), dtype=np.complex128)
    kspace[col_off + src_lo:col_off + src_hi, rows, :] = arr[src_lo:src_hi][:, rows - offset, :]

    read_os = _header_float(hdr, ("Meas", "flReadoutOSFactor"), ("Dicom", "flReadoutOSFactor"))
    read_os = int(read_os) if read_os in (1.0, 2.0) else 1
    facts.set("read_os", read_os, Src.HEADER, "Meas.flReadoutOSFactor")
    if read_os > 1:
        kspace = crop_readout_oversampling(kspace, expected_nx=n_col_full // read_os,
                                           readout_axis=0, os_factor=read_os)

    # ── Calibration lines (integrated reference scan) ─────────────────────────
    calib_rows = np.unique(lin[lines["ref"][sel]]) + offset
    ref_mode = (yaps.get("sPat", {}) or {}).get("ucRefScanMode")
    if meta.refscan is not None and meta.refscan_lines.get("Lin", np.empty(0)).size:
        if ref_mode == 2:   # integrated: same readouts as the image, merge them
            kspace, extra_rows = _merge_refscan(kspace, meta, slice_idx, offset, col_off,
                                                ncol, n_col_full, read_os)
            calib_rows = np.union1d(calib_rows, extra_rows)
            rows = np.union1d(rows, extra_rows)
        else:
            facts.warnings.append(f"Separate reference scan (mode {ref_mode}) not used yet.")
    facts.set("ref_mode", {1: "none", 2: "integrated", 4: "separate"}.get(ref_mode, ref_mode),
              Src.HEADER, "MeasYaps.sPat.ucRefScanMode")

    # ── Coil noise prewhitening (linear in the coils: empty lines stay zero) ──
    noise = getattr(meta, "noise", None)
    if not PREWHITEN:
        facts.set("prewhitened", False, Src.SETTING, "switched off")
    elif noise is None or noise.ndim != 2 or noise.shape[1] != n_coils:
        facts.set("prewhitened", False, Src.NOISE, "no matching noise-adjust data in the file")
    else:
        try:
            W, max_corr = noise_whitener(noise)
        except np.linalg.LinAlgError:
            facts.set("prewhitened", False, Src.NOISE, "noise covariance not positive definite")
        else:
            kspace = np.tensordot(kspace, W, axes=([2], [1]))      # coil c ← Σ_d W[c, d] k_d
            facts.set("prewhitened", True, Src.NOISE,
                      f"{noise.shape[0]} samples; largest coil correlation {max_corr:.2f}")

    # ── Line categories ───────────────────────────────────────────────────────
    centre = n_full // 2
    cat = np.zeros(n_full, dtype=np.uint8)
    cat[rows] |= np.uint8(LineCat.ACQ)
    cat[calib_rows] |= np.uint8(LineCat.CALIB)
    pf_code = kspace_hdr.get("ucPhasePartialFourier", 16)
    pf = facts.set("partial_fourier", _PF_FRACTION.get(pf_code, str(pf_code)), Src.HEADER,
                   "MeasYaps.sKSpace.ucPhasePartialFourier")
    below, above = centre - rows.min(), rows.max() - centre
    pi_lo, pi_hi = 0, n_full
    if pf != "off":
        if above < below:
            pi_hi = int(rows.max()) + 1
        else:
            pi_lo = int(rows.min())
        cat[:pi_lo] |= np.uint8(LineCat.PF_MISSING)
        cat[pi_hi:] |= np.uint8(LineCat.PF_MISSING)
    gaps = (cat & np.uint8(LineCat.ACQ | LineCat.PF_MISSING)) == 0
    cat[gaps] |= np.uint8(LineCat.SKIP_ACCEL)

    # DC cross-check: the strongest acquired line should be the header's centre.
    energy = np.sum(np.abs(kspace) ** 2, axis=(0, 2))
    dc_row = int(np.argmax(energy))
    if abs(dc_row - centre) > 1:
        facts.warnings.append(f"Strongest line {dc_row} is not the k-space centre {centre}.")

    if calib_rows.size:
        calib_range = (int(calib_rows.min()), int(calib_rows.max()) + 1)
    else:
        half = int(min(below, above))
        calib_range = (centre - half, centre + half + 1)

    # ── Phase oversampling and pixel size of the final image ──────────────────
    phase_os = _header_float(hdr, ("Meas", "flPhaseOS")) or 0.0
    phase_os = facts.set("phase_os", max(phase_os, 0.0), Src.HEADER, "Meas.flPhaseOS")
    n_pe_image = min(n_full, int(round(n_full / (1.0 + phase_os))))
    facts.set("n_pe_image", n_pe_image, Src.DERIVED, "lines kept after phase-oversampling crop")
    n_read_image = kspace.shape[0]
    facts.set("n_read_image", n_read_image, Src.DERIVED)
    spacing = _pixel_spacing(yaps, n_read_image, n_pe_image)
    if spacing:
        facts.set("pixel_spacing_mm", spacing, Src.HEADER, "sSliceArray FOV / matrix")

    asm = Assembled(kspace=kspace, line_category=cat, calib_range=calib_range,
                    pi_range=(pi_lo, pi_hi), n_pe_image=n_pe_image,
                    pixel_spacing_mm=spacing, facts=facts, slice_idx=slice_idx)
    r_accel, r_pf, _ = asm.r_parts()
    facts.set("n_pe_acquired", int(asm.pe_mask.sum()), Src.MDH)
    facts.set("accel_r", round(r_accel, 2), Src.DERIVED)
    facts.set("r_partial_fourier", round(r_pf, 3), Src.DERIVED)
    emit(f"  [ACQ] {facts.summary()}")
    for w in facts.warnings:
        emit(f"  [ACQ] WARNING: {w}")
    return asm


# ─────────────────────────────────────────────────────────────────────────────
# Pulseq: line geometry from the played .seq or the scan's trajectory JSON
# ─────────────────────────────────────────────────────────────────────────────

# A JSON line order is replaced by the TSE echo-train plan only when the image
# placed by the plan has this much lower entropy (nats) — a JSON may be shared
# by an SE and a TSE scan and then carry the SE order. On the lab's scans the
# wrong order costs 0.5–0.8 nats.
_ORDER_DECISION_NATS = 0.1


def acquire_pulseq(
    dat_path: str,
    *,
    slice_selector_cb: Optional[Callable[[int], int]] = None,
    slice_idx: Optional[int] = None,
    json_path: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[Assembled]:
    """
    Assemble one slice of a Pulseq scan from the k-space position of each
    readout, taken from (1) the played .seq — found by its signature, or
    located by the user when a request handler is set — else (2) the
    trajectory JSON written with the scan. None when neither is available or
    neither matches the data; the caller then keeps its existing path.
    """
    from . import seq_geometry as sg
    from .utils import ensure_twix_importable

    def emit(msg: str) -> None:
        if log:
            log(msg)

    info = sg.locate_played_seq(dat_path, log)
    json_path = json_path or sg.find_line_json(dat_path)
    if not info["path"]:
        if info["signature"]:
            emit(f"  [ACQ] {info['note']} ({info['name']}).")
        if not json_path:
            emit("  [ACQ] No .seq or trajectory JSON for this Pulseq scan — line placement "
                 "is inferred from the file name and line count.")
            return None

    emit(f"Reading TWIX data: {Path(dat_path).name}")
    ensure_twix_importable()
    from read_siemens_scans.twix_reading import read_raw_twix_data
    raw, _dims = read_raw_twix_data(str(dat_path), fix_oversampling=False)[:2]
    n_slices = 1
    if raw.ndim == 4:                                  # (slices, lines, coils, read)
        n_slices = raw.shape[0]
        if slice_idx is None:
            slice_idx = int(slice_selector_cb(n_slices)) if slice_selector_cb else n_slices // 2
        raw = raw[slice_idx]
        emit(f"  Multi-slice scan ({n_slices} slices) — using slice {slice_idx}")
    if raw.ndim != 3:
        return None
    n_lin, _n_coils, ncol = raw.shape

    sources = []
    if info["path"]:
        sources.append(("seq", info["path"]))
    if json_path:
        sources.append(("json", json_path))
    for kind, path in sources:
        name = Path(path).name
        try:
            if kind == "seq":
                geo = sg.seq_line_geometry(path)
                src, detail = Src.SEQ, f"{name} (signature {info['signature']})"
            else:
                geo = sg.json_line_geometry(path, ncol)
                src, detail = Src.TRAJ_JSON, name
        except ImportError:
            emit("  [ACQ] pypulseq is not installed — the played .seq cannot be read.")
            continue
        except (OSError, ValueError, KeyError) as exc:
            emit(f"  [ACQ] {name}: {exc}")
            continue
        if geo["samples_per_adc"] != ncol:
            emit(f"  [ACQ] {name}: {geo['samples_per_adc']} samples per line, the data has "
                 f"{ncol} — not used.")
            continue
        pe = _per_slice_lines(geo["pe_index"], n_lin, n_slices)
        if pe is None:
            emit(f"  [ACQ] {name}: {geo['pe_index'].size} lines, the data has {n_lin} per "
                 f"slice ({n_slices} slice(s)) — not used.")
            continue
        if src is Src.TRAJ_JSON:
            pe, detail = _check_echo_order(dat_path, raw, pe, detail, emit)
        asm = assemble_pulseq(raw, pe, geo, src=src, detail=detail, slice_idx=slice_idx,
                              n_slices=n_slices, log=log)
        if asm is not None:
            return asm
    emit("  [ACQ] Falling back to line placement inferred from the file name and line count.")
    return None


def assemble_pulseq(
    raw: np.ndarray,
    pe: np.ndarray,
    geo: Dict[str, Any],
    *,
    src: Src,
    detail: str,
    slice_idx: Optional[int] = None,
    n_slices: int = 1,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[Assembled]:
    """
    Place Pulseq readouts (raw: lines × coils × samples, acquisition order) on
    their phase-encode grid. ``pe`` is each readout's phase-encode index (k = 0
    → 0); ``geo`` carries readout oversampling, direction and FOV. None when
    every line is recorded more than once (averages or unlabelled slices).
    """
    from .utils import crop_readout_oversampling

    def emit(msg: str) -> None:
        if log:
            log(msg)

    facts = AcqFacts()
    facts.set("family", "pulseq", Src.HEADER, "Config.SequenceFileName")
    facts.set("geometry_source", src.value, src, detail)
    n_lin, n_coils, ncol = raw.shape
    if n_slices > 1:
        facts.set("n_slices", n_slices, Src.MDH, f"slice {slice_idx} reconstructed")
    facts.set("n_coils", n_coils, Src.MDH)
    facts.set("n_adc_samples", ncol, Src.MDH)

    # ── Repeated readouts of a line: keep the first (later ones are fillers) ──
    pe = np.asarray(pe, dtype=int)
    uniq, first, counts = np.unique(pe, return_index=True, return_counts=True)
    if uniq.size < pe.size and counts.min() > 1:
        emit(f"  [ACQ] Every line is recorded {counts.min()}+ times (averages or slices the "
             f"file does not label) — not assembled from the {src.value}.")
        return None
    if uniq.size < pe.size:
        facts.warnings.append(
            f"{pe.size - uniq.size} repeated readout(s) of line(s) "
            f"{uniq[counts > 1].tolist()[:5]} not used (the first acquisition is kept).")
    keep = np.sort(first)
    pe = pe[keep]

    # ── Readout: k increasing along the array, then remove oversampling ──────
    arr = np.transpose(raw[keep], (2, 0, 1)).astype(np.complex128)     # (read, line, coil)
    reversed_read = bool(geo.get("read_reversed"))
    if reversed_read:
        arr = arr[::-1]
    facts.set("read_reversed", reversed_read, src,
              "samples run from +k to −k; reversed" if reversed_read else "")
    ratio = float(geo.get("readout_os") or 1.0)
    read_os = int(round(ratio))
    if read_os not in (1, 2) or abs(ratio - read_os) > 0.1:
        facts.warnings.append(f"Readout/phase k-step ratio {ratio:.2f} — oversampling not removed.")
        read_os = 1
    facts.set("read_os", read_os, src, f"Δk_phase / Δk_read = {ratio:.3f}")

    # ── Phase-encode grid ─────────────────────────────────────────────────────
    n_read_img = ncol // read_os
    lo, hi = int(pe.min()), int(pe.max())
    span = hi - lo + 1
    steps = np.diff(uniq)
    max_step = int(steps.max()) if steps.size else 1
    one_sided = abs(-lo - (hi + 1)) > max_step + 1     # one half of k-space much shorter
    if one_sided:
        n_full = 2 * max(-lo, hi + 1)
        how = "symmetric about k = 0 around the longer half (partial Fourier)"
    elif span <= n_read_img and n_read_img - span <= max_step + 1:
        n_full = n_read_img
        how = "square pixels: phase-encode FOV / readout resolution"
    else:
        n_full = span + span % 2
        how = "span of the acquired lines"
    offset = n_full // 2
    if lo + offset < 0:
        offset = -lo
    elif hi + offset >= n_full:
        offset = n_full - 1 - hi
    facts.set("n_pe_full", n_full, Src.DERIVED, how)
    facts.set("pe_centre_line", offset, src, "row of k = 0")
    if offset != n_full // 2:
        facts.warnings.append(
            f"k = 0 placed {offset - n_full // 2:+d} line(s) from the grid centre to fit the "
            f"lines (a linear phase; magnitude unaffected).")
    if geo.get("pe_half_line"):
        facts.set("pe_half_line", True, src,
                  "lines sampled half a line off k = 0 (a linear phase; magnitude unaffected)")
    facts.set("swapped", bool(geo.get("swapped")), src, "phase encoding on the x gradient")

    rows = pe + offset
    kspace = np.zeros((ncol, n_full, n_coils), dtype=np.complex128)
    kspace[:, rows, :] = arr
    if read_os > 1:
        kspace = crop_readout_oversampling(kspace, expected_nx=n_read_img,
                                           readout_axis=0, os_factor=read_os)
    facts.set("prewhitened", False, Src.SETTING, "Pulseq scans carry no noise-adjust data")

    # ── Line categories, calibration block ────────────────────────────────────
    cat = np.zeros(n_full, dtype=np.uint8)
    cat[rows] |= np.uint8(LineCat.ACQ)
    pi_lo, pi_hi = 0, n_full
    pf = "off"
    if one_sided:
        if hi + 1 < -lo:
            pi_hi = int(rows.max()) + 1
        else:
            pi_lo = int(rows.min())
        cat[:pi_lo] |= np.uint8(LineCat.PF_MISSING)
        cat[pi_hi:] |= np.uint8(LineCat.PF_MISSING)
        frac = (pi_hi - pi_lo) / n_full
        pf = f"{round(frac * 8)}/8" if abs(frac * 8 - round(frac * 8)) < 0.15 else f"{frac:.2f}"
    facts.set("partial_fourier", pf, src, "acquired extent about k = 0")
    gaps = (cat & np.uint8(LineCat.ACQ | LineCat.PF_MISSING)) == 0
    cat[gaps] |= np.uint8(LineCat.SKIP_ACCEL)
    inner_gaps = gaps[int(rows.min()):int(rows.max()) + 1].any()
    if inner_gaps:
        run = _longest_run(np.unique(rows), centre=offset)
        cat[run] |= np.uint8(LineCat.CALIB)
        calib_range = (int(run[0]), int(run[-1]) + 1)
    else:
        half = int(min(offset - rows.min(), rows.max() - offset))
        calib_range = (offset - half, offset + half + 1)

    energy = np.sum(np.abs(kspace) ** 2, axis=(0, 2))
    dc_row = int(np.argmax(energy))
    if abs(dc_row - offset) > 1:
        facts.warnings.append(f"Strongest line {dc_row} is not the k-space centre {offset}.")

    spacing = None
    fov_pe, fov_read_os = geo.get("fov_phase_mm"), geo.get("fov_read_os_mm")
    if fov_pe and fov_read_os:
        spacing = (round(fov_read_os / ncol, 4), round(fov_pe / n_full, 4))
        facts.set("pixel_spacing_mm", spacing, src, "FOV from the k-space step / matrix")
    facts.set("n_read_image", kspace.shape[0], Src.DERIVED)

    asm = Assembled(kspace=kspace, line_category=cat, calib_range=calib_range,
                    pi_range=(pi_lo, pi_hi), n_pe_image=n_full, pixel_spacing_mm=spacing,
                    facts=facts, slice_idx=slice_idx)
    r_accel, r_pf, _ = asm.r_parts()
    facts.set("n_pe_acquired", int(asm.pe_mask.sum()), src)
    facts.set("accel_r", round(r_accel, 2), Src.DERIVED)
    facts.set("r_partial_fourier", round(r_pf, 3), Src.DERIVED)
    emit(f"  [ACQ] Line placement from the {src.value}: {detail}")
    emit(f"  [ACQ] {facts.summary()}")
    for w in facts.warnings:
        emit(f"  [ACQ] WARNING: {w}")
    return asm


def _per_slice_lines(pe: np.ndarray, n_lin: int, n_slices: int) -> Optional[np.ndarray]:
    """The phase-encode index of each of one slice's n_lin readouts, or None
    when the geometry does not describe this data."""
    pe = np.asarray(pe)
    if pe.size == n_lin:
        return pe
    if n_slices > 1 and pe.size == n_lin * n_slices:
        per_line = pe.reshape(n_lin, n_slices)         # slices interleaved within a line
        if np.all(per_line == per_line[:, :1]):
            return per_line[:, 0]
        per_slice = pe.reshape(n_slices, n_lin)        # one slice after the other
        if np.all(per_slice == per_slice[:1]):
            return per_slice[0]
    return None


def _longest_run(rows: np.ndarray, centre: int) -> np.ndarray:
    """Longest run of consecutive rows (ties: the one nearest the centre)."""
    breaks = np.where(np.diff(rows) != 1)[0] + 1
    runs = np.split(rows, breaks)
    return max(runs, key=lambda r: (r.size, -abs((r[0] + r[-1]) / 2 - centre)))


def image_entropy(lines: np.ndarray, pe: np.ndarray) -> float:
    """
    Entropy (nats) of the root-sum-of-squares image when the readouts
    (lines × coils × samples) are placed at ``pe`` — the usual autofocus
    measure: a wrong line order spreads signal into ghosts and raises it.
    Only comparable between placements of the same lines.
    """
    from .imaging import ifft2c
    uniq, first = np.unique(np.asarray(pe, dtype=int), return_index=True)
    n = 2 * max(-int(uniq.min()), int(uniq.max()) + 1)
    k = np.zeros((lines.shape[2], n, lines.shape[1]), dtype=np.complex128)
    k[:, uniq + n // 2, :] = np.transpose(lines[first], (2, 0, 1))
    img = np.sqrt(np.sum(np.abs(ifft2c(k)) ** 2, axis=2)).ravel()
    p = img[img > 0] / img.sum()
    return float(-np.sum(p * np.log(p)))


def _check_echo_order(dat_path, raw, pe, detail, emit):
    """
    A trajectory JSON can be shared by an SE and a TSE scan of the same
    pattern and then carry the SE order. For a TSE scan, compare it with the
    echo-train plan on the same lines and keep the JSON unless the plan gives
    a clearly sharper image (lower entropy).
    """
    from .tse_reorder import detect_tse_params, tse_acq_to_line
    tse = detect_tse_params(dat_path)
    if tse is None:
        return pe, detail
    n = 2 * max(-int(pe.min()), int(pe.max()) + 1)
    if n % tse["necho"]:
        return pe, detail
    mask = np.zeros(n, dtype=bool)
    mask[pe + n // 2] = True
    plan = tse_acq_to_line(n, tse["necho"], tse["TE1"], tse["TEeff"], pe_mask=mask) - n // 2
    if plan.size != pe.size or np.array_equal(plan, pe) \
            or not np.array_equal(np.sort(plan), np.sort(pe)):
        return pe, detail
    h_json, h_plan = image_entropy(raw, pe), image_entropy(raw, plan)
    if h_json - h_plan > _ORDER_DECISION_NATS:
        emit(f"  [ACQ] The JSON line order is not this TSE scan's echo order (image entropy "
             f"{h_json:.2f} vs {h_plan:.2f} with the echo-train plan) — using the plan.")
        return plan, f"{detail} (lines), TSE echo-train plan (order)"
    return pe, detail


def espirit_calib_width(asm: Assembled, wanted: int) -> int:
    """
    ESPIRiT calibration width for this grid. BART ecalib and sigpy
    EspiritCalib both calibrate on the region centred on the grid, rows
    (n − w)//2 … + w; it must hold measured calibration lines only, so this is
    the widest w ≤ wanted whose centred region lies inside ``calib_range``
    (a block that is off-centre by a line — some Pulseq grids — narrows it).
    """
    n = len(asm.line_category)
    lo, hi = asm.calib_range
    top = min(int(wanted), hi - lo)
    for w in range(top, 0, -1):
        start = (n - w) // 2
        if start >= lo and start + w <= hi:
            return w
    return top


def finalize_image(img: np.ndarray, asm: Optional[Assembled], axis: int = 1) -> np.ndarray:
    """Crop phase oversampling so the image shows the prescribed field of view."""
    if asm is None:
        return img
    return crop_center(img, asm.n_pe_image, axis)


def result_extras(asm: Optional[Assembled]) -> Dict[str, Any]:
    """Keys every pipeline adds to its result when the layer was used."""
    if asm is None:
        return {}
    r_accel, r_pf, _ = asm.r_parts()
    return {
        "acq_facts": asm.facts.to_dict(),
        "acq_summary": asm.facts.summary(),
        "line_category": asm.line_category.copy(),
        "pixel_spacing_mm": asm.pixel_spacing_mm,
        "geometry": {"n_read": int(asm.kspace.shape[0]), "n_pe_full": len(asm.line_category),
                     "n_pe_image": asm.n_pe_image, "pi_range": asm.pi_range,
                     "calib_range": asm.calib_range},
        "r_accel": r_accel,
        "r_pf": r_pf,
    }


def result_summary_line(result: Dict[str, Any]) -> Optional[str]:
    """
    One line describing a result's acquisition, shared by the result tab and
    the PDF, e.g. 'PF 7/8 · 235/269 lines · R 1 · phase OS 0% · readout OS 2×
    · partial Fourier: zerofill (34 lines zero) · pixel 0.417×0.595 mm'.
    None when the result was not built from the file's metadata (Pulseq).
    """
    summary = result.get("acq_summary")
    if not summary:
        return None
    parts = [summary]
    pf = result.get("pf_method")
    if pf and not str(pf).startswith("none"):
        parts.append(f"partial Fourier: {pf}")
    spacing = result.get("pixel_spacing_mm")
    if spacing:
        parts.append(f"pixel {spacing[0]:.3f}×{spacing[1]:.3f} mm")
    return " · ".join(parts)


def protocol_signature(dat_path: str, slice_idx: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    What makes two Siemens acquisitions comparable, read from the header:
    sequence program, TE(s)/TR, slice position/orientation/thickness and FOV.
    None for Pulseq (placeholder protocol) and unreadable files — those cannot
    be judged from the header.
    """
    from .utils import read_twix_header, sequence_family
    if sequence_family(dat_path) != "siemens":
        return None
    hdr = read_twix_header(dat_path) or {}
    yaps = hdr.get("MeasYaps", {}) or {}
    try:
        slices = yaps["sSliceArray"]["asSlice"]
        sl = slices[slice_idx if slice_idx is not None and slice_idx < len(slices) else 0]
    except (KeyError, IndexError, TypeError):
        return None

    def vec(d):   # twixtools omits zero components
        d = d or {}
        return tuple(float(d.get(k, 0.0)) for k in ("dSag", "dCor", "dTra"))

    n_te = int(yaps.get("lContrasts", 1) or 1)
    return {
        "sequence": str(hdr.get("Config", {}).get("SequenceFileName", "")).split("\\")[-1],
        "te_ms": tuple(t / 1000.0 for t in (yaps.get("alTE") or [])[:n_te]),
        "tr_ms": float((yaps.get("alTR") or [0])[0]) / 1000.0,
        "position_mm": vec(sl.get("sPosition")),
        "normal": vec(sl.get("sNormal")),
        "thickness_mm": float(sl.get("dThickness", 0.0)),
        "fov_mm": (float(sl.get("dReadoutFOV", 0.0)), float(sl.get("dPhaseFOV", 0.0))),
    }


def acquisition_mismatch(recon_dat: str, reference_dat: str,
                         slice_idx: Optional[int] = None) -> Optional[List[str]]:
    """
    Reasons the reference is a different acquisition than the recon ([] when
    they match), or None when either header cannot be trusted (Pulseq,
    unreadable) — then the caller cannot judge from the header.
    """
    a = protocol_signature(recon_dat, slice_idx)
    b = protocol_signature(reference_dat, slice_idx)
    if a is None or b is None:
        return None
    issues = []
    if a["sequence"] != b["sequence"]:
        issues.append(f"different sequence ({a['sequence']} vs {b['sequence']})")
    if (len(a["te_ms"]) != len(b["te_ms"])
            or not np.allclose(a["te_ms"], b["te_ms"], rtol=0.01, atol=0.05)):
        issues.append(f"different TE ({_fmt(a['te_ms'])} vs {_fmt(b['te_ms'])} ms)")
    if not np.isclose(a["tr_ms"], b["tr_ms"], rtol=0.01, atol=0.05):
        issues.append(f"different TR ({a['tr_ms']:g} vs {b['tr_ms']:g} ms)")
    if (not np.allclose(a["position_mm"], b["position_mm"], atol=0.5)
            or not np.allclose(a["normal"], b["normal"], atol=1e-3)
            or not np.isclose(a["thickness_mm"], b["thickness_mm"], atol=0.05)):
        issues.append("different slice position/orientation/thickness")
    if not np.allclose(a["fov_mm"], b["fov_mm"], atol=0.5):
        issues.append(f"different FOV ({_fmt(a['fov_mm'])} vs {_fmt(b['fov_mm'])} mm)")
    return issues


def _fmt(values) -> str:
    return "/".join(f"{v:g}" for v in values)


def _header_float(hdr: dict, *paths: Tuple[str, str]) -> Optional[float]:
    for section, key in paths:
        try:
            return float(hdr[section][key])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _pixel_spacing(yaps: dict, n_read: int, n_pe: int) -> Optional[Tuple[float, float]]:
    try:
        sl = yaps["sSliceArray"]["asSlice"][0]
        return (round(float(sl["dReadoutFOV"]) / n_read, 4),
                round(float(sl["dPhaseFOV"]) / n_pe, 4))
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
        return None


def _merge_refscan(kspace, meta, slice_idx, offset, col_off, ncol, n_col_full, read_os):
    """Add integrated-reference readouts that are not already image lines."""
    from .utils import crop_readout_oversampling
    ref = meta.refscan
    ref_axes = list(meta.refscan_axes)
    if "Sli" in ref_axes:
        ref = np.take(ref, slice_idx, axis=ref_axes.index("Sli"))
        ref_axes.remove("Sli")
        ref_lin = meta.refscan_lines["Lin"][meta.refscan_lines["Sli"] == slice_idx]
    else:
        ref_lin = meta.refscan_lines["Lin"]
    if "Cha" not in ref_axes:
        ref = np.expand_dims(ref, axis=len(ref_axes))
        ref_axes.append("Cha")
    ref = np.transpose(ref, (ref_axes.index("Col"), ref_axes.index("Lin"), ref_axes.index("Cha")))
    if ref.shape[0] != ncol:
        raise AcquisitionConflict("Reference-scan readout length differs from the image's.")
    ref_lin = np.unique(ref_lin)
    rows = ref_lin + offset
    acquired = np.sum(np.abs(kspace) ** 2, axis=(0, 2)) > 0
    new = ~acquired[rows]
    block = np.zeros((n_col_full, rows.size, ref.shape[2]), dtype=np.complex128)
    lo, hi = max(0, -col_off), min(ncol, n_col_full - col_off)
    block[col_off + lo:col_off + hi] = ref[lo:hi][:, ref_lin, :]
    if read_os > 1:
        block = crop_readout_oversampling(block, expected_nx=n_col_full // read_os,
                                          readout_axis=0, os_factor=read_os)
    kspace[:, rows[new], :] = block[:, new, :]
    return kspace, rows
