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
"3D"), so Pulseq and unreadable files are not assembled here:
``acquire_cartesian`` returns None and callers keep their existing path.
"""

from dataclasses import dataclass
from enum import Enum, IntFlag
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .imaging import crop_center


class Src(str, Enum):
    """Where an acquisition fact came from."""
    HEADER = "protocol header"
    MDH = "per-line header"
    DATA = "inferred from data"
    DERIVED = "derived"


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
    log: Optional[Callable[[str], None]] = None,
) -> Optional[Assembled]:
    """
    Read a Siemens Cartesian 2-D file and assemble one slice on the full grid.

    Returns None for Pulseq files and files whose header cannot be read — the
    caller then keeps its existing path. Raises UnsupportedAcquisition /
    AcquisitionConflict when the file cannot be reconstructed correctly.
    """
    from .utils import ensure_twix_importable, sequence_family
    if sequence_family(dat_path) != "siemens":
        return None
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
    n_col_full = 2 * max(centre_col, ncol - centre_col)
    col_off = n_col_full // 2 - centre_col
    facts.set("n_adc_samples", ncol, Src.MDH, "SamplesInScan")
    facts.set("asymmetric_echo", n_col_full != ncol, Src.MDH,
              f"CenterCol {centre_col} of {ncol} samples")
    # The reader's line axis is indexed by the Lin counter, so array row
    # (row - offset) holds full-grid row `row`.
    kspace = np.zeros((n_col_full, n_full, n_coils), dtype=np.complex128)
    kspace[col_off:col_off + ncol, rows, :] = arr[:, rows - offset, :]

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
        return float(sl["dReadoutFOV"]) / n_read, float(sl["dPhaseFOV"]) / n_pe
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
    block[col_off:col_off + ncol] = ref[:, ref_lin, :]
    if read_os > 1:
        block = crop_readout_oversampling(block, expected_nx=n_col_full // read_os,
                                          readout_axis=0, os_factor=read_os)
    kspace[:, rows[new], :] = block[:, new, :]
    return kspace, rows
