"""
Shared utilities: CFL I/O, BART runner, trajectory helpers, scan-type detection.
BART only runs on Linux/WSL. On native Windows, run_bart() bridges calls into
WSL and translates Windows paths to /mnt/<drive>/... (see _bart_command).
"""

import functools
import json
import logging as _logging
import os
import re
import subprocess
import sys
import tempfile
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np


def ensure_twix_importable() -> None:
    """
    Make the sibling ``read_siemens_scans`` package importable.

    The TWIX reader lives next to AnalyzeMRIDataApp/ (not installed as a
    package); app.py adds it to sys.path at startup, but recon modules must
    also work when imported directly (tests, batch_report, scripts).
    """
    root = Path(__file__).resolve().parent.parent.parent
    twix_dir = str(root / "read_siemens_scans")
    if twix_dir not in sys.path:
        sys.path.insert(0, twix_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Scan-type enum & auto-detection
# ─────────────────────────────────────────────────────────────────────────────

class ScanType(Enum):
    BAND = auto()
    FULLY_SAMPLED = auto()
    EQUISPACED = auto()
    POISSON = auto()
    RETRO_UNDERSAMPLE = auto()
    UNKNOWN = auto()


# Regex patterns matched against the stem (filename without extension)
_PATTERNS = [
    (ScanType.FULLY_SAMPLED, re.compile(r"fully_sampled",  re.IGNORECASE)),
    (ScanType.EQUISPACED,    re.compile(r"equispaced",     re.IGNORECASE)),
    (ScanType.POISSON,       re.compile(r"poisson",        re.IGNORECASE)),
    # band_se_<n> / band_tse_<n> — any non-negative index (not limited to 0–15)
    (ScanType.BAND,          re.compile(r"band_(t?se)_\d+$", re.IGNORECASE)),
]


def auto_detect_scan_type(filename: str) -> ScanType:
    """Return the ScanType for a given .dat filename."""
    stem = Path(filename).stem
    for stype, pat in _PATTERNS:
        if pat.search(stem):
            return stype
    return ScanType.UNKNOWN


def extract_band_index(filename: str) -> Optional[int]:
    """Extract the band index from a band filename (e.g. band_se_3 → 3)."""
    m = re.search(r"band_t?se_(\d+)", Path(filename).stem, re.IGNORECASE)
    return int(m.group(1)) if m else None


def read_twix_header(dat_path: str) -> Optional[dict]:
    """
    Protocol header of the last measurement in a TWIX file, or None if unreadable.

    Parses just the ASCII protocol (``parse_data=False``, ~0.1–0.3 s) and is
    cached per (path, modification time, size), so every caller shares one read.
    Treat the returned dict as read-only.
    """
    try:
        st = os.stat(dat_path)
    except OSError:
        return None
    return _read_twix_header_cached(str(dat_path), st.st_mtime_ns, st.st_size)


@functools.lru_cache(maxsize=64)
def _read_twix_header_cached(dat_path: str, _mtime_ns: int, _size: int) -> Optional[dict]:
    try:
        import twixtools
        meas = twixtools.read_twix(
            dat_path, parse_data=False, parse_pmu=False,
            parse_geometry=False, verbose=False,
        )
        return meas[-1]["hdr"]
    except Exception:
        return None


def sequence_family(dat_path: str) -> Optional[str]:
    """
    Which kind of sequence produced a TWIX file: "pulseq", "siemens", or None.

    Decided from the sequence program that ran (``Config.SequenceFileName``,
    e.g. ``%CustomerSeq%\\pulseq`` vs ``%SiemensSeq%\\tse``). Pulseq protocol
    parameters (TR, matrix, oversampling, ...) are interpreter placeholders,
    but this field is always true. "siemens" covers every non-Pulseq sequence:
    those label each line with its true phase-encode index. None when the
    header cannot be read (callers keep their previous behaviour).
    """
    hdr = read_twix_header(dat_path)
    if hdr is None:
        return None
    for section, key in (("Config", "SequenceFileName"),
                         ("Dicom", "tSequenceFileName"),
                         ("Meas", "tSequenceFileName")):
        try:
            name = str(hdr[section][key])
        except (KeyError, TypeError):
            continue
        if name:
            return "pulseq" if "pulseq" in name.lower() else "siemens"
    return None


def probe_num_slices(dat_path: str) -> int:
    """
    Number of slices in a TWIX file, read cheaply from the protocol header only.

    The slice count is ``sSliceArray.lSize`` (equivalently ``lSliceArraySize``).
    Returns 1 when the header has no slice field or cannot be read.
    """
    try:
        hdr = read_twix_header(dat_path)
        if hdr is None:
            return 1
        # Preferred: nested MeasYaps → sSliceArray.lSize
        n = (hdr.get("MeasYaps", {}) or {}).get("sSliceArray", {}).get("lSize")
        if not n:
            # Fallbacks: flat Meas/Dicom keys, then a raw-text regex.
            for sect in ("Meas", "Dicom"):
                v = (hdr.get(sect, {}) or {}).get("lSliceArraySize")
                if v:
                    n = v
                    break
        if not n:
            m = re.search(r"sSliceArray\.lSize\s*=\s*(\d+)",
                          hdr.get("MeasYaps_raw", "") or "")
            n = int(m.group(1)) if m else 1
        return max(1, int(n))
    except Exception:
        return 1


def find_generic_traj_json(dat_path: str) -> Optional[str]:
    """
    Find a trajectory JSON for an UNKNOWN (non-catalogued) scan.

    Convention: place either a ``<scanname>.json`` (same stem as the .dat) or a
    ``Trajectory.json`` next to the .dat file or in its parent folder — the
    format is the same {index: [[kx...],[ky...]]} JSON the MATLAB sequence
    builder writes (Bands.json / PoissonDisc.json), read with index "0".

    The search is deliberately narrow so a Bands.json / PoissonDisc.json meant
    for another scan is never mis-paired with an unknown one. Returns the first
    match or None.
    """
    dat = Path(dat_path)
    stem = dat.stem
    for folder in (dat.parent, dat.parent.parent):
        candidate = folder / f"{stem}.json"
        if candidate.is_file():
            return str(candidate)
    for folder in (dat.parent, dat.parent.parent):
        for candidate in sorted(folder.glob("[Tt]rajectory.json")):
            if candidate.is_file():
                return str(candidate)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CFL / BART helpers
# ─────────────────────────────────────────────────────────────────────────────

def writecfl(name: str, array: np.ndarray) -> None:
    """Write a NumPy array to BART .cfl + .hdr format (Fortran-order)."""
    arr = np.asfortranarray(np.asarray(array, dtype=np.complex64))
    dims = list(arr.shape) + [1] * (16 - len(arr.shape))
    with open(name + ".hdr", "w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(map(str, dims)) + "\n")
    np.ravel(arr, order="F").view(np.float32).tofile(name + ".cfl")


def readcfl(name: str) -> np.ndarray:
    """Read a BART .cfl + .hdr file into a NumPy complex64 array."""
    with open(name + ".hdr", "r") as f:
        f.readline()  # '# Dimensions'
        dims = list(map(int, f.readline().split()))
    # Strip trailing 1s
    while dims and dims[-1] == 1:
        dims.pop()
    data = np.fromfile(name + ".cfl", dtype=np.complex64)
    return np.reshape(data, dims, order="F")


# True when running as a native Windows process (incl. a frozen PyInstaller exe).
_IS_WINDOWS = os.name == "nt"


def _no_window_kwargs() -> dict:
    """
    subprocess kwargs that stop a child console window from flashing on Windows.

    On the native-Windows build every BART call launches ``wsl.exe`` (a console
    program); from a ``--windowed`` GUI process each launch pops a black console
    window open and closed. CREATE_NO_WINDOW (plus a hidden STARTUPINFO) suppress
    it. No-op on Linux/macOS.
    """
    if not _IS_WINDOWS:
        return {}
    kwargs: dict = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    try:
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0  # SW_HIDE
        kwargs["startupinfo"] = si
    except Exception:
        pass
    return kwargs

# Matches a Windows absolute path (drive letter + separator), e.g. C:\ or C:/ .
# Deliberately requires a slash after the colon so BART regularization args such
# as "W:7:0:0.01" or grid specs like "-x128:128:1" are NOT mistaken for paths.
_WIN_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _bart_command(args) -> list:
    """
    Build the argv used to launch BART.

    On Linux/WSL, BART is on PATH and invoked directly. On native Windows there
    is no bart.exe, so the call is bridged into WSL and any Windows absolute-path
    arguments (the temp .cfl files) are translated to ``/mnt/<drive>/…`` so BART
    inside WSL can read/write the same files.

    BART is invoked directly (not via a login shell) so its stdout stays clean —
    commands like ``bitmask`` are parsed from stdout, and sourcing ~/.bashrc
    (conda banners, MOTD, etc.) could corrupt it. Because BART is usually NOT on
    the bare WSL PATH, set MRI_BART_BIN to its absolute WSL path, e.g.
    ``/home/<user>/bart/bart``. A source-built BART run by absolute path derives
    its own toolbox path, so no BART_TOOLBOX_PATH is normally needed.

    Overridable via environment:
      MRI_WSL_EXE  — the WSL launcher (default "wsl.exe")
      MRI_BART_BIN — the BART executable name/path inside WSL (default "bart")
    """
    str_args = [str(a) for a in args]
    if _IS_WINDOWS:
        translated = [win_to_wsl(a) if _WIN_PATH_RE.match(a) else a for a in str_args]
        wsl_exe = os.environ.get("MRI_WSL_EXE", "wsl.exe")
        bart_bin = os.environ.get("MRI_BART_BIN", "bart")
        return [wsl_exe, bart_bin, *translated]
    # MRI_BART_BIN is honoured on every platform, so a BART built outside PATH
    # (e.g. ~/bart/bart) can be used without shell profile changes.
    return [os.environ.get("MRI_BART_BIN", "bart"), *str_args]


def run_bart(*args, check: bool = True) -> str:
    """
    Run a BART command. On Linux/WSL BART is called directly; on native Windows
    the call is bridged into WSL (see :func:`_bart_command`).

    Returns stdout as a string; raises RuntimeError on failure (if check=True).
    """
    cmd = _bart_command(args)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, **_no_window_kwargs()
        )
    except FileNotFoundError as exc:
        # wsl.exe (or bart) not found — give an actionable message rather than a
        # bare FileNotFoundError, since this is the common Windows setup issue.
        if _IS_WINDOWS:
            raise RuntimeError(
                "Could not launch BART via WSL. This build routes BART calls "
                "through WSL (there is no native Windows BART). Ensure WSL is "
                "installed and a distro has BART on its PATH:\n"
                "  wsl --install         (once, then reboot)\n"
                "  # inside WSL: install/build BART so `bart version` works\n"
                f"  (launcher tried: {cmd[0]!r})"
            ) from exc
        raise RuntimeError(
            "Could not launch BART. Ensure BART is installed and on PATH "
            "(`bart version` should work)."
        ) from exc
    if check and result.returncode != 0:
        raise RuntimeError(
            f"BART command failed:\n  cmd : {' '.join(cmd)}\n  err : {result.stderr.strip()}"
        )
    return result.stdout


def bart_bitmask(*dims) -> str:
    """Return the integer bitmask string for a set of BART dimension indices."""
    out = run_bart("bitmask", *dims)
    return out.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_trajectory(json_path: str, idx: str) -> np.ndarray:
    """
    Load one trajectory from a BART JSON file.

    Parameters
    ----------
    json_path : str
        Path to Bands.json / PoissonDisc.json / Undersampled4equispaces.json
    idx : str
        Key in the JSON (e.g. "0", "1", …).

    Returns
    -------
    trajectory : np.ndarray, shape (N, 2)
        Raw kx/ky trajectory samples in scanner units.
    """
    with open(json_path, "r") as f:
        all_data = json.load(f)
    traj_list = all_data[str(idx)]
    # JSON stores trajectory as (coord_dim, N_samples) — same convention as the
    # notebook: `band_0_traj_np = np.array(...[idx]); trajectory = band_0_traj_np.T`
    traj_np = np.array(traj_list).T        # (N_samples, coord_dim)
    return traj_np[:, :2]                  # keep kx, ky only → (N_samples, 2)


def scale_trajectory(traj: np.ndarray) -> np.ndarray:
    """
    Normalise trajectory to the [-0.5, 0.5) range expected by BART NUFFT.

    Parameters
    ----------
    traj : (N, 2)  raw kx/ky samples

    Returns
    -------
    scaled : (N, 2)
    """
    max_val = np.max(np.abs(traj))
    return traj / (2.0 * max_val)


def make_bart_trajectory(
    traj_raw: np.ndarray,
    dim1: int,
    dim2: int,
    *,
    N_grid: int,
    margin: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Build the 3×dim1×dim2 BART trajectory array from a raw (N, 2) trajectory,
    correctly scaled to BART grid-index units.

    BART nufft expects trajectory coordinates in [-N/2, N/2).
    We scale so that the maximum trajectory sample maps to N/2 - margin,
    staying safely inside the valid range.

    Parameters
    ----------
    traj_raw : (N, 2)  raw kx/ky samples in physical units (e.g. m⁻¹)
    dim1, dim2 : int   k-space dimensions (readout lines × readout samples)
    N_grid : int       BART oversampled grid size (must match -x arg to bart nufft)
    margin : float     safety margin in grid pixels (default 1.0)

    Returns
    -------
    bart_traj : (3, dim1, dim2) BART trajectory array
    pixel_size : float  physical size of one reconstructed pixel (same units as 1/traj_raw)
    """
    k_max = np.max(np.abs(traj_raw))
    scale = (N_grid / 2.0 - margin) / k_max
    traj_bart = traj_raw * scale                                            # (N, 2)
    traj_3d = np.hstack([traj_bart, np.zeros((traj_bart.shape[0], 1))])    # (N, 3)
    bart_traj = traj_3d.T.reshape(3, dim1, dim2)                           # (3, dim1, dim2)
    pixel_size = (1.0 / (2.0 * k_max)) * (N_grid / 2.0 - margin) / (N_grid / 2.0)
    return bart_traj, pixel_size


# ─────────────────────────────────────────────────────────────────────────────
# Temp-dir context manager for BART intermediate files
# ─────────────────────────────────────────────────────────────────────────────

class BartTempDir:
    """
    Context manager that creates a temporary directory for BART .cfl/.hdr files
    and removes it on exit.

    Usage::

        with BartTempDir() as tmp:
            writecfl(f"{tmp}/traj", traj_array)
            run_bart("nufft", "-a", f"{tmp}/traj", ...)
    """

    def __init__(self, prefix: str = "bart_mri_"):
        self._prefix = prefix
        self._dir: Optional[tempfile.TemporaryDirectory] = None

    def __enter__(self) -> str:
        self._dir = tempfile.TemporaryDirectory(prefix=self._prefix)
        return self._dir.name

    def __exit__(self, *_) -> None:
        if self._dir is not None:
            self._dir.cleanup()


def win_to_wsl(path: str) -> str:
    """
    Convert a Windows path (C:\\Users\\...) to a WSL path (/mnt/c/Users/...).
    Already-WSL paths are returned unchanged.
    """
    path = str(path)
    if path.startswith("/"):
        return path                         # already WSL-style
    path = path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":  # C:/...
        drive = path[0].lower()
        return f"/mnt/{drive}" + path[2:]
    return path


_os_logger = _logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Readout oversampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def readout_os_factor_from_header(dat_path: str) -> Optional[int]:
    """
    Readout oversampling factor recorded in a TWIX header (1 or 2), or None.

    Siemens sequences store it as ``flReadoutOSFactor``. Pulseq headers always
    carry the interpreter's placeholder (1.0) whatever the .seq actually
    sampled — e.g. 640 samples for a 320 matrix — so for Pulseq this returns
    None and the caller falls back to inferring the factor from the data.
    """
    if sequence_family(dat_path) != "siemens":
        return None
    hdr = read_twix_header(dat_path)
    for section in ("Meas", "Dicom"):
        try:
            factor = float(hdr[section]["flReadoutOSFactor"])
        except Exception:
            continue
        if factor in (1.0, 2.0):
            return int(factor)
    return None


def crop_readout_oversampling(
    kspace: np.ndarray,
    expected_nx: int,
    readout_axis: int,
    os_factor: Optional[int] = None,
    strict: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
    dat_path: Optional[str] = None,
) -> np.ndarray:
    """
    Crop readout oversampling from a raw k-space array.

    Parameters
    ----------
    kspace : np.ndarray
        Raw k-space; shape is typically (Nacq, Ncoils, Nreadout).
    expected_nx : int
        Expected readout size after cropping (= the prescribed matrix Nx).
        Only used to guess the OS factor when the TWIX header cannot.
    readout_axis : int
        Array axis that corresponds to the readout dimension.
    os_factor : int or None
        Oversampling factor.  ``None`` (default) → read it from the TWIX
        header of ``dat_path``; if unavailable, auto-detect via
        ``round(actual_nread / expected_nx)``.  Only 1 (no-op) and 2
        (central-half crop) are supported.
    strict : bool
        If True, raise ``ValueError`` when the resolved factor is not 1 or 2.
        If False (default), raise only for factors > 2; for ambiguous small
        values log a warning and return the array unchanged.
    log_fn : callable(str) or None
        Progress-callback used by the calling recon module.  When provided,
        OS decisions are reported through it *and* the module logger.
    dat_path : str or None
        TWIX file the array was read from. Its header gives the true OS
        factor; guessing from ``expected_nx`` (usually the PE line count)
        fails on non-square, partial-Fourier or phase-oversampled scans
        (e.g. readout 768 vs 235 PE lines → "factor 3").

    Returns
    -------
    kspace_cropped : np.ndarray  (same dtype and number of dims)
    """
    actual_nread = kspace.shape[readout_axis]
    source = "caller"
    if os_factor is None and dat_path is not None:
        hdr_factor = readout_os_factor_from_header(str(dat_path))
        # A header claiming 2x on an odd-length readout cannot be right.
        if hdr_factor == 2 and actual_nread % 2:
            hdr_factor = None
        if hdr_factor is not None:
            os_factor, source = hdr_factor, "TWIX header"
    if os_factor is None:
        os_factor = round(actual_nread / expected_nx)
        source = f"expected_nx={expected_nx}"

    def _emit(msg: str) -> None:
        _os_logger.info(msg)
        if log_fn is not None:
            log_fn(msg)

    if os_factor == 1:
        _emit(f"[OS] readout={actual_nread}, OS factor=1 (from {source}) — no crop.")
        return kspace

    if os_factor == 2:
        # Readout oversampling lives in the IMAGE domain: the 2x ADC samples
        # give a 2x-wider readout FOV at full resolution, while k-space still
        # spans the full k-extent. The crop must therefore be done in the
        # image domain (IFFT readout -> crop central FOV -> FFT back).
        #
        # Chopping the central Nx samples out of k-space (the previous
        # implementation) instead discarded the outer half of a fully
        # populated k-space, which halved the resolution AND sharply
        # truncated real signal -> Gibbs ringing on every sharp edge.
        #
        # Crop to actual_nread // os_factor — the TRUE de-oversampled readout
        # size — NOT to expected_nx. `expected_nx` is only a hint used to
        # *detect* the OS factor; on non-square Siemens scans the real readout
        # matrix differs from the PE matrix that callers pass as expected_nx
        # (e.g. readout 532/512 vs PE 240). Cropping the image to expected_nx
        # would then shrink the readout FOV below the object and wrap it
        # (aliasing). actual_nread // os_factor reproduces the scanner's own
        # OS removal exactly (532 -> 266, 512 -> 256).
        cropped_nx = actual_nread // os_factor
        start = actual_nread // 2 - cropped_nx // 2
        end   = start + cropped_nx
        slices = [slice(None)] * kspace.ndim
        slices[readout_axis] = slice(start, end)

        ax = readout_axis
        img = np.fft.fftshift(
            np.fft.ifft(np.fft.ifftshift(kspace, axes=ax), axis=ax),
            axes=ax,
        )
        img = img[tuple(slices)]
        result = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(img, axes=ax), axis=ax),
            axes=ax,
        ).astype(kspace.dtype, copy=False)
        _emit(
            f"[OS] readout {actual_nread} → {cropped_nx} "
            f"(OS factor={os_factor} from {source}, "
            f"axis={readout_axis}, image-domain crop)."
        )
        return result

    msg = (
        f"[OS] Unsupported oversampling factor={os_factor} "
        f"(actual_nread={actual_nread}, expected_nx={expected_nx}). "
        f"Only factors 1 and 2 are supported."
    )
    if strict or os_factor > 2:
        raise ValueError(msg)
    _emit(f"WARNING: {msg}  Proceeding without crop.")
    return kspace
