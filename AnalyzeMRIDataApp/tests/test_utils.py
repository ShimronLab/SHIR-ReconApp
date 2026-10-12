"""Unit tests for processing.utils: CFL I/O, path translation, scan detection,
readout-oversampling crop."""

import numpy as np
import pytest

from processing.utils import (
    ScanType,
    auto_detect_scan_type,
    crop_readout_oversampling,
    extract_band_index,
    readcfl,
    win_to_wsl,
    writecfl,
)


# ── CFL round-trip ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", [(8,), (8, 6), (8, 6, 4), (8, 6, 1, 4), (8, 6, 1, 4, 2)])
def test_writecfl_readcfl_roundtrip(tmp_path, shape):
    rng = np.random.default_rng(0)
    arr = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    name = str(tmp_path / "data")
    writecfl(name, arr)
    back = readcfl(name)
    # readcfl strips trailing singleton dims, so compare on squeezed content.
    assert np.allclose(np.squeeze(back), np.squeeze(arr), atol=1e-6)


def test_writecfl_header_dims(tmp_path):
    arr = np.zeros((4, 5, 1, 3), dtype=np.complex64)
    name = str(tmp_path / "k")
    writecfl(name, arr)
    with open(name + ".hdr") as f:
        f.readline()
        dims = list(map(int, f.readline().split()))
    assert dims[:4] == [4, 5, 1, 3]
    assert len(dims) == 16 and all(d == 1 for d in dims[4:])


# ── Windows→WSL path translation ─────────────────────────────────────────────

@pytest.mark.parametrize("win,wsl", [
    (r"C:\Users\x\a.cfl", "/mnt/c/Users/x/a.cfl"),
    ("C:/Users/x/a.cfl", "/mnt/c/Users/x/a.cfl"),
    ("D:\\data\\scan", "/mnt/d/data/scan"),
    ("/mnt/c/already/wsl", "/mnt/c/already/wsl"),   # already-WSL passthrough
])
def test_win_to_wsl(win, wsl):
    assert win_to_wsl(win) == wsl


def test_bart_command_leaves_regularization_args(monkeypatch):
    """The drive-letter guard lives in _bart_command's _WIN_PATH_RE (it requires
    a slash after the colon), so BART reg args like 'W:7:0:0.01' are never fed
    to win_to_wsl. On non-Windows, _bart_command passes args through verbatim."""
    from processing.utils import _WIN_PATH_RE, _bart_command
    monkeypatch.delenv("MRI_BART_BIN", raising=False)   # default binary name
    assert _WIN_PATH_RE.match("W:7:0:0.01") is None      # not treated as a path
    assert _WIN_PATH_RE.match(r"C:\x") is not None        # real path is
    assert _bart_command(["pics", "-R", "W:7:0:0.01"])[0] == "bart"  # Linux passthrough
    assert "W:7:0:0.01" in _bart_command(["pics", "-R", "W:7:0:0.01"])


# ── Scan-type detection ──────────────────────────────────────────────────────

@pytest.mark.parametrize("name,expected", [
    ("meas_MID001_fully_sampled.dat", ScanType.FULLY_SAMPLED),
    ("equispaced_R4_acs24.dat", ScanType.EQUISPACED),
    ("Siemens_t1_fl2d_tra_with_4_equispaced_R4_acs24.dat", ScanType.EQUISPACED),
    ("PoissonDisc.dat", ScanType.POISSON),
    ("meas_MID785_band_se_0.dat", ScanType.BAND),
    ("meas_MID793_band_tse_8.dat", ScanType.BAND),
    ("random_scan.dat", ScanType.UNKNOWN),
])
def test_auto_detect_scan_type(name, expected):
    assert auto_detect_scan_type(name) == expected


@pytest.mark.parametrize("name,idx", [
    ("meas_band_se_0.dat", 0),
    ("meas_band_se_15.dat", 15),
    ("meas_band_tse_8.dat", 8),
    ("fully_sampled.dat", None),
    # No 0–15 cap: any non-negative index is supported.
    ("meas_band_se_23.dat", 23),
    ("meas_band_tse_150.dat", 150),
])
def test_extract_band_index(name, idx):
    assert extract_band_index(name) == idx


@pytest.mark.parametrize("name", [
    "meas_band_se_16.dat", "meas_band_se_47.dat", "meas_band_tse_255.dat",
])
def test_band_detection_not_capped_at_15(name):
    """Band scan-type detection accepts indices well beyond 15."""
    assert auto_detect_scan_type(name) == ScanType.BAND


# ── Readout-oversampling crop ────────────────────────────────────────────────

def test_crop_os_factor1_noop():
    ksp = np.ones((16, 4, 64), dtype=np.complex64)
    out = crop_readout_oversampling(ksp, expected_nx=64, readout_axis=2)
    assert out.shape == ksp.shape
    assert out is ksp                     # untouched


def test_crop_os_factor2_halves_readout():
    """2x oversampled readout: an image narrower than the FOV survives the
    image-domain crop; the output readout is actual_nread // 2."""
    n_read_os = 128
    # Build k-space whose image support fits in the central half (no wrap).
    img = np.zeros((8, 3, n_read_os), dtype=np.complex64)
    img[:, :, n_read_os // 4: n_read_os // 4 + n_read_os // 2] = (
        np.random.default_rng(0).standard_normal((8, 3, n_read_os // 2))
    )
    ksp = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(img, axes=2), axis=2), axes=2)
    out = crop_readout_oversampling(ksp, expected_nx=64, readout_axis=2)
    assert out.shape == (8, 3, 64)


def test_crop_os_factor3_raises():
    ksp = np.ones((8, 3, 192), dtype=np.complex64)
    with pytest.raises(ValueError):
        crop_readout_oversampling(ksp, expected_nx=64, readout_axis=2)
