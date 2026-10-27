"""
Shared test scaffolding for the SHIR test suite.

Path setup mirrors app.py: the app dir itself (so `processing` / `ui` import)
and the sibling read_siemens_scans/ package. Qt runs offscreen and matplotlib
on Agg so every test is headless.

Key fixtures
------------
phantom_kspace   : synthetic multi-coil Shepp-Logan k-space in the TWIX reader's
                   native layout (Nacq_PE, Ncoils, Nreadout), plus ground truth.
fake_twix_reader : factory that monkeypatches read_raw_twix_data to return a
                   given k-space array (recon modules import it lazily inside
                   the function body, so patching the module attribute works).
sample_scan_dir  : path to the real phantom scan folder (skips if unavailable).
"""

import os
import sys
from pathlib import Path

# Headless Qt + matplotlib — must be set before any Qt/matplotlib import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

APP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_DIR.parent
for p in (str(APP_DIR), str(REPO_ROOT / "read_siemens_scans")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _isolate_qsettings(tmp_path, monkeypatch):
    """Route QSettings to a temp dir so tests never touch real user settings."""
    from PyQt5.QtCore import QSettings
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path))
    yield

# Real sample data (a scan folder with dat_files/); tests marked `realdata`
# skip unless SHIR_SAMPLE_SCAN_DIR points to an existing folder.
SAMPLE_SCAN_DIR = Path(os.environ.get("SHIR_SAMPLE_SCAN_DIR", "/nonexistent/shir-sample-scans"))


def fft2c(img: np.ndarray) -> np.ndarray:
    """Centered 2-D FFT over axes (0, 1) — inverse of the app's _ifft2c."""
    return np.fft.ifftshift(
        np.fft.fft2(np.fft.fftshift(img, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )


def ifft2c(ks: np.ndarray) -> np.ndarray:
    """Centered 2-D IFFT over axes (0, 1) — same convention as the app."""
    return np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(ks, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )


def make_phantom(nread: int = 64, nphase: int = 64, ncoils: int = 4):
    """
    Build a Shepp-Logan phantom with smooth birdcage coil sensitivities.

    Returns
    -------
    image : (nread, nphase) float — ground-truth magnitude image
    maps  : (nread, nphase, ncoils) complex — coil sensitivities
    ksp   : (nread, nphase, ncoils) complex — fully-sampled k-space
    """
    import sigpy.mri as mr
    import sigpy.sim

    image = np.abs(sigpy.sim.shepp_logan((nread, nphase), dtype=np.complex64))
    maps = np.moveaxis(
        mr.sim.birdcage_maps((ncoils, nread, nphase), dtype=np.complex64), 0, -1
    )
    coil_imgs = image[:, :, None] * maps
    ksp = fft2c(coil_imgs).astype(np.complex128)
    return image, maps, ksp


def to_reader_layout(ksp: np.ndarray) -> np.ndarray:
    """(Nread, Nphase, Ncoils) k-space → TWIX reader layout (Nacq_PE, Ncoils, Nread)."""
    return np.transpose(ksp, (1, 2, 0))


@pytest.fixture
def phantom_kspace():
    """dict with image / maps / ksp (recon layout) / reader (TWIX reader layout)."""
    image, maps, ksp = make_phantom()
    return {
        "image": image,
        "maps": maps,
        "ksp": ksp,                       # (Nread, Nphase, Ncoils)
        "reader": to_reader_layout(ksp),  # (Nacq_PE, Ncoils, Nread)
    }


def install_fake_twix_reader(monkeypatch, kspace):
    """
    Patch read_siemens_scans.twix_reading.read_raw_twix_data.

    ``kspace`` is either a single array (returned for every path) or a dict of
    {filename_substring: array} matched against the requested path (so e.g. an
    equispaced scan and its fully-sampled reference return different data).
    """
    import read_siemens_scans.twix_reading as twix_reading

    def _fake(path, fix_oversampling=False, **kwargs):
        if isinstance(kspace, dict):
            for key, arr in kspace.items():
                if key in str(path):
                    return arr.copy(), None
            raise FileNotFoundError(f"fake reader: no entry matches {path!r}")
        return kspace.copy(), None

    monkeypatch.setattr(twix_reading, "read_raw_twix_data", _fake)
    return _fake


@pytest.fixture
def fake_twix_reader(monkeypatch):
    """Factory: fake_twix_reader(kspace) — see install_fake_twix_reader."""
    def _install(kspace):
        return install_fake_twix_reader(monkeypatch, kspace)

    return _install


@pytest.fixture
def sample_scan_dir():
    if not SAMPLE_SCAN_DIR.is_dir():
        pytest.skip(f"sample scan folder not available: {SAMPLE_SCAN_DIR}")
    return SAMPLE_SCAN_DIR


def bart_on_path() -> bool:
    """True when BART can run (used by the `bart` marker's skipif)."""
    import shutil
    return shutil.which(os.environ.get("MRI_BART_BIN", "bart")) is not None


def pytest_collection_modifyitems(config, items):
    skip_bart = pytest.mark.skipif(not bart_on_path(), reason="BART not on PATH")
    skip_real = pytest.mark.skipif(
        not SAMPLE_SCAN_DIR.is_dir(), reason="sample scan folder not available"
    )
    for item in items:
        if "bart" in item.keywords:
            item.add_marker(skip_bart)
        if "realdata" in item.keywords:
            item.add_marker(skip_real)
