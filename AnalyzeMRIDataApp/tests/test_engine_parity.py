"""BART and SigPy must run the same reconstruction: every engine parameter is
passed explicitly (no silent library defaults), and SENSE on the synthetic
phantom agrees between the two engines."""

import numpy as np
import pytest

from conftest import make_phantom
from processing import recon_backend
from processing.imaging import espirit_normalise, metrics_vs_reference


def test_sigpy_espirit_gets_every_parameter(monkeypatch):
    """SigPy's EspiritCalib defaults (crop 0.95, thresh 0.02, kernel 6) must never
    apply silently: the app passes calib width, crop, kernel and threshold."""
    import sigpy.mri as mr
    captured = {}

    class FakeCalib:
        def __init__(self, ksp, **kwargs):
            captured.update(kwargs)
            self.shape = ksp.shape

        def run(self):
            return np.ones(self.shape, dtype=np.complex64)

    monkeypatch.setattr(mr.app, "EspiritCalib", FakeCalib)
    monkeypatch.setattr(recon_backend, "use_bart", lambda: False)
    recon_backend.espirit_maps(np.ones((16, 16, 1, 2), np.complex64), 8, crop=0.8)
    for key in ("calib_width", "crop", "kernel_width", "thresh"):
        assert key in captured, key
    assert captured["crop"] == 0.8 and captured["kernel_width"] == recon_backend.ESPIRIT_KERNEL


def test_bart_commands_carry_every_parameter(monkeypatch):
    """ecalib gets -r -c -m -k -t and pics always gets -i (BART's own default
    is not relied on)."""
    calls = []
    monkeypatch.setattr(recon_backend, "use_bart", lambda: True)
    monkeypatch.setattr(recon_backend, "run_bart", lambda *args, **kw: calls.append(args))
    monkeypatch.setattr(recon_backend, "readcfl", lambda path: np.zeros((4, 4, 1, 2), np.complex64))
    k4d = np.ones((4, 4, 1, 2), np.complex64)
    recon_backend.espirit_maps(k4d, 4, crop=0.8)
    recon_backend.pics(k4d, k4d, soft_sense=True)
    ecalib, pics = calls
    assert ecalib[0] == "ecalib"
    for flag in ("-r4:4:1", "-c0.8", "-m1", f"-k{recon_backend.ESPIRIT_KERNEL}",
                 f"-t{recon_backend.ESPIRIT_THRESH_BART}"):
        assert flag in ecalib, flag
    assert pics[0] == "pics" and f"-i{recon_backend.PICS_ITERS_DEFAULT}" in pics


@pytest.mark.bart
def test_bart_and_sigpy_sense_agree_on_phantom():
    _, _, ksp = make_phantom(96, 96, 8)
    mask = np.zeros(96, bool)
    mask[::2] = True
    mask[40:56] = True
    k4d = (ksp * mask[None, :, None])[:, :, None, :].astype(np.complex64)
    k4d_norm, _ = espirit_normalise(k4d)
    images = {}
    try:
        for mode in ("bart", "sigpy"):
            recon_backend.set_backend_mode(mode)
            maps = recon_backend.espirit_maps(k4d_norm, 16, num_maps=1, crop=0.8)
            images[mode] = np.abs(recon_backend.pics(k4d_norm, maps, soft_sense=True))
    finally:
        recon_backend.set_backend_mode("auto")
    bart, sigpy = images["bart"], images["sigpy"]
    sigpy = sigpy * (np.percentile(bart, 95) / np.percentile(sigpy, 95))
    assert metrics_vs_reference(sigpy, bart)["ssim"] > 0.95
