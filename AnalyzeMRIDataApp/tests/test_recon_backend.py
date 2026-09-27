"""recon_backend: sigpy path shapes + recovery; BART parity behind the @bart mark."""

import numpy as np
import pytest

from processing import recon_backend as rb
from conftest import ifft2c, make_phantom


def _equispaced_undersample(ksp_recon_layout, R=2, acs=16):
    """Zero every R-th PE line, keep a central ACS block. Returns k4d BART layout."""
    nread, nphase, ncoils = ksp_recon_layout.shape
    mask = np.zeros(nphase, dtype=bool)
    c = nphase // 2
    mask[np.arange(c, nphase, R)] = True
    mask[np.arange(c, -1, -R)] = True
    mask[c - acs // 2: c + acs // 2] = True
    k = ksp_recon_layout.copy()
    k[:, ~mask, :] = 0
    return k[:, :, np.newaxis, :].astype(np.complex64), mask


def _masked_nmse(recon, ref):
    from processing.imaging import metrics_vs_reference
    return metrics_vs_reference(recon, ref)["nmse"]


# ── sigpy path (no BART needed) ──────────────────────────────────────────────

def test_sigpy_espirit_maps_shape():
    rb.set_backend_mode("sigpy")
    _, _, ksp = make_phantom()
    k4d = ksp[:, :, np.newaxis, :].astype(np.complex64)
    maps = rb.espirit_maps(k4d, calib=24, num_maps=1)
    assert maps.shape[:2] == (64, 64)
    assert maps.shape[-1] == 4          # coils last (BART layout)


def test_sigpy_sense_recovers_phantom():
    """Pure SENSE on R=2 undersampled 4-coil data should reconstruct the
    phantom to a small masked NMSE."""
    rb.set_backend_mode("sigpy")
    image, maps, ksp = make_phantom()
    ref = image * np.sqrt(np.sum(np.abs(maps) ** 2, axis=-1))

    k4d, _ = _equispaced_undersample(ksp, R=2, acs=20)
    # normalise k-space like the pipelines do
    import sigpy as sp
    scale = float(np.percentile(np.abs(sp.rss(sp.ifft(k4d, axes=(0, 1)), axes=-1)).ravel(), 95))
    k4d_n = k4d / (scale + 1e-12)
    maps_bart = rb.espirit_maps(k4d_n, calib=20, num_maps=1)
    img = np.abs(np.squeeze(rb.pics(k4d_n, maps_bart, soft_sense=True, iters=30)))
    # rescale to reference for a fair comparison
    img = img * (np.percentile(ref, 95) / (np.percentile(img, 95) + 1e-12))
    assert _masked_nmse(img, ref) < 0.1


def test_backend_mode_switch():
    rb.set_backend_mode("sigpy")
    assert rb.use_bart() is False
    assert rb.active_backend_label() == "sigpy"
    with pytest.raises(ValueError):
        rb.set_backend_mode("nonsense")


# ── BART parity (skipped automatically without BART) ─────────────────────────

@pytest.mark.bart
def test_bart_available_true_when_on_path():
    assert rb.bart_available() is True


@pytest.mark.bart
def test_bart_cfl_fft_roundtrip(tmp_path):
    """writecfl → bart fft → bart fft -i → readcfl recovers the input
    (convention-independent check of CFL I/O + the BART bridge)."""
    import os

    from processing.utils import BartTempDir, readcfl, run_bart, writecfl

    rng = np.random.default_rng(0)
    img = (rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))).astype(np.complex64)
    with BartTempDir("test_") as tmp:
        inp = os.path.join(tmp, "i")
        fwd = os.path.join(tmp, "k")
        inv = os.path.join(tmp, "o")
        writecfl(inp, img)
        bitmask = run_bart("bitmask", "0", "1").strip()
        run_bart("fft", bitmask, inp, fwd)
        run_bart("fft", "-i", bitmask, fwd, inv)
        back = readcfl(inv)
    # BART's fft/fft -i pair is non-unitary: it scales by N (= product of the
    # transformed dims), so divide it out before comparing.
    n = img.shape[0] * img.shape[1]
    assert np.allclose(back / n, img, atol=1e-4 * np.abs(img).max())


@pytest.mark.bart
def test_bart_sense_recovers_phantom():
    rb.set_backend_mode("bart")
    image, maps, ksp = make_phantom()
    ref = image * np.sqrt(np.sum(np.abs(maps) ** 2, axis=-1))
    k4d, _ = _equispaced_undersample(ksp, R=2, acs=20)
    import sigpy as sp
    scale = float(np.percentile(np.abs(sp.rss(sp.ifft(k4d, axes=(0, 1)), axes=-1)).ravel(), 95))
    k4d_n = k4d / (scale + 1e-12)
    maps_bart = rb.espirit_maps(k4d_n, calib=20, num_maps=1)
    img = np.abs(np.squeeze(rb.pics(k4d_n, maps_bart, soft_sense=True, iters=30)))
    img = img * (np.percentile(ref, 95) / (np.percentile(img, 95) + 1e-12))
    assert _masked_nmse(img, ref) < 0.1
    rb.set_backend_mode("auto")
