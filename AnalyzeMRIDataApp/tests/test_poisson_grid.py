"""Poisson-disc Cartesian grid placement (processing.poisson_recon)."""

import json

import numpy as np
import pytest

from conftest import ifft2c, install_fake_twix_reader, to_reader_layout


@pytest.mark.parametrize("frac", [0.0, 0.34])
def test_grid_from_readout_when_outer_ky_unsampled(phantom_kspace, monkeypatch,
                                                   tmp_path, frac):
    """A VD pattern that never samples the outermost PE lines (and carries a
    sub-step ky centring shift) must still land on the full prescribed matrix
    — it used to be sized from the ky span, shrinking the grid and making the
    readout-OS crop halve a non-oversampled readout (zoomed-in, half-FOV)."""
    from processing.poisson_recon import reconstruct_poisson

    ksp = phantom_kspace["ksp"]                       # (64 read, 64 phase, coils)
    n = ksp.shape[0]
    dk = 1000.0 / n
    rng = np.random.default_rng(0)
    lines = np.unique(np.concatenate([
        np.arange(n // 2 - 6, n // 2 + 6),            # fully-sampled centre
        rng.choice(np.arange(10, n - 12), 14, replace=False),   # never the edges
    ]))
    install_fake_twix_reader(monkeypatch, to_reader_layout(ksp)[lines])

    kx = (np.arange(n) - n // 2) * dk
    ky = (lines - n // 2 + frac) * dk
    traj = np.stack([np.tile(kx, lines.size), np.repeat(ky, n)])   # (2, N)
    js = tmp_path / "PoissonDisc.json"
    js.write_text(json.dumps({"0": traj.tolist()}))

    res = reconstruct_poisson("PoissonDisc.dat", str(js), use_zerofill=True)

    masked = np.zeros_like(ksp)
    masked[:, lines, :] = ksp[:, lines, :]
    expected = np.sqrt(np.sum(np.abs(ifft2c(masked)) ** 2, axis=2))
    assert res["combined_img"].shape == (n, n)
    np.testing.assert_allclose(res["combined_img"], expected, rtol=1e-4, atol=1e-6)
    assert res["params"]["n_phase"] == n
