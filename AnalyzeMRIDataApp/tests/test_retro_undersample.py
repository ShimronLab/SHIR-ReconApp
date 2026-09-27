"""Retro-undersampling simulator: mask builders + pipeline regression tests."""

import numpy as np
import pytest

from processing.retro_undersample import (
    build_equispaced_mask,
    build_vd_mask,
    reconstruct_retro,
)


# ── Mask builders ────────────────────────────────────────────────────────────

def test_equispaced_mask_basic():
    n, R, acs = 256, 4, 24
    mask = build_equispaced_mask(n, R, acs)
    center = n // 2
    assert mask.shape == (n,)
    assert mask[center]                                   # center line sampled
    assert mask[center - acs // 2: center + acs // 2].all()   # ACS block full
    actual_r = n / mask.sum()
    assert 2.0 < actual_r <= R                            # ACS lowers actual R below target


def test_equispaced_mask_step():
    """Outside the ACS block, sampled lines step by exactly R."""
    n, R, acs = 256, 4, 24
    mask = build_equispaced_mask(n, R, acs)
    center = n // 2
    idx = np.where(mask)[0]
    right = idx[idx >= center + acs]
    assert np.all(np.diff(right) == R)


def test_vd_mask_seed_reproducible():
    n, R, pd, acs = 256, 4.0, 1.0, 24
    m1 = build_vd_mask(n, R, pd, acs, rng=np.random.default_rng(42))
    m2 = build_vd_mask(n, R, pd, acs, rng=np.random.default_rng(42))
    m3 = build_vd_mask(n, R, pd, acs, rng=np.random.default_rng(43))
    assert np.array_equal(m1, m2)
    assert not np.array_equal(m1, m3)     # different seed → different mask


def test_vd_mask_fixed_lines_and_budget():
    n, R, acs = 256, 4.0, 24
    mask = build_vd_mask(n, R, 1.0, acs, rng=np.random.default_rng(0))
    center = n // 2
    assert mask[0] and mask[n - 1]                          # outer edges fixed
    assert mask[center - acs // 2: center + acs // 2].all() # ACS fixed
    assert mask.sum() == round(n / R)                       # exact line budget


# ── Pipeline regression tests (fake TWIX reader) ─────────────────────────────

def test_retro_recon_none_regression(phantom_kspace, fake_twix_reader):
    """recon_method='none' (the UI default) must return a complete result dict.

    Regression: sens_maps_out was only assigned inside the recon!=none branch
    but referenced unconditionally in the return dict → UnboundLocalError.
    """
    fake_twix_reader(phantom_kspace["reader"])
    result = reconstruct_retro(
        "fully_sampled.dat", mask_type="equispaced", R=4.0,
        acs_lines=16, recon_method="none", tse=False,
    )
    assert result["sens_maps"] is None
    assert result["recon_image"] is None
    assert result["rmse"] is None
    assert result["zf_image"].shape == (64, 64)
    assert result["fully_sampled_image"].shape == (64, 64)
    assert result["pe_mask"].sum() < 64                    # actually undersampled
    assert result["method"] == "none"


def test_retro_recon_grappa(phantom_kspace, fake_twix_reader):
    fake_twix_reader(phantom_kspace["reader"])
    result = reconstruct_retro(
        "fully_sampled.dat", mask_type="equispaced", R=2.0,
        acs_lines=16, recon_method="grappa", kernel_size=(5, 5),
        grappa_lambda=0.01, tse=False,
    )
    assert result["recon_image"] is not None
    assert result["recon_image"].shape == (64, 64)
    assert np.isfinite(result["rmse"])
    assert np.isfinite(result["ssim"])
    # GRAPPA at R=2 with ACS on 4 coils should beat plain zero-fill.
    fs = result["fully_sampled_image"].astype(float)
    scale = np.percentile(fs, 95) + 1e-12
    zf_rmse = float(np.sqrt(np.mean((result["zf_image"] / scale - fs / scale) ** 2)))
    assert result["rmse"] < zf_rmse


def test_retro_swap_pe_read(phantom_kspace, fake_twix_reader):
    """swap_pe_read undersamples the readout axis instead of phase."""
    fake_twix_reader(phantom_kspace["reader"])
    result = reconstruct_retro(
        "fully_sampled.dat", mask_type="equispaced", R=4.0,
        acs_lines=16, recon_method="none", swap_pe_read=True, tse=False,
    )
    ksp = result["undersampled_kspace_rss"]
    # Zeroed rows along axis 0 (readout) — column sums nonzero everywhere,
    # row sums zero where the mask is off.
    row_energy = ksp.sum(axis=1)
    assert (row_energy == 0).any()
    assert result["swap_pe_read"] is True


def test_retro_vd_seed_recorded(phantom_kspace, fake_twix_reader):
    """rng_seed makes VD simulations reproducible end-to-end."""
    fake_twix_reader(phantom_kspace["reader"])
    kwargs = dict(mask_type="vd", R=4.0, power_decay=1.0, acs_lines=16,
                  recon_method="none", rng_seed=7, tse=False)
    r1 = reconstruct_retro("fully_sampled.dat", **kwargs)
    r2 = reconstruct_retro("fully_sampled.dat", **kwargs)
    assert np.array_equal(r1["pe_mask"], r2["pe_mask"])


def test_retro_multislice_uses_selector(phantom_kspace, fake_twix_reader):
    """4-D input routes through slice_selector_cb."""
    reader = phantom_kspace["reader"]
    multi = np.stack([reader * 0.5, reader, reader * 2.0])   # 3 slices
    fake_twix_reader(multi)
    calls = []

    def selector(n):
        calls.append(n)
        return 1

    result = reconstruct_retro(
        "fully_sampled.dat", mask_type="equispaced", R=4.0,
        acs_lines=16, recon_method="none", tse=False,
        slice_selector_cb=selector,
    )
    assert calls == [3]
    assert result["zf_image"].shape == (64, 64)
