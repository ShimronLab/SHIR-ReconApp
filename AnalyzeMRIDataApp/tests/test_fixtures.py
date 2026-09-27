"""Self-tests for the synthetic-phantom fixtures the rest of the suite relies on."""

import numpy as np

from conftest import ifft2c


def test_phantom_shapes(phantom_kspace):
    p = phantom_kspace
    assert p["image"].shape == (64, 64)
    assert p["maps"].shape == (64, 64, 4)
    assert p["ksp"].shape == (64, 64, 4)
    assert p["reader"].shape == (64, 4, 64)


def test_phantom_kspace_roundtrip(phantom_kspace):
    """RSS of the IFFT of the synthetic k-space must recover the phantom."""
    p = phantom_kspace
    coil_imgs = ifft2c(p["ksp"])
    rss = np.sqrt(np.sum(np.abs(coil_imgs) ** 2, axis=-1))
    ref = p["image"] * np.sqrt(np.sum(np.abs(p["maps"]) ** 2, axis=-1))
    assert np.allclose(rss, ref, atol=1e-6)


def test_reader_layout_matches_pipeline_transpose(phantom_kspace):
    """transpose(2,0,1) of reader layout must equal the (Nread,Nphase,Ncoils) k-space."""
    p = phantom_kspace
    assert np.array_equal(np.transpose(p["reader"], (2, 0, 1)), p["ksp"])


def test_fake_twix_reader(phantom_kspace, fake_twix_reader):
    fake_twix_reader(phantom_kspace["reader"])
    from read_siemens_scans.twix_reading import read_raw_twix_data
    ksp, dims = read_raw_twix_data("/nonexistent/file.dat", fix_oversampling=False)
    assert ksp.shape == (64, 4, 64)
    assert dims is None
