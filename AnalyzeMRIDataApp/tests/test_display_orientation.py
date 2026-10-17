"""Non-Cartesian results share the Cartesian (read/kx, phase/ky) frame, and every
tab draws it the same way: readout (kx) left→right, phase encode (ky) bottom→top."""

import numpy as np
import pytest

N = 32


def _asymmetric_image():
    img = np.zeros((N, N))
    img[6:12, 18:28] = 1.0                     # off-centre, not symmetric under transpose
    return img


def _corr(a, b):
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


def test_gridding_keeps_the_trajectory_frame():
    """Adjoint NUFFT of a full Cartesian trajectory returns the image as is —
    axis 0 follows kx, axis 1 follows ky — not transposed."""
    sp = pytest.importorskip("sigpy")
    from processing import recon_backend
    img = _asymmetric_image()
    ksp = sp.fft(img, center=True)                                   # [kx, ky]
    kx, ky = np.meshgrid(np.arange(N) - N // 2, np.arange(N) - N // 2, indexing="ij")
    traj = np.stack([kx, ky, np.zeros_like(kx)]).astype(float)       # (3, dim1, dim2)
    recon_backend.set_backend_mode("sigpy")
    try:
        imgs, cart = recon_backend.nufft_adjoint_to_cartesian(traj, ksp[None], N, N)
    finally:
        recon_backend.set_backend_mode("auto")
    out = np.abs(imgs[..., 0])
    assert _corr(out, img) > 0.9 > _corr(out, img.T)
    assert _corr(np.abs(cart[:, :, 0, 0]), np.abs(ksp)) > 0.9


def test_display_aspect_keeps_physical_proportions():
    from processing.imaging import display_aspect
    # square acquisition, square pixels: square in image and k-space, any rotation
    for turns in range(4):
        assert display_aspect((256, 256), None, turns) == 1.0
        assert display_aspect((256, 256), None, turns, kspace=True) == 1.0
    # 0.5 × 1.0 mm pixels (read, phase), drawn rot90 (phase down the rows)
    assert display_aspect((128, 256), (0.5, 1.0), 1) == 2.0
    # its k-space: FOV 128 mm in both → Δk equal → square k-space pixels
    assert display_aspect((128, 256), (0.5, 1.0), 1, kspace=True) == 1.0
    # 256 read × 128 phase at 1 mm: half the phase FOV → Δk_phase twice Δk_read
    assert display_aspect((128, 256), None, 1, kspace=True) == 2.0
    assert display_aspect((256, 128), None, 2, kspace=True) == 0.5


@pytest.mark.gui
def test_band_tab_draws_like_the_cartesian_tabs_and_turns_the_trajectory(qtbot):
    from ui.results_viewer import ReconTab
    img = _asymmetric_image()
    t = np.linspace(-0.5, 0.5, 50)
    traj = np.column_stack([t, 0.5 * t])                             # band at atan(0.5)
    res = {"scan_type": "BAND", "label": "band", "combined_img": img,
           "kspace_combined": np.fft.fftshift(np.fft.fft2(img)), "traj_scaled": traj}
    tab = ReconTab(res, "band")
    qtbot.addWidget(tab)

    def panel(prefix):
        return next(a for a in tab._fig.axes if a.get_title().startswith(prefix))

    np.testing.assert_allclose(panel("Combined").images[0].get_array(), np.rot90(img))
    xy = np.asarray(panel("Trajectory").collections[0].get_offsets())
    np.testing.assert_allclose(xy, traj)                             # kx →, ky ↑ (no −ky)
    # proportions: square image and k-space, square ±kmax trajectory box
    assert panel("Combined").get_aspect() == 1.0
    assert panel("K-space").get_aspect() == 1.0
    t_ax = panel("Trajectory")
    assert t_ax.get_aspect() == 1.0 and t_ax.get_xlim() == t_ax.get_ylim() == (-0.5, 0.5)

    tab._on_rotate()                                                 # 90° CCW, like the images
    np.testing.assert_allclose(panel("Combined").images[0].get_array(), np.rot90(img, 2))
    xy = np.asarray(panel("Trajectory").collections[0].get_offsets())
    np.testing.assert_allclose(xy, np.column_stack([-traj[:, 1], traj[:, 0]]))
