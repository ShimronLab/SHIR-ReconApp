"""Offscreen GUI smoke tests: MainWindow, tab lifecycle, view-toggle safety."""

import numpy as np
import pytest

from conftest import install_fake_twix_reader, make_phantom, to_reader_layout

pytestmark = pytest.mark.gui


@pytest.fixture
def main_window(qtbot):
    from ui.main_window import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def _equispaced_result(monkeypatch):
    """Run the real equispaced GRAPPA pipeline on synthetic undersampled data."""
    from processing.equispaced_recon import _build_grappa_mask_computed
    from processing.equispaced_recon import reconstruct_equispaced

    image, maps, ksp = make_phantom()
    full_reader = to_reader_layout(ksp)                     # (64, 4, 64)
    mask, idx, _, _ = _build_grappa_mask_computed(64, 4, 16, lambda _m: None)
    under_reader = full_reader[idx]                         # acquired lines only

    install_fake_twix_reader(monkeypatch, {
        "equispaced": under_reader,
        "fully_sampled": full_reader,
    })
    result = reconstruct_equispaced(
        "equispaced_R4_acs16.dat", method="grappa", acs_lines=16, r_factor=4,
        kernel_size=(5, 5), lamda=0.01, tse=False,
        reference_dat_path="fully_sampled.dat",
    )
    result["scan_type"] = "EQUISPACED"
    result["label"] = "T — Equispaced (GRAPPA)"
    result["dat_path"] = "equispaced_R4_acs16.dat"
    result["folder_key"] = "T"
    return result


def test_mainwindow_constructs(main_window):
    assert main_window._results == []
    assert main_window._act_stop.isEnabled() is False


def test_get_params_contract(main_window):
    """Keys the recon dispatch depends on must exist in get_params()."""
    p = main_window._params_panel.get_params()
    required = {
        "grid_size", "scan_fov_m", "crop_size", "traj_margin", "calib_width",
        "recon_backend", "equi_method", "equi_mask_method", "acs_lines",
        "r_factor", "kernel_size", "grappa_lambda", "sense_lambda",
        "sense_max_iter", "sense_l2", "espirit_num_maps", "fullysamp_mode",
        "tse", "tse_necho", "tse_te1", "tse_teeff", "poisson_mode",
        "lambda_wav", "max_iter", "unknown_mode", "sim_mask_type", "sim_R",
        "sim_acs_lines", "sim_power_decay", "sim_recon_method",
        "sim_kernel_size", "sim_grappa_lambda", "sim_sense_iters",
        "sim_sense_lambda", "sim_swap_pe_read", "save_dir", "gen_pdf",
        "save_npy", "espirit_crop", "pf_method", "prewhiten", "seq_library_dir",
    }
    missing = required - set(p)
    assert not missing, f"get_params() lost keys: {missing}"


def test_seq_library_setting_persists(qtbot, tmp_path):
    from PyQt5.QtCore import QSettings
    from ui.params_panel import ParamsPanel
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    first = ParamsPanel()
    qtbot.addWidget(first)
    first.edit_seq_library.setText(str(tmp_path))
    first.save_persistent(settings)
    second = ParamsPanel()
    qtbot.addWidget(second)
    second.load_persistent(settings)
    assert second.get_params()["seq_library_dir"] == str(tmp_path)


def test_tab_close_removes_result(main_window, monkeypatch):
    result = _equispaced_result(monkeypatch)
    main_window._on_result_ready(result)
    assert len(main_window._results) == 1
    assert main_window._results_viewer.count() == 1

    main_window._results_viewer._on_tab_close_requested(0)
    assert main_window._results_viewer.count() == 0
    assert main_window._results == []          # closed tab dropped from saves


def test_view_toggles_restore_and_flag(main_window, monkeypatch):
    """Center/Crop toggles must flag the result as modified and restore the
    pristine reconstruction when switched off (regression: exports silently
    contained transformed data)."""
    result = _equispaced_result(monkeypatch)
    main_window._on_result_ready(result)
    tab = main_window._results_viewer.widget(0)
    original = np.asarray(result["grappa_image"]).copy()

    tab._btn_crop.setChecked(True)
    tab._toggle_eq_crop()
    assert result["_view_modified"] is True
    assert result["grappa_image"].shape != original.shape   # cropped view
    assert np.array_equal(result["_pristine_recon"], original)

    tab._btn_crop.setChecked(False)
    tab._toggle_eq_crop()
    assert result["_view_modified"] is False
    assert np.array_equal(result["grappa_image"], original)  # restored


def test_all_coils_feedback_when_absent(main_window, monkeypatch):
    """All Coils on a tab without per-coil data reports instead of no-oping."""
    result = _equispaced_result(monkeypatch)      # equispaced has no imgs_per_coil
    main_window._on_result_ready(result)
    assert main_window._results_viewer.show_all_coils_current() is False
    main_window._on_all_coils()
    assert "No per-coil images" in main_window._status_label.text()


class _WaitingWorker:
    """Stands in for a ReconWorker blocked on the slice dialog."""
    def __init__(self, job):
        self._job = job
        self.choice = None

    def set_slice_choice(self, idx):
        self.choice = idx


def test_slice_dialog_queues_extra_slices(main_window, monkeypatch):
    """Ticking several slices runs the first in the current job and queues the
    rest right after it, each tagged with its slice for a unique tab label."""
    import ui.main_window as mw
    monkeypatch.setattr(mw, "_ask_slices", lambda _parent, _n: [2, 5, 7])
    later_job = {"dat_path": "other.dat", "scan_type": "FULLY_SAMPLED"}
    main_window._job_queue = [later_job]
    main_window._jobs_total = 2
    worker = _WaitingWorker({"dat_path": "multi.dat", "scan_type": "FULLY_SAMPLED",
                             "folder_key": "k-space"})
    main_window._active_worker = worker

    main_window._on_slice_selection_needed(11)

    assert worker.choice == 2
    assert worker._job["slice_idx"] == 2 and worker._job["n_slices"] == 11
    queued = main_window._job_queue
    assert [j.get("slice_idx") for j in queued] == [5, 7, None]
    assert queued[-1] is later_job                       # other scans keep their place
    assert all(j["dat_path"] == "multi.dat" for j in queued[:2])
    assert main_window._jobs_total == 4
    assert "slice 6/11" in mw._make_label(queued[0])     # labels are 1-based


def test_ask_slices_dialog_builds(main_window, monkeypatch):
    """The real dialog constructs and, accepted untouched, returns the middle slice."""
    import ui.main_window as mw
    monkeypatch.setattr(mw.QDialog, "exec_", lambda _self: mw.QDialog.Accepted)
    assert mw._ask_slices(main_window, 11) == [5]
    monkeypatch.setattr(mw.QDialog, "exec_", lambda _self: mw.QDialog.Rejected)
    assert mw._ask_slices(main_window, 11) == []


def test_slice_dialog_cancel_uses_middle_slice(main_window, monkeypatch):
    import ui.main_window as mw
    monkeypatch.setattr(mw, "_ask_slices", lambda _parent, _n: [])
    main_window._job_queue = []
    main_window._jobs_total = 1
    worker = _WaitingWorker({"dat_path": "multi.dat", "scan_type": "FULLY_SAMPLED"})
    main_window._active_worker = worker

    main_window._on_slice_selection_needed(11)

    assert worker.choice == 5
    assert main_window._job_queue == []
