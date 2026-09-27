"""FolderPanel: simulator target-confirmation dialog + loose/dat_files loading."""

import pytest

pytestmark = pytest.mark.gui


def _fs_jobs():
    return [
        {"dat_path": "/x/dat_files/fully_sampled_a.dat", "folder_key": "SE_1mm",
         "scan_type": "FULLY_SAMPLED"},
        {"dat_path": "/x/dat_files/fully_sampled_b.dat", "folder_key": "SE_1mm",
         "scan_type": "FULLY_SAMPLED"},
        {"dat_path": "/y/dat_files/fully_sampled_c.dat", "folder_key": "SE_3mm",
         "scan_type": "FULLY_SAMPLED"},
    ]


def test_simulate_dialog_all_selected_by_default(qtbot):
    from ui.folder_panel import _SimulateSelectDialog
    dlg = _SimulateSelectDialog(_fs_jobs())
    qtbot.addWidget(dlg)
    assert len(dlg.selected_jobs()) == 3          # all pre-ticked
    assert dlg._buttons.button(dlg._buttons.Ok).isEnabled()


def test_simulate_dialog_labels_slices(qtbot):
    """Per-slice jobs of one multi-slice scan must be distinguishable."""
    from ui.folder_panel import _SimulateSelectDialog
    jobs = [
        {"dat_path": "/x/dat_files/siemens_fs.dat", "folder_key": "S",
         "scan_type": "FULLY_SAMPLED", "slice_idx": 0, "n_slices": 5},
        {"dat_path": "/x/dat_files/siemens_fs.dat", "folder_key": "S",
         "scan_type": "FULLY_SAMPLED", "slice_idx": 4, "n_slices": 5},
    ]
    dlg = _SimulateSelectDialog(jobs)
    qtbot.addWidget(dlg)
    labels = [cb.text() for cb in dlg._checkboxes]
    assert "slice 1/5" in labels[0]
    assert "slice 5/5" in labels[1]
    assert labels[0] != labels[1]


def test_simulate_dialog_untick_filters(qtbot):
    from ui.folder_panel import _SimulateSelectDialog
    jobs = _fs_jobs()
    dlg = _SimulateSelectDialog(jobs)
    qtbot.addWidget(dlg)
    dlg._checkboxes[1].setChecked(False)          # drop the middle one
    sel = dlg.selected_jobs()
    assert [j["dat_path"] for j in sel] == [jobs[0]["dat_path"], jobs[2]["dat_path"]]


def test_simulate_dialog_clear_all_disables_ok(qtbot):
    from ui.folder_panel import _SimulateSelectDialog
    dlg = _SimulateSelectDialog(_fs_jobs())
    qtbot.addWidget(dlg)
    dlg._set_all(False)
    assert dlg.selected_jobs() == []
    assert not dlg._buttons.button(dlg._buttons.Ok).isEnabled()   # can't proceed with none
    dlg._set_all(True)
    assert dlg._buttons.button(dlg._buttons.Ok).isEnabled()


def test_on_simulate_emits_only_confirmed(qtbot, monkeypatch):
    """_on_simulate must emit exactly the dialog-confirmed subset, re-typed."""
    from PyQt5.QtWidgets import QDialog
    from ui.folder_panel import FolderPanel
    import ui.folder_panel as fp

    panel = FolderPanel()
    qtbot.addWidget(panel)

    jobs = _fs_jobs()
    monkeypatch.setattr(panel, "_collect_jobs", lambda: [dict(j) for j in jobs])

    # Stand-in dialog: accept, returning only the first + third scans.
    class _FakeDlg:
        def __init__(self, fs_jobs, parent=None):
            self._fs = fs_jobs
        def exec_(self):
            return QDialog.Accepted
        def selected_jobs(self):
            return [self._fs[0], self._fs[2]]

    monkeypatch.setattr(fp, "_SimulateSelectDialog", _FakeDlg)

    emitted = []
    panel.simulate_requested.connect(lambda js: emitted.append(js))
    panel._on_simulate()

    assert len(emitted) == 1
    sent = emitted[0]
    assert len(sent) == 2
    assert all(j["scan_type"] == "RETRO_UNDERSAMPLE" for j in sent)
    assert [j["dat_path"] for j in sent] == [jobs[0]["dat_path"], jobs[2]["dat_path"]]


def test_on_simulate_cancel_emits_nothing(qtbot, monkeypatch):
    from PyQt5.QtWidgets import QDialog
    from ui.folder_panel import FolderPanel
    import ui.folder_panel as fp

    panel = FolderPanel()
    qtbot.addWidget(panel)
    monkeypatch.setattr(panel, "_collect_jobs", lambda: _fs_jobs())

    class _FakeDlg:
        def __init__(self, fs_jobs, parent=None): pass
        def exec_(self): return QDialog.Rejected
        def selected_jobs(self): return []

    monkeypatch.setattr(fp, "_SimulateSelectDialog", _FakeDlg)
    emitted = []
    panel.simulate_requested.connect(lambda js: emitted.append(js))
    panel._on_simulate()
    assert emitted == []                          # Cancel → no run
