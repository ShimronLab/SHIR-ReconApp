"""Stop/abort mechanism: JobAborted must cancel jobs cleanly."""

import numpy as np
import pytest

from processing.constants import JobAborted

pytestmark = pytest.mark.gui   # ReconWorker is a QObject → needs a QApplication


def test_jobaborted_survives_broad_except():
    """Processing code is full of `except Exception` fallbacks; the abort must
    not be swallowed by them (JobAborted derives from BaseException)."""
    def log(_msg):
        raise JobAborted()

    with pytest.raises(JobAborted):
        try:
            log("stage 1")
        except Exception:                     # a typical fallback block
            pytest.fail("JobAborted was swallowed by `except Exception`")


def test_worker_abort_no_result(qapp, phantom_kspace, fake_twix_reader):
    """An aborted worker emits no result, logs [ABORTED], and still finishes."""
    fake_twix_reader(phantom_kspace["reader"])
    from ui.main_window import ReconWorker

    job = {"scan_type": "RETRO_UNDERSAMPLE", "dat_path": "fully_sampled.dat",
           "folder_key": "T"}
    params = {"sim_mask_type": "equispaced", "sim_R": 4.0, "sim_acs_lines": 16,
              "sim_recon_method": "none", "recon_backend": "sigpy", "tse": False}

    worker = ReconWorker(job, params)
    events = {"results": [], "errors": [], "msgs": [], "finished": 0}
    worker.result_ready.connect(lambda r: events["results"].append(r))
    worker.error_signal.connect(lambda m: events["errors"].append(m))
    worker.progress_msg.connect(lambda m: events["msgs"].append(m))
    worker.finished.connect(lambda: events.__setitem__("finished", events["finished"] + 1))

    worker.abort()          # Stop pressed before the job starts
    worker.run()            # run synchronously (no thread needed for the test)

    assert events["results"] == []
    assert events["errors"] == []
    assert events["finished"] == 1
    assert any("[ABORTED]" in m for m in events["msgs"])


def test_slice_dialog_before_full_read(qapp, phantom_kspace, fake_twix_reader, monkeypatch):
    """Multi-slice scans must ask for the slice via the cheap header probe
    BEFORE the full TWIX read (the old flow parsed the whole file first,
    stalling for tens of seconds before the dialog appeared)."""
    import numpy as np

    import processing.utils as putils
    from ui.main_window import ReconWorker

    order = []

    monkeypatch.setattr(
        putils, "probe_num_slices",
        lambda _p: (order.append("probe"), 3)[1],
    )

    reader = phantom_kspace["reader"]
    multi = np.stack([reader, reader, reader])
    real_fake = fake_twix_reader(multi)          # installs the fake reader

    import read_siemens_scans.twix_reading as twix_reading

    def _tracking_read(path, fix_oversampling=False, **kw):
        order.append("read")
        return real_fake(path, fix_oversampling, **kw)

    monkeypatch.setattr(twix_reading, "read_raw_twix_data", _tracking_read)

    job = {"scan_type": "RETRO_UNDERSAMPLE", "dat_path": "fully_sampled.dat",
           "folder_key": "T"}
    params = {"sim_mask_type": "equispaced", "sim_R": 4.0, "sim_acs_lines": 16,
              "sim_recon_method": "none", "recon_backend": "sigpy", "tse": False}
    worker = ReconWorker(job, params)

    def _answer(n_slices):
        order.append("dialog")
        assert n_slices == 3
        worker.set_slice_choice(1)

    worker.slice_selection_needed.connect(_answer)
    results = []
    worker.result_ready.connect(results.append)
    worker.run()

    assert order[:3] == ["probe", "dialog", "read"]   # ask first, read after
    assert len(results) == 1


def test_worker_completes_without_abort(qapp, phantom_kspace, fake_twix_reader):
    """Sanity: the same job completes and emits exactly one result when not aborted."""
    fake_twix_reader(phantom_kspace["reader"])
    from ui.main_window import ReconWorker

    job = {"scan_type": "RETRO_UNDERSAMPLE", "dat_path": "fully_sampled.dat",
           "folder_key": "T"}
    params = {"sim_mask_type": "equispaced", "sim_R": 4.0, "sim_acs_lines": 16,
              "sim_recon_method": "none", "recon_backend": "sigpy", "tse": False}

    worker = ReconWorker(job, params)
    events = {"results": [], "errors": [], "finished": 0}
    worker.result_ready.connect(lambda r: events["results"].append(r))
    worker.error_signal.connect(lambda m: events["errors"].append(m))
    worker.finished.connect(lambda: events.__setitem__("finished", events["finished"] + 1))

    worker.run()

    assert events["errors"] == []
    assert len(events["results"]) == 1
    assert events["finished"] == 1
    result = events["results"][0]
    assert result["scan_type"] == "RETRO_UNDERSAMPLE"
    assert result["folder_key"] == "T"
    assert result["zf_image"].shape == (64, 64)
