"""
MainWindow — top-level application window.

Layout
------
  Toolbar: [Open Folder] [Run] [Stop] [All Coils] [Clear Tabs] [About]
  ┌──────────┬────────────────────────────────┬──────────┐
  │  Folder  │       ResultsViewer            │  Params  │
  │  Panel   │         (tabs)                 │  Panel   │
  │          │                                │          │
  ├──────────┴────────────────────────────────┴──────────┤
  │  Progress bar  |  Log console                        │
  └──────────────────────────────────────────────────────┘

Worker thread
─────────────
  One QThread per job; emits progress_msg(str), result_ready(dict), finished().
  Multiple jobs run sequentially to avoid overwhelming BART.
"""

import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PyQt5.QtCore import QObject, QSettings, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QTextCursor
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .folder_panel   import FolderPanel
from .params_panel   import ParamsPanel
from .results_viewer import ResultsViewer
from .report_generator import generate_pdf
from processing.constants import JobAborted, method_label
from processing.constants import (
    SCAN_BAND,
    SCAN_EQUISPACED,
    SCAN_FULLY_SAMPLED,
    SCAN_POISSON,
    SCAN_RETRO_UNDERSAMPLE,
    SCAN_UNKNOWN,
)
from resources import resource_path
from version import __version__


# ─────────────────────────────────────────────────────────────────────────────
# Worker (runs a single reconstruction job in a background thread)
# ─────────────────────────────────────────────────────────────────────────────

class ReconWorker(QObject):
    """Runs one reconstruction job; emits signals back to the main thread."""
    progress_msg          = pyqtSignal(str)
    result_ready          = pyqtSignal(dict)       # carries result + meta on success
    error_signal          = pyqtSignal(str)        # carries error message
    finished              = pyqtSignal()
    slice_selection_needed = pyqtSignal(int)       # n_slices — request dialog on main thread
    mask_fallback_needed  = pyqtSignal()           # no JSON traj — request mask-choice dialog
    seq_file_needed       = pyqtSignal(dict)       # played .seq not found — ask where it is

    def __init__(self, job: Dict[str, Any], params: Dict[str, Any]):
        super().__init__()
        self._job    = job
        self._params = params
        self._abort  = False
        self._slice_event  = threading.Event()
        self._slice_choice: Optional[int] = None
        self._mask_event   = threading.Event()
        self._mask_choice: Optional[str] = None
        self._seq_event    = threading.Event()
        self._seq_choice: Optional[str] = None

    def abort(self) -> None:
        self._abort = True
        self._slice_event.set()  # unblock if waiting for slice selection
        self._mask_event.set()   # unblock if waiting for mask-fallback choice
        self._seq_event.set()    # unblock if waiting for a .seq file

    def set_seq_choice(self, path: Optional[str]) -> None:
        """Called from the main thread to unblock _request_seq_file()."""
        self._seq_choice = path
        self._seq_event.set()

    def _request_seq_file(self, info: Dict[str, Any]) -> Optional[str]:
        """Block the worker thread and ask the main thread to locate the .seq."""
        self._seq_event.clear()
        self._seq_choice = None
        self.seq_file_needed.emit(info)
        self._seq_event.wait()
        if self._abort:
            raise JobAborted()
        return self._seq_choice

    def set_slice_choice(self, idx: int) -> None:
        """Called from the main thread to unblock _request_slice()."""
        self._slice_choice = idx
        self._slice_event.set()

    def _request_slice(self, n_slices: int) -> int:
        """Block the worker thread and ask the main thread to show a dialog."""
        self._slice_event.clear()
        self._slice_choice = None
        self.slice_selection_needed.emit(n_slices)
        self._slice_event.wait()          # releases main-thread event loop
        if self._abort:                   # Stop pressed while waiting → cancel,
            raise JobAborted()            # don't reconstruct the middle slice
        return self._slice_choice if self._slice_choice is not None else (n_slices // 2)

    def set_mask_choice(self, choice: str) -> None:
        """Called from the main thread to unblock _request_mask_fallback()."""
        self._mask_choice = choice
        self._mask_event.set()

    def _request_mask_fallback(self) -> str:
        """Block the worker thread and ask the main thread which mask to build."""
        self._mask_event.clear()
        self._mask_choice = None
        self.mask_fallback_needed.emit()
        self._mask_event.wait()           # releases main-thread event loop
        if self._abort:
            raise JobAborted()
        return self._mask_choice if self._mask_choice is not None else "analytical"

    def run(self) -> None:
        job    = self._job
        params = self._params
        stype  = job.get("scan_type", SCAN_UNKNOWN)

        def cb(msg: str) -> None:
            # Stop support: every recon function reports progress between its
            # stages through this callback, so raising here cancels the job at
            # the next stage boundary (an in-flight BART/sigpy stage finishes).
            if self._abort:
                raise JobAborted()
            self.progress_msg.emit(msg)

        # Scan types whose recon consumes a slice selector (Band/Poisson are
        # single-slice NUFFT recons).
        _slice_capable = (SCAN_FULLY_SAMPLED, SCAN_EQUISPACED,
                          SCAN_RETRO_UNDERSAMPLE, SCAN_UNKNOWN)

        try:
            # Add processing package to path (needed when running from 'ui' sub-dir)
            _ensure_processing_on_path()

            # Slice callback: when this job targets a specific slice (multi-slice
            # tree selection), force it silently. Otherwise ask the user NOW,
            # via a cheap header-only probe (~0.3 s), BEFORE the full TWIX read —
            # multi-slice Siemens files take tens of seconds to parse, and the
            # old flow only popped the dialog afterwards, stalling the queue on
            # unattended runs.
            if job.get("slice_idx") is not None:
                _forced = int(job["slice_idx"])
                slice_cb = lambda _n, _i=_forced: _i          # noqa: E731
            elif stype in _slice_capable:
                from processing.utils import probe_num_slices
                n_slices = probe_num_slices(job["dat_path"])
                if n_slices > 1:
                    cb(f"Multi-slice scan detected ({n_slices} slices) — choose a slice…")
                    _forced = int(self._request_slice(n_slices))
                    slice_cb = lambda _n, _i=_forced: _i      # noqa: E731
                else:
                    # Header said single-slice (or probe failed): keep the
                    # in-recon dialog as a safety net for 4-D data.
                    slice_cb = self._request_slice
            else:
                slice_cb = self._request_slice

            # Select the reconstruction engine (BART vs sigpy) for this job.
            from processing import acquisition, recon_backend, seq_geometry
            recon_backend.set_backend_mode(params.get("recon_backend", "auto"))
            recon_backend.set_espirit_threshold(params.get("espirit_thresh"))
            acquisition.set_prewhitening(params.get("prewhiten", True))
            seq_geometry.set_library_dir(params.get("seq_library_dir") or None)
            seq_geometry.set_seq_request_handler(self._request_seq_file)
            cb(f"Recon engine: {recon_backend.active_backend_label()} "
               f"(mode={recon_backend.get_backend_mode()}, "
               f"BART available={recon_backend.bart_available()})")

            if stype == SCAN_BAND:
                from processing.band_recon import reconstruct_band
                result = reconstruct_band(
                    dat_path       = job["dat_path"],
                    bands_json_path= job["json_path"],
                    band_idx       = job["band_idx"],
                    grid_size      = params["grid_size"],
                    calib_width    = params["calib_width"],
                    scan_fov_m     = params["scan_fov_m"],
                    crop_size      = params.get("crop_size", 0) or None,
                    traj_margin    = params["traj_margin"],
                    espirit_crop   = params.get("espirit_crop", 0.8),
                    tse            = params.get("tse"),
                    progress_cb    = cb,
                )

            elif stype == SCAN_FULLY_SAMPLED:
                from processing.fullysamp_recon import reconstruct_fullysamp
                result = reconstruct_fullysamp(
                    dat_path    = job["dat_path"],
                    use_sense   = params.get("fullysamp_mode", "rss") == "sense",
                    pf_method   = params.get("pf_method", "zerofill"),
                    calib_width = params.get("calib_width", 24),
                    espirit_crop = params.get("espirit_crop", 0.8),
                    tse         = params.get("tse"),
                    necho       = params.get("tse_necho"),
                    te1         = params.get("tse_te1"),
                    teeff       = params.get("tse_teeff"),
                    progress_cb = cb,
                    slice_selector_cb = slice_cb,
                )

            elif stype == SCAN_EQUISPACED:
                from processing.equispaced_recon import reconstruct_equispaced
                ref_path = job.get("reference_dat_path")
                result = reconstruct_equispaced(
                    dat_path            = job["dat_path"],
                    method              = params.get("equi_method", "grappa"),
                    json_path           = job.get("json_path"),
                    mask_method         = params.get("equi_mask_method", "analytical"),
                    acs_lines           = params["acs_lines"],
                    r_factor            = params["r_factor"],
                    kernel_size         = tuple(params["kernel_size"]),
                    lamda               = params["grappa_lambda"],
                    calib_width         = params.get("calib_width", 24),
                    sense_lambda        = params.get("sense_lambda", 0.01),
                    sense_max_iter      = params.get("sense_max_iter", 100),
                    sense_l2            = params.get("sense_l2", False),
                    espirit_num_maps    = params.get("espirit_num_maps", 2),
                    espirit_crop        = params.get("espirit_crop", 0.8),
                    tse                 = params.get("tse"),
                    necho               = params.get("tse_necho"),
                    te1                 = params.get("tse_te1"),
                    teeff               = params.get("tse_teeff"),
                    reference_dat_path  = ref_path,
                    progress_cb         = cb,
                    slice_selector_cb   = slice_cb,
                    mask_fallback_cb    = self._request_mask_fallback,
                    pf_method           = params.get("pf_method", "zerofill"),
                )

            elif stype == SCAN_POISSON:
                from processing.poisson_recon import reconstruct_poisson
                result = reconstruct_poisson(
                    dat_path           = job["dat_path"],
                    poisson_json_path  = job["json_path"],
                    grid_size          = params["grid_size"],
                    calib_width        = params["calib_width"],
                    scan_fov_m         = params["scan_fov_m"],
                    crop_size          = params.get("crop_size", 0) or None,
                    traj_margin        = params["traj_margin"],
                    espirit_crop       = params.get("espirit_crop", 0.8),
                    lambda_wav         = params["lambda_wav"],
                    max_iter           = params["max_iter"],
                    use_zerofill       = params.get("poisson_mode", "pics") == "zerofill",
                    tse                = params.get("tse"),
                    reference_dat_path = job.get("reference_dat_path"),
                    progress_cb        = cb,
                    slice_selector_cb  = slice_cb,
                )

            elif stype == SCAN_RETRO_UNDERSAMPLE:
                from processing.retro_undersample import reconstruct_retro
                result = reconstruct_retro(
                    dat_path        = job["dat_path"],
                    mask_type       = params.get("sim_mask_type",    "equispaced"),
                    R               = params.get("sim_R",            4.0),
                    power_decay     = params.get("sim_power_decay",  1.0),
                    acs_lines       = params.get("sim_acs_lines",    24),
                    recon_method    = params.get("sim_recon_method", "none"),
                    kernel_size     = tuple(params.get("sim_kernel_size",    (5, 5))),
                    grappa_lambda   = params.get("sim_grappa_lambda",   0.1),
                    calib_width     = params.get("calib_width",     24),
                    sense_lambda    = params.get("sim_sense_lambda",    0.01),
                    sense_max_iter  = params.get("sim_sense_iters",  100),
                    sense_l2        = params.get("sense_l2",        False),
                    espirit_num_maps = params.get("espirit_num_maps", 2),
                    espirit_crop    = params.get("espirit_crop",    0.8),
                    swap_pe_read    = params.get("sim_swap_pe_read", False),
                    rng_seed        = params.get("sim_rng_seed"),
                    pf_method       = params.get("pf_method", "zerofill"),
                    tse             = params.get("tse"),
                    necho           = params.get("tse_necho"),
                    te1             = params.get("tse_te1"),
                    teeff           = params.get("tse_teeff"),
                    progress_cb     = cb,
                    slice_selector_cb = slice_cb,
                )

            else:
                # UNKNOWN (or anything unforeseen) — generic fallback so the
                # user always gets an image instead of an error.
                cb(f"[WARN] Unrecognized scan type '{stype}' — using generic "
                   f"fallback recon (mode={params.get('unknown_mode', 'auto')}).")
                from processing.generic_recon import reconstruct_generic
                result = reconstruct_generic(
                    dat_path    = job["dat_path"],
                    json_path   = job.get("json_path"),
                    mode        = params.get("unknown_mode", "auto"),
                    pf_method   = params.get("pf_method", "zerofill"),
                    grid_size   = params["grid_size"],
                    calib_width = params.get("calib_width", 24),
                    scan_fov_m  = params["scan_fov_m"],
                    crop_size   = params.get("crop_size", 0) or None,
                    traj_margin = params["traj_margin"],
                    espirit_crop = params.get("espirit_crop", 0.8),
                    lambda_wav  = params.get("lambda_wav", 0.01),
                    max_iter    = params.get("max_iter", 100),
                    tse         = params.get("tse"),
                    necho       = params.get("tse_necho"),
                    te1         = params.get("tse_te1"),
                    teeff       = params.get("tse_teeff"),
                    progress_cb = cb,
                    slice_selector_cb = slice_cb,
                )

            result["scan_type"] = stype
            result["label"]     = _make_label(job)
            # Record which engine actually ran (BART vs sigpy) for the log,
            # PDF parameter summary, and engine-aware display labels.
            result["recon_engine"] = recon_backend.active_backend_label()
            # Append recon mode suffix for Equispaced tabs
            if stype == SCAN_EQUISPACED:
                method = result.get("method", "grappa")
                result["label"] += f" ({method_label(method)})"
            # Append recon mode suffix for Poisson tabs
            if stype == SCAN_POISSON:
                mode = result.get("recon_mode", "pics")
                suffix = " (ZF)" if mode == "zerofill" else " (CS)"
                result["label"] += suffix
            if stype == SCAN_RETRO_UNDERSAMPLE:
                mtype  = result.get("mask_type", "equispaced").upper()
                actual = result.get("actual_r", 0)
                method = result.get("method", "none")
                msuffix = "" if method == "none" else f" {method_label(method)}"
                result["label"] += f" [{mtype} R={actual:.1f}×{msuffix}]"
            if stype not in (SCAN_BAND, SCAN_FULLY_SAMPLED, SCAN_EQUISPACED, SCAN_POISSON,
                             SCAN_RETRO_UNDERSAMPLE):
                suffix_map = {"nufft": " (NUFFT RSS)", "cs": " (CS)",
                              "cartesian": " (RSS)"}
                result["label"] += suffix_map.get(result.get("recon_path"), " (RSS)")
            result["dat_path"]   = job["dat_path"]
            result["folder_key"] = job.get("folder_key", "")
            # The exact parameters this result was made with (the panel may
            # change before the report is saved).
            result["run_params"] = dict(params)
            self.result_ready.emit(result)

        except JobAborted:
            self.progress_msg.emit("[ABORTED] Job cancelled by user.")

        except Exception as exc:
            self.error_signal.emit(
                f"[ERROR] {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )

        finally:
            try:
                from processing import seq_geometry
                seq_geometry.set_seq_request_handler(None)   # never call a finished worker
            except ImportError:
                pass
            self.finished.emit()


def _ensure_processing_on_path() -> None:
    app_dir = str(Path(__file__).resolve().parent.parent)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)


def _ask_slices(parent: QWidget, n_slices: int) -> List[int]:
    """
    Checkable slice list with the middle slice pre-ticked. Returns the ticked
    slice indices in ascending order, or [] when cancelled / nothing ticked.
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("Multi-Slice Scan Detected")
    layout = QVBoxLayout(dlg)
    layout.addWidget(QLabel(
        f"This scan contains {n_slices} slices (0–{n_slices - 1}).\n"
        "Tick the slices to reconstruct — each one gets its own result tab:"
    ))

    slice_list = QListWidget(dlg)
    for k in range(n_slices):
        item = QListWidgetItem(f"slice {k}")
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if k == n_slices // 2 else Qt.Unchecked)
        slice_list.addItem(item)
    layout.addWidget(slice_list)

    def set_all(state) -> None:
        for i in range(slice_list.count()):
            slice_list.item(i).setCheckState(state)

    row = QHBoxLayout()
    for text, state in (("All", Qt.Checked), ("None", Qt.Unchecked)):
        btn = QPushButton(text, dlg)
        btn.clicked.connect(lambda _checked=False, s=state: set_all(s))
        row.addWidget(btn)
    row.addStretch(1)
    layout.addLayout(row)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dlg)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    if dlg.exec_() != QDialog.Accepted:
        return []
    return [i for i in range(slice_list.count())
            if slice_list.item(i).checkState() == Qt.Checked]


def _make_label(job: Dict) -> str:
    folder_key = job.get("folder_key", "")
    stype      = job.get("scan_type", "")
    bidx       = job.get("band_idx")
    name       = Path(job.get("dat_path", "")).stem
    if stype == SCAN_BAND and bidx is not None:
        base = f"{folder_key} — Band {bidx}"
    elif stype == SCAN_EQUISPACED:
        base = f"{folder_key} — Equispaced"
    elif stype == SCAN_RETRO_UNDERSAMPLE:
        base = f"{folder_key} — Retro Sim"
    elif stype == SCAN_UNKNOWN:
        # Several unknown files may share a folder — keep the stem in the label.
        base = f"{folder_key} — {name}"
    else:
        base = f"{folder_key} — {stype.replace('_', ' ').title()}"
    # Per-slice suffix keeps every slice's tab label unique (multi-slice runs).
    if job.get("n_slices", 1) and job.get("slice_idx") is not None:
        base += f" — slice {int(job['slice_idx']) + 1}/{int(job['n_slices'])}"
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Application main window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            f"SHIR — System for Hybrid Image Reconstruction  ·  v{__version__}"
        )
        try:
            icon = QIcon(resource_path("Graphics", "logo.png"))
            if not icon.isNull():
                self.setWindowIcon(icon)
        except Exception:
            pass   # missing graphics must never block launch
        self.resize(1400, 860)

        self._job_queue: List[Dict]  = []
        self._params: Dict           = {}
        self._results: List[Dict]    = []   # accumulate for PDF
        self._failed_jobs: List[str] = []   # labels of jobs that errored this run
        self._jobs_total             = 0
        self._active_thread: Optional[QThread] = None
        self._active_worker: Optional[ReconWorker] = None

        self._build_ui()
        self._build_toolbar()
        self._build_status_bar()
        self._restore_settings()

    # ─── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Three-pane splitter
        self._splitter = QSplitter(Qt.Horizontal)
        self._folder_panel = FolderPanel()
        self._results_viewer = ResultsViewer()
        self._params_panel = ParamsPanel()

        self._splitter.addWidget(self._folder_panel)
        self._splitter.addWidget(self._results_viewer)
        self._splitter.addWidget(self._params_panel)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)
        self._splitter.setSizes([440, 820, 280])

        # Bottom: progress + log
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate spinner
        self._progress.setVisible(False)
        self._progress.setFixedHeight(18)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(130)
        self._log.setStyleSheet("QTextEdit { font-family: monospace; font-size: 11px; }")

        bottom = QVBoxLayout()
        bottom.setSpacing(2)
        bottom.addWidget(self._progress)
        bottom.addWidget(self._log)

        main_layout.addWidget(self._splitter, stretch=1)
        main_layout.addLayout(bottom)

        # Signals
        self._folder_panel.run_requested.connect(self._on_run_requested)
        self._folder_panel.simulate_requested.connect(self._on_simulate_requested)
        self._params_panel.save_output_requested.connect(self._on_save_output)
        self._results_viewer.result_closed.connect(self._on_result_closed)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        # Plain-text labels: emoji glyphs render as empty boxes on systems
        # without an emoji font (common on minimal Linux/WSL installs).
        act_open = QAction("Open Folder", self)
        act_open.setToolTip("Open a scan folder to analyse")
        act_open.triggered.connect(self._folder_panel._on_open_folder)
        tb.addAction(act_open)

        tb.addSeparator()

        self._act_run = QAction("Run Selected", self)
        self._act_run.setToolTip("Start processing checked scans")
        self._act_run.triggered.connect(self._folder_panel._on_run)
        tb.addAction(self._act_run)

        self._act_simulate = QAction("Undersampling Simulator", self)
        self._act_simulate.setToolTip(
            "Retrospectively undersample a fully-sampled scan and reconstruct.\n"
            "Check at least one Fully-Sampled scan first."
        )
        self._act_simulate.triggered.connect(self._folder_panel._on_simulate)
        tb.addAction(self._act_simulate)

        self._act_stop = QAction("Stop", self)
        self._act_stop.setToolTip(
            "Cancel the remaining queue; the current job stops at its next "
            "pipeline stage"
        )
        self._act_stop.setEnabled(False)
        self._act_stop.triggered.connect(self._on_stop)
        tb.addAction(self._act_stop)

        tb.addSeparator()

        self._act_coils = QAction("All Coils", self)
        self._act_coils.setToolTip("Open all-coils viewer for current tab")
        self._act_coils.setEnabled(False)
        self._act_coils.triggered.connect(self._on_all_coils)
        tb.addAction(self._act_coils)

        self._act_clear = QAction("Clear Tabs", self)
        self._act_clear.setToolTip("Remove all result tabs")
        self._act_clear.setEnabled(False)
        self._act_clear.triggered.connect(self._on_clear_results)
        tb.addAction(self._act_clear)

        tb.addSeparator()

        act_about = QAction("About", self)
        act_about.triggered.connect(self._on_about)
        tb.addAction(act_about)

    def _build_status_bar(self) -> None:
        self._status_label = QLabel("Ready")
        self.statusBar().addPermanentWidget(self._status_label)

    # ─── Job execution ────────────────────────────────────────────────────────

    def _start_job_queue(self, jobs: List[Dict], banner: str) -> None:
        """Shared run/simulate entry: snapshot params, validate, start the queue."""
        params = self._params_panel.get_params()
        self._params = params
        self._npy_warned = False
        self._warn_questionable_params(params)
        if (params.get("gen_pdf") or params.get("save_npy")) \
                and not self._check_save_dir_writable(params.get("save_dir", "")):
            return

        self._job_queue = list(jobs)
        self._failed_jobs = []
        self._jobs_total = len(jobs)
        self._log_msg(f"{banner}Queued {len(jobs)} job(s). Starting…")
        self._run_next_job()

    def _on_run_requested(self, jobs: List[Dict]) -> None:
        if self._active_thread is not None and self._active_thread.isRunning():
            QMessageBox.warning(self, "Busy", "A reconstruction is already running.\nPlease wait or stop it first.")
            return
        # Attach fully-sampled reference to undersampled jobs in same folder
        jobs = _attach_references(jobs)
        self._start_job_queue(jobs, "")

    def _on_simulate_requested(self, jobs: List[Dict]) -> None:
        """Handle jobs from the Simulate button (RETRO_UNDERSAMPLE scan type)."""
        if self._active_thread is not None and self._active_thread.isRunning():
            QMessageBox.warning(self, "Busy", "A reconstruction is already running.\nPlease wait or stop it first.")
            return
        self._start_job_queue(jobs, "[Simulate] ")

    def _warn_questionable_params(self, params: Dict) -> None:
        """Log (never block) warnings for technically-valid-but-unusual values."""
        for key, name in (("kernel_size", "GRAPPA kernel"),
                          ("sim_kernel_size", "Simulator GRAPPA kernel")):
            k = params.get(key) or ()
            if any(int(v) % 2 == 0 for v in k):
                self._log_msg(
                    f"⚠ {name} {tuple(k)} has an even dimension — GRAPPA kernels "
                    f"are normally odd (e.g. 5×5) so they centre on the target sample.",
                    error=True,
                )
        grid = int(params.get("grid_size", 0))
        if grid and (grid & (grid - 1)) != 0:
            self._log_msg(
                f"⚠ Grid size {grid} is not a power of 2 — NUFFT gridding is "
                f"fastest and best-tested at 256/512/1024.",
                error=True,
            )

    def _check_save_dir_writable(self, save_dir: str) -> bool:
        """Create/verify the save directory before a run that will write to it."""
        try:
            p = Path(save_dir).expanduser()
            p.mkdir(parents=True, exist_ok=True)
            if os.access(str(p), os.W_OK):
                return True
            raise OSError("no write permission")
        except OSError as exc:
            QMessageBox.warning(
                self, "Save directory not writable",
                f"Cannot write to the Save directory:\n{save_dir}\n\n({exc})\n\n"
                "Choose another directory in the Output panel, or untick the "
                "PDF/.npy save options.",
            )
            return False

    def _run_next_job(self) -> None:
        if not self._job_queue:
            self._on_all_jobs_done()
            return

        job = self._job_queue.pop(0)
        self._current_job_label = _make_label(job)
        job_no = self._jobs_total - len(self._job_queue)
        self._log_msg(f"\n── [{job_no}/{self._jobs_total}] {self._current_job_label} ──")
        self._set_running(True)
        if self._jobs_total > 1:
            # Overall queue progress; the busy state of the current job is
            # conveyed by the status text and streaming log.
            self._progress.setRange(0, self._jobs_total)
            self._progress.setValue(job_no - 1)
            self._progress.setFormat(f"Job {job_no} of {self._jobs_total}")
        else:
            self._progress.setRange(0, 0)   # indeterminate spinner
        self._status_label.setText(
            f"Job {job_no}/{self._jobs_total}: {self._current_job_label}"
        )

        worker = ReconWorker(job, self._params)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress_msg.connect(self._log_msg)
        worker.result_ready.connect(self._on_result_ready)
        worker.error_signal.connect(self._on_worker_error)
        worker.slice_selection_needed.connect(self._on_slice_selection_needed)
        worker.mask_fallback_needed.connect(self._on_mask_fallback_needed)
        worker.seq_file_needed.connect(self._on_seq_file_needed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._run_next_job)   # chain next job
        thread.start()

        self._active_thread = thread
        self._active_worker = worker

    def _on_result_ready(self, result: Dict) -> None:
        label = self._unique_label(result.get("label", "Result"))
        result["label"] = label
        self._results.append(result)
        self._results_viewer.add_result(result, label)
        self._log_msg(f"  ✓  {label}")
        self._update_result_actions()

        # Auto-save .npy if requested
        if self._params.get("save_npy"):
            self._save_npy(result)

    def _on_result_closed(self, result: object) -> None:
        """A tab was closed — drop its result so PDFs/.npy saves and memory
        stay in sync with what is on screen."""
        self._results = [r for r in self._results if r is not result]
        self._update_result_actions()

    def _on_slice_selection_needed(self, n_slices: int) -> None:
        """
        Main-thread slot: ask which slice(s) to reconstruct. The waiting worker
        reconstructs the first one; the others are queued right after it as
        per-slice jobs (the same jobs the tree's "Select slices…" creates).
        """
        chosen = _ask_slices(self, n_slices)
        if not chosen:
            chosen = [n_slices // 2]
            self._log_msg(f"  No slice chosen — using middle slice {chosen[0]}")
        else:
            self._log_msg(f"  Slice(s) selected: {', '.join(map(str, chosen))}")

        worker = getattr(self, "_active_worker", None)
        if worker is None:
            return
        # Tag the running job with its slice so its tab label names the slice
        # (the worker builds the label from this dict when it finishes).
        job = worker._job
        job["slice_idx"], job["n_slices"] = chosen[0], n_slices
        self._current_job_label = _make_label(job)
        extra = [dict(job, slice_idx=k) for k in chosen[1:]]
        if extra:
            self._job_queue[0:0] = extra
            self._jobs_total += len(extra)
            self._log_msg(f"  Queued {len(extra)} more slice job(s) for this scan.")
        worker.set_slice_choice(chosen[0])

    def _on_mask_fallback_needed(self) -> None:
        """Main-thread slot: ask the user which mask to build when no JSON trajectory is found."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("No JSON Trajectory Found")
        box.setText(
            "No undersampling JSON trajectory was found for this scan.\n"
            "Choose how to build the phase-encode sampling mask:"
        )
        btn_analytical = box.addButton("Analytical (computed)", QMessageBox.AcceptRole)
        btn_bart       = box.addButton("BART pattern",          QMessageBox.RejectRole)
        box.setDefaultButton(btn_analytical)
        box.exec_()

        choice = "bart_pattern" if box.clickedButton() is btn_bart else "analytical"
        self._log_msg(f"  Mask fallback: {choice}")
        if hasattr(self, "_active_worker") and self._active_worker is not None:
            self._active_worker.set_mask_choice(choice)

    def _on_seq_file_needed(self, info: Dict) -> None:
        """Main-thread slot: the played .seq of a Pulseq scan was not found."""
        worker = getattr(self, "_active_worker", None)
        name = Path(str(info.get("name") or "the sequence")).name
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Sequence file not found")
        box.setText(
            f"This Pulseq scan was acquired with <b>{name}</b>, but that exact file "
            f"(signature {info.get('signature')}) was not found next to the scan or in "
            f"the .seq library.<br><br>"
            "The .seq gives the true position of every k-space line (echo order, "
            "direction). Without it, the trajectory JSON or the file name and line "
            "count are used instead.")
        btn_browse = box.addButton("Locate .seq…", QMessageBox.AcceptRole)
        box.addButton("Continue without", QMessageBox.RejectRole)
        box.setDefaultButton(btn_browse)
        box.exec_()
        path = None
        if box.clickedButton() is btn_browse:
            path, _ = QFileDialog.getOpenFileName(
                self, f"Locate {name}", str(Path(self._params.get("seq_library_dir") or "~")
                                             .expanduser()),
                "Pulseq sequence (*.seq)")
            path = path or None
        self._log_msg(f"  .seq: {path}" if path else "  .seq not located — using the fallback.")
        if worker is not None:
            worker.set_seq_choice(path)

    def _on_worker_error(self, msg: str) -> None:
        self._failed_jobs.append(getattr(self, "_current_job_label", "?"))
        self._log_msg(msg, error=True)
        self._status_label.setText("Job failed — see log")

    def _set_running(self, running: bool) -> None:
        """Single place that flips every run-state control (toolbar actions,
        folder-panel buttons, progress bar)."""
        self._act_run.setEnabled(not running)
        self._act_simulate.setEnabled(not running)
        self._act_stop.setEnabled(running)
        self._progress.setVisible(running)
        self._folder_panel.set_running(running)

    def _update_result_actions(self) -> None:
        """Enable result-dependent actions only when there is something to act on."""
        has = bool(self._results)
        self._act_coils.setEnabled(has)
        self._act_clear.setEnabled(has)

    def _on_all_jobs_done(self) -> None:
        self._set_running(False)
        # Clear references so the deleted C++ Qt objects are not accessed on the
        # next run (calling .isRunning() on a deleteLater'd QThread crashes the app).
        self._active_thread = None
        self._active_worker = None

        if self._failed_jobs:
            self._status_label.setText(
                f"Done — {len(self._results)} result(s), "
                f"{len(self._failed_jobs)} job(s) FAILED"
            )
            self._log_msg(
                f"\n⚠ Finished with {len(self._failed_jobs)} failed job(s).",
                error=True,
            )
            QMessageBox.warning(
                self, "Some jobs failed",
                f"{len(self._failed_jobs)} of {self._jobs_total} job(s) failed:\n\n"
                + "\n".join(f"• {lbl}" for lbl in self._failed_jobs)
                + "\n\nSee the log console for details.",
            )
        else:
            self._status_label.setText(f"Done — {len(self._results)} result(s)")
            self._log_msg("\n✓ All jobs complete.")

        if self._params.get("gen_pdf") and self._results:
            self._generate_and_save_pdf()

    def _on_stop(self) -> None:
        self._job_queue.clear()
        if self._active_worker is not None:
            self._active_worker.abort()
        self._log_msg("⚠ Stop requested — cancelling at the next pipeline stage.")
        self._act_stop.setEnabled(False)

    # ─── Settings persistence ─────────────────────────────────────────────────

    def _restore_settings(self) -> None:
        """Restore window geometry + persisted panel settings. A corrupt or
        missing setting must never block launch."""
        try:
            s = QSettings("Technion", "SHIR")
            geo = s.value("window/geometry")
            if geo is not None:
                self.restoreGeometry(geo)
            split = s.value("window/splitter")
            if split is not None:
                self._splitter.restoreState(split)
            self._params_panel.load_persistent(s)
        except Exception:
            pass
        self._fit_to_screen()

    def _fit_to_screen(self) -> None:
        """Keep the whole window on its screen: no wider or taller than the
        available area (a saved geometry from a larger monitor, or panel
        minimum widths, must not push it past the edge), and fully inside it."""
        try:
            screen = self.screen() or QApplication.primaryScreen()
            if screen is None or self.isMaximized() or self.isFullScreen():
                return
            avail = screen.availableGeometry()
            frame, inner = self.frameGeometry(), self.geometry()
            dw, dh = frame.width() - inner.width(), frame.height() - inner.height()
            w = min(inner.width(), avail.width() - dw)
            h = min(inner.height(), avail.height() - dh)
            if (w, h) != (inner.width(), inner.height()):
                self.resize(w, h)
            frame = self.frameGeometry()
            x = min(max(frame.x(), avail.x()), avail.x() + avail.width() - frame.width())
            y = min(max(frame.y(), avail.y()), avail.y() + avail.height() - frame.height())
            if (x, y) != (frame.x(), frame.y()):
                self.move(max(x, avail.x()), max(y, avail.y()))
        except Exception:
            pass   # window placement must never block launch

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not getattr(self, "_fitted_once", False):
            self._fitted_once = True
            QTimer.singleShot(0, self._fit_to_screen)     # after the first layout pass

    def _save_settings(self) -> None:
        try:
            s = QSettings("Technion", "SHIR")
            s.setValue("window/geometry", self.saveGeometry())
            s.setValue("window/splitter", self._splitter.saveState())
            self._params_panel.save_persistent(s)
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        """Shut down cleanly: never destroy a QThread that is still running."""
        self._save_settings()
        thread = self._active_thread
        if thread is None or not thread.isRunning():
            event.accept()
            return

        reply = QMessageBox.question(
            self, "Reconstruction running",
            "A reconstruction is still running.\nAbort it and quit?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            event.ignore()
            return

        self._job_queue.clear()
        try:
            # Don't chain into the next job while shutting down.
            thread.finished.disconnect(self._run_next_job)
        except TypeError:
            pass
        if self._active_worker is not None:
            self._active_worker.abort()   # also unblocks slice/mask dialog waits
        thread.quit()
        if not thread.wait(15000):
            # An in-flight BART/sigpy stage can take a while; give the user the
            # choice instead of hanging forever or crashing Qt.
            force = QMessageBox.question(
                self, "Still running",
                "The current reconstruction stage has not finished yet.\n"
                "Force quit anyway? (Temporary files may be left behind.)",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if force != QMessageBox.Yes:
                event.ignore()
                return
        event.accept()

    # ─── Output saving ─────────────────────────────────────────────────────────

    def _inject_display_settings(self) -> None:
        """Copy each tab's W/L + brightness + rotation onto its result dict."""
        display_map = self._results_viewer.get_display_settings_map()
        for r in self._results:
            settings = display_map.get(r.get("label", ""), {})
            r["_wl_min"]     = settings.get("wl_min", 0)
            r["_wl_max"]     = settings.get("wl_max", 100)
            r["_brightness"] = settings.get("brightness", 1.0)
            r["_rotation"]   = settings.get("rotation", 0)

    @staticmethod
    def _group_by_folder(results: List[Dict]) -> "Dict[str, List[Dict]]":
        groups: Dict[str, List[Dict]] = {}
        for r in results:
            fkey = r.get("folder_key") or Path(r.get("dat_path", "unknown")).parent.parent.name
            groups.setdefault(fkey, []).append(r)
        return groups

    @staticmethod
    def _pick_combined(result: Dict):
        """First available combined image (explicit None check — arrays are not truthy)."""
        for key in ("combined_img", "image_rss", "grappa_image", "sense_image"):
            img = result.get(key)
            if img is not None:
                return img
        return None

    def _generate_and_save_pdf(self) -> None:
        """Auto-save one PDF per folder into the run's save_dir (end-of-run path)."""
        save_dir = Path(self._params.get("save_dir", "."))
        save_dir.mkdir(parents=True, exist_ok=True)

        self._inject_display_settings()

        for fkey, group_results in self._group_by_folder(self._results).items():
            pdf_path = str(save_dir / f"report_{fkey}.pdf")
            try:
                generate_pdf(
                    results       = group_results,
                    output_path   = pdf_path,
                    folder_name   = fkey,
                    params_summary= self._params,
                )
                self._log_msg(f"  PDF saved: {pdf_path}")
            except Exception as e:
                self._log_msg(f"  ⚠ PDF generation failed for {fkey}: {e}")

    def _on_save_output(self) -> None:
        """Manual save: prompt per folder and write the checked outputs (PDF/.npy)."""
        if not self._results:
            QMessageBox.information(self, "No results", "Run some reconstructions first.")
            return

        # Read LIVE panel state (not the params captured at run start), so the
        # current Save directory and checkbox choices are honoured.
        p        = self._params_panel.get_params()
        save_dir = p.get("save_dir", ".")
        gen_pdf  = bool(p.get("gen_pdf"))
        save_npy = bool(p.get("save_npy"))

        if not gen_pdf and not save_npy:
            QMessageBox.information(
                self, "Nothing selected",
                "Check 'Generate PDF report' and/or 'Save combined images (.npy)' first.",
            )
            return

        if gen_pdf:
            self._inject_display_settings()

        # View toggles (Center K-space / Crop to BBox) replace the displayed
        # arrays. The PDF deliberately matches the screen; for .npy exports let
        # the user choose between the displayed and the raw reconstruction.
        use_raw_npy = False
        if save_npy and any(r.get("_view_modified") for r in self._results):
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle("Save .npy — modified view active")
            box.setText(
                "Some result tabs have active view transforms\n"
                "(Center K-space / Crop to BBox).\n\n"
                "Save the .npy images as displayed (modified) or as the\n"
                "original raw reconstruction?"
            )
            btn_mod = box.addButton("As displayed (modified)", QMessageBox.AcceptRole)
            btn_raw = box.addButton("Raw reconstruction", QMessageBox.RejectRole)
            box.setDefaultButton(btn_mod)
            box.exec_()
            use_raw_npy = box.clickedButton() is btn_raw
            self._log_msg(
                f"  .npy export: {'raw reconstruction' if use_raw_npy else 'as displayed'}"
            )

        saved: List[str] = []
        for fkey, group_results in self._group_by_folder(self._results).items():
            chosen, _ = QFileDialog.getSaveFileName(
                self, f"Save Output — {fkey}",
                str(Path(save_dir) / fkey), "All Files (*)",
            )
            if not chosen:
                self._log_msg(f"  Save cancelled for {fkey} — skipped.")
                continue

            stem    = Path(chosen).stem
            out_dir = Path(chosen).parent
            out_dir.mkdir(parents=True, exist_ok=True)

            if gen_pdf:
                pdf_path = str(out_dir / f"{stem}.pdf")
                try:
                    generate_pdf(
                        results       = group_results,
                        output_path   = pdf_path,
                        folder_name   = fkey,
                        # Parameters of the run that produced these results, not
                        # the panel's current state.
                        params_summary= group_results[0].get("run_params", p),
                    )
                    saved.append(pdf_path)
                    self._log_msg(f"  PDF saved: {pdf_path}")
                except Exception as e:
                    self._log_msg(f"  ⚠ PDF generation failed for {fkey}: {e}")

            if save_npy:
                multi = len(group_results) > 1
                for r in group_results:
                    if use_raw_npy and r.get("_view_modified") \
                            and r.get("_pristine_recon") is not None:
                        combined = r["_pristine_recon"]
                    else:
                        combined = self._pick_combined(r)
                    if combined is None:
                        continue
                    if multi:
                        label = r.get("label", "result").replace(" ", "_").replace("—", "-")
                        npy_path = out_dir / f"{stem}_{label}.npy"
                    else:
                        npy_path = out_dir / f"{stem}.npy"
                    np.save(str(npy_path), np.squeeze(combined))
                    saved.append(str(npy_path))
                    self._log_msg(f"  .npy saved: {npy_path}")

        if saved:
            self._status_label.setText(f"Saved {len(saved)} file(s)")
            QMessageBox.information(
                self, "Save output",
                f"Saved {len(saved)} file(s):\n" + "\n".join(saved),
            )

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _unique_label(self, label: str) -> str:
        """
        Ensure the label is unique among current results, suffixing " (2)", " (3)"…
        Duplicate labels would collide in the per-tab display-settings map and
        overwrite each other's auto-saved .npy files.
        """
        existing = {r.get("label") for r in self._results}
        if label not in existing:
            return label
        n = 2
        while f"{label} ({n})" in existing:
            n += 1
        return f"{label} ({n})"

    def _save_npy(self, result: Dict) -> None:
        try:
            save_dir = Path(self._params.get("save_dir", "."))
            save_dir.mkdir(parents=True, exist_ok=True)
            label = result.get("label", "result").replace(" ", "_").replace("—", "-")
            combined = self._pick_combined(result)
            if combined is not None:
                npy_path = save_dir / f"{label}.npy"
                np.save(str(npy_path), np.squeeze(combined))
                self._log_msg(f"  .npy saved: {npy_path}")
        except OSError as exc:
            # A bad/unwritable save dir must not kill the run; warn once.
            self._log_msg(f"  ⚠ .npy auto-save failed: {exc}", error=True)
            if not getattr(self, "_npy_warned", False):
                self._npy_warned = True
                self._log_msg(
                    "  ⚠ Check the Save directory in the Output panel.", error=True
                )

    def _on_all_coils(self) -> None:
        if not self._results_viewer.show_all_coils_current():
            self._status_label.setText("No per-coil images on the current tab")

    def _on_clear_results(self) -> None:
        self._results_viewer.clear_all()
        self._results.clear()
        self._update_result_actions()
        self._log_msg("All result tabs cleared.")

    def _log_msg(self, msg: str, error: bool = False) -> None:
        self._log.moveCursor(QTextCursor.End)
        if error:
            import html
            body = html.escape(msg).replace("\n", "<br>")
            self._log.insertHtml(f'<span style="color:#c0392b;">{body}</span><br>')
        else:
            self._log.insertPlainText(msg + "\n")
        self._log.moveCursor(QTextCursor.End)

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About SHIR",
            f"<b>SHIR</b> — System for Hybrid Image Reconstruction<br>"
            f"v{__version__}<br><br>"
            "Multi-pipeline reconstruction of Siemens TWIX .dat files.<br><br>"
            "Supported modes:<br>"
            "&nbsp;• Band (NUFFT + ESPIRiT + SENSE)<br>"
            "&nbsp;• Fully-sampled (IFFT + RSS or ESPIRiT + SENSE)<br>"
            "&nbsp;• Equispaced (GRAPPA / SENSE / CS-SENSE / ESPIRiT ± L1/L2/TV)<br>"
            "&nbsp;• Poisson-disc CS (PICS, L1-wavelet)<br>"
            "&nbsp;• Retrospective undersampling simulator<br>"
            "&nbsp;• Unknown scans — generic fallback (NUFFT / CS / Cartesian RSS)<br><br>"
            "Reconstruction engine: <b>BART</b> when available (native or via WSL "
            "on Windows), otherwise the built-in <b>sigpy</b> engine — see the "
            "Recon engine selector in the parameters panel.<br><br>"
            "Developed by Saar Drive, Shimron Lab, Technion.<br><br>"
            "<i>SHIR is named after — and dedicated to — Shir, my wife, in "
            "gratitude for her long and unconditional support on my journey to "
            "become a researcher.</i>",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _attach_references(jobs: List[Dict]) -> List[Dict]:
    """
    For EQUISPACED jobs, find a suitable fully-sampled reference in the same folder.

    Two reference categories:
      * custom ("mine")  — file containing "fully_sampled" in the name
      * Siemens SE       — file matching Siemens SE patterns (*_se_tra*.dat,
                           *t1_se*, *_se_*tra*) but NOT a GRAPPA file

    Siemens GRAPPA scans (filenames containing "t1_fl2d_tra", "Siemens_", or
    "_equispaced_R") are routed to the Siemens SE reference; all other scans
    are routed to the custom fully-sampled reference.

    When a folder holds several fully-sampled scans, the one from the same
    sequence family (a Pulseq scan is not scored against a Siemens product
    scan that happens to be named "fully_sampled") and with matching
    swapPE-ness is preferred: a swapPE reference has readout and phase
    swapped, so pairing it with a non-swapPE scan (or vice versa) compares a
    rotated/mirrored image and wrecks the metrics.
    """
    import re as _re
    _SIEMENS_GRAPPA = _re.compile(
        r"(Siemens_|t1_fl2d_tra|_fl2d_tra)", _re.IGNORECASE
    )
    _SIEMENS_SE = _re.compile(
        r"(t1_se_tra|_se_tra|t1_se_)", _re.IGNORECASE
    )
    _GRAPPA_SE_EXCLUSION = _re.compile(
        r"(grappa|fl2d|equispaced|poisson|band)", _re.IGNORECASE
    )

    def _is_swap(path: str) -> bool:
        return "swappe" in Path(path).name.lower()

    # Build per-folder reference maps
    fs_map: Dict[str, List[str]]   = {}   # custom fully-sampled (all candidates)
    siemens_se_map: Dict[str, str] = {}   # Siemens SE

    for j in jobs:
        dat_dir = str(Path(j["dat_path"]).parent)
        fname   = Path(j["dat_path"]).name
        if j["scan_type"] == SCAN_FULLY_SAMPLED:
            fs_map.setdefault(dat_dir, []).append(j["dat_path"])
        elif _SIEMENS_SE.search(fname) and not _GRAPPA_SE_EXCLUSION.search(fname):
            siemens_se_map[dat_dir] = j["dat_path"]

    # Disk scan: look for files not already in the job list
    for j in jobs:
        dat_dir = Path(j["dat_path"]).parent
        dat_dir_str = str(dat_dir)

        # sorted() keeps the pick deterministic when several candidates exist
        # (bare glob order is filesystem-dependent).
        if dat_dir_str not in fs_map:
            candidates = sorted(dat_dir.glob("*fully_sampled*.dat"))
            if candidates:
                fs_map[dat_dir_str] = [str(c) for c in candidates]

        if dat_dir_str not in siemens_se_map:
            for f in sorted(dat_dir.glob("*.dat")):
                if _SIEMENS_SE.search(f.name) and not _GRAPPA_SE_EXCLUSION.search(f.name):
                    siemens_se_map[dat_dir_str] = str(f)
                    break

    from processing.utils import sequence_family

    def _custom_ref(dat_dir: str, scan_path: str) -> Optional[str]:
        """Fully-sampled reference from the same sequence family (Pulseq vs
        Siemens product, read from the header) with matching swapPE-ness;
        else the closest available."""
        cands = sorted(fs_map.get(dat_dir, []))
        if not cands:
            return None
        family = sequence_family(scan_path)
        return min(cands, key=lambda c: (sequence_family(c) != family,
                                         _is_swap(c) != _is_swap(scan_path), c))

    for j in jobs:
        # A user-set reference (folder-panel "Set reference") is never overwritten.
        if j.get("reference_override"):
            continue
        if j["scan_type"] == SCAN_EQUISPACED:
            dat_dir = str(Path(j["dat_path"]).parent)
            fname   = Path(j["dat_path"]).name
            if _SIEMENS_GRAPPA.search(fname):
                # Siemens GRAPPA scan → prefer Siemens SE, fall back to custom
                ref = siemens_se_map.get(dat_dir) or _custom_ref(dat_dir, j["dat_path"])
            else:
                # Custom scan → prefer custom fully-sampled
                ref = _custom_ref(dat_dir, j["dat_path"]) or siemens_se_map.get(dat_dir)
            if ref:
                j["reference_dat_path"] = ref
        elif j["scan_type"] == SCAN_POISSON:
            # Poisson-disc is non-Cartesian; just take any fully-sampled in the
            # folder (custom preferred, else the Siemens SE) for comparison.
            dat_dir = str(Path(j["dat_path"]).parent)
            ref = _custom_ref(dat_dir, j["dat_path"]) or siemens_se_map.get(dat_dir)
            if ref:
                j["reference_dat_path"] = ref
    return jobs
