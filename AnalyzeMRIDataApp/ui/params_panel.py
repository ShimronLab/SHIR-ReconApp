"""
ParamsPanel — right-side panel for all reconstruction parameters.

Groups (top to bottom): Common (NUFFT/ESPIRiT + Recon engine) · Fully Sampled ·
TSE echo order · Equispaced Reconstruction · Poisson Disc Reconstruction ·
Unknown Scans (generic fallback) · Undersampling Simulator · Output.

Method combos enable/disable their dependent widgets; all values are read as a
plain dict via .get_params(). Method names are engine-neutral — whether BART or
sigpy runs the ESPIRiT/SENSE-family methods is chosen by the "Recon engine"
selector (processing.recon_backend), not by the method name.
"""

from pathlib import Path

from PyQt5.QtCore import QStandardPaths, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)


class ParamsPanel(QWidget):
    """Right-side parameter panel."""

    save_output_requested = pyqtSignal()   # emitted by the "Save output" button

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(420)   # wide enough for the longest method names

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        vbox  = QVBoxLayout(inner)
        vbox.setSpacing(8)
        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.addWidget(QLabel("<b>Reconstruction Parameters</b>"))
        outer.addWidget(scroll)

        # ── Common ────────────────────────────────────────────────────────────
        grp_common = QGroupBox("Common (NUFFT gridding & engine)")
        form_common = QFormLayout(grp_common)

        self.spin_grid = QSpinBox()
        self.spin_grid.setRange(64, 4096)
        self.spin_grid.setSingleStep(64)
        self.spin_grid.setValue(512)
        self.spin_grid.setToolTip("BART NUFFT oversampled grid size (power of 2 recommended)")
        form_common.addRow("Grid size:", self.spin_grid)

        self.dspin_fov_mm = QDoubleSpinBox()
        self.dspin_fov_mm.setRange(10.0, 500.0)
        self.dspin_fov_mm.setDecimals(1)
        self.dspin_fov_mm.setSingleStep(1.0)
        self.dspin_fov_mm.setValue(256.0)
        self.dspin_fov_mm.setToolTip(
            "Scan field-of-view in mm.  Used to auto-compute the crop size "
            "so it matches the trajectory scaling."
        )
        form_common.addRow("Scan FOV (mm):", self.dspin_fov_mm)

        self.spin_crop = QSpinBox()
        self.spin_crop.setRange(0, 4096)
        self.spin_crop.setSingleStep(2)
        self.spin_crop.setValue(0)
        self.spin_crop.setSpecialValueText("Auto (from FOV)")
        self.spin_crop.setToolTip(
            "Image-domain crop size (pixels) for NUFFT recons (Band / Poisson).\n"
            "0 = Auto: compute from Scan FOV and the trajectory scaling.\n"
            "Set a value to force a fixed output size (e.g. 220 or 320)."
        )
        form_common.addRow("Crop size (px):", self.spin_crop)

        self.dspin_traj_margin = QDoubleSpinBox()
        self.dspin_traj_margin.setRange(0.0, 32.0)
        self.dspin_traj_margin.setDecimals(1)
        self.dspin_traj_margin.setSingleStep(0.5)
        self.dspin_traj_margin.setValue(1.0)
        self.dspin_traj_margin.setToolTip(
            "Margin (grid-index pixels) subtracted from N/2 when scaling the "
            "trajectory so samples stay within BART's exclusive upper bound."
        )
        form_common.addRow("Traj margin (px):", self.dspin_traj_margin)

        self.combo_backend = QComboBox()
        self.combo_backend.addItem("Auto (BART if available)", "auto")
        self.combo_backend.addItem("Force BART",               "bart")
        self.combo_backend.addItem("Force sigpy (no BART)",    "sigpy")
        self.combo_backend.setToolTip(
            "Reconstruction engine for ESPIRiT / SENSE / CS / NUFFT methods.\n"
            "Auto: use BART (via WSL) if available, else fall back to sigpy.\n"
            "Force BART: always use BART (errors/degrades if unavailable).\n"
            "Force sigpy: always use the pure-Python sigpy engine — no WSL/BART "
            "needed (results differ slightly from BART; slower)."
        )
        form_common.addRow("Recon engine:", self.combo_backend)

        vbox.addWidget(grp_common)

        # ── ESPIRiT (shared by every method that estimates coil maps) ──────────
        grp_espirit = QGroupBox("ESPIRiT coil maps (all methods, both engines)")
        form_espirit = QFormLayout(grp_espirit)

        self.spin_calib = QSpinBox()
        self.spin_calib.setRange(4, 256)
        self.spin_calib.setSingleStep(2)
        self.spin_calib.setValue(24)
        self.spin_calib.setToolTip(
            "ESPIRiT calibration region width (pixels). Applies to every method "
            "that uses ESPIRiT coil maps: fully-sampled SENSE, equispaced "
            "SENSE/CS/ESPIRiT, Poisson-disc CS, non-Cartesian band, and the "
            "simulator."
        )
        form_espirit.addRow("ESPIRiT calib width:", self.spin_calib)

        self.dspin_espirit_crop = QDoubleSpinBox()
        self.dspin_espirit_crop.setRange(0.0, 1.0)
        self.dspin_espirit_crop.setDecimals(2)
        self.dspin_espirit_crop.setSingleStep(0.05)
        self.dspin_espirit_crop.setValue(0.8)
        self.dspin_espirit_crop.setToolTip(
            "ESPIRiT eigenvalue mask threshold, applied to BOTH engines "
            "(BART ecalib -c, sigpy EspiritCalib crop) across ALL ESPIRiT "
            "methods (fully-sampled SENSE, equispaced, Poisson CS, band, "
            "simulator).\n"
            "Lower keeps more (low-SNR) voxels; higher masks more aggressively. "
            "0.8 matches BART's default and keeps the two engines comparable."
        )
        form_espirit.addRow("ESPIRiT crop:", self.dspin_espirit_crop)

        vbox.addWidget(grp_espirit)

        # ── Fully Sampled ─────────────────────────────────────────────────────
        grp_fs = QGroupBox("Fully Sampled")
        form_fs = QFormLayout(grp_fs)

        self.combo_fs_mode = QComboBox()
        self.combo_fs_mode.addItem("ESPIRiT + SENSE",            "sense")
        self.combo_fs_mode.addItem("Root Sum of Squares (RSS)",  "rss")
        self.combo_fs_mode.setToolTip(
            "Choose how coils are combined for fully-sampled data.\n"
            "RSS is fast and parameter-free.\n"
            "ESPIRiT + SENSE uses BART to estimate sensitivity maps and "
            "produces a SENSE-combined image (uses ESPIRiT calib width above)."
        )
        form_fs.addRow("Combination mode:", self.combo_fs_mode)

        vbox.addWidget(grp_fs)

        # ── TSE echo un-scramble ──────────────────────────────────────────────
        grp_tse = QGroupBox("TSE echo order")
        form_tse = QFormLayout(grp_tse)

        self.chk_tse_auto = QCheckBox("Auto-detect from filename")
        self.chk_tse_auto.setChecked(True)
        self.chk_tse_auto.setToolTip(
            "TSE scans are acquired in echo-train order (no LIN label), so the\n"
            "Cartesian recons (Fully-Sampled, Equispaced) must un-scramble each\n"
            "line to its true PE position. Auto: 'tse' in the filename → necho=16,\n"
            "contrast t1→echo1 / t2→echo8, TE1=12 ms. NUFFT recons (Band/Poisson)\n"
            "never need this. Untick to force the overrides below."
        )
        form_tse.addRow(self.chk_tse_auto)

        self.spin_tse_necho = QSpinBox()
        self.spin_tse_necho.setRange(1, 64)
        self.spin_tse_necho.setValue(16)
        self.spin_tse_necho.setToolTip("Turbo factor (echoes per shot). 1 = SE (no un-scramble).")
        form_tse.addRow("necho (override):", self.spin_tse_necho)

        self.dspin_tse_te1 = QDoubleSpinBox()
        self.dspin_tse_te1.setRange(0.1, 500.0)
        self.dspin_tse_te1.setDecimals(1)
        self.dspin_tse_te1.setValue(12.0)
        self.dspin_tse_te1.setToolTip("Echo spacing TE1 in ms (override).")
        form_tse.addRow("TE1 ms (override):", self.dspin_tse_te1)

        self.dspin_tse_teeff = QDoubleSpinBox()
        self.dspin_tse_teeff.setRange(0.1, 2000.0)
        self.dspin_tse_teeff.setDecimals(1)
        self.dspin_tse_teeff.setValue(96.0)
        self.dspin_tse_teeff.setToolTip(
            "Effective TE in ms (override). round(TEeff/TE1) picks the central "
            "echo (12→echo1 T1, 96→echo8 T2)."
        )
        form_tse.addRow("TEeff ms (override):", self.dspin_tse_teeff)

        # Overrides are only active when auto-detect is off
        def _toggle_tse_overrides(checked: bool) -> None:
            for w in (self.spin_tse_necho, self.dspin_tse_te1, self.dspin_tse_teeff):
                w.setEnabled(not checked)
        self.chk_tse_auto.toggled.connect(_toggle_tse_overrides)
        _toggle_tse_overrides(self.chk_tse_auto.isChecked())

        vbox.addWidget(grp_tse)

        # ── Equispaced ────────────────────────────────────────────────────────
        grp_grappa = QGroupBox("Equispaced Reconstruction")
        form_grappa = QFormLayout(grp_grappa)

        self.combo_equi_method = QComboBox()
        self.combo_equi_method.addItem("GRAPPA",             "grappa")
        self.combo_equi_method.addItem("GRAPPA + SENSE",     "grappa_sense")
        self.combo_equi_method.addItem("Pure SENSE",         "sense")
        self.combo_equi_method.addItem("CS-SENSE",           "sense_cs")
        self.combo_equi_method.addItem("ESPIRiT",            "espirit")
        self.combo_equi_method.addItem("ESPIRiT + L1",       "espirit_l1")
        self.combo_equi_method.addItem("ESPIRiT + L2",       "espirit_l2")
        self.combo_equi_method.addItem("ESPIRiT + TV",       "espirit_tv")
        self.combo_equi_method.addItem("ESPIRiT + L1-Wav",   "espirit_l1wav")
        self.combo_equi_method.setToolTip(
            "GRAPPA: parallel-imaging k-space interpolation (pygrappa; RSS coil combination).\n"
            "GRAPPA + SENSE: GRAPPA k-space fill + ESPIRiT sensitivity-weighted coil combination.\n"
            "Pure SENSE: 1 ESPIRiT map set + PICS -S (no regularisation).\n"
            "CS-SENSE: 1 ESPIRiT map set + PICS with L1 wavelet regularisation.\n"
            "ESPIRiT: multi-map ESPIRiT operator (PICS, no regularisation).\n"
            "ESPIRiT + L1: multi-map ESPIRiT operator + L1 wavelet (legacy -l1 flag).\n"
            "ESPIRiT + L2: ESPIRiT + L2 regularisation (-R Q:λ).\n"
            "ESPIRiT + TV: ESPIRiT + Total Variation (-R T:7:0:λ).\n"
            "ESPIRiT + L1-Wav: ESPIRiT + L1 wavelet (-R W:7:0:λ).\n\n"
            "SENSE/ESPIRiT methods run on the engine chosen under 'Recon engine'\n"
            "(BART or the built-in sigpy fallback)."
        )
        form_grappa.addRow("Method:", self.combo_equi_method)

        self.combo_equi_mask_method = QComboBox()
        self.combo_equi_mask_method.addItem("Analytical (JSON / sequence)", "analytical")
        self.combo_equi_mask_method.addItem("BART pattern (k-space zeros)",  "bart_pattern")
        self.combo_equi_mask_method.setToolTip(
            "Analytical: derive PE mask from the JSON trajectory file or the\n"
            "  equispaced sequence formula. Fast; no BART required.\n"
            "BART pattern: run 'bart pattern' on the filled k-space to extract\n"
            "  the sampling mask — identical to the demo notebook approach.\n"
            "  Requires BART; falls back to analytical if BART is unavailable."
        )
        form_grappa.addRow("Mask extraction:", self.combo_equi_mask_method)

        self.spin_acs = QSpinBox()
        self.spin_acs.setRange(4, 128)
        self.spin_acs.setSingleStep(4)
        self.spin_acs.setValue(24)
        self.spin_acs.setToolTip(
            "Number of auto-calibration signal (ACS) lines at the k-space centre.\n"
            "Used to build the analytical PE mask for every equispaced method."
        )
        form_grappa.addRow("ACS lines:", self.spin_acs)

        self.spin_r = QSpinBox()
        self.spin_r.setRange(2, 8)
        self.spin_r.setValue(4)
        self.spin_r.setToolTip(
            "Under-sampling acceleration factor for the analytical PE mask.\n"
            "Auto-overridden when the filename contains _R<n> (e.g. equispaced_R4)."
        )
        form_grappa.addRow("R-factor:", self.spin_r)

        kr_widget = QWidget()
        kr_layout = QHBoxLayout(kr_widget)
        kr_layout.setContentsMargins(0, 0, 0, 0)
        self.spin_k_rows = QSpinBox()
        self.spin_k_rows.setRange(1, 15)
        self.spin_k_rows.setSingleStep(2)
        self.spin_k_rows.setValue(5)
        self.spin_k_rows.setFixedWidth(55)
        self.spin_k_cols = QSpinBox()
        self.spin_k_cols.setRange(1, 15)
        self.spin_k_cols.setSingleStep(2)
        self.spin_k_cols.setValue(5)
        self.spin_k_cols.setFixedWidth(55)
        kr_layout.addWidget(self.spin_k_rows)
        kr_layout.addWidget(QLabel("×"))
        kr_layout.addWidget(self.spin_k_cols)
        kr_layout.addStretch()
        form_grappa.addRow("Kernel size:", kr_widget)

        self.dspin_grappa_lambda = QDoubleSpinBox()
        self.dspin_grappa_lambda.setRange(0.0, 10.0)
        self.dspin_grappa_lambda.setDecimals(4)
        self.dspin_grappa_lambda.setSingleStep(0.01)
        self.dspin_grappa_lambda.setValue(0.01)
        self.dspin_grappa_lambda.setToolTip(
            "Tikhonov regularisation weight (lamda) passed to pygrappa.\n"
            "Higher values smooth the result; 0 = no regularisation."
        )
        form_grappa.addRow("λ GRAPPA (lamda):", self.dspin_grappa_lambda)

        # CS-SENSE specific parameters
        self.dspin_sense_lambda = QDoubleSpinBox()
        self.dspin_sense_lambda.setRange(0.0001, 10.0)
        self.dspin_sense_lambda.setDecimals(4)
        self.dspin_sense_lambda.setSingleStep(0.001)
        self.dspin_sense_lambda.setValue(0.01)
        self.dspin_sense_lambda.setToolTip("L1 wavelet regularisation weight for CS-SENSE PICS")
        form_grappa.addRow("λ CS-SENSE:", self.dspin_sense_lambda)

        self.spin_sense_iters = QSpinBox()
        self.spin_sense_iters.setRange(10, 500)
        self.spin_sense_iters.setSingleStep(10)
        self.spin_sense_iters.setValue(100)
        self.spin_sense_iters.setToolTip("Number of PICS CG iterations (Pure SENSE and CS-SENSE)")
        form_grappa.addRow("SENSE iterations:", self.spin_sense_iters)

        self.chk_sense_l2 = QCheckBox("Add L2 regularisation")
        self.chk_sense_l2.setChecked(False)
        self.chk_sense_l2.setToolTip(
            "When checked, runs BART PICS -S -l2 instead of -S only.\n"
            "Applies to Pure SENSE only."
        )
        form_grappa.addRow("", self.chk_sense_l2)

        self.spin_espirit_num_maps = QSpinBox()
        self.spin_espirit_num_maps.setRange(1, 4)
        self.spin_espirit_num_maps.setValue(2)
        self.spin_espirit_num_maps.setToolTip(
            "Number of ESPIRiT map sets extracted by BART ecalib.\n"
            "1 = equivalent to single-map SENSE.\n"
            "2 = true ESPIRiT (handles FOV<object and phase wraps).\n"
            "Applies to ESPIRiT and ESPIRiT+L1 methods only."
        )
        form_grappa.addRow("ESPIRiT map sets:", self.spin_espirit_num_maps)

        # Wire method combo to show/hide relevant widgets
        self.combo_equi_method.currentIndexChanged.connect(self._on_equi_method_changed)
        self._on_equi_method_changed()   # set initial visibility

        vbox.addWidget(grp_grappa)

        # ── Poisson CS ────────────────────────────────────────────────────────
        grp_poisson = QGroupBox("Poisson Disc Reconstruction")
        form_poisson = QFormLayout(grp_poisson)

        self.combo_poisson_mode = QComboBox()
        self.combo_poisson_mode.addItem("Zero-Fill (RSS)",    "zerofill")
        self.combo_poisson_mode.addItem("CS PICS (wavelet)",  "pics")
        self.combo_poisson_mode.setToolTip(
            "CS PICS: compressed-sensing iterative reconstruction (slow, high quality).\n"
            "Zero-Fill: RSS of adjoint-NUFFT result (fast, no regularisation)."
        )
        form_poisson.addRow("Recon mode:", self.combo_poisson_mode)

        self.dspin_lambda = QDoubleSpinBox()
        self.dspin_lambda.setRange(0.0001, 10.0)
        self.dspin_lambda.setDecimals(4)
        self.dspin_lambda.setSingleStep(0.001)
        self.dspin_lambda.setValue(0.01)
        self.dspin_lambda.setToolTip("Wavelet L1 regularisation weight for PICS")
        form_poisson.addRow("λ wavelet:", self.dspin_lambda)

        self.spin_iters = QSpinBox()
        self.spin_iters.setRange(10, 500)
        self.spin_iters.setSingleStep(10)
        self.spin_iters.setValue(100)
        self.spin_iters.setToolTip("Number of PICS CG iterations")
        form_poisson.addRow("Max iterations:", self.spin_iters)

        # Disable CS-only widgets when Zero-Fill is selected
        self.combo_poisson_mode.currentIndexChanged.connect(self._on_poisson_mode_changed)

        vbox.addWidget(grp_poisson)

        # ── Unknown scans (generic fallback) ─────────────────────────────────
        grp_unknown = QGroupBox("Unknown Scans (generic fallback)")
        form_unknown = QFormLayout(grp_unknown)

        self.combo_unknown_mode = QComboBox()
        self.combo_unknown_mode.addItem("Auto (JSON → NUFFT, else RSS)", "auto")
        self.combo_unknown_mode.addItem("Cartesian IFFT + RSS",          "cartesian_rss")
        self.combo_unknown_mode.addItem("NUFFT (trajectory JSON)",       "nufft_json")
        self.combo_unknown_mode.addItem("NUFFT (assume radial)",         "nufft_radial")
        self.combo_unknown_mode.addItem("CS (mask from k-space)",        "cs_pattern")
        self.combo_unknown_mode.setToolTip(
            "How to reconstruct scans whose type is not recognised from the filename.\n"
            "Auto: if a trajectory JSON is found next to the .dat (<scanname>.json or\n"
            "  Trajectory.json — or one picked via right-click in the folder tree),\n"
            "  grid it with NUFFT + RSS; otherwise plain Cartesian IFFT + RSS.\n"
            "Cartesian IFFT + RSS: assume Cartesian sampling.\n"
            "NUFFT (trajectory JSON): force the NUFFT path (falls back to RSS if\n"
            "  no JSON is found or it mismatches the data).\n"
            "NUFFT (assume radial): generate a uniform radial trajectory with\n"
            "  `bart traj -r` (spokes × samples taken from the data shape).\n"
            "CS (mask from k-space): for undersampled Cartesian data — extract the\n"
            "  sampling mask with `bart pattern` and run ESPIRiT + PICS L1-wavelet\n"
            "  (uses the Poisson λ wavelet / max iterations above).\n"
            "Every mode falls back to Cartesian IFFT + RSS on failure."
        )
        form_unknown.addRow("Recon mode:", self.combo_unknown_mode)

        vbox.addWidget(grp_unknown)

        # ── Undersampling Simulator ──────────────────────────────
        grp_sim = QGroupBox("Undersampling Simulator")
        form_sim = QFormLayout(grp_sim)

        self.combo_sim_mask = QComboBox()
        self.combo_sim_mask.addItem("Equispaced",             "equispaced")
        self.combo_sim_mask.addItem("Variable Density (VD)", "vd")
        self.combo_sim_mask.setToolTip(
            "Equispaced: uniform undersampling anchored to k-space centre.\n"
            "Variable Density: higher density near centre following a power-law PDF."
        )
        form_sim.addRow("Mask type:", self.combo_sim_mask)

        self.dspin_sim_R = QDoubleSpinBox()
        self.dspin_sim_R.setRange(2.0, 16.0)
        self.dspin_sim_R.setDecimals(1)
        self.dspin_sim_R.setSingleStep(0.5)
        self.dspin_sim_R.setValue(4.0)
        self.dspin_sim_R.setToolTip("Target acceleration factor R")
        form_sim.addRow("R (acceleration):", self.dspin_sim_R)

        self.spin_sim_acs = QSpinBox()
        self.spin_sim_acs.setRange(0, 64)
        self.spin_sim_acs.setSingleStep(4)
        self.spin_sim_acs.setValue(24)
        self.spin_sim_acs.setToolTip("Fully-sampled ACS lines at the k-space centre")
        form_sim.addRow("ACS lines:", self.spin_sim_acs)

        self._sim_power_label = QLabel("Power decay:")
        self.dspin_sim_power = QDoubleSpinBox()
        self.dspin_sim_power.setRange(0.25, 4.0)
        self.dspin_sim_power.setDecimals(2)
        self.dspin_sim_power.setSingleStep(0.25)
        self.dspin_sim_power.setValue(1.0)
        self.dspin_sim_power.setToolTip(
            "PDF exponent for Variable Density: 1=linear, 2=parabolic\n"
            "(only visible when VD is selected)"
        )
        form_sim.addRow(self._sim_power_label, self.dspin_sim_power)

        self._sim_seed_label = QLabel("Random seed:")
        self.spin_sim_seed = QSpinBox()
        self.spin_sim_seed.setRange(0, 2**31 - 1)
        self.spin_sim_seed.setValue(0)
        self.spin_sim_seed.setSpecialValueText("Random")
        self.spin_sim_seed.setToolTip(
            "Seed for the Variable-Density random line selection.\n"
            "0 = Random (a new mask every run); any other value makes the\n"
            "VD mask reproducible. Recorded in the result parameters and PDF."
        )
        form_sim.addRow(self._sim_seed_label, self.spin_sim_seed)

        self.combo_sim_recon = QComboBox()
        self.combo_sim_recon.addItem("None (mask & ZF only)",  "none")
        self.combo_sim_recon.addItem("GRAPPA",                 "grappa")
        self.combo_sim_recon.addItem("gGRAPPA (PyTorch)",      "ggrappa")
        self.combo_sim_recon.addItem("Pure SENSE",             "sense")
        self.combo_sim_recon.addItem("CS-SENSE",               "sense_cs")
        self.combo_sim_recon.addItem("ESPIRiT",                "espirit")
        self.combo_sim_recon.addItem("ESPIRiT + L1",           "espirit_l1")
        self.combo_sim_recon.addItem("ESPIRiT + L2",           "espirit_l2")
        self.combo_sim_recon.addItem("ESPIRiT + TV",           "espirit_tv")
        self.combo_sim_recon.addItem("ESPIRiT + L1-Wav",       "espirit_l1wav")
        self.combo_sim_recon.setToolTip(
            "Reconstruction method to apply to the retrospectively undersampled k-space.\n"
            "SENSE/ESPIRiT methods run on the selected Recon engine (BART or sigpy);\n"
            "GRAPPA uses pygrappa, gGRAPPA needs the optional torch install."
        )
        form_sim.addRow("Recon method:", self.combo_sim_recon)

        # ── Simulator GRAPPA params ───────────────────────────────────────────
        self._sim_grappa_widget = QWidget()
        _sg_form = QFormLayout(self._sim_grappa_widget)
        _sg_form.setContentsMargins(8, 2, 0, 2)
        _sg_kr = QWidget()
        _sg_kr_layout = QHBoxLayout(_sg_kr)
        _sg_kr_layout.setContentsMargins(0, 0, 0, 0)
        self.spin_sim_k_rows = QSpinBox()
        self.spin_sim_k_rows.setRange(1, 15)
        self.spin_sim_k_rows.setSingleStep(2)
        self.spin_sim_k_rows.setValue(5)
        self.spin_sim_k_rows.setFixedWidth(55)
        self.spin_sim_k_cols = QSpinBox()
        self.spin_sim_k_cols.setRange(1, 15)
        self.spin_sim_k_cols.setSingleStep(2)
        self.spin_sim_k_cols.setValue(5)
        self.spin_sim_k_cols.setFixedWidth(55)
        _sg_kr_layout.addWidget(self.spin_sim_k_rows)
        _sg_kr_layout.addWidget(QLabel("×"))
        _sg_kr_layout.addWidget(self.spin_sim_k_cols)
        _sg_kr_layout.addStretch()
        _sg_kr.setToolTip(
            "GRAPPA kernel size (rows × cols).\n"
            "gGRAPPA uses it as (rows, 1, cols) — 4×5 recommended."
        )
        _sg_form.addRow("Kernel size:", _sg_kr)
        self.dspin_sim_grappa_lambda = QDoubleSpinBox()
        self.dspin_sim_grappa_lambda.setRange(0.0, 10.0)
        self.dspin_sim_grappa_lambda.setDecimals(4)
        self.dspin_sim_grappa_lambda.setSingleStep(0.01)
        self.dspin_sim_grappa_lambda.setValue(0.01)
        self.dspin_sim_grappa_lambda.setToolTip("Tikhonov regularisation λ for GRAPPA")
        _sg_form.addRow("λ GRAPPA:", self.dspin_sim_grappa_lambda)
        form_sim.addRow(self._sim_grappa_widget)

        # ── Simulator SENSE / CS-SENSE params ────────────────────────────────
        self._sim_sense_widget = QWidget()
        _ss_form = QFormLayout(self._sim_sense_widget)
        _ss_form.setContentsMargins(8, 2, 0, 2)
        self.dspin_sim_sense_lambda = QDoubleSpinBox()
        self.dspin_sim_sense_lambda.setRange(0.0001, 10.0)
        self.dspin_sim_sense_lambda.setDecimals(4)
        self.dspin_sim_sense_lambda.setSingleStep(0.001)
        self.dspin_sim_sense_lambda.setValue(0.01)
        self.dspin_sim_sense_lambda.setToolTip("L1 wavelet λ for CS-SENSE (BART PICS -l1)")
        _ss_form.addRow("λ CS-SENSE:", self.dspin_sim_sense_lambda)
        self.spin_sim_sense_iters = QSpinBox()
        self.spin_sim_sense_iters.setRange(10, 500)
        self.spin_sim_sense_iters.setSingleStep(10)
        self.spin_sim_sense_iters.setValue(100)
        self.spin_sim_sense_iters.setToolTip("BART PICS CG iterations (Pure SENSE and CS-SENSE)")
        _ss_form.addRow("Iterations:", self.spin_sim_sense_iters)
        form_sim.addRow(self._sim_sense_widget)

        # Wire visibility
        self.combo_sim_mask.currentIndexChanged.connect(self._on_sim_mask_changed)
        self._on_sim_mask_changed()  # set initial visibility
        self.combo_sim_recon.currentIndexChanged.connect(self._on_sim_recon_changed)
        self._on_sim_recon_changed()  # set initial visibility

        self.chk_sim_swap_pe_read = QCheckBox("Swap Phase/Read (L-R undersampling)")
        self.chk_sim_swap_pe_read.setChecked(False)
        self.chk_sim_swap_pe_read.setToolTip(
            "When unchecked (default): undersamples the Phase Encode axis (A-P).\
\nWhen checked: undersamples the Readout axis (L-R), simulating a Swap Phase/Read acquisition."
        )
        form_sim.addRow(self.chk_sim_swap_pe_read)

        vbox.addWidget(grp_sim)
        # ── Output ────────────────────────────────────────────────────────────
        grp_out = QGroupBox("Output")
        form_out = QFormLayout(grp_out)

        # Save directory
        save_row = QWidget()
        save_layout = QHBoxLayout(save_row)
        save_layout.setContentsMargins(0, 0, 0, 0)
        self.edit_save_dir = QLineEdit()
        # Default to a user-writable location: the app folder is read-only in
        # installed/frozen builds (Program Files, .app bundle), so <app>/results
        # would fail there. Created lazily on first save.
        _docs = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        default_save = str(Path(_docs) / "SHIR" if _docs else Path.home() / "SHIR")
        self.edit_save_dir.setText(default_save)
        self.edit_save_dir.setToolTip(
            "Directory where results and reports are saved\n"
            "(created on first save if it does not exist)"
        )
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(28)
        btn_browse.clicked.connect(self._browse_save_dir)
        save_layout.addWidget(self.edit_save_dir)
        save_layout.addWidget(btn_browse)
        form_out.addRow("Save directory:", save_row)

        self.chk_pdf     = QCheckBox("Generate PDF report")
        self.chk_pdf.setChecked(False)
        self.chk_npy     = QCheckBox("Save combined images (.npy)")
        self.chk_npy.setChecked(False)
        form_out.addRow(self.chk_pdf)
        form_out.addRow(self.chk_npy)

        self.btn_save_output = QPushButton("Save Output…")
        self.btn_save_output.setToolTip(
            "Save the checked outputs — PDF report and/or .npy images — to a chosen location"
        )
        self.btn_save_output.clicked.connect(self.save_output_requested.emit)
        form_out.addRow(self.btn_save_output)

        vbox.addWidget(grp_out)
        vbox.addStretch()
    # ─── Simulator mask type visibility ──────────────────────────────────────────

    def _on_sim_mask_changed(self) -> None:
        is_vd = self.combo_sim_mask.currentData() == "vd"
        self._sim_power_label.setVisible(is_vd)
        self.dspin_sim_power.setVisible(is_vd)
        self._sim_seed_label.setVisible(is_vd)
        self.spin_sim_seed.setVisible(is_vd)

    def _on_sim_recon_changed(self) -> None:
        m = self.combo_sim_recon.currentData()
        _bart_methods = ("sense", "sense_cs", "espirit", "espirit_l1", "espirit_l2", "espirit_tv", "espirit_l1wav")
        _reg_methods  = ("sense_cs", "espirit_l1", "espirit_l2", "espirit_tv", "espirit_l1wav")
        self._sim_grappa_widget.setVisible(m in ("grappa", "ggrappa"))
        self._sim_sense_widget.setVisible(m in _bart_methods)
        self.dspin_sim_sense_lambda.setEnabled(m in _reg_methods)

    # ─── Equispaced method visibility ────────────────────────────────────────

    def _on_equi_method_changed(self) -> None:
        m = self.combo_equi_method.currentData()
        is_grappa   = (m in ("grappa", "grappa_sense"))
        is_espirit  = (m in ("espirit", "espirit_l1", "espirit_l2", "espirit_tv", "espirit_l1wav"))
        # ACS lines and R-factor build the analytical PE mask for EVERY
        # equispaced method (not just GRAPPA) — keep them enabled always.
        # GRAPPA-only widgets
        self.spin_k_rows.setEnabled(is_grappa)
        self.spin_k_cols.setEnabled(is_grappa)
        self.dspin_grappa_lambda.setEnabled(is_grappa)
        # Regularisation lambda: CS-SENSE and all regularised ESPIRiT variants
        self.dspin_sense_lambda.setEnabled(
            m in ("sense_cs", "espirit_l1", "espirit_l2", "espirit_tv", "espirit_l1wav")
        )
        # L2 option is Pure SENSE only
        self.chk_sense_l2.setEnabled(m == "sense")
        # Iterations: all SENSE-family methods
        self.spin_sense_iters.setEnabled(
            m in ("sense", "sense_cs", "espirit", "espirit_l1", "espirit_l2", "espirit_tv", "espirit_l1wav")
        )
        # ESPIRiT map sets: ESPIRiT methods only
        self.spin_espirit_num_maps.setEnabled(is_espirit)

    # ─── Browse helper ────────────────────────────────────────────────────

    def _on_poisson_mode_changed(self) -> None:
        is_pics = self.combo_poisson_mode.currentData() == "pics"
        self.dspin_lambda.setEnabled(is_pics)
        self.spin_iters.setEnabled(is_pics)

    def _browse_save_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if folder:
            self.edit_save_dir.setText(folder)

    # ─── Settings persistence ─────────────────────────────────────────────────

    def load_persistent(self, settings) -> None:
        """Restore the sticky output/engine choices from QSettings."""
        d = settings.value("output/save_dir", "", str)
        if d:
            self.edit_save_dir.setText(d)
        self.chk_pdf.setChecked(settings.value("output/gen_pdf", False, bool))
        self.chk_npy.setChecked(settings.value("output/save_npy", False, bool))
        engine = settings.value("common/recon_backend", "", str)
        idx = self.combo_backend.findData(engine)
        if idx >= 0:
            self.combo_backend.setCurrentIndex(idx)

    def save_persistent(self, settings) -> None:
        settings.setValue("output/save_dir", self.edit_save_dir.text())
        settings.setValue("output/gen_pdf", self.chk_pdf.isChecked())
        settings.setValue("output/save_npy", self.chk_npy.isChecked())
        settings.setValue("common/recon_backend", self.combo_backend.currentData())

    # ─── Public API ───────────────────────────────────────────────────────────

    def get_params(self) -> dict:
        """Return all parameters as a flat dict."""
        return {
            # common
            "grid_size":    self.spin_grid.value(),
            "scan_fov_m":   self.dspin_fov_mm.value() * 1e-3,
            "crop_size":    self.spin_crop.value(),   # 0 → auto (from FOV)
            "traj_margin":  self.dspin_traj_margin.value(),
            "calib_width":  self.spin_calib.value(),
            "espirit_crop": self.dspin_espirit_crop.value(),   # both engines
            "recon_backend": self.combo_backend.currentData(),   # auto | bart | sigpy
            # equispaced
            "equi_method":       self.combo_equi_method.currentData(),
            "equi_mask_method":  self.combo_equi_mask_method.currentData(),
            "acs_lines":         self.spin_acs.value(),
            "r_factor":       self.spin_r.value(),
            "kernel_size":    (self.spin_k_rows.value(), self.spin_k_cols.value()),
            "grappa_lambda":  self.dspin_grappa_lambda.value(),
            "sense_lambda":   self.dspin_sense_lambda.value(),
            "sense_max_iter": self.spin_sense_iters.value(),
            "sense_l2":         self.chk_sense_l2.isChecked(),
            "espirit_num_maps": self.spin_espirit_num_maps.value(),
            # fully sampled
            "fullysamp_mode": self.combo_fs_mode.currentData(),
            # TSE echo un-scramble (None → recon auto-detects from filename)
            "tse":       None if self.chk_tse_auto.isChecked() else (self.spin_tse_necho.value() > 1),
            "tse_necho": None if self.chk_tse_auto.isChecked() else self.spin_tse_necho.value(),
            "tse_te1":   None if self.chk_tse_auto.isChecked() else self.dspin_tse_te1.value() * 1e-3,
            "tse_teeff": None if self.chk_tse_auto.isChecked() else self.dspin_tse_teeff.value() * 1e-3,
            # poisson
            "poisson_mode": self.combo_poisson_mode.currentData(),
            "lambda_wav":   self.dspin_lambda.value(),
            "max_iter":     self.spin_iters.value(),
            # unknown scans (generic fallback)
            "unknown_mode": self.combo_unknown_mode.currentData(),
            # simulator
            "sim_mask_type":    self.combo_sim_mask.currentData(),
            "sim_R":            self.dspin_sim_R.value(),
            "sim_acs_lines":    self.spin_sim_acs.value(),
            "sim_power_decay":  self.dspin_sim_power.value(),
            "sim_recon_method": self.combo_sim_recon.currentData(),
            # 0 shows as "Random" in the UI → no fixed seed
            "sim_rng_seed":     self.spin_sim_seed.value() or None,
            "sim_kernel_size":  (self.spin_sim_k_rows.value(), self.spin_sim_k_cols.value()),
            "sim_grappa_lambda": self.dspin_sim_grappa_lambda.value(),
            "sim_sense_iters":  self.spin_sim_sense_iters.value(),
            "sim_sense_lambda": self.dspin_sim_sense_lambda.value(),
            "sim_swap_pe_read": self.chk_sim_swap_pe_read.isChecked(),
            # output
            "save_dir":     self.edit_save_dir.text(),
            "gen_pdf":      self.chk_pdf.isChecked(),
            "save_npy":     self.chk_npy.isChecked(),
        }
