# SHIR — System for Hybrid Image Reconstruction

**SHIR** (the app formerly described here as *AnalyzeMRIDataApp* — that is still the source folder
name) is a PyQt5 desktop application for multi-pipeline MRI reconstruction and visualisation of
Siemens TWIX `.dat` files. It runs from source (Linux / WSL / macOS) **and** as a standalone app
packaged for **Windows (`SHIR.exe`)** and **macOS (`SHIR.app`)** — see [`Compile/`](Compile/).

**BART is optional.** Reconstruction can run on **BART** (exact numerics) or a built-in **sigpy**
engine that needs no external install; a **Recon engine** selector (Auto / Force BART / Force sigpy)
chooses between them, and Auto falls back to sigpy automatically when BART is absent. On Windows,
BART is invoked through WSL; on macOS/Linux it runs natively.

> **End-user & packaging guides live in [`Compile/`](Compile/):** [`instructions.md`](Compile/instructions.md)
> (how to use the app), [`demands.md`](Compile/demands.md) / [`demands_mac.md`](Compile/demands_mac.md)
> (Windows / macOS setup), and [`compilation_instruction.md`](Compile/compilation_instruction.md)
> (building the standalone app).

Part of the wider [MRI Acquisition & Reconstruction Toolkit](../README.md).

---

## Table of Contents

1. [Architecture](#architecture)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Usage Walkthrough](#usage-walkthrough)
5. [Reconstruction Pipelines](#reconstruction-pipelines)
6. [Parameter Reference](#parameter-reference)
7. [Scan Folder Format](#scan-folder-format)
8. [Module Structure](#module-structure)
9. [Known Constraints](#known-constraints)
10. [Credits & Licensing](#credits--licensing)

---

## Architecture

```
Toolbar:  Open │ Run Selected │ Simulate │ Stop │ All Coils │ Clear Tabs │ About
┌────────────────┬──────────────────────────────┬───────────────┐
│  FolderPanel   │   ResultsViewer (tabbed)     │  ParamsPanel  │
│  (left)        │   matplotlib canvases        │  (right)      │
├────────────────┴──────────────────────────────┴───────────────┤
│  ProgressBar                                                   │
│  Log console (QTextEdit)                                       │
└────────────────────────────────────────────────────────────────┘
```

**Module dependency chain:**

```
app.py
  └── MainWindow  (ui/main_window.py)
        ├── FolderPanel     (ui/folder_panel.py)     ← folder/file/band selection
        ├── ResultsViewer   (ui/results_viewer.py)   ← tabbed image display
        ├── ParamsPanel     (ui/params_panel.py)     ← all reconstruction parameters
        ├── ReportGenerator (ui/report_generator.py) ← PDF export
        └── ReconWorker  (background QThread)
              ├── band_recon.py           (processing/)
              ├── fullysamp_recon.py      (processing/)
              ├── equispaced_recon.py     (processing/)
              ├── poisson_recon.py        (processing/)
              ├── retro_undersample.py    (processing/)  ← simulator
              └── utils.py                (processing/)  ← BART, CFL I/O, trajectories
                    └── read_siemens_scans/               ← TWIX reader
```

---

## Installation

### 1. Python packages

```bash
pip install -r requirements.txt
```

Key packages: `PyQt5`, `numpy`, `scipy`, `matplotlib`, `sigpy`, `pygrappa`,
`twixtools`, `scikit-image` (`numba` comes in via sigpy).  
See [requirements.txt](requirements.txt) for pinned versions and
[requirements-dev.txt](requirements-dev.txt) for the test/build tools.

### 2. BART (Berkeley Advanced Reconstruction Toolbox) — *optional*

BART is **optional**: the built-in **sigpy** engine reconstructs everything without it (choose
*Force sigpy* or *Auto* in the Recon-engine selector). Install BART only for its exact numerics.

```bash
sudo apt install bart          # Ubuntu/Debian WSL   (or build from source)
```

Build from source: https://mrirecon.github.io/bart . When present, BART must be on the `PATH`
(Linux/macOS) or reachable via WSL (Windows). In the packaged app, set `MRI_BART_BIN` to BART's
path if it isn't on the default PATH — see [`Compile/demands.md`](Compile/demands.md).

### 3. GUI display (WSL)

| Environment | Setup |
|---|---|
| **WSL2 + Windows 11** | WSLg is built-in — no extra setup needed |
| **WSL2 + Windows 10** | Install [VcXsrv](https://sourceforge.net/projects/vcxsrv/), then `export DISPLAY=:0` |

`app.py` auto-detects `DISPLAY` / `WAYLAND_DISPLAY` and sets the matplotlib backend to `Qt5Agg`.

---

## Quick Start

```bash
# In a WSL terminal
cd /mnt/c/Users/<you>/OneDrive\ -\ Technion/תואר\ שני/CheckDatFiles/AnalyzeMRIDataApp
python app.py
```

---

## Usage Walkthrough

### 1. Open a scan folder
Click **Open Folder** (or **Add Folder**) in the toolbar or folder panel and select any of:
- A parent scan folder containing a `dat_files/` sub-folder (e.g. `SE_1mm/`),
- A `dat_files/` folder directly, **or**
- Any folder containing loose `.dat` files (trajectory JSONs are looked up alongside them).

The app auto-detects each `.dat` file type from the filename stem:

| Type | Filename pattern | Colour |
|---|---|---|
| Band | `band_se_N` / `band_tse_N` | Green |
| Fully-Sampled | `*fully_sampled*` | Blue |
| Equispaced | `*equispaced*` (R factor read from `_R<n>`, e.g. `equispaced_R4_acs24.dat`) | Yellow |
| PoissonDisc | `*poisson*` | Red/pink |
| Unknown | anything else | Grey — reconstructed with the **generic fallback** (below) |

### 2. Select files to process
- Check/uncheck folders, scan types, or individual files in the tree; each **Band** file
  (`band_se_0` … `band_se_15`) is its own checkable tree item.
- Use **Select All** / **Clear All** for bulk selection.

### 3. Configure parameters
Adjust parameters in the **right panel** before running (see [Parameter Reference](#parameter-reference)).

### 4. Run
Click **Run Selected** (toolbar or folder panel).  
Jobs execute sequentially in a background thread; progress is shown in the log console.  
Click **Stop** to cancel after the current job finishes.

### 4b. Retrospective Undersampling Simulation
To simulate undersampling on an already-acquired fully-sampled dataset:
1. Check one or more **Fully-Sampled** files in the tree.
2. Configure the **Undersampling Simulator** group in the right panel (see [Parameter Reference](#parameter-reference)).
3. Click **Undersampling Simulator** in the toolbar.
4. A confirmation dialog lists the checked fully-sampled scans (all pre-ticked).
   **Untick any you don't want**, then click **Proceed** — each remaining scan
   is undersampled with the current settings and gets its own result tab.

The app retrospectively applies the chosen mask to the fully-sampled k-space and — optionally — runs a reconstruction, all without re-scanning.

### Multi-slice & manual reference (right-click a scan)
- **Select slices…** — expand a multi-slice scan into per-slice checkboxes (with
  an "All slices" master); each ticked slice becomes its own result tab
  (`… — slice k/N`). Cartesian recons only (Band/Poisson are single-slice).
- **Set reference ▸** — override the automatic fully-sampled reference used for
  the Equispaced/Poisson difference + SSIM/NMSE/PSNR (pick a detected
  fully-sampled scan, browse for a `.dat`, or restore Auto).

### 5. View results
Each completed reconstruction appears as a new **tab** in the results viewer:
- **W/L sliders** — adjust image window/level (min%/max% of data range).
- **⟳ Rotate 90°** — rotate all images in the tab (per-tab); the chosen
  orientation is also applied in the PDF report.
- **matplotlib toolbar** — zoom, pan, save individual figures.
- **All Coils** button — open a grid of all per-coil images for the active tab.

### 6. Save output
In the **Output** group (right panel), tick **Generate PDF report** and/or **Save combined
images (.npy)**, then click **💾 Save output**. A **Save As** dialog opens **per scan
folder**, starting in the current **Save directory** and pre-filled with the folder's name
(editable). The name you confirm for a folder becomes the prefix for that folder's files:

- **PDF** — one per folder: `<name>.pdf` (a multi-page report for that folder's results).
- **`.npy`** — one per result: `<name>.npy`, or `<name>_<result-label>.npy` when a folder
  has several results.

The two checkboxes also **auto-save** at the end of **Run Selected** — that path writes
`results/report_<ScanFolder>.pdf` and `<label>.npy` into the **Save directory**. Use
**💾 Save output** to save (or re-save to a chosen location) at any time after a run,
including toggling `.npy` on *after* the run has finished.

---

## Reconstruction Pipelines

### Reconstruction engine (BART / sigpy)

All ESPIRiT (`ecalib`), SENSE/CS (`pics`) and NUFFT steps below are dispatched through
`processing/recon_backend.py`, which has a **BART** implementation and a **sigpy** fallback
(`EspiritCalib`, `SenseRecon`, `L1WaveletRecon`, `TotalVariationRecon`, `nufft_adjoint`). The
**Recon engine** selector chooses `auto` (BART if available, else sigpy) / `bart` / `sigpy`. BART
commands are unchanged when BART is active; sigpy results are close but not identical. GRAPPA
(pygrappa) and RSS use neither engine.

### Band (non-Cartesian, NUFFT + ESPIRiT + SENSE)

```
read_raw_twix_data(dat)  +  load_trajectory(Bands.json, band_idx)
  │
  ├─ scale trajectory to [-0.5, 0.5)  →  build (3, dim1, dim2) BART CFL
  │
  ▼
bart nufft -a   (adjoint NUFFT on oversampled grid  [grid_size × grid_size])
  │
  ├─ crop to [crop_size × crop_size]
  ├─ bart fft (Cartesian FFT of NUFFT image)
  └─ normalise
  │
  ▼
bart ecalib -r calib_width   (ESPIRiT sensitivity maps)
  │
  ▼
bart pics -S   (SENSE coil combination)  →  final image
```

### Fully-Sampled (IFFT2 + RSS  or  ESPIRiT + SENSE)

```
read_raw_twix_data(dat)  →  permute to (Nread, Nphase, Ncoils)
  │
  ├─ [RSS mode]   centred IFFT2  →  Root-Sum-of-Squares  →  final image
  │
  └─ [SENSE mode] normalise k-space
                   →  bart ecalib  →  bart pics -S  →  final image
```

### Equispaced (GRAPPA / GRAPPA+SENSE / SENSE / CS-SENSE / ESPIRiT ±L1/L2/TV)

```
read_raw_twix_data(dat)
  │
  ├─ reconstruct PE mask: equispaced lines from centre + ACS block (acs_lines)
  ├─ place acquired lines into zero-filled k-space  (Nread × Nphase × Ncoils)
  │
  ▼   choose Method:
  ├─ GRAPPA         → pygrappa.mdgrappa (kernel rows×cols) → IFFT2 → RSS
  ├─ GRAPPA+SENSE   → pygrappa fill + ESPIRiT maps → optimal coil combine
  ├─ SENSE / CS-SENSE → ESPIRiT maps → pics -S [ -l1 -r λ ]
  └─ ESPIRiT (+L1/L2/TV) → ESPIRiT maps → pics [ -R W/Q/T:λ ]
  │
  └─ [optional]  load reference .dat  →  compute RMSE / PSNR / SSIM vs. reference
```

The ESPIRiT-map, SENSE and CS/PICS steps run on the selected **Recon engine**
(BART or sigpy — see below). GRAPPA (pygrappa) needs neither.

### Poisson-Disc (Compressed Sensing, BART PICS)

```
read_raw_twix_data(dat)  +  load_trajectory(PoissonDisc.json)
  │
  ├─ [PICS mode]
  │    same NUFFT → crop → fft → normalise steps as Band pipeline
  │    →  bart ecalib  →  bart pics -S -l1 -r λ  (L1-wavelet CS)  →  final image
  │
  └─ [zero-fill mode]  adjoint NUFFT only  →  RSS  →  final image (no BART PICS)
```

### Unknown scans (generic fallback)

Any `.dat` whose type isn't recognised from its filename is still reconstructed
(`processing/generic_recon.py` → `reconstruct_generic()`), so users always get
an image instead of an error. The **Unknown Scans** parameter group selects the
mode:

```
Auto (default)   trajectory JSON found → NUFFT + RSS, else Cartesian IFFT + RSS
Cartesian RSS    centred IFFT2 + RSS (delegates to reconstruct_fullysamp)
NUFFT (JSON)     adjoint NUFFT with a trajectory JSON + RSS (zero-fill)
NUFFT (radial)   adjoint NUFFT with a uniform-radial trajectory from `bart traj -r`
CS (pattern)     sampling mask from `bart pattern` → ESPIRiT + PICS L1-wavelet
```

Trajectory discovery (`utils.find_generic_traj_json`): a `<scanname>.json` or
`Trajectory.json` next to the `.dat` or in its parent folder — same format as
`Bands.json`. A file can also be assigned a JSON manually by right-clicking it in
the folder tree. Every mode degrades gracefully to Cartesian RSS on failure, and
a trajectory/data sample-count mismatch warns rather than crashing.

Alternatively, right-click an unknown file → **Reconstruct as** to route it
through a known pipeline (Fully-Sampled / Equispaced / Poisson-Disc) instead of
the generic fallback; the override is stored on the job's `scan_type` so the
existing dispatch in `main_window` handles it unchanged.

### Retrospective Undersampling Simulator

Applies a synthetic 1-D PE undersampling mask to a **fully-sampled** dataset and
optionally reconstructs the result. Useful for evaluating mask designs and reconstruction
quality without acquiring new data.

```
read_raw_twix_data(fully_sampled.dat)
  │
  ├─ build PE mask
  │    ├─ Equispaced: step by round(R) from centre outward + ACS block
  │    │   (port of MATLAB generate_equispaced_1D)
  │    └─ Variable Density (VD): PDF = (1−|x|)^power + 0.05 floor,
  │        fixed ACS + edges, numpy weighted random sampling
  │        (port of MATLAB generate_variable_density_1D)
  │
  ├─ apply mask  →  undersampled k-space  →  zero-filled RSS image
  │
  └─ [optional reconstruction]
       ├─ GRAPPA  →  pygrappa mdgrappa  →  RSS image
       ├─ Pure SENSE  →  bart ecalib  →  bart pics -S
       └─ CS-SENSE   →  bart ecalib  →  bart pics -l1 -r λ

Outputs (per result tab — 3 × 3 grid):
  Row 0: Fully-Sampled reference  │  Zero-Filled  │  2-D PE mask
  Row 1: Reconstructed image      │  Abs diff     │  Simulation params + RMSE/PSNR
  Row 2: FS k-space (log)         │  Undersampled k-space (log)  │  Recon k-space (log)
```

---

## Parameter Reference

| Section | Parameter | Default | Applies to | Description |
|---|---|---|---|---|
| Common | Grid size | 512 | Band, Poisson | NUFFT oversampled grid dimension (power of 2) |
| Common | Scan FOV (mm) | 256 | Band, Poisson | Field-of-view in mm for trajectory scaling |
| Common | Traj margin (px) | 1.0 | Band, Poisson | Margin subtracted from N/2 when scaling trajectory |
| Common | ESPIRiT calib width | 24 | Band, Poisson, FullySamp (SENSE), Simulator (SENSE) | Calibration region width for ESPIRiT sensitivity maps |
| Common | **Recon engine** | Auto | all BART/sigpy methods | Auto (BART if available, else sigpy) / Force BART / Force sigpy |
| Fully Sampled | Combination mode | RSS | Fully-Sampled | RSS (fast) or ESPIRiT + SENSE coil combination |
| Equispaced Recon | Method | GRAPPA | Equispaced | GRAPPA / GRAPPA+SENSE / Pure SENSE / CS-SENSE / ESPIRiT / ESPIRiT +L1 / +L2 / +TV / +L1-Wav |
| Equispaced Recon | Mask method | analytical | Equispaced | Build PE mask analytically (R/ACS) or from the data's sampling pattern |
| Equispaced Recon | ESPIRiT maps | 2 | Equispaced (ESPIRiT) | Number of ESPIRiT map sets (BART multi-map; sigpy uses 1) |
| Equispaced Recon | ACS lines | 24 | Equispaced | Auto-calibration signal lines in centre of k-space |
| Equispaced Recon | R-factor | 4 | Equispaced | Phase-encode acceleration factor |
| Equispaced Recon | Kernel size | 5 × 5 | Equispaced (GRAPPA) | GRAPPA convolution kernel (rows × cols; odd values recommended) |
| Equispaced Recon | λ GRAPPA | 0.01 | Equispaced (GRAPPA) | Tikhonov regularisation weight for pygrappa |
| Equispaced Recon | λ CS-SENSE | 0.01 | Equispaced (CS-SENSE) | L1 wavelet regularisation weight for BART PICS |
| Equispaced Recon | SENSE iterations | 100 | Equispaced (SENSE/CS-SENSE) | BART PICS CG iterations |
| Poisson Disc Recon | Recon mode | Zero-Fill | Poisson | Zero-Fill (RSS) or CS PICS (L1-wavelet) |
| Poisson Disc Recon | λ wavelet | 0.01 | Poisson (PICS) | Wavelet L1 regularisation weight for BART PICS |
| Poisson Disc Recon | Max iterations | 100 | Poisson (PICS) | Conjugate-gradient iterations for BART PICS |
| **Undersampling Simulator** | **Mask type** | Equispaced | Simulator | Equispaced or Variable Density (VD) |
| Undersampling Simulator | R (acceleration) | 4.0 | Simulator | Target acceleration factor |
| Undersampling Simulator | ACS lines | 24 | Simulator | Fully-sampled lines at k-space centre |
| Undersampling Simulator | Power decay | 1.0 | Simulator (VD) | PDF exponent: 1=linear, 2=parabolic (VD only) |
| Undersampling Simulator | Random seed | Random | Simulator (VD) | Seed for VD line selection; a fixed value makes the mask reproducible (VD only) |
| Undersampling Simulator | Recon method | None | Simulator | None / GRAPPA / gGRAPPA / Pure SENSE / CS-SENSE / ESPIRiT ± L1/L2/TV |
| Undersampling Simulator | Kernel size | 5 × 5 | Simulator (GRAPPA) | GRAPPA convolution kernel |
| Undersampling Simulator | λ GRAPPA | 0.01 | Simulator (GRAPPA) | Tikhonov regularisation weight |
| Undersampling Simulator | Iterations | 100 | Simulator (SENSE/CS-SENSE) | BART PICS CG iterations |
| Undersampling Simulator | λ CS-SENSE | 0.01 | Simulator (CS-SENSE) | L1 wavelet regularisation weight |
| Output | Save directory | `~/Documents/SHIR` | All | Default directory for PDF reports and `.npy` image files; also the starting folder for the **Save output** dialog. Created on first save; verified writable at run start |
| Output | Generate PDF report | — | All | Include a PDF report when saving (auto-saved after all jobs complete, and saved by **Save output**) |
| Output | Save combined images (.npy) | — | All | Include combined images as `.npy` when saving (auto-saved after all jobs complete, and saved by **Save output**) |
| Output | 💾 Save output | — | All | Save the checked outputs per scan folder; prompts for a location/name (pre-filled from the folder), works any time after a run |

---

## Scan Folder Format

The app expects the following layout (produced by `CreateSeqFiles/`):

```
SE_1mm/                              ← parent scan folder
├── dat_files/
│   ├── band_se_0.dat ... band_se_15.dat
│   ├── fully_sampled.dat
│   ├── equispaced_R4_acs24.dat      ← R factor read from the "_R4" in the name
│   └── PoissonDisc.dat
├── Bands.json                       ← rotated-band trajectories, keyed by index
│                                       ("0", "1", … — typically 16, but any count)
├── PoissonDisc.json                 ← Poisson-disc 2D trajectory
└── Undersampled4equispaced.json     ← <R>× equispaced + ACS PE mask (R matches the .dat)
```

`utils.auto_detect_scan_type(filename)` uses regex matching on the filename stem to determine scan
type; `utils.extract_band_index(filename)` parses the trailing integer for band files.

`FolderPanel` looks for JSON files in the parent of `dat_files/` (i.e. the scan folder root):

| Scan type | JSON looked up |
|---|---|
| Band | `Bands.json` |
| PoissonDisc | `PoissonDisc.json` |
| Equispaced | `Undersampled<R>equispaced.json` (R from the `.dat` name; `equispaces`/lower-case variants also tried) |
| Fully-Sampled | *(no JSON needed)* |

---

## Module Structure

```
AnalyzeMRIDataApp/
├── app.py                     ← entry point; splash screen, QApplication, MainWindow
├── requirements.txt
├── processing/
│   ├── constants.py           ← scan-type + method wire constants, METHOD_LABELS,
│   │                             method_label(), JobAborted (Stop control-flow)
│   ├── imaging.py             ← shared ifft2c/fft2c/rss/norm01 + metrics_vs_reference
│   ├── utils.py               ← ScanType enum, BART subprocess wrapper (WSL bridge on
│   │                             Windows), CFL I/O, trajectory loaders, BartTempDir,
│   │                             ensure_twix_importable()
│   ├── recon_backend.py       ← BART⇄sigpy backend abstraction: bart_available(),
│   │                             espirit_maps(), pics(), nufft_adjoint_to_cartesian()
│   ├── band_recon.py          ← reconstruct_band()         — NUFFT + ESPIRiT + SENSE
│   ├── fullysamp_recon.py     ← reconstruct_fullysamp()    — IFFT2 + RSS / SENSE
│   ├── equispaced_recon.py    ← reconstruct_equispaced()   — GRAPPA / SENSE / CS-SENSE
│   ├── poisson_recon.py       ← reconstruct_poisson()      — NUFFT + PICS L1-wavelet
│   ├── generic_recon.py       ← reconstruct_generic()      — UNKNOWN-scan fallback
│   │                             (Cartesian RSS / NUFFT JSON / radial / CS pattern)
│   ├── reference.py           ← load_reference_rss / compare_to_reference
│   │                             (shared fully-sampled comparison: Equispaced + Poisson)
│   └── retro_undersample.py   ← reconstruct_retro()        — retrospective simulator
│                                   build_equispaced_mask()  — port of MATLAB function
│                                   build_vd_mask()          — port of MATLAB function
├── resources.py               ← resource_path(): dev + frozen (PyInstaller) asset paths
├── version.py                 ← __version__ (title bar / About / build banner)
├── Graphics/                  ← Intro.png (splash) + logo.png (window / exe icon)
├── tests/                    ← pytest suite (see Testing below)
└── ui/
    ├── main_window.py         ← MainWindow + ReconWorker (background QThread)
    ├── folder_panel.py        ← FolderPanel (scan tree); run/simulate signals
    ├── params_panel.py        ← ParamsPanel; exposes get_params() → dict
    ├── results_viewer.py      ← ResultsViewer (tabbed matplotlib canvases)
    └── report_generator.py    ← PDF multi-page report export
```

Each `reconstruct_*()` function accepts a `progress_cb` callback so `ReconWorker` can emit live
log messages without blocking the GUI thread. **Stop** cancels the queue and interrupts the
running job at its next pipeline stage (the callback raises `JobAborted`); an in-flight BART or
sigpy stage finishes first.

---

## Testing

A pytest suite lives in [`tests/`](tests/). Install the dev tools and run it:

```bash
pip install -r requirements-dev.txt
pytest                          # from AnalyzeMRIDataApp/
```

Markers (registered in [`pytest.ini`](pytest.ini)) gate the environment-specific tests:

| Marker | Needs | Skipped automatically when… |
|---|---|---|
| *(unmarked)* | numpy/scipy/sigpy/pygrappa | — always run (headless, synthetic phantoms) |
| `bart` | BART on `PATH` | BART is absent |
| `gui` | a Qt platform (pytest-qt) | runs offscreen; needs pytest-qt |
| `realdata` | the local sample scan folder | the folder is not present |

```bash
pytest -m "not bart and not realdata"    # CI-friendly subset (no BART, no data)
pytest -m realdata                        # end-to-end on real .dat scans
```

Tests run headless (`QT_QPA_PLATFORM=offscreen`, matplotlib `Agg`); Qt settings are
redirected to a temp dir so they never touch your real preferences. The synthetic
tests use a Shepp-Logan phantom with birdcage coil sensitivities and monkeypatch the
TWIX reader, so no scanner data is required for the bulk of the suite.

---

## Known Constraints

- **BART is optional.** With the Recon engine on *Force sigpy* (or *Auto* with no BART), all
  methods run on the built-in sigpy engine — results are close to BART's but not identical, and
  sigpy is CPU-only (slower). *Force BART* requires BART on `PATH` (Linux/macOS) or via WSL
  (Windows). If BART is installed somewhere off `PATH`, point SHIR at it with the
  **`MRI_BART_BIN`** environment variable (honoured on every platform; on Windows it names the
  binary *inside* WSL). The standalone builds exclude **torch**, so the torch-based **gGRAPPA**
  simulator method is unavailable there.
- **TSE folders** (`TSE_1mm`, `TSE_3mm`) contain Band data only; Fully-Sampled, Equispaced and
  Poisson-Disc scan types will not appear for these folders.
- Intermediate BART `.cfl` / `.hdr` files are written to a **temporary directory** and deleted
  automatically via `BartTempDir` after each reconstruction.
- WSL2 on Windows 11 supports **WSLg** natively. Older setups require VcXsrv with
  `export DISPLAY=:0` in the WSL shell before running.
- GRAPPA reconstruction assumes a fixed **4× equispaced** under-sampling pattern generated by
  `CreateSeqFiles/`. Other acceleration factors require re-generated JSON and matching R-factor
  in the parameter panel.
- The **Retrospective Undersampling Simulator** requires the selected file to be detected as
  **Fully-Sampled** type. It does not use any JSON trajectory file — the mask is synthesised
  entirely from the simulator parameters.
- The VD mask uses **numpy weighted random sampling** in place of the MATLAB Poisson-disc
  algorithm; the probability distribution matches faithfully but individual sample positions
  will differ from MATLAB. Set the simulator's **Random seed** field to a fixed value for a
  reproducible VD mask (leave it on *Random* for a fresh mask each run).

---

## Distributing the app

To share the app with other groups:

1. **Build** the standalone app on the target OS (PyInstaller can't
   cross-compile) — see [`Compile/compilation_instruction.md`](Compile/compilation_instruction.md)
   (Windows) / [`compilation_for_mac.py`](Compile/compilation_for_mac.py) (macOS).
   The build bundles `Graphics/` and sets `logo.png` as the exe/app icon.
2. **Zip** the `--onedir` output folder together with the runtime prerequisites
   ([`Compile/demands.md`](Compile/demands.md) — WSL + BART on Windows) and the
   user guide ([`Compile/instructions.md`](Compile/instructions.md)).
3. Tag the version (`version.py` → shown in the title bar / About dialog) and
   attach the zip to a **GitHub Release**.

Further options worth considering for wider distribution (not yet set up):

- A **GitHub Actions** `windows-latest` job that builds the zip on each release
  tag — this also sidesteps the "build outside OneDrive" restriction entirely.
- An **Inno Setup** installer wrapping the onedir folder (Start-menu shortcut
  with the logo, optional WSL check).
- A **Zenodo** DOI on the tagged release, plus a small anonymised demo `.dat`
  (+ trajectory JSON) so recipients can validate their WSL/BART setup.
- **Licensing is settled**: SHIR is released under **GPL-3.0**
  (Copyright © 2026 Saar Drive, Shimron Lab, Technion) — see the repository
  [`LICENSE`](../LICENSE) and [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md).
  GPL-3.0 is compatible with the bundled **PyQt5 (GPL v3)** for both source and
  binary distribution. (Migrating to PySide6/LGPL remains possible later if a
  more permissive app license is ever wanted.)

---

## Credits & Licensing

### Application

Developed by **Saar Drive** (Shimron Lab, Technion — Israel Institute of
Technology). Copyright © 2026 Saar Drive. Licensed under **GPL-3.0** — see
[`LICENSE`](../LICENSE) and [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md).

### Third-Party Libraries Used

| Library | License | Role |
|---|---|---|
| [BART](https://mrirecon.github.io/bart) | BSD-style | NUFFT, ESPIRiT, PICS (core reconstruction engine) |
| [pygrappa](https://github.com/mckib2/pygrappa) | MIT | GRAPPA kernel calibration & reconstruction |
| [twixtools](https://github.com/pehses/twixtools) | MIT | Siemens TWIX raw data parsing |
| [sigpy](https://github.com/mikgroup/sigpy) | BSD 3-Clause | Built-in reconstruction engine (ESPIRiT / SENSE / CS / NUFFT) |
| PyQt5 (GPL v3), numpy, scipy, matplotlib, scikit-image | various open-source | GUI, numerics, visualisation |

Full pinned dependency list: [requirements.txt](requirements.txt)

### TWIX Reader
The `read_siemens_scans` module is adapted from original code by **Orel Tsioni**.

### Sequence Design (companion component)
The Pulseq sequence files and JSON trajectories consumed by this app were created by
**Saar Drive & Alon Granek** — see the [root README](../README.md#licensing) for details.
