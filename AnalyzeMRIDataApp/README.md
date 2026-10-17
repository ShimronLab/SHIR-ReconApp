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
`twixtools`, `scikit-image` (`numba` comes in via sigpy), and `pypulseq` (reads the played
Pulseq `.seq`; optional at runtime — imported lazily). Install pypulseq with
`pip install --no-deps pypulseq==1.5.0.post1` if pip tries to change the pinned numpy/scipy.  
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

Siemens **product** scans keep their scanner names (`meas_MID…_t1_se_cor.dat`), so they show
up as *Unknown*. Their geometry does not depend on the name — it is read from the file (see
[How SHIR reads a scan](#how-shir-reads-a-scan)). A fully-sampled product scan reconstructs
directly; for a GRAPPA (PAT) scan, right-click → **Reconstruct as ▸ Equispaced**.

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
  fully-sampled scan, browse for a `.dat`, or restore Auto). Auto prefers the
  fully-sampled scan from the same sequence family (Pulseq vs Siemens product,
  read from the header), then the one with the same swapPE-ness as the scan (a
  `swapPE` reference has readout/phase swapped, i.e. a rotated image).

### 5. View results
Each completed reconstruction appears as a new **tab** in the results viewer.
Every tab and PDF page draws the data the same way, straight from the acquisition
frame: **readout (kx) left → right, phase encode (ky) bottom → top**. Band and
Poisson tabs use the same axes for their trajectory plot, so a band lies at the
same angle in the k-space panel and the trajectory panel (band *i* of 16 at
*i* × 11.25°). No rotation is applied to the data; use **Rotate** to turn the view.
Panels keep their physical proportions rather than stretching to the window: images
use the real pixel size (square pixels when it is unknown), k-space panels the
matching k-space step, and the trajectory plot a square ±kmax box.
- **W/L sliders** — adjust image window/level (min%/max% of data range).
- **⟳ Rotate 90°** — rotate all images in the tab (per-tab), together with the
  trajectory plot and the All-coils window; the chosen orientation is also
  applied in the PDF report.
- **Ref orient** (Equispaced/Poisson with a reference) — orientation of the
  fully-sampled reference relative to the recon. **Auto** tries all 8
  rotations/flips and keeps the best-correlated one (shown in the dropdown and
  under the metrics; a near-tie is flagged "close call"); pick another entry to
  force it. Only the reference moves, and the difference map + SSIM/NMSE/PSNR
  are recomputed instantly (no re-run).
- **matplotlib toolbar** — zoom, pan, save individual figures.
- **All Coils** button — open a grid of all per-coil images for the active tab.

For scans whose geometry is read from the file (Siemens product scans):
- **Facts line** under the controls — a one-line summary, e.g.
  `PF 7/8 · 235/269 lines · R 1 · phase OS 0% · readout OS 2× · prewhitened`.
- **Acquisition…** — the full table of acquisition facts, each with the **source** it came
  from (header, per-readout tag, noise scan, data, derived, setting); doubtful values are flagged.
  Band and Poisson tabs have it too: played `.seq` and signature, trajectory JSON and key,
  readouts × samples, k-step / FOV / k max, and the band's angle, NUFFT grid and pixel size
  (band) or matrix, sampled lines, R and ACS block (Poisson).
- **Sampling strip** — a thin coloured bar *beside* the k-space (never drawn over it): blue
  acquired, green calibration, grey skipped by acceleration, orange never acquired (partial
  Fourier).
- **Show filled k-space** — switch the k-space panel between what was acquired and what the
  reconstruction filled in.
- **Physical aspect** — images are drawn with the true pixel spacing (e.g. 0.42 × 0.60 mm),
  not stretched to the panel; k-space with the matching k-space step.
- **Score anyway** — appears when the reference is not comparable (see
  [Reference comparability](#reference-comparability)); scores computed this way are labelled.

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

### How SHIR reads a scan

Every Cartesian pipeline (Fully-Sampled, Equispaced, Simulator, generic CS, reference loading)
starts from the same step, `processing/acquisition.py` → `acquire_cartesian()`. What it trusts
depends on the **sequence family**, read from `Config.SequenceFileName` in the TWIX header:

| Family | Geometry comes from | Notes |
|---|---|---|
| **Siemens product** | the protocol header + each readout's own tags | Line numbers, k-space centre line/column, partial Fourier, readout and phase oversampling, PAT reference lines, FOV and pixel spacing. Nothing is guessed from the filename. |
| **Pulseq** | the played `.seq` → else the scan's trajectory JSON → else the legacy path (filename, line count, `tse_reorder`) | Pulseq headers carry placeholder protocol values (matrix 128, OS 1, "3D"), so they are **not** used. See [Pulseq: the played `.seq`](#pulseq-the-played-seq). |
| Unreadable header | the legacy path | |

For product scans the k-space is placed on its **full phase-encode grid** (`lPhaseEncodingLines`,
centred on `CenterLin`) — lines the protocol never acquired stay zero and are labelled:

- **Partial Fourier** — missing lines are filled by the *Partial-Fourier lines* setting:
  **Zero-fill** (default), **Homodyne** or **POCS**. GRAPPA fills only the accelerated gaps;
  SENSE/CS treat the partial-Fourier lines as unsampled. The ESPIRiT calibration region is
  limited to the symmetric centre.
- **Asymmetric echo** — the readout is zero-padded symmetric about `CenterCol`.
- **Oversampling** — readout oversampling (`flReadoutOSFactor`) is cropped in the image domain;
  phase oversampling (`flPhaseOS`) is cropped last, after reconstruction and before metrics.
- **Integrated PAT reference lines** (`ucRefScanMode` 2) are merged and used as GRAPPA/ESPIRiT
  calibration. *Separate* reference scans (mode 4) still use the legacy path.
- **Noise prewhitening** — the noise-adjust scan stored in the file is used to decorrelate the
  coils before any coil processing (on by default; Siemens only).

SHIR **refuses rather than guesses** when a product file is outside what it handles:

| Message | Cause |
|---|---|
| `3-D acquisitions are not supported yet.` | `ucDimension` = 3D |
| `Unsupported data dimensions [...]` | multi-echo, repetitions, sets, averages … |
| `Readouts disagree on CenterLin / CenterCol / ncol` | the file contradicts itself |
| `Line numbers … do not fit the N-line grid` | line counters outside the protocol grid |
| `This accelerated scan has no calibration (reference) lines …` | PAT scan without reference lines (Equispaced) |

`python tools/acq_report.py <folder>` prints the fact sheet of every `.dat` under a folder.

#### Pulseq: the played `.seq`

A Pulseq header still names the `.seq` that was played (`Meas.tFree`) and its signature
(`Dicom.tSequenceVariant` = the `[SIGNATURE] Hash` line of the `.seq`). `processing/seq_geometry.py`
finds that **exact** file by signature — a same-name file with a different hash (regenerated
after the scan) is never used. Searched, in order:

1. the `.dat` folder and its parent;
2. `seq_files/` or any `*_seq/` folder in the `.dat` folder, its parent or grandparent;
3. the **.seq library** folder set in the *Pulseq scans* parameter group (searched with its
   sub-folders; remembered between sessions);
4. if still not found, the app **asks** (once per sequence per session) — *Locate .seq…*
   accepts the file only if its signature matches, and its folder is searched for the rest of
   the session; *Continue without* moves on to the next source.

The `.seq` (read with `pypulseq`) gives every readout's k-space position, and the lines are
placed by it the same way as for product scans: the true phase-encode index of every line (so
any echo-train order, e.g. a centre-out TSE, is placed correctly), the direction of both axes
(k increasing along the array), the readout oversampling (Δk_phase / Δk_read), the FOV and the
pixel size. Lines recorded twice keep the first acquisition (filler echoes are dropped, with a
warning); a scan where every line repeats (averages, unlabelled slices) is not assembled this
way. Half-line ky grids and lines one step off the grid centre are placed with a linear phase
only (magnitude unchanged).

**Without the `.seq`**, the trajectory JSON written with the scan (`FullySampled_<tags>.json` /
`Undersampled<R>equispace(d|s)[_swapPE].json` in the scan folder) carries the same per-line k
positions — on the lab's data it agrees with the `.seq` line for line. One caveat is handled: a
JSON shared by an SE and a TSE scan carries the SE order, so for a TSE scan the JSON order is
compared with the echo-train plan and the one giving the sharper image (lower entropy, by at
least 0.1 nats) is used. **With neither**, the legacy placement (file name, line count, TSE
plan) is used and the log says so. `python tools/seq_parity.py` compares the legacy placement
with the `.seq` for every Pulseq scan.

### Reference comparability

Before a fully-sampled reference is scored against a reconstruction, two Siemens product
files are compared: sequence program, TE, TR, slice position/orientation/thickness and FOV.
If they differ, SSIM/NMSE/PSNR are **withheld** and the tab (and the PDF) states why, e.g.
`different TE (12 vs 96 ms)`. **Score anyway** computes them regardless, marked *not
comparable*. Pulseq pairs (placeholder headers) keep the filename-based warning only.

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
  │    └─ Variable Density (VD): PDF = (1−|x|)^power + 0.05 floor,
  │        fixed ACS + edges, numpy weighted random sampling
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
| ESPIRiT | ESPIRiT crop | 0.8 | all ESPIRiT methods | Eigenvalue mask threshold, same in both engines |
| ESPIRiT | ESPIRiT threshold | Engine default | all ESPIRiT methods | Null-space threshold; *Engine default* = BART 0.001 / sigpy 0.02, a value applies to both |
| Siemens scans | Partial-Fourier lines | Zero-fill | all Cartesian (Siemens product) | Zero-fill / Homodyne / POCS for lines a partial-Fourier protocol never acquired. No effect on Pulseq scans |
| Siemens scans | Noise prewhitening | On | all Cartesian (Siemens product) | Decorrelate coils with the file's noise-adjust scan before coil processing |
| Pulseq scans | .seq library | — | Pulseq scans | Extra folder searched for the played `.seq`, which sets the line geometry |
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

The app expects the following layout (the trajectory/mask JSONs are inputs you supply
alongside the raw `.dat` data, in the format shown):

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
├── Undersampled4equispaced.json     ← <R>× equispaced + ACS PE mask (R matches the .dat)
└── seq_files/                       ← optional: the Pulseq .seq files that were played
    └── *.seq                           (matched by signature, so names may differ)
```

Siemens **product** scans need none of the JSONs or naming conventions — a folder of
`meas_MID….dat` files is enough. For Pulseq scans, keep the exact `.seq` of every scan
(the generator script can overwrite it with a new signature later); see
[Pulseq: the played `.seq`](#pulseq-the-played-seq).

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
│   │                             (shared fully-sampled comparison + comparability gate)
│   ├── acquisition.py         ← acquire_cartesian(): full-grid k-space from the header
│   │                             (Siemens) or the played .seq / trajectory JSON (Pulseq),
│   │                             line categories, facts with sources; prewhitening;
│   │                             finalize_image()
│   ├── partial_fourier.py     ← pf_fill(): zero-fill / homodyne / POCS
│   ├── seq_geometry.py        ← played-.seq lookup by signature (+ asking the user),
│   │                             pypulseq / trajectory-JSON line geometry
│   ├── tse_reorder.py         ← Pulseq TSE echo-train plan (legacy path, JSON check)
│   └── retro_undersample.py   ← reconstruct_retro()        — retrospective simulator
│                                   build_equispaced_mask()  — port of MATLAB function
│                                   build_vd_mask()          — port of MATLAB function
├── tools/                     ← developer scripts (not shipped in the app)
│   ├── golden.py              ← capture / compare real-scan outputs (regression net)
│   ├── seq_parity.py          ← legacy Pulseq geometry vs the played .seq, per scan
│   └── acq_report.py          ← print the acquisition fact sheet of every .dat
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
| `golden` | a captured baseline (`SHIR_GOLDEN_DIR`) | the variable is unset |

```bash
pytest -m "not bart and not realdata"    # CI-friendly subset (no BART, no data)
pytest -m realdata                        # end-to-end on real .dat scans
```

**Golden regression (real scans).** [`tools/golden.py`](tools/golden.py) runs a manifest of real
Pulseq scans through every pipeline and engine and stores the outputs, so a change can be
checked against a known-good commit:

```bash
python tools/golden.py capture --app-root <clean repo checkout> --out ~/shir_golden/<rev>
python tools/golden.py compare --app-root .. --baseline ~/shir_golden/<rev> [--fast]
SHIR_GOLDEN_DIR=~/shir_golden/<rev> pytest -m golden     # SHIR_GOLDEN_FULL=1 for all cases
```

`--app-root` is the repository root (the folder holding `AnalyzeMRIDataApp/` and
`read_siemens_scans/`). Each case reports `PASS` (bit-identical), `PASS~` (within tolerance),
`SAME-ERROR`, `MIRRORED`, `CHANGED` or `FAIL`. A Pulseq case still on the legacy path must
match (any `FAIL` exits non-zero); a Pulseq case now placed by its `.seq`/JSON is `MIRRORED`
when it is the baseline mirrored along phase encode (the legacy placement had the direction
reversed) and `CHANGED` otherwise; Siemens product cases are `CHANGED` when the header-driven
geometry changed them.

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
- **Siemens product scans:** 2-D Cartesian, single echo, one repetition. 3-D, multi-echo,
  repetitions/sets and self-contradicting files are refused with a message (see
  [How SHIR reads a scan](#how-shir-reads-a-scan)). Scans with a *separate* PAT reference
  scan still use the legacy geometry. Expect differences from the scanner's own DICOM images:
  the scanner also applies filters, 2-D distortion correction, prescan normalise and
  interpolation, which SHIR does not.
- **Pulseq scans without their `.seq` or trajectory JSON** fall back to the legacy placement
  (file name + line count + TSE plan). Compared with the `.seq` (`tools/seq_parity.py`), that
  placement is right for the lab's newer sequences, but **mirrored** along phase encode for the
  older SE ("band") sequences and the `swapPE` scans, and wrong for a TSE whose echo order it
  cannot infer (e.g. 266 readouts, not a multiple of the echo-train length). Keep every scan's
  `.seq`. Pulseq scans are not prewhitened (no noise-adjust data); multi-slice Pulseq scans are
  placed by the `.seq` only when every slice has the same line pattern.
- **ESPIRiT thresholds differ between engines by default:** BART `ecalib` uses `-t 0.001`, sigpy
  `EspiritCalib` uses `thresh 0.02` (each library's default, passed explicitly). This is one
  source of the small BART/sigpy differences; set **ESPIRiT threshold** to use one value in both.
- **ESPIRiT calibrates on a grid-centred region.** When a `.seq` places the calibration block a
  line off the grid centre, the region is narrowed (e.g. block rows 114–138 on a 256 grid:
  25 → 23 lines) so it holds measured lines only; the log's `calib=` value is the width used.
- **TSE folders** (`TSE_1mm`, `TSE_3mm`) contain Band data only; Fully-Sampled, Equispaced and
  Poisson-Disc scan types will not appear for these folders.
- Intermediate BART `.cfl` / `.hdr` files are written to a **temporary directory** and deleted
  automatically via `BartTempDir` after each reconstruction.
- WSL2 on Windows 11 supports **WSLg** natively. Older setups require VcXsrv with
  `export DISPLAY=:0` in the WSL shell before running.
- GRAPPA reconstruction assumes a **regular equispaced** under-sampling pattern — it is *not* tied
  to a fixed acceleration. With a `.seq`, trajectory JSON or Siemens header, the mask, R and
  calibration block are measured from the file. Only on the legacy path does R come from the
  filename (e.g. `equispaced_R4`), and it must then match the **R-factor** in the parameter panel. Non-equispaced (e.g. variable-density) patterns are better served
  by the SENSE/CS methods; the retrospective simulator lets you pick R (and mask type) freely.
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
4. **Demo data** goes on the same Release as a separate zip asset (up to 2 GB per file), not
   into the repository: a single TWIX file is typically 200–300 MB, over GitHub's 100 MB
   per-file limit, and `*.dat` is git-ignored. Include the protocol PDF, and the `.seq` files for
   Pulseq scans, and link the asset from this README.

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
The `read_siemens_scans` module is adapted from **Alon Granek**'s repository, which is
based on **Orel Tsioni**'s [`raw_kspace_recon`](https://github.com/orelts/raw_kspace_recon).
