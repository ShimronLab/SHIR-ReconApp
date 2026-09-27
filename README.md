# SHIR — System for Hybrid Image Reconstruction

**SHIR** is a PyQt5 desktop application for multi-pipeline MRI reconstruction and
visualisation of Siemens TWIX `.dat` files. It runs from source (Linux / WSL / macOS)
and can be packaged as a standalone app for **Windows (`SHIR.exe`)** and
**macOS (`SHIR.app`)**.

This repository is the **standalone reconstruction app**, extracted from a larger
research toolkit. It contains everything needed to run and build the app — no
pulse-sequence generation code is included.

Supported reconstruction pipelines include fully-sampled RSS / ESPIRiT (SENSE),
equispaced GRAPPA / SENSE / CS-SENSE / ESPIRiT, non-Cartesian (band) and
Poisson-disc compressed sensing, plus a retrospective under-sampling simulator.
PDF reports are generated per scan folder, including **all coil sensitivity maps**.

---

## Repository layout

```
SHIR-ReconApp/
├── AnalyzeMRIDataApp/        # the app (source folder name kept for stability)
│   ├── app.py               # entry point
│   ├── processing/          # reconstruction pipelines & backends (BART / sigpy)
│   ├── ui/                  # PyQt5 windows, panels, report/PDF generator
│   ├── Graphics/            # logo / splash
│   ├── Compile/             # PyInstaller spec + build scripts & instructions
│   ├── testing/             # development notebooks (PE-mask comparison, …)
│   ├── requirements.txt     # pinned Python dependencies
│   └── README.md            # full app documentation (usage, pipelines, params)
├── read_siemens_scans/      # TWIX (.dat) reader package — REQUIRED, imported by the app
├── ggrappa/                 # vendored optional dep for the "gGRAPPA (PyTorch)" option
└── run_mri_app.bat          # Windows launcher (WSL + conda)
```

> `read_siemens_scans/` must remain a **sibling** of `AnalyzeMRIDataApp/` — the app
> resolves it by relative path at startup.

Full usage, pipeline, and parameter documentation lives in
[`AnalyzeMRIDataApp/README.md`](AnalyzeMRIDataApp/README.md). Packaging / build guides
are in [`AnalyzeMRIDataApp/Compile/`](AnalyzeMRIDataApp/Compile/).

---

## Installation

```bash
# 1. Create the environment (conda recommended)
conda create -n mri_toolkit python=3.11 -y
conda activate mri_toolkit

# 2. Core dependencies
pip install -r AnalyzeMRIDataApp/requirements.txt
```

### Optional components

- **BART** (exact recon numerics) — optional. Without it, SHIR falls back to a
  built-in **sigpy** engine. The in-app *Recon engine* selector (Auto / Force BART /
  Force sigpy) chooses between them. On Windows, BART is invoked through WSL.

- **gGRAPPA (PyTorch)** — the `gGRAPPA` recon option needs the vendored `ggrappa/`
  package plus PyTorch (CPU build is fine):

  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install --no-deps ./ggrappa
  ```

  `--no-deps` is deliberate: `ggrappa` pins an older SciPy that would otherwise
  downgrade the environment. Without this, the `gGRAPPA` option simply falls back
  to zero-fill; every other pipeline works normally.

---

## Running

**From source (Linux / WSL / macOS):**

```bash
conda activate mri_toolkit
cd AnalyzeMRIDataApp
python app.py
```

**Windows:** double-click `run_mri_app.bat` (launches under WSL + the `mri_toolkit`
conda env).

To build a standalone executable, see
[`AnalyzeMRIDataApp/Compile/compilation_instruction.md`](AnalyzeMRIDataApp/Compile/compilation_instruction.md).

---

## Credits & licensing

SHIR is developed in the Shimron Lab, Technion. The vendored `ggrappa/` package is
third-party (see `ggrappa/LICENSE`). See `AnalyzeMRIDataApp/README.md` for full
credits.
