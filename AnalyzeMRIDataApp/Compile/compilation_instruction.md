# Compiling SHIR into a standalone Windows `.exe`

This guide explains how to build SHIR into a standalone,
double-clickable Windows application using the `compilation.py` script in this
folder. It also documents every error we hit during the first build and how to
fix it.

> **Who runs this:** the developer packaging the app. End users do **not** need
> any of this — they just need the prerequisites in [`demands.md`](demands.md).

---

## 1. Key facts before you start

| Fact | Why it matters |
|------|----------------|
| **PyInstaller is not a cross-compiler.** | You must run `compilation.py` on **native Windows** Python (PowerShell/cmd), *not* from WSL. Building in WSL produces a Linux binary, not a `.exe`. |
| **Build outside OneDrive.** | OneDrive locks files mid-build → `PermissionError`. Build from a local path like `C:\dev\`. |
| **BART is Linux-only.** | The compiled Windows app calls BART **through WSL** at runtime. BART is not bundled. See [`demands.md`](demands.md). |
| **torch is excluded (`--no-torch`).** | torch's DLLs fail to load on locked-down/managed machines and are only needed for the optional `ggrappa` recon method. Excluding it makes the build smaller, faster, and reliable. |
| **Output is one folder (`--onedir`).** | A folder containing `SHIR.exe` + dependencies. Distribute it as a zip. |
| **App icon comes from `Graphics/logo.png`.** | The script passes `--icon logo.png`; PyInstaller **≥ 5** auto-converts the PNG to `.ico` via **Pillow** (already in `requirements.txt`). No committed `.ico` needed. If Pillow is missing the build still succeeds with the default icon. |
| **`Graphics/` is bundled.** | `--add-data Graphics` ships `Intro.png` (splash) + `logo.png` so the frozen exe finds them via `resources.resource_path()`. |

---

## 2. One-time environment setup (on native Windows)

Use Python **3.11–3.13** (a conda env or a venv both work). Example with conda:

```powershell
conda create -n mri_toolkit python=3.13
conda activate mri_toolkit
```

Install the build tool + app dependencies:

```powershell
pip install pyinstaller

# ggrappa: install WITHOUT its deps (its pinned scipy==1.13 tries to build from
# source and fails with "Unknown compiler(s)" — we don't need it anyway).
pip install git+https://github.com/mind-inria/ggrappa.git --no-deps

# The scientific stack (torch is optional and excluded from the final build):
pip install PyQt5 matplotlib pillow scikit-image pygrappa sigpy twixtools mri-nufft finufft numba h5py opencv-python networkx imageio pyyaml pandas
```

**Sanity check** (fast, catches broken installs before a long build):

```powershell
python -c "import numpy, scipy, sigpy, numba, PyQt5, matplotlib, twixtools, skimage; print('all imports OK')"
```

---

## 3. Copy the repo to a local path (avoid OneDrive)

```powershell
robocopy "C:\Users\<you>\OneDrive - Technion\...\MRI_Acquisition_Reconstruction_Toolkit" "C:\dev\MRI_Toolkit" /E /XD .git Compile\build Compile\compilation_product
cd C:\dev\MRI_Toolkit
```

The conda env works from any location, so no reinstall is needed.

---

## 4. Build

### Test build (keep the console to see errors)
```powershell
python AnalyzeMRIDataApp\Compile\compilation.py --console --no-torch
```

### Final build for distribution (windowed, no console)
```powershell
python AnalyzeMRIDataApp\Compile\compilation.py --no-torch
```

Output:
```
AnalyzeMRIDataApp\Compile\compilation_product\SHIR\SHIR.exe
```

### Script flags
| Flag | Effect |
|------|--------|
| `--no-torch` | **Always use.** Excludes torch + ggrappa. Keeps all BART/sigpy/pygrappa recon working; disables only the `ggrappa` method. |
| `--console` | Keep only while testing — prints Python tracebacks to a console window. Drop it for the final build. |
| `--allow-non-windows` | Deliberately build on a non-Windows platform (produces a non-`.exe`; for testing the script only). |

---

## 5. Distribute

Confirm the **windowed** exe launches and runs one full recon (windowed builds
show nothing if they crash before the GUI opens, so verify once). Then zip it:

```powershell
Compress-Archive -Path "C:\dev\MRI_Toolkit\AnalyzeMRIDataApp\Compile\compilation_product\SHIR\*" -DestinationPath "C:\dev\SHIR.zip"
```

Recipients unzip and run `SHIR.exe`. They must satisfy the runtime
prerequisites in [`demands.md`](demands.md) (WSL + BART) for BART-based recon.

---

## 6. Troubleshooting — every error we hit, and the fix

### 6.1 `ERROR: Hidden import 'read_siemens_scans.twix_reading' not found` / "not a package"
- **Cause:** `read_siemens_scans` is a **nested** package
  (`repo/read_siemens_scans/read_siemens_scans/`). PyInstaller was pointed at
  the outer folder, which has no `__init__.py`.
- **Fix (already in `compilation.py`):** the import path is set to
  `REPO_ROOT/read_siemens_scans` (the parent of the actual package), matching
  what `app.py` does at runtime. Nothing to do unless you move the package.

### 6.2 `PermissionError: [WinError 5] Access is denied: ...\Compile\build\...\localpycs`
- **Cause:** OneDrive (or antivirus/Application Control) locking build files.
- **Fix:** Build from a local path (`C:\dev\...`), not OneDrive. If you must
  build in place: pause OneDrive syncing, kill stray `python.exe`, delete
  `Compile\build`, and retry.

### 6.3 `OSError: [WinError 1114] ... Error loading ...\torch\lib\c10.dll` (during a recon)
- **Cause:** `sigpy` imports `torch` at import time. On managed machines
  torch's DLLs fail to initialize (also seen as `WinError 4551 Application
  Control policy has blocked this file`). Because `sigpy` only catches
  `ImportError`, the `OSError` propagates and kills the recon.
- **Fix:** Build with **`--no-torch`**. Without torch, `sigpy` gets a clean
  `ImportError`, disables its torch backend, and all BART/sigpy/pygrappa recon
  works. (The `ggrappa` method becomes unavailable — it needs torch, which
  wasn't loading anyway.)

### 6.4 `RuntimeError: BART command failed: ... bart: command not found`
- **Cause:** The app bridges BART calls into WSL, but BART isn't on WSL's
  default PATH (e.g. it's a source build at `/home/<user>/bart/bart`).
- **Fix:** Set a Windows env var to BART's absolute WSL path (no rebuild
  needed — the app reads it at runtime):
  ```powershell
  setx MRI_BART_BIN /home/<user>/bart/bart
  ```
  Open a fresh terminal afterward. See [`demands.md`](demands.md) for the full
  BART setup. Override the launcher with `MRI_WSL_EXE` if `wsl.exe` isn't on PATH.

### 6.5 `AttributeError: 'NoneType' object has no attribute 'write'` (only in the windowed build)
- **Cause:** A `--windowed` PyInstaller build has no console, so
  `sys.stdout`/`sys.stderr` are `None`. Libraries that print (twixtools
  progress, tqdm) crash on the first write.
- **Fix (already in `app.py`):** a startup guard redirects `None` streams to a
  log file. If you see this again, ensure that guard runs **before** any import
  that prints. Debug windowed-only issues via the log:
  ```powershell
  type $env:TEMP\SHIR.log
  ```

### 6.6 `error: metadata-generation-failed` building scipy (while installing ggrappa)
- **Cause:** ggrappa pins `scipy==1.13.*`, which tries to compile from source
  and fails (`Unknown compiler(s)`, no MSVC).
- **Fix:** Install ggrappa with `--no-deps` (step 2). The resulting
  `ggrappa 0.1.0 requires scipy==1.13.*` dependency-conflict warning is
  harmless — modern scipy works fine.

---

## 7. How the build script works (for maintainers)

`compilation.py` invokes PyInstaller programmatically (no separate `.spec`
file). Key pieces:

- **Paths:** `APP_DIR` (has `app.py`) and `TWIX_PARENT` (parent of the nested
  `read_siemens_scans` package) are added via `--paths`.
- **`COLLECT_ALL`:** pulls data/binaries/submodules for packages PyInstaller
  can't fully trace (numba, llvmlite, finufft, mri_nufft, sigpy, pygrappa,
  twixtools, skimage, scipy, matplotlib, read_siemens_scans).
- **`TORCH_PKGS`:** `torch` + `ggrappa`, added to the build only when
  `--no-torch` is **absent**, otherwise excluded.
- **`HIDDEN_IMPORTS`:** lazily/dynamically imported modules
  (`read_siemens_scans.twix_reading`, `PyQt5.sip`, the matplotlib Qt5 backend).
- **`EXCLUDES`:** dev-only bloat (jupyter/notebook/IPython/pytest/tkinter).
- **Pre-flight checks:** aborts early if not on Windows (unless
  `--allow-non-windows`), if the entry point or the nested package is missing,
  or if PyInstaller isn't installed.

To tune the build, edit the `COLLECT_ALL` / `HIDDEN_IMPORTS` / `EXCLUDES` lists
at the top of the script and rerun.
