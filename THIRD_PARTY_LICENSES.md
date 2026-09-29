# Third-Party Licenses & Credits

SHIR (Copyright © 2026 Saar Drive, Shimron Lab, Technion — Israel Institute of
Technology) is licensed under **GPL-3.0** (see [LICENSE](LICENSE)). It builds
on the following third-party components. Each remains under its own license;
nothing in SHIR's license changes their terms.

## Bundled / vendored in this repository

| Component | Location | License | Role |
|---|---|---|---|
| gGRAPPA | `ggrappa/` | BSD 3-Clause (per its bundled [`ggrappa/LICENSE`](ggrappa/LICENSE); note its `pyproject.toml` metadata says MIT — the LICENSE file governs) | Optional torch-based GRAPPA for the undersampling simulator |
| read_siemens_scans | `read_siemens_scans/` | Adapted from **Alon Granek**'s repository (Shimron Lab), which is based on **Orel Tsioni**'s [`raw_kspace_recon`](https://github.com/orelts/raw_kspace_recon) | Siemens TWIX `.dat` reading (wraps twixtools) |

## External runtime dependencies (installed via pip, not bundled in the repo)

| Library | License | Role |
|---|---|---|
| [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) | **GPL v3** (or Riverbank commercial) | GUI toolkit. PyQt5's GPL is the reason SHIR itself is GPL-3.0: binaries that bundle PyQt5 must be distributed under GPL-compatible terms. |
| [numpy](https://numpy.org) | BSD 3-Clause | Numerics |
| [scipy](https://scipy.org) | BSD 3-Clause | Numerics, image resize/morphology |
| [matplotlib](https://matplotlib.org) | PSF-based (BSD-compatible) | All figure rendering + PDF reports |
| [sigpy](https://github.com/mikgroup/sigpy) | BSD 3-Clause | Built-in reconstruction engine (ESPIRiT, SENSE, CS, NUFFT) |
| [pygrappa](https://github.com/mckib2/pygrappa) | MIT | GRAPPA kernel calibration & reconstruction |
| [twixtools](https://github.com/pehses/twixtools) | MIT | Siemens TWIX raw-data parsing |
| [scikit-image](https://scikit-image.org) | BSD 3-Clause | SSIM and image metrics |
| [PyTorch](https://pytorch.org) *(optional)* | BSD-style | Backend for the optional gGRAPPA method |

## External binary (invoked as a subprocess, never bundled)

| Tool | License | Role |
|---|---|---|
| [BART](https://mrirecon.github.io/bart) — Berkeley Advanced Reconstruction Toolbox | BSD 3-Clause | Preferred reconstruction engine (`nufft`, `ecalib`, `pics`, …). SHIR calls the user's own BART installation as an external process; BART is not distributed with SHIR. |
