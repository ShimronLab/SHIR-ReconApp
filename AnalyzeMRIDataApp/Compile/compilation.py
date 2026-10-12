#!/usr/bin/env python3
"""
Build script — package AnalyzeMRIDataApp into a standalone executable.

Run this on a NATIVE WINDOWS Python interpreter (PowerShell / cmd) to produce a
Windows ``.exe``. PyInstaller is not a cross-compiler: running this from WSL /
Linux yields a Linux binary, not an ``.exe``.

    py -m venv .venv-build
    .venv-build\\Scripts\\activate
    pip install -r AnalyzeMRIDataApp\\requirements.txt ^
                -r AnalyzeMRIDataApp\\requirements-dev.txt
    python AnalyzeMRIDataApp\\Compile\\compilation.py --no-torch

See Compile/compilation_instruction.md for the full step-by-step guide.
Output lands in ``AnalyzeMRIDataApp/Compile/compilation_product/SHIR/``.

Notes
-----
* Package mode is ``--onedir`` (one folder + exe): reliable for the heavy
  scientific stack (numba/scipy). Distribute by zipping the output folder.
* Build with ``--no-torch`` (recommended): torch + ggrappa are excluded, which
  keeps all BART/sigpy/pygrappa recon working and only disables the optional
  ``ggrappa`` method. Omit ``--no-torch`` (and pip-install torch + ggrappa
  first) only if you specifically need that method.
* BART is a Linux/WSL-only external binary and cannot be bundled into a native
  Windows exe. Instead, run_bart() (processing/utils.py) bridges BART calls into
  WSL on Windows, so BART-based recon (pics/ecalib/nufft/pattern) works PROVIDED
  the target machine has WSL installed with BART on its PATH (`wsl bart version`
  must succeed). The exe itself is standalone; BART is used via the system WSL.

Usage
-----
    python compilation.py                # windowed build (no console)
    python compilation.py --console      # keep a console for first-run debugging
    python compilation.py --allow-non-windows   # deliberate Linux test build
"""

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

# ── Paths (all relative to this script) ─────────────────────────────────────
COMPILE_DIR = Path(__file__).resolve().parent
APP_DIR = COMPILE_DIR.parent                    # contains app.py
REPO_ROOT = APP_DIR.parent                      # repo root
# The read_siemens_scans PACKAGE is nested one level down (repo has an outer
# read_siemens_scans/ folder that CONTAINS the actual read_siemens_scans/ pkg).
# The parent that must be on the import path is this outer folder — mirroring
# what app.py adds to sys.path at runtime (TWIX_DIR).
TWIX_PARENT = REPO_ROOT / "read_siemens_scans"  # parent dir of the package
ENTRY = APP_DIR / "app.py"
DIST = COMPILE_DIR / "compilation_product"
WORK = COMPILE_DIR / "build"
SPEC = COMPILE_DIR
APP_NAME = "SHIR"

# ── Packaging configuration ─────────────────────────────────────────────────
# Collect data files + binaries + submodules for packages PyInstaller cannot
# fully resolve on its own (compiled extensions, dynamic imports, plugins).
COLLECT_ALL = [
    "read_siemens_scans",
    "numba",
    "llvmlite",
    "sigpy",
    "pygrappa",
    "twixtools",
    "pypulseq",
    "skimage",
    "scipy",
    "matplotlib",
]

# torch (+ ggrappa, which needs it) are only used by the 'ggrappa' recon method.
# They are large and, on locked-down machines, torch's DLLs may be blocked by an
# Application Control policy (WinError 1114/4551). Build with --no-torch to drop
# them: sigpy then gets a clean ImportError and disables its torch backend, so
# all BART/sigpy/pygrappa recon still works.
TORCH_PKGS = ["torch", "ggrappa"]

# Modules imported lazily / dynamically that static analysis misses.
HIDDEN_IMPORTS = [
    "read_siemens_scans.twix_reading",
    "PyQt5.sip",
    "matplotlib.backends.backend_qt5agg",
]

# Dev-only bloat pulled in by requirements.txt but never used by the app.
EXCLUDES = [
    "jupyter",
    "notebook",
    "jupyterlab",
    "jupyterlab_server",
    "jupyter_server",
    "IPython",
    "ipykernel",
    "ipython",
    "nbconvert",
    "nbformat",
    "nbclient",
    "pytest",
    "tkinter",
]


def _preflight(allow_non_windows: bool) -> None:
    """Fail early with actionable messages if the environment is wrong."""
    if sys.platform != "win32" and not allow_non_windows:
        sys.exit(
            "ERROR: not running on native Windows Python (sys.platform="
            f"{sys.platform!r}).\n"
            "PyInstaller cannot cross-compile — this would produce a non-Windows\n"
            "binary. Run this script from a native Windows Python, or pass\n"
            "--allow-non-windows for a deliberate test build on this platform."
        )

    if not ENTRY.is_file():
        sys.exit(f"ERROR: entry point not found: {ENTRY}")

    # The importable package is TWIX_PARENT/read_siemens_scans/__init__.py .
    twix_init = TWIX_PARENT / "read_siemens_scans" / "__init__.py"
    if not twix_init.is_file():
        sys.exit(
            f"ERROR: read_siemens_scans package not found at {twix_init}.\n"
            "Expected the nested layout REPO_ROOT/read_siemens_scans/read_siemens_scans/."
        )

    if importlib.util.find_spec("PyInstaller") is None:
        sys.exit("ERROR: PyInstaller is not installed. Run: pip install pyinstaller")

    # torch/ggrappa are bundled by default; warn (don't abort) if missing so
    # the build still runs for users who intentionally dropped them.
    for opt in ("torch", "ggrappa"):
        if importlib.util.find_spec(opt) is None:
            print(
                f"WARNING: '{opt}' is not importable; the ggrappa recon method "
                "will be unavailable in the build. (pip install torch ggrappa)"
            )

    # Pillow is needed by PyInstaller (≥ 5) to convert the PNG --icon to .ico.
    if importlib.util.find_spec("PIL") is None:
        print(
            "WARNING: Pillow is not installed; the exe will keep PyInstaller's "
            "default icon. (pip install pillow)"
        )


def _build_args(console: bool, include_torch: bool) -> list[str]:
    """Assemble the PyInstaller command-line arguments."""
    import os

    collect = list(COLLECT_ALL)
    excludes = list(EXCLUDES)
    if include_torch:
        collect += TORCH_PKGS
    else:
        # Drop torch/ggrappa entirely. sigpy imports torch at import time but
        # falls back gracefully on ImportError, so BART/sigpy recon still works.
        excludes += TORCH_PKGS

    args = [
        str(ENTRY),
        "--name", APP_NAME,
        "--onedir",
        "--noconfirm",
        "--clean",
        "--console" if console else "--windowed",
        "--distpath", str(DIST),
        "--workpath", str(WORK),
        "--specpath", str(SPEC),
        "--paths", str(APP_DIR),
        "--paths", str(TWIX_PARENT),
    ]

    # Bundle Graphics/ (splash + logo) so resource_path() finds it when frozen.
    graphics = APP_DIR / "Graphics"
    if graphics.is_dir():
        args += ["--add-data", f"{graphics}{os.pathsep}Graphics"]

    # App icon (exe / taskbar). PyInstaller ≥ 5 converts the PNG to .ico/.icns
    # automatically via Pillow; skipped on Linux test builds where it is
    # meaningless.
    icon_png = graphics / "logo.png"
    if icon_png.is_file() and sys.platform in ("win32", "darwin"):
        args += ["--icon", str(icon_png)]

    for pkg in collect:
        args += ["--collect-all", pkg]
    for mod in HIDDEN_IMPORTS:
        args += ["--hidden-import", mod]
    for mod in excludes:
        args += ["--exclude-module", mod]
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AnalyzeMRIDataApp into a standalone app.")
    parser.add_argument(
        "--console", action="store_true",
        help="Keep a console window (recommended for the first debug build).",
    )
    parser.add_argument(
        "--allow-non-windows", action="store_true",
        help="Permit building on a non-Windows platform (produces a non-.exe binary).",
    )
    parser.add_argument(
        "--no-torch", action="store_true",
        help="Exclude torch + ggrappa (smaller/faster build; disables the ggrappa "
             "recon method but keeps all BART/sigpy/pygrappa methods working).",
    )
    opts = parser.parse_args()

    _preflight(opts.allow_non_windows)

    # Fresh output each run.
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    DIST.mkdir(parents=True, exist_ok=True)

    if opts.no_torch:
        print("torch/ggrappa EXCLUDED from this build (--no-torch).")
    args = _build_args(opts.console, include_torch=not opts.no_torch)
    print("Running PyInstaller with:\n  " + " ".join(args) + "\n")

    import PyInstaller.__main__
    PyInstaller.__main__.run(args)

    exe = DIST / APP_NAME / (f"{APP_NAME}.exe" if sys.platform == "win32" else APP_NAME)
    print("\n" + "=" * 70)
    if exe.exists():
        print(f"BUILD SUCCEEDED\nExecutable: {exe}")
        print("Distribute by zipping the folder:", DIST / APP_NAME)
    else:
        print("BUILD FINISHED but expected executable not found at:", exe)
        print("Check the PyInstaller output above for errors.")
    print("=" * 70)


if __name__ == "__main__":
    main()
