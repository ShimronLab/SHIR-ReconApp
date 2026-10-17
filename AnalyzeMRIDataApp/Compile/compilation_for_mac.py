#!/usr/bin/env python3
"""
Build script — package AnalyzeMRIDataApp into a standalone macOS app bundle.

Run this on a NATIVE macOS Python interpreter to produce a ``SHIR.app``
bundle. PyInstaller is not a cross-compiler: you cannot build a macOS app from
Windows or Linux — it must run on a Mac.

    python3 -m venv .venv-build
    source .venv-build/bin/activate
    pip install -r AnalyzeMRIDataApp/requirements.txt \\
                -r AnalyzeMRIDataApp/requirements-dev.txt
    # ggrappa (optional; needs torch). Install with --no-deps to skip its scipy pin:
    #   pip install torch && pip install --no-deps ./ggrappa
    python AnalyzeMRIDataApp/Compile/compilation_for_mac.py --no-torch

Output lands in ``AnalyzeMRIDataApp/Compile/compilation_product/SHIR.app``.

Notes
-----
* Package mode is ``--onedir`` inside a ``.app`` bundle — reliable for the heavy
  scientific stack. Distribute by zipping the ``.app``.
* Unlike Windows, **BART runs natively on macOS** — no WSL bridge is needed.
  ``processing/utils.run_bart`` calls ``bart`` directly on any non-Windows OS, so
  if BART is on the PATH the app uses it; otherwise it falls back to the built-in
  sigpy engine automatically. Nothing extra to configure here.
* ``--no-torch`` drops torch + ggrappa (recommended): smaller/faster build; only
  the optional ``ggrappa`` recon method becomes unavailable.

Architecture & distribution
---------------------------
* A build on Apple Silicon (arm64) will NOT run on Intel Macs and vice-versa.
  Build on the target architecture, or pass ``--target-arch universal2`` IF your
  Python and all wheels are universal2 (most aren't — arm64/x86_64 is simpler).
* macOS Gatekeeper blocks unsigned apps downloaded to another Mac. For lab use,
  recipients right-click → Open once, or run:
      xattr -dr com.apple.quarantine SHIR.app
  For public distribution, codesign + notarize with an Apple Developer ID.

Usage
-----
    python3 compilation_for_mac.py --no-torch              # windowed .app (recommended)
    python3 compilation_for_mac.py --no-torch --console    # plain terminal binary for debugging
    python3 compilation_for_mac.py --no-torch --target-arch universal2
    python3 compilation_for_mac.py --allow-non-mac         # deliberate off-Mac test build
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
TWIX_PARENT = REPO_ROOT / "read_siemens_scans"  # parent dir of the package
ENTRY = APP_DIR / "app.py"
DIST = COMPILE_DIR / "compilation_product"
WORK = COMPILE_DIR / "build"
SPEC = COMPILE_DIR
APP_NAME = "SHIR"
BUNDLE_ID = "com.shimronlab.shir"

# ── Packaging configuration (identical to the Windows build) ────────────────
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

# torch (+ ggrappa) are only used by the optional 'ggrappa' recon method and are
# large; --no-torch drops them and sigpy falls back gracefully on ImportError.
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


def _require_runtime_packages() -> None:
    """
    Abort if a package in COLLECT_ALL is not importable. PyInstaller's
    collect_all bundles nothing for a missing package and the build still
    succeeds, so the app would ship without it — e.g. without pypulseq, Pulseq
    scans silently fall back from the played .seq to the trajectory file.
    """
    missing = [pkg for pkg in COLLECT_ALL
               if pkg != "read_siemens_scans"            # found via TWIX_PARENT, not installed
               and importlib.util.find_spec(pkg) is None]
    if missing:
        sys.exit(
            f"ERROR: required package(s) not installed in this Python: {', '.join(missing)}.\n"
            "Install the runtime requirements first:  pip install -r requirements.txt"
        )


def _preflight(allow_non_mac: bool) -> None:
    """Fail early with actionable messages if the environment is wrong."""
    if sys.platform != "darwin" and not allow_non_mac:
        sys.exit(
            "ERROR: not running on native macOS Python (sys.platform="
            f"{sys.platform!r}).\n"
            "PyInstaller cannot cross-compile — a macOS .app must be built on a\n"
            "Mac. Run this on macOS, or pass --allow-non-mac for a deliberate\n"
            "off-Mac test build (produces a non-.app binary)."
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

    _require_runtime_packages()
    for opt in ("torch", "ggrappa"):
        if importlib.util.find_spec(opt) is None:
            print(
                f"WARNING: '{opt}' is not importable; the ggrappa recon method "
                "will be unavailable in the build. (drop it with --no-torch)"
            )

    # Pillow is needed by PyInstaller (≥ 5) to convert the PNG --icon to .icns.
    if importlib.util.find_spec("PIL") is None:
        print(
            "WARNING: Pillow is not installed; the app will keep PyInstaller's "
            "default icon. (pip install pillow)"
        )


def _build_args(console: bool, include_torch: bool, target_arch: str) -> list:
    """Assemble the PyInstaller command-line arguments for macOS."""
    collect = list(COLLECT_ALL)
    excludes = list(EXCLUDES)
    if include_torch:
        collect += TORCH_PKGS
    else:
        excludes += TORCH_PKGS

    args = [
        str(ENTRY),
        "--name", APP_NAME,
        "--onedir",
        "--noconfirm",
        "--clean",
        # --windowed builds the .app bundle; --console makes a plain unix binary.
        "--console" if console else "--windowed",
        "--distpath", str(DIST),
        "--workpath", str(WORK),
        "--specpath", str(SPEC),
        "--paths", str(APP_DIR),
        "--paths", str(TWIX_PARENT),
        "--osx-bundle-identifier", BUNDLE_ID,
    ]
    if target_arch:
        args += ["--target-arch", target_arch]

    # Bundle Graphics/ (splash + logo) so resource_path() finds it when frozen.
    import os
    graphics = APP_DIR / "Graphics"
    if graphics.is_dir():
        args += ["--add-data", f"{graphics}{os.pathsep}Graphics"]

    # App icon: PyInstaller ≥ 5 converts the PNG to .icns via Pillow.
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


def _result_path(console: bool):
    """Where PyInstaller should have written the app for this build type."""
    if console or sys.platform != "darwin":
        # Plain onedir binary (no .app bundle).
        return DIST / APP_NAME / APP_NAME
    return DIST / f"{APP_NAME}.app"       # macOS windowed → .app bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AnalyzeMRIDataApp into a standalone macOS app.")
    parser.add_argument(
        "--console", action="store_true",
        help="Build a plain terminal binary (prints tracebacks) instead of a .app bundle.",
    )
    parser.add_argument(
        "--allow-non-mac", action="store_true",
        help="Permit building on a non-macOS platform (produces a non-.app binary).",
    )
    parser.add_argument(
        "--no-torch", action="store_true",
        help="Exclude torch + ggrappa (smaller/faster; disables only the ggrappa "
             "recon method — all sigpy/BART/pygrappa methods keep working).",
    )
    parser.add_argument(
        "--target-arch", default="", choices=["", "x86_64", "arm64", "universal2"],
        help="PyInstaller target architecture. Default: native. 'universal2' "
             "requires a universal2 Python and universal2 wheels.",
    )
    opts = parser.parse_args()

    _preflight(opts.allow_non_mac)

    # Fresh output each run.
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    DIST.mkdir(parents=True, exist_ok=True)

    if opts.no_torch:
        print("torch/ggrappa EXCLUDED from this build (--no-torch).")
    args = _build_args(opts.console, include_torch=not opts.no_torch,
                       target_arch=opts.target_arch)
    print("Running PyInstaller with:\n  " + " ".join(args) + "\n")

    import PyInstaller.__main__
    PyInstaller.__main__.run(args)

    result = _result_path(opts.console)
    print("\n" + "=" * 70)
    if result.exists():
        print(f"BUILD SUCCEEDED\nApp: {result}")
        if result.suffix == ".app":
            print("Distribute by zipping the .app bundle:", result)
            print("On the target Mac, if Gatekeeper complains, right-click → Open,")
            print(f"or run:  xattr -dr com.apple.quarantine '{result.name}'")
        else:
            print("Distribute by zipping the folder:", DIST / APP_NAME)
    else:
        print("BUILD FINISHED but expected app not found at:", result)
        print("Check the PyInstaller output above for errors.")
    print("=" * 70)


if __name__ == "__main__":
    main()
