"""
Resource-path resolution that works both in dev mode and in a frozen
PyInstaller build.

In dev mode resources live next to this file (AnalyzeMRIDataApp/Graphics/…).
In a frozen --onedir build, files added with --add-data land under
``sys._MEIPASS`` (the _internal/ folder for PyInstaller ≥ 6) with the same
relative layout, so a single helper covers both.
"""

import sys
from pathlib import Path

_BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def resource_path(*parts: str) -> str:
    """Absolute path of a bundled resource, e.g. resource_path("Graphics", "logo.png")."""
    return str(_BASE.joinpath(*parts))
