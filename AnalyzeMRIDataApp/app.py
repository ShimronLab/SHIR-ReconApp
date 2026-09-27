#!/usr/bin/env python3
"""
SHIR — System for Hybrid Image Reconstruction — entry point.

Run this script from a Linux/WSL terminal where BART is installed:

    cd /mnt/c/<path>/AnalyzeMRIDataApp
    python app.py

Requirements
------------
  pip install PyQt5 matplotlib numpy scipy sigpy pygrappa
  apt install bart          (or build from source)
"""

import os
import sys
import time
from pathlib import Path


# ── 0. Windowed-build stdout/stderr guard ─────────────────────────────────
# When frozen with PyInstaller --windowed there is no console, so sys.stdout and
# sys.stderr are None. Any library that writes to them (twixtools progress,
# tqdm, print) then crashes with "'NoneType' object has no attribute 'write'".
# Redirect them to a log file so those writes are harmless (and debuggable).
# Must run BEFORE importing matplotlib / PyQt5 / twixtools below.
if sys.stdout is None or sys.stderr is None:
    import tempfile
    try:
        _logf = open(
            os.path.join(tempfile.gettempdir(), "SHIR.log"),
            "a", buffering=1, encoding="utf-8", errors="replace",
        )
    except Exception:
        _logf = open(os.devnull, "w")
    if sys.stdout is None:
        sys.stdout = _logf
    if sys.stderr is None:
        sys.stderr = _logf


# ── 1. Make 'processing' and 'ui' importable ──────────────────────────────
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# ── 2. Add read_siemens_scans package to path ──────────────────────────────
TWIX_DIR = APP_DIR.parent / "read_siemens_scans"
if TWIX_DIR.exists() and str(TWIX_DIR) not in sys.path:
    sys.path.insert(0, str(TWIX_DIR))

# ── 3. Qt display setup for WSL/Linux (no-op on native Linux) ─────────────
#   If running under WSLg, DISPLAY is set automatically.
#   If using an older WSL1/WSL2 with a manual X server, you may need to run:
#       export DISPLAY=:0
if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    os.environ.setdefault("DISPLAY", ":0")

# ── 4. Launch app ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Qt5Agg")

from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap, QBitmap, QFont, QPainter, QColor, QIcon
from PyQt5.QtCore import Qt, QTimer

from resources import resource_path
from version import __version__

# Minimum time the splash stays visible (seconds); the actual lifetime is tied
# to the real load (importing/constructing MainWindow), whichever is longer.
MIN_SPLASH_S = 1.0


def _fallback_splash_pixmap() -> QPixmap:
    """Code-drawn splash pixmap, used when Graphics/Intro.png is unavailable."""
    pix = QPixmap(540, 200)
    pix.fill(QColor("#1e2a38"))
    painter = QPainter(pix)
    painter.setPen(QColor("#e8f0fe"))
    font = QFont("Arial", 22, QFont.Bold)
    painter.setFont(font)
    painter.drawText(pix.rect(), Qt.AlignCenter, "SHIR")
    font2 = QFont("Arial", 11)
    painter.setFont(font2)
    painter.drawText(
        0, 140, 540, 40, Qt.AlignCenter,
        "System for Hybrid Image Reconstruction"
    )
    painter.end()
    return pix


def _rounded_mask(w: int, h: int, radius: int) -> "QBitmap":
    """1-bit mask of a rounded rectangle, for clipping the splash window corners."""
    bmp = QBitmap(w, h)
    bmp.fill(Qt.color0)                       # transparent (masked out)
    p = QPainter(bmp)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(Qt.color1)                     # opaque (kept)
    p.setPen(Qt.color1)
    p.drawRoundedRect(0, 0, w, h, radius, radius)
    p.end()
    return bmp


def _make_splash() -> QSplashScreen:
    """
    Splash from Graphics/Intro.png — enlarged and clipped to the image's rounded
    corners (the PNG is RGB with light corners, so an unclipped window would show
    a rectangle with white corners). Code-drawn fallback if the image is missing.
    """
    pix = QPixmap()
    try:
        pix = QPixmap(resource_path("Graphics", "Intro.png"))
        if not pix.isNull():
            # Source is 1254×1254 — scale up to a prominent on-screen size.
            side = 620
            screen = QApplication.primaryScreen()
            if screen is not None:
                side = min(640, int(screen.availableGeometry().height() * 0.55))
            pix = pix.scaled(side, side, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    except Exception:
        pix = QPixmap()

    rounded = not pix.isNull()
    if pix.isNull():
        pix = _fallback_splash_pixmap()

    splash = QSplashScreen(pix, Qt.WindowStaysOnTopHint)
    if rounded:
        # Clip the light corners so only the curved dark card shows.
        try:
            radius = max(1, int(min(pix.width(), pix.height()) * 0.08))
            splash.setMask(_rounded_mask(pix.width(), pix.height(), radius))
        except Exception:
            pass
    splash.showMessage(
        f"Loading SHIR v{__version__}…",
        Qt.AlignBottom | Qt.AlignHCenter, Qt.white,
    )
    return splash


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("SHIR")
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")

    # App icon: title bar + taskbar (all windows). Missing file → no-op.
    try:
        icon = QIcon(resource_path("Graphics", "logo.png"))
        if not icon.isNull():
            app.setWindowIcon(icon)
    except Exception:
        pass
    if os.name == "nt":
        # Windows groups taskbar entries by AppUserModelID; without an explicit
        # one, unfrozen runs show the Python icon instead of the app logo.
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "Technion.SHIR"
            )
        except Exception:
            pass

    # Global stylesheet
    app.setStyleSheet("""
        QMainWindow, QWidget { font-size: 12px; }
        QGroupBox { font-weight: bold; margin-top: 8px; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; }
        QToolBar { spacing: 4px; }
        QTabBar::tab { padding: 4px 10px; }
        QTabBar::tab:selected { background: #2d9e4f; color: white; font-weight: bold; }
        QProgressBar { border: 1px solid #aaa; border-radius: 4px; height: 16px; }
        QProgressBar::chunk { background: #2d9e4f; }
    """)

    splash = _make_splash()
    splash.show()
    app.processEvents()

    # The heavy part: importing ui.main_window pulls in matplotlib backends,
    # numpy/scipy etc. Imported here (not at module scope) so the splash is
    # already visible while it happens.
    t0 = time.monotonic()
    from ui.main_window import MainWindow
    window = MainWindow()

    remaining_ms = max(0, int((MIN_SPLASH_S - (time.monotonic() - t0)) * 1000))

    def _reveal() -> None:
        window.show()
        splash.finish(window)

    QTimer.singleShot(remaining_ms, _reveal)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
