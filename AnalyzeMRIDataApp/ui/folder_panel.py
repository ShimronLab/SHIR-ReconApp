"""
FolderPanel — left-side panel for scan-folder selection and file picking.

Layout
------
  [Open Folder…] [Add Folder…]
  ─────────────────────────────
  QTreeWidget  (checkable; folder → scan-type group → file [→ per-slice])
    └ SE_1mm
        └ [✓] Band            band_se_0.dat … band_se_15.dat
        └ [✓] Fully-Sampled   fully_sampled.dat
        └ [✓] Equispaced      equispaced_R4_acs24.dat
        └ [✓] PoissonDisc     PoissonDisc.dat
    └ SE_3mm  …
  ─────────────────────────────
  [Select All] [Clear All] [Remove Folder]          [Run Selected ▶]

Accepts a scan folder with a dat_files/ sub-folder, a dat_files/ folder
directly, or any folder containing loose .dat files. Right-click a file for
"Reconstruct as", per-slice selection, manual reference, or trajectory JSON.

Emits
-----
  run_requested(list[dict])      — checked jobs: {dat_path, scan_type,
      json_path|None, band_idx|None, folder_key, [slice_idx, n_slices,
      reference_dat_path, reference_override]}
  simulate_requested(list[dict]) — fully-sampled jobs re-typed RETRO_UNDERSAMPLE
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import QSettings, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from processing.constants import (
    SCAN_EQUISPACED,
    SCAN_FULLY_SAMPLED,
    SCAN_POISSON,
    SCAN_RETRO_UNDERSAMPLE,
    SCAN_UNKNOWN,
)
from processing.utils import (
    ScanType,
    auto_detect_scan_type,
    extract_band_index,
    find_generic_traj_json,
    probe_num_slices,
)

# Scan types whose recon supports slice selection (accept slice_selector_cb).
# Band/Poisson are single-slice NUFFT recons — no per-slice UI.
_SLICE_CAPABLE = {SCAN_FULLY_SAMPLED, SCAN_EQUISPACED, SCAN_UNKNOWN, SCAN_RETRO_UNDERSAMPLE}
# Scan types that compare against a fully-sampled reference.
_REFERENCE_CONSUMERS = {SCAN_EQUISPACED, SCAN_POISSON}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_LABELS = {
    ScanType.BAND:          "Band",
    ScanType.FULLY_SAMPLED: "Fully-Sampled",
    ScanType.EQUISPACED:    "Equispaced",
    ScanType.POISSON:       "PoissonDisc",
    ScanType.UNKNOWN:       "Unknown (generic RSS)",
}

_TYPE_COLORS = {
    ScanType.BAND:          "#d4edda",
    ScanType.FULLY_SAMPLED: "#cce5ff",
    ScanType.EQUISPACED:    "#fff3cd",
    ScanType.POISSON:       "#f8d7da",
    ScanType.UNKNOWN:       "#e2e3e5",
}

_JSON_NAMES = {
    ScanType.BAND:    "Bands.json",
    ScanType.POISSON: "PoissonDisc.json",
}

# Recon types offered by the "Reconstruct as" context-menu (mainly for Unknown
# scans, but available on any file). Value = the scan_type job key the dispatch
# in main_window routes on. Band is omitted: it needs a band index + Bands.json
# that a lone/unknown file does not have.
_RECON_AS_OPTIONS = [
    ("Generic fallback (auto)", SCAN_UNKNOWN),
    ("Fully-Sampled (IFFT + RSS / SENSE)", SCAN_FULLY_SAMPLED),
    ("Equispaced (GRAPPA / SENSE / …)", SCAN_EQUISPACED),
    ("Poisson-Disc CS (needs trajectory JSON)", SCAN_POISSON),
]
_RECON_AS_LABELS = {key: label for label, key in _RECON_AS_OPTIONS}

# Naming patterns tried in order for EQUISPACED JSONs, {R} replaced with factor
_EQUI_JSON_PATTERNS = [
    "Undersampled{R}equispaced.json",
    "Undersampled{R}equispaces.json",
    "undersampled{R}equispaced.json",
    "undersampled{R}equispaces.json",
]


def _extract_r_from_filename(filename: str) -> Optional[int]:
    """Parse the R factor from filenames like equispaced_R2_acs24.dat."""
    m = re.search(r'_[Rr](\d+)', Path(filename).stem)
    return int(m.group(1)) if m else None


def _json_search_dirs(dat_folder: Path) -> list:
    """Where trajectory JSONs may live: the scan root (parent of dat_files/)
    for the classic layout, or the folder itself for a loose-.dat layout."""
    return [dat_folder.parent, dat_folder]


def _find_equispaced_json(dat_folder: Path, dat_filename: str) -> Optional[str]:
    """Find the equispaced trajectory JSON matching the R factor in the filename."""
    r = _extract_r_from_filename(dat_filename)
    for folder in _json_search_dirs(dat_folder):
        if r is not None:
            for pat in _EQUI_JSON_PATTERNS:
                candidate = folder / pat.format(R=r)
                if candidate.exists():
                    return str(candidate)
    # Fall back to any undersampled JSON in the search folders
    for folder in _json_search_dirs(dat_folder):
        for candidate in sorted(folder.glob("[Uu]ndersampled*.json")):
            return str(candidate)
    return None


def _find_json(dat_folder: Path, scan_type: ScanType) -> Optional[str]:
    """Look for the trajectory/mask JSON near the scan folder."""
    json_name = _JSON_NAMES.get(scan_type)
    if not json_name:
        return None
    for folder in _json_search_dirs(dat_folder):
        candidate = folder / json_name
        if candidate.exists():
            return str(candidate)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Undersampling-simulator target confirmation
# ─────────────────────────────────────────────────────────────────────────────

class _SimulateSelectDialog(QDialog):
    """
    Confirm which fully-sampled scans the undersampling simulator will process.

    Shows one pre-ticked checkbox per candidate so the user can review the list
    and untick any scan before running — the checked subset is returned by
    ``selected_jobs()``.
    """

    def __init__(self, fs_jobs: List[Dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Undersampling Simulator — select scans")
        self.setMinimumWidth(420)
        self._jobs = fs_jobs

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Retrospectively undersample these fully-sampled scan(s) with the "
            "current simulator settings.\nUntick any you don't want, then Proceed:"
        ))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        self._checkboxes: List[QCheckBox] = []
        for job in fs_jobs:
            name = Path(job.get("dat_path", "")).name
            folder = job.get("folder_key", "")
            # Per-slice jobs (multi-slice scans expanded via "Select slices…")
            # share one filename — distinguish them by their slice number.
            if job.get("slice_idx") is not None:
                name += f" — slice {int(job['slice_idx']) + 1}/{int(job.get('n_slices', 0))}"
            cb = QCheckBox(f"{name}    ({folder})" if folder else name)
            cb.setChecked(True)
            cb.toggled.connect(self._update_ok_state)
            self._checkboxes.append(cb)
            inner_layout.addWidget(cb)
        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        btn_row = QHBoxLayout()
        btn_all  = QPushButton("Select All")
        btn_none = QPushButton("Clear All")
        btn_all.clicked.connect(lambda: self._set_all(True))
        btn_none.clicked.connect(lambda: self._set_all(False))
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._buttons.button(QDialogButtonBox.Ok).setText("Proceed")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)
        self._update_ok_state()

    def _set_all(self, state: bool) -> None:
        for cb in self._checkboxes:
            cb.setChecked(state)

    def _update_ok_state(self) -> None:
        any_checked = any(cb.isChecked() for cb in self._checkboxes)
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(any_checked)

    def selected_jobs(self) -> List[Dict]:
        return [job for job, cb in zip(self._jobs, self._checkboxes) if cb.isChecked()]


# ─────────────────────────────────────────────────────────────────────────────
# Main FolderPanel widget
# ─────────────────────────────────────────────────────────────────────────────

class FolderPanel(QWidget):
    """Left panel for folder/file selection."""

    # Emitted when user clicks "Run Selected ▶"
    # payload: list of job dicts
    run_requested = pyqtSignal(list)

    # Emitted when user clicks "Simulate ▶" — only FULLY_SAMPLED jobs, scan_type overridden
    simulate_requested = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._nslices_cache: Dict[str, int] = {}   # dat_path → slice count

        # ── Top buttons ──────────────────────────────────────────────────────
        btn_open = QPushButton("Open Folder…")
        btn_open.setToolTip("Select a dat_files folder (or its parent scan folder)")
        btn_open.clicked.connect(self._on_open_folder)

        btn_add  = QPushButton("Add Folder…")
        btn_add.setToolTip("Add another scan folder to the list")
        btn_add.clicked.connect(self._on_add_folder)

        btn_row1 = QHBoxLayout()
        btn_row1.addWidget(btn_open)
        btn_row1.addWidget(btn_add)

        # ── Tree ─────────────────────────────────────────────────────────────
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Scan", "File", "JSON"])
        self._tree.setColumnWidth(0, 160)
        self._tree.setColumnWidth(1, 220)
        self._tree.setColumnWidth(2, 120)
        self._tree.setAlternatingRowColors(True)
        self._tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Cascade check state from a category/folder node down to its child scans
        self._tree.itemChanged.connect(self._on_item_changed)
        # Right-click on a file item: manually assign a trajectory JSON
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_tree_context_menu)

        # ── Bottom row ────────────────────────────────────────────────────────
        btn_selall = QPushButton("Select All")
        btn_selall.clicked.connect(self._on_select_all)
        btn_clrall = QPushButton("Clear All")
        btn_clrall.clicked.connect(self._on_clear_all)
        btn_remove = QPushButton("Remove Folder")
        btn_remove.clicked.connect(self._on_remove_folder)
        self._btn_run = QPushButton("Run Selected ▶")
        self._btn_run.setStyleSheet("background:#2d9e4f;color:white;font-weight:bold;")
        self._btn_run.clicked.connect(self._on_run)

        btn_row2 = QHBoxLayout()
        btn_row2.addWidget(btn_selall)
        btn_row2.addWidget(btn_clrall)
        btn_row2.addWidget(btn_remove)
        btn_row2.addStretch()
        btn_row2.addWidget(self._btn_run)

        # ── Main layout ───────────────────────────────────────────────────────
        lbl = QLabel("<b>Scan Folders</b>")
        layout = QVBoxLayout(self)
        layout.addWidget(lbl)
        layout.addLayout(btn_row1)
        layout.addWidget(self._tree, stretch=3)
        layout.addLayout(btn_row2)
        self.setMinimumWidth(440)

    # ─── Folder loading ───────────────────────────────────────────────────────

    def set_running(self, running: bool) -> None:
        """Disable run/folder controls while a reconstruction queue is active."""
        self._btn_run.setEnabled(not running)

    def _on_open_folder(self) -> None:
        if self._tree.topLevelItemCount() > 0:
            reply = QMessageBox.question(
                self, "Replace folder list?",
                "Opening a folder replaces the current folder list and its "
                "selections.\nUse 'Add Folder…' to keep them.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return
            self._tree.clear()
        self._add_folder_dialog()

    def _on_add_folder(self) -> None:
        self._add_folder_dialog()

    def _add_folder_dialog(self) -> None:
        settings = QSettings("Technion", "SHIR")
        start_dir = settings.value("folders/last_dir", "", str) or ""
        folder = QFileDialog.getExistingDirectory(
            self, "Select scan folder (dat_files/ layout or loose .dat files)",
            start_dir,
        )
        if folder:
            settings.setValue("folders/last_dir", folder)
            self._load_folder(Path(folder))

    def _loaded_dat_folders(self) -> set:
        """dat-folder paths already present as top-level tree entries."""
        out = set()
        for i in range(self._tree.topLevelItemCount()):
            out.add(self._tree.topLevelItem(i).text(1))
        return out

    def _load_folder(self, folder: Path) -> None:
        """
        Accept, in order of preference:
          * a ``dat_files/`` folder directly,
          * a parent scan folder containing ``dat_files/``,
          * any folder with loose ``*.dat`` files directly inside it.
        """
        if folder.name == "dat_files":
            dat_folder = folder
        elif (folder / "dat_files").is_dir():
            dat_folder = folder / "dat_files"
        else:
            dat_folder = folder                     # loose-.dat layout

        dat_files = sorted(dat_folder.glob("*.dat"))
        if not dat_files:
            QMessageBox.warning(
                self, "No .dat files",
                f"No .dat files found in:\n{folder}\n\n"
                "Select a scan folder (with a dat_files/ sub-folder), a "
                "dat_files/ folder itself, or a folder containing .dat files.",
            )
            return

        if str(dat_folder) in self._loaded_dat_folders():
            QMessageBox.information(
                self, "Already loaded",
                f"This folder is already in the list:\n{dat_folder}",
            )
            return

        # Scan-folder display name: the parent for a dat_files/ layout, the
        # folder itself for a loose layout.
        folder_key = (dat_folder.parent.name if dat_folder.name == "dat_files"
                      else dat_folder.name)
        self._add_tree_folder(folder_key, dat_folder, dat_files)

    def _add_tree_folder(
        self, folder_key: str, dat_folder: Path, dat_files: List[Path]
    ) -> None:
        """Build tree items for one scan folder."""
        # Top-level item
        root_item = QTreeWidgetItem(self._tree, [folder_key, str(dat_folder), ""])
        root_item.setFont(0, QFont("", -1, QFont.Bold))
        root_item.setCheckState(0, Qt.Checked)
        root_item.setExpanded(True)

        # Group files by detected type
        groups: Dict[ScanType, List[Path]] = {}
        for f in dat_files:
            st = auto_detect_scan_type(f.name)
            groups.setdefault(st, []).append(f)

        for stype, files in sorted(groups.items(), key=lambda x: x[0].name):
            type_label = _TYPE_LABELS.get(stype, stype.name.replace("_", " ").title())
            color = QColor(_TYPE_COLORS.get(stype, "#e2e3e5"))

            if stype in (ScanType.EQUISPACED, ScanType.UNKNOWN):
                # JSON is resolved per-file (R factor / generic-trajectory naming)
                json_path = ""
                type_json_display = "per-file"
            else:
                json_path = _find_json(dat_folder, stype) or ""
                type_json_display = Path(json_path).name if json_path else "⚠ missing"

            type_item = QTreeWidgetItem(root_item, [type_label, "", type_json_display])
            type_item.setCheckState(0, Qt.Checked)
            type_item.setBackground(0, color)
            type_item.setBackground(2, color)
            type_item.setExpanded(stype == ScanType.BAND)

            for f in sorted(files):
                bidx = extract_band_index(f.name)
                if stype == ScanType.EQUISPACED:
                    file_json = _find_equispaced_json(dat_folder, f.name) or ""
                elif stype == ScanType.UNKNOWN:
                    # Missing is legitimate here — the generic recon falls back
                    # to Cartesian IFFT + RSS when no trajectory is found.
                    file_json = find_generic_traj_json(str(f)) or ""
                else:
                    file_json = json_path
                file_item = QTreeWidgetItem(
                    type_item,
                    ["", f.name, Path(file_json).name if file_json else ""],
                )
                file_item.setCheckState(0, Qt.Checked)
                file_item.setData(0, Qt.UserRole, {
                    "dat_path":       str(f),
                    "scan_type":      stype.name,
                    "_detected_type": stype.name,   # for the "Reconstruct as" reset
                    "json_path":      file_json or None,
                    "band_idx":       bidx,
                    "folder_key":     folder_key,
                })

        self._tree.resizeColumnToContents(1)

    # ─── Context menu (recon-type, slices, trajectory JSON, reference) ─────────

    def _on_tree_context_menu(self, pos) -> None:
        """Right-click on a file item → recon-type / slices / JSON / reference."""
        item = self._tree.itemAt(pos)
        if item is None:
            return
        data = item.data(0, Qt.UserRole)
        if not data:                      # folder/type nodes have no job data
            return
        # Slice children themselves carry a slice_idx — no per-slice menu.
        if data.get("slice_idx") is not None:
            return

        menu = QMenu(self)
        stype = data.get("scan_type", SCAN_UNKNOWN)

        # ── Reconstruct as ▸ (override the recon type; mainly for Unknown) ────
        submenu = menu.addMenu("Reconstruct as")
        detected = data.get("_detected_type", stype)
        recon_acts = {}
        for label, key in _RECON_AS_OPTIONS:
            text = label + ("   (auto-detected)" if key == detected else "")
            act = submenu.addAction(text)
            act.setCheckable(True)
            act.setChecked(key == stype)
            recon_acts[act] = key

        # ── Select slices… (multi-slice scans only) ──────────────────────────
        act_slices = None
        if stype in _SLICE_CAPABLE:
            menu.addSeparator()
            act_slices = menu.addAction(
                "Collapse slices (single)" if data.get("_slice_group")
                else "Select slices…"
            )

        # ── Set reference ▸ (undersampled scans) ─────────────────────────────
        ref_acts, act_ref_browse, act_ref_auto = {}, None, None
        if stype in _REFERENCE_CONSUMERS:
            refmenu = menu.addMenu("Set reference")
            for label, path in self._fully_sampled_candidates():
                a = refmenu.addAction(label)
                a.setCheckable(True)
                a.setChecked(data.get("reference_dat_path") == path)
                ref_acts[a] = path
            refmenu.addSeparator()
            act_ref_browse = refmenu.addAction("Browse for .dat…")
            act_ref_auto = refmenu.addAction("Auto (default)")
            act_ref_auto.setCheckable(True)
            act_ref_auto.setChecked(not data.get("reference_override"))

        # ── Trajectory JSON ──────────────────────────────────────────────────
        menu.addSeparator()
        act_set   = menu.addAction("Choose trajectory JSON…")
        act_clear = menu.addAction("Clear trajectory JSON")
        act_clear.setEnabled(bool(data.get("json_path")))

        chosen = menu.exec_(self._tree.viewport().mapToGlobal(pos))
        if chosen is None:
            return

        if chosen in recon_acts:
            self._set_recon_type(item, data, recon_acts[chosen])
        elif chosen is act_slices:
            self._toggle_slice_selection(item, data)
        elif chosen in ref_acts:
            self._set_reference(item, data, ref_acts[chosen])
        elif chosen is act_ref_browse:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select fully-sampled reference",
                str(Path(data["dat_path"]).parent), "TWIX (*.dat)"
            )
            if path:
                self._set_reference(item, data, path)
        elif chosen is act_ref_auto:
            self._set_reference(item, data, None)
        elif chosen is act_set:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select trajectory JSON",
                str(Path(data["dat_path"]).parent), "JSON files (*.json)"
            )
            if path:
                data["json_path"] = path
                item.setData(0, Qt.UserRole, data)
                item.setText(2, Path(path).name)
        elif chosen is act_clear:
            data["json_path"] = None
            item.setData(0, Qt.UserRole, data)
            item.setText(2, "")

    def _set_recon_type(self, item: QTreeWidgetItem, data: Dict, new_type: str) -> None:
        """Override a file's recon type and reflect it in the file-name column."""
        data["scan_type"] = new_type
        item.setData(0, Qt.UserRole, data)
        name = Path(data["dat_path"]).name
        if new_type != data.get("_detected_type", new_type):
            item.setText(1, f"{name}   [as {_RECON_AS_LABELS.get(new_type, new_type)}]")
        else:
            item.setText(1, name)

    # ─── Slice selection ──────────────────────────────────────────────────────

    def _toggle_slice_selection(self, item: QTreeWidgetItem, data: Dict) -> None:
        """Expand a multi-slice scan into per-slice checkboxes (or collapse back)."""
        name = Path(data["dat_path"]).name
        recon_suffix = ""
        if data.get("scan_type") != data.get("_detected_type", data.get("scan_type")):
            recon_suffix = f"   [as {_RECON_AS_LABELS.get(data['scan_type'], data['scan_type'])}]"

        if data.get("_slice_group"):                      # collapse → single job
            item.takeChildren()
            data.pop("_slice_group", None)
            item.setData(0, Qt.UserRole, data)
            item.setText(1, name + recon_suffix)
            return

        dat = data["dat_path"]
        n = self._nslices_cache.get(dat)
        if n is None:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                n = probe_num_slices(dat)
            finally:
                QApplication.restoreOverrideCursor()
            self._nslices_cache[dat] = n
        if n <= 1:
            QMessageBox.information(
                self, "Single-slice scan",
                f"{name} has a single slice — nothing to select."
            )
            return

        # One checkable child per slice; the parent (this file item) becomes the
        # "All slices" master via the existing checkbox cascade.
        self._tree.blockSignals(True)
        try:
            for k in range(n):
                child = QTreeWidgetItem(item, ["", f"slice {k}", ""])
                cdata = dict(data)
                cdata.pop("_slice_group", None)
                cdata["slice_idx"] = k
                cdata["n_slices"] = n
                child.setData(0, Qt.UserRole, cdata)
                child.setCheckState(0, Qt.Checked)
            data["_slice_group"] = True
            item.setData(0, Qt.UserRole, data)
            item.setText(1, f"{name}{recon_suffix}   (all slices — untick to choose)")
            item.setExpanded(True)
            item.setCheckState(0, Qt.Checked)
        finally:
            self._tree.blockSignals(False)

    # ─── Reference selection ──────────────────────────────────────────────────

    def _fully_sampled_candidates(self):
        """(label, dat_path) for every fully-sampled file item in the tree."""
        out, seen = [], set()

        def rec(node):
            d = node.data(0, Qt.UserRole)
            if (d and d.get("scan_type") == SCAN_FULLY_SAMPLED
                    and d.get("slice_idx") is None):
                p = d["dat_path"]
                if p not in seen:
                    seen.add(p)
                    out.append((f"{Path(p).name}  ({d.get('folder_key', '')})", p))
            for i in range(node.childCount()):
                rec(node.child(i))

        for i in range(self._tree.topLevelItemCount()):
            rec(self._tree.topLevelItem(i))
        return out

    def _set_reference(self, item: QTreeWidgetItem, data: Dict,
                       path: Optional[str]) -> None:
        """Set/clear the manual fully-sampled reference on a scan (+ slice kids)."""
        def apply(d):
            if path is None:                      # Auto → drop the override
                d.pop("reference_dat_path", None)
                d.pop("reference_override", None)
            else:
                d["reference_dat_path"] = path
                d["reference_override"] = True
            return d

        item.setData(0, Qt.UserRole, apply(data))
        item.setToolTip(1, "" if path is None else f"reference: {Path(path).name}")
        for i in range(item.childCount()):
            ch = item.child(i)
            cd = ch.data(0, Qt.UserRole)
            if cd:
                ch.setData(0, Qt.UserRole, apply(cd))

    # ─── Selection helpers ────────────────────────────────────────────────────

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """Cascade a category/folder check down to its scans and refresh ancestors."""
        if column != 0:
            return
        self._tree.blockSignals(True)
        try:
            state = item.checkState(0)
            if state != Qt.PartiallyChecked:
                self._set_children_state(item, state)
            self._update_parent_state(item.parent())
        finally:
            self._tree.blockSignals(False)

    def _set_children_state(self, item: QTreeWidgetItem, state: Qt.CheckState) -> None:
        for i in range(item.childCount()):
            child = item.child(i)
            child.setCheckState(0, state)
            self._set_children_state(child, state)

    def _update_parent_state(self, parent: Optional[QTreeWidgetItem]) -> None:
        while parent is not None:
            states = [parent.child(i).checkState(0) for i in range(parent.childCount())]
            if states and all(s == Qt.Checked for s in states):
                parent.setCheckState(0, Qt.Checked)
            elif states and all(s == Qt.Unchecked for s in states):
                parent.setCheckState(0, Qt.Unchecked)
            else:
                parent.setCheckState(0, Qt.PartiallyChecked)
            parent = parent.parent()

    def _set_all_checks(self, state: Qt.CheckState) -> None:
        self._tree.blockSignals(True)
        try:
            def recurse(item):
                item.setCheckState(0, state)
                for i in range(item.childCount()):
                    recurse(item.child(i))
            for i in range(self._tree.topLevelItemCount()):
                recurse(self._tree.topLevelItem(i))
        finally:
            self._tree.blockSignals(False)

    def _on_select_all(self) -> None:
        self._set_all_checks(Qt.Checked)

    def _on_clear_all(self) -> None:
        self._set_all_checks(Qt.Unchecked)

    def _on_remove_folder(self) -> None:
        """Remove the selected top-level folder from the tree."""
        selected = self._tree.selectedItems()
        roots_to_remove = []
        for item in selected:
            # Walk up to find top-level
            top = item
            while top.parent():
                top = top.parent()
            if top not in roots_to_remove:
                roots_to_remove.append(top)
        for item in roots_to_remove:
            idx = self._tree.indexOfTopLevelItem(item)
            if idx >= 0:
                self._tree.takeTopLevelItem(idx)

    # ─── Build job list & emit ────────────────────────────────────────────────

    def _collect_jobs(self) -> List[Dict]:
        """Walk the tree and collect all checked leaf file items."""
        jobs = []

        def recurse(item):
            data = item.data(0, Qt.UserRole)
            # A slice-group parent is only a container; its per-slice children
            # carry the jobs, so don't emit the parent's own job.
            if data and not data.get("_slice_group") and item.checkState(0) == Qt.Checked:
                # Check parent is also checked
                parent = item.parent()
                while parent:
                    if parent.checkState(0) == Qt.Unchecked:
                        return
                    parent = parent.parent()
                jobs.append(dict(data))  # copy
            for i in range(item.childCount()):
                recurse(item.child(i))

        for i in range(self._tree.topLevelItemCount()):
            recurse(self._tree.topLevelItem(i))
        return jobs

    def _on_run(self) -> None:
        jobs = self._collect_jobs()
        if not jobs:
            QMessageBox.warning(self, "Nothing selected", "Please check at least one scan to run.")
            return
        self.run_requested.emit(jobs)

    def _on_simulate(self) -> None:
        """Confirm the checked FULLY_SAMPLED scans, then emit them as RETRO_UNDERSAMPLE."""
        all_jobs = self._collect_jobs()
        fs_jobs  = [j for j in all_jobs if j.get("scan_type") == SCAN_FULLY_SAMPLED]
        if not fs_jobs:
            QMessageBox.warning(
                self, "No fully-sampled data",
                "Please check at least one Fully-Sampled scan to simulate undersampling."
            )
            return

        # Confirm which scans to undersample (all pre-selected; user can untick).
        dlg = _SimulateSelectDialog(fs_jobs, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        chosen = dlg.selected_jobs()
        if not chosen:
            return

        for j in chosen:
            j["scan_type"] = SCAN_RETRO_UNDERSAMPLE
        self.simulate_requested.emit(chosen)
