"""
FolderPanel — left-side panel for scan-folder selection and file/band picking.

Layout
------
  [Open Folder…] [Add Folder…]
  ─────────────────────────────
  QTreeWidget
    └ SE_1mm  (dat_files/)
        └ [✓] Band         band_se_0  …  band_se_15
            └ Band indices: [0] [1] … [15]  (checkboxes)
        └ [✓] Fully-Sampled  (auto-detected file)
        └ [✓] GRAPPA
        └ [✓] PoissonDisc
    └ SE_3mm  …
  ─────────────────────────────
  [Select All]  [Clear All]  [Run Selected ▶]

Emits
-----
  run_requested(list[dict])   — list of job dicts with keys:
      scan_folder, scan_type, dat_path, json_path (or None), band_idx (or None)
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
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


def _find_equispaced_json(dat_folder: Path, dat_filename: str) -> Optional[str]:
    """Find the equispaced trajectory JSON matching the R factor in the filename."""
    parent = dat_folder.parent
    r = _extract_r_from_filename(dat_filename)
    if r is not None:
        for pat in _EQUI_JSON_PATTERNS:
            candidate = parent / pat.format(R=r)
            if candidate.exists():
                return str(candidate)
    # Fall back to any undersampled JSON in the parent folder
    for candidate in sorted(parent.glob("[Uu]ndersampled*.json")):
        return str(candidate)
    return None


def _find_json(dat_folder: Path, scan_type: ScanType) -> Optional[str]:
    """Look for the trajectory/mask JSON in the scan parent folder."""
    json_name = _JSON_NAMES.get(scan_type)
    if not json_name:
        return None
    parent = dat_folder.parent
    candidate = parent / json_name
    return str(candidate) if candidate.exists() else None


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

    def _on_open_folder(self) -> None:
        self._tree.clear()
        self._add_folder_dialog()

    def _on_add_folder(self) -> None:
        self._add_folder_dialog()

    def _add_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select scan folder or dat_files folder"
        )
        if folder:
            self._load_folder(Path(folder))

    def _load_folder(self, folder: Path) -> None:
        """
        Accept either a dat_files/ folder directly or a parent scan folder
        (which should then have a dat_files/ sub-folder).
        """
        dat_folder = folder if folder.name == "dat_files" else folder / "dat_files"
        if not dat_folder.exists():
            QMessageBox.warning(
                self, "Folder not found",
                f"Could not find dat_files/ in:\n{folder}\n\n"
                "Please select either a dat_files/ folder directly or its parent."
            )
            return

        dat_files = sorted(dat_folder.glob("*.dat"))
        if not dat_files:
            QMessageBox.warning(self, "No .dat files", f"No .dat files found in:\n{dat_folder}")
            return

        folder_key = dat_folder.parent.name   # e.g. SE_1mm
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
        """Collect only FULLY_SAMPLED jobs and emit them as RETRO_UNDERSAMPLE."""
        all_jobs = self._collect_jobs()
        fs_jobs  = [j for j in all_jobs if j.get("scan_type") == SCAN_FULLY_SAMPLED]
        if not fs_jobs:
            QMessageBox.warning(
                self, "No fully-sampled data",
                "Please check at least one Fully-Sampled scan to simulate undersampling."
            )
            return
        for j in fs_jobs:
            j["scan_type"] = SCAN_RETRO_UNDERSAMPLE
        self.simulate_requested.emit(fs_jobs)
