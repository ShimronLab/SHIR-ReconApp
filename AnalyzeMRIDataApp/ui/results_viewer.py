"""
ResultsViewer — QTabWidget with one embedded matplotlib canvas per reconstruction.
Each tab shows relevant plots and has W/L (window/level) sliders + Export PNG button.
"""

from typing import Any, Dict

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


# Default parameters for the anatomy mask used by "Masked Metrics".
MASK_THRESHOLD_DEFAULT  = 0.1
MASK_DILATION_DEFAULT   = 10


from processing.constants import method_display_label as _method_display_label
from processing.constants import (
    SCAN_BAND,
    SCAN_EQUISPACED,
    SCAN_FULLY_SAMPLED,
    SCAN_POISSON,
    SCAN_RETRO_UNDERSAMPLE,
    SCAN_UNKNOWN,
)


class _ImageSelectDialog(QDialog):
    """Checklist dialog: user picks which subplot images to include in the export."""

    def __init__(self, titles: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select images to export")
        self.setMinimumWidth(340)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select which images to include in the export:"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        self._checkboxes: list = []
        for title in titles:
            cb = QCheckBox(title)
            cb.setChecked(True)
            self._checkboxes.append(cb)
            inner_layout.addWidget(cb)
        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        btn_row = QHBoxLayout()
        btn_all  = QPushButton("Select All")
        btn_none = QPushButton("Deselect All")
        btn_all.clicked.connect(lambda: [cb.setChecked(True)  for cb in self._checkboxes])
        btn_none.clicked.connect(lambda: [cb.setChecked(False) for cb in self._checkboxes])
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_indices(self) -> list:
        return [i for i, cb in enumerate(self._checkboxes) if cb.isChecked()]


# Reference-orientation choices, labelled as the operation the user SEES on screen
# (relative to the reference as loaded). Keys are the array-level ops of
# processing.reference.ORIENTATIONS; since a tab draws arrays with a base rot90
# (plus its Rotate button), a screen op maps to a different array op — see
# _screen_to_array_orient.
_ORIENT_LABELS = {
    "none":          "As loaded",
    "rot90":         "Rotate 90° CCW",
    "rot180":        "Rotate 180°",
    "rot270":        "Rotate 90° CW",
    "fliplr":        "Flip left-right",
    "flipud":        "Flip up-down",
    "transpose":     "Mirror ↘ diagonal",
    "antitranspose": "Mirror ↙ diagonal",
}


def _screen_to_array_orient(screen_key: str, k: int) -> str:
    """Array op A such that drawing A(ref) rotated by k shows screen op S of the
    drawn reference: rot90(A(x), k) == S(rot90(x, k))."""
    from processing.reference import orient_image
    x = np.arange(15.0).reshape(3, 5)          # asymmetric probe
    want = orient_image(np.rot90(x, k), screen_key)
    for a in _ORIENT_LABELS:
        got = np.rot90(orient_image(x, a), k)
        if got.shape == want.shape and np.array_equal(got, want):
            return a
    raise AssertionError(f"no array op for screen op {screen_key} (k={k})")


def _array_to_screen_orient(array_key: str, k: int) -> str:
    """Inverse of :func:`_screen_to_array_orient`."""
    for sk in _ORIENT_LABELS:
        if _screen_to_array_orient(sk, k) == array_key:
            return sk
    raise AssertionError(f"no screen op for array op {array_key} (k={k})")


def _norm(img: np.ndarray) -> np.ndarray:
    arr = np.abs(np.squeeze(img)).astype(float)
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)


def _build_anatomy_mask(image: np.ndarray,
                        threshold: float = MASK_THRESHOLD_DEFAULT,
                        dilation_iterations: int = MASK_DILATION_DEFAULT) -> np.ndarray:
    """Binary mask of the anatomy: threshold the normalised magnitude, fill
    holes, keep the largest connected component, then dilate.

    Returns a 2-D boolean array the same shape as the squeezed input.
    """
    from scipy.ndimage import (
        binary_dilation,
        binary_fill_holes,
        generate_binary_structure,
        label,
    )

    img = np.squeeze(image)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D slice after squeeze, got shape={img.shape}")

    img_mag = np.abs(img) if np.iscomplexobj(img) else img
    img_mag = np.asarray(img_mag, dtype=np.float32)
    img_min = float(img_mag.min())
    img_max = float(img_mag.max())
    img_norm = (img_mag - img_min) / (img_max - img_min + 1e-8)

    mask = img_norm > threshold
    mask_filled = binary_fill_holes(mask)

    structure = generate_binary_structure(mask_filled.ndim, 1)
    labeled, num = label(mask_filled, structure=structure)
    if num > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        largest_label = int(sizes.argmax())
        largest = labeled == largest_label
    else:
        largest = mask_filled

    if dilation_iterations and dilation_iterations > 0:
        largest = binary_dilation(largest, iterations=int(dilation_iterations))
    return largest


def _masked_metrics(recon: np.ndarray, ref: np.ndarray,
                    threshold: float, dilation_iterations: int):
    """Compute SSIM / PSNR / NMSE evaluated ONLY inside the anatomy mask.

    The mask is derived from the reference (ground-truth anatomy). Returns
    ``(mask, nmse, psnr, ssim)`` where ``mask`` is aligned to ``recon`` (after
    any resize of ``ref`` to match). ``ssim`` may be ``None`` if it cannot be
    computed.
    """
    from skimage.metrics import structural_similarity as _ssim

    recon = np.abs(np.squeeze(recon)).astype(float)
    ref   = np.abs(np.squeeze(ref)).astype(float)
    if recon.shape != ref.shape:
        from scipy.ndimage import zoom
        zf = (recon.shape[0] / ref.shape[0], recon.shape[1] / ref.shape[1])
        ref = zoom(ref, zf, order=1)

    mask = _build_anatomy_mask(ref, threshold, dilation_iterations)

    # Rescale the recon to the reference (95th-pct) first, so a recon on an
    # arbitrary intensity scale (e.g. PICS output) is comparable — then normalise
    # both by the reference scale. NMSE is a ratio (scale-invariant); SSIM/PSNR
    # are not, so the rescale matters. For equispaced the factor is ≈1 (no change).
    r95   = float(np.percentile(recon.ravel(), 95)) + 1e-12
    scale = float(np.percentile(ref.ravel(), 95)) + 1e-12
    g  = recon * (scale / r95) / scale
    rr = ref / scale

    sel = mask
    if not sel.any():
        return mask, None, None, None

    diff    = (g - rr)[sel]
    ref_sel = rr[sel]
    nmse = float(np.sum(diff ** 2) / (np.sum(ref_sel ** 2) + 1e-12))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    psnr = float(20.0 * np.log10(1.0 / (rmse + 1e-12)))

    # Masked SSIM: compute the full SSIM map, then average it over the mask.
    try:
        _, ssim_map = _ssim(g, rr, data_range=1.0, full=True)
        ssim = float(ssim_map[sel].mean())
    except Exception:
        ssim = None

    return mask, nmse, psnr, ssim


# ─────────────────────────────────────────────────────────────────────────────
# Per-tab canvas widget
# ─────────────────────────────────────────────────────────────────────────────

class ReconTab(QWidget):
    """
    A single reconstruction tab: matplotlib canvas + W/L sliders + toolbar.
    """

    def __init__(self, result: Dict[str, Any], title: str, parent=None):
        super().__init__(parent)
        self._result = result
        self._title  = title
        self._fig    = Figure(figsize=(14, 9), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Navigation toolbar (zoom, pan, save)
        self._nav = NavToolbar(self._canvas, self)

        # W/L controls
        wl_label = QLabel("W/L  min%:")
        self._slider_min = QSlider(Qt.Horizontal)
        self._slider_min.setRange(0, 100)
        self._slider_min.setValue(0)
        self._slider_min.setFixedWidth(120)
        self._slider_min.setToolTip("Adjust display minimum (percentile)")

        wl_label2 = QLabel("max%:")
        self._slider_max = QSlider(Qt.Horizontal)
        self._slider_max.setRange(0, 100)
        self._slider_max.setValue(100)
        self._slider_max.setFixedWidth(120)
        self._slider_max.setToolTip("Adjust display maximum (percentile)")

        self._slider_min.valueChanged.connect(self._update_clim)
        self._slider_max.valueChanged.connect(self._update_clim)

        # Brightness control (applies only to main reconstructed images)
        bright_label = QLabel("Bright:")
        self._slider_brightness = QSlider(Qt.Horizontal)
        self._slider_brightness.setRange(10, 500)
        self._slider_brightness.setValue(100)
        self._slider_brightness.setFixedWidth(120)
        self._slider_brightness.setToolTip(
            "Adjust brightness of main reconstructed images (10%–500%)\n"
            "Does not affect k-space, sensitivity maps, or difference images."
        )
        self._brightness_label = QLabel("100%")
        self._brightness_label.setFixedWidth(38)
        self._slider_brightness.valueChanged.connect(self._update_clim)
        self._slider_brightness.valueChanged.connect(
            lambda v: self._brightness_label.setText(f"{v}%")
        )

        # Rotate button — 90° per click, per tab (also applied to the PDF report)
        self._rotation = 0   # number of 90° (CCW) turns, 0–3
        btn_rotate = QPushButton("⟳ Rotate 90°")
        btn_rotate.setToolTip(
            "Rotate every image in this tab 90° (click again to keep turning).\n"
            "Rotation is separate per tab and is also used in the PDF report."
        )
        btn_rotate.clicked.connect(self._on_rotate)

        # Reference orientation — orients ONLY the fully-sampled reference to the
        # recon and re-scores (metrics + difference map). Auto picks the best of
        # the 8 rotations/flips at recon time; a manual choice overrides it.
        self._lbl_ref_orient = QLabel("Ref orient:")
        self._combo_ref_orient = QComboBox()
        self._combo_ref_orient.addItem("Auto", "auto")
        for _key, _lbl in _ORIENT_LABELS.items():
            self._combo_ref_orient.addItem(_lbl, _key)
        self._combo_ref_orient.setToolTip(
            "Orientation of the fully-sampled reference relative to the recon.\n"
            "A scan whose readout/phase axes differ from the reference's comes\n"
            "out rotated or mirrored, so the reference is re-oriented to match\n"
            "it BEFORE SSIM/NMSE/PSNR and the difference map are computed.\n"
            "Auto: best of the 8 rotations/flips by correlation (default).\n"
            "Other entries force that rotation/flip of the reference as loaded.\n"
            "Only the reference moves; the recon is untouched. Also used in the\n"
            "PDF report."
        )
        self._combo_ref_orient.activated.connect(self._on_ref_orient_changed)

        # Shown only when the reference is a different acquisition (metrics withheld).
        self._btn_score_anyway = QPushButton("Score anyway")
        self._btn_score_anyway.setCheckable(True)
        self._btn_score_anyway.setChecked(bool(self._result.get("_score_anyway")))
        self._btn_score_anyway.setToolTip(
            "The reference is a different acquisition (sequence, TE/TR, slice, FOV\n"
            "or image grid differ), so SSIM/NMSE/PSNR were withheld. Tick to\n"
            "compute them anyway — they are then labelled as NOT a valid\n"
            "comparison, on screen and in the PDF."
        )
        self._btn_score_anyway.clicked.connect(self._toggle_score_anyway)

        # Export button
        btn_export = QPushButton("Export PNG…")
        btn_export.clicked.connect(self._export_png)

        # ── Equispaced-only post-hoc alignment buttons (composable toggles) ──
        self._eq_center_on = False
        self._eq_crop_on   = False
        self._eq_base      = None   # captured once on first equispaced draw
        self._btn_center = QPushButton("Center K-space")
        self._btn_center.setCheckable(True)
        self._btn_center.setToolTip(
            "Center the k-space (roll DC to the array centre) and recompute "
            "RMSE/PSNR/SSIM on the centred images.\n"
            "Recon pipeline is untouched. Composes with 'Crop to BBox'."
        )
        self._btn_center.clicked.connect(self._toggle_eq_center)
        self._btn_crop = QPushButton("Crop to BBox")
        self._btn_crop.setCheckable(True)
        self._btn_crop.setToolTip(
            "Crop a tight bounding box around the reconstruction and the "
            "reference (same size) and recompute RMSE/PSNR/SSIM.\n"
            "Composes with 'Center K-space'."
        )
        self._btn_crop.clicked.connect(self._toggle_eq_crop)

        # ── Masked-metrics controls (anatomy mask: threshold + dilation) ──
        self._eq_masked_on = False
        self._btn_masked = QPushButton("Masked Metrics")
        self._btn_masked.setCheckable(True)
        self._btn_masked.setToolTip(
            "Recompute SSIM / PSNR / NMSE using only the anatomy region.\n"
            "A binary mask is built from the reference: threshold the normalised\n"
            "magnitude, fill holes, keep the largest component, then dilate.\n"
            "Composes with 'Center K-space' and 'Crop to BBox'."
        )
        self._btn_masked.clicked.connect(self._toggle_eq_masked)

        self._spin_thr = QDoubleSpinBox()
        self._spin_thr.setRange(0.0, 1.0)
        self._spin_thr.setSingleStep(0.05)
        self._spin_thr.setDecimals(2)
        self._spin_thr.setValue(MASK_THRESHOLD_DEFAULT)
        self._spin_thr.setFixedWidth(64)
        self._spin_thr.setToolTip("Mask threshold on the normalised magnitude (0–1).")
        self._spin_thr.valueChanged.connect(self._on_mask_param_changed)

        self._spin_dil = QSpinBox()
        self._spin_dil.setRange(0, 50)
        self._spin_dil.setSingleStep(1)
        self._spin_dil.setValue(MASK_DILATION_DEFAULT)
        self._spin_dil.setFixedWidth(54)
        self._spin_dil.setToolTip("Number of binary-dilation iterations applied to the mask.")
        self._spin_dil.valueChanged.connect(self._on_mask_param_changed)

        self._lbl_thr = QLabel("Thr:")
        self._lbl_dil = QLabel("Dil:")

        # Toolbar row
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(self._nav)
        ctrl_row.addStretch()
        ctrl_row.addWidget(wl_label)
        ctrl_row.addWidget(self._slider_min)
        ctrl_row.addWidget(wl_label2)
        ctrl_row.addWidget(self._slider_max)
        ctrl_row.addSpacing(12)
        ctrl_row.addWidget(bright_label)
        ctrl_row.addWidget(self._slider_brightness)
        ctrl_row.addWidget(self._brightness_label)
        ctrl_row.addSpacing(12)
        ctrl_row.addWidget(self._btn_center)
        ctrl_row.addWidget(self._btn_crop)
        ctrl_row.addSpacing(12)
        ctrl_row.addWidget(self._btn_masked)
        ctrl_row.addWidget(self._lbl_thr)
        ctrl_row.addWidget(self._spin_thr)
        ctrl_row.addWidget(self._lbl_dil)
        ctrl_row.addWidget(self._spin_dil)
        ctrl_row.addSpacing(12)
        ctrl_row.addWidget(self._lbl_ref_orient)
        ctrl_row.addWidget(self._combo_ref_orient)
        ctrl_row.addWidget(self._btn_score_anyway)
        ctrl_row.addSpacing(12)
        ctrl_row.addWidget(btn_rotate)
        ctrl_row.addWidget(btn_export)

        layout = QVBoxLayout(self)
        layout.addLayout(ctrl_row)
        layout.addWidget(self._canvas)

        # Draw initial figure
        self._axes_images = []  # (ax, raw_data) pairs for W/L update
        self._draw()

        # Center/Crop alignment is equispaced-only; the ROI (anatomy-mask) metrics
        # are available for equispaced AND Poisson (both compare to a reference).
        _is_eq   = self._result.get("scan_type") == SCAN_EQUISPACED
        _is_pois = self._result.get("scan_type") == SCAN_POISSON
        _has_ref = self._result.get("reference_image") is not None
        _masked_ok = (_is_eq or _is_pois) and _has_ref
        self._btn_center.setVisible(_is_eq and _has_ref)
        self._btn_crop.setVisible(_is_eq and _has_ref)
        for w in (self._btn_masked, self._lbl_thr, self._spin_thr,
                  self._lbl_dil, self._spin_dil):
            w.setVisible(_masked_ok)
        if _masked_ok:
            self._capture_eq_base()
        _orient_ok = _masked_ok and self._result.get("reference_image_raw") is not None
        self._lbl_ref_orient.setVisible(_orient_ok)
        self._combo_ref_orient.setVisible(_orient_ok)
        if _orient_ok:
            self._sync_ref_orient_combo()
        self._btn_score_anyway.setVisible(_orient_ok and bool(
            self._result.get("reference_blocked_reason") or self._result.get("_score_anyway")))

    # ─── Drawing ─────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self._fig.clear()
        self._axes_images.clear()
        stype = self._result.get("scan_type", "")

        if stype == SCAN_BAND or stype == SCAN_POISSON:
            self._draw_nufft()
        elif stype == SCAN_FULLY_SAMPLED:
            self._draw_fullysamp()
        elif stype == SCAN_EQUISPACED:
            self._draw_equispaced()
        elif stype == SCAN_RETRO_UNDERSAMPLE:
            self._draw_retro()
        elif self._result.get("combined_img") is not None:
            self._draw_nufft()          # generic NUFFT-RSS / CS result
        elif self._result.get("image_rss") is not None:
            self._draw_fullysamp()      # generic Cartesian-RSS result
        else:
            ax = self._fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Display not implemented\nfor type: {stype}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

        self._apply_rotation()
        self._fig.suptitle(self._title, fontsize=11, fontweight="bold", y=0.99)
        self._canvas.draw()

    def _on_rotate(self) -> None:
        """Rotate every image in this tab by another 90° and redraw."""
        self._rotation = (self._rotation + 1) % 4
        self._sync_ref_orient_combo()
        self._draw()

    # ─── Reference orientation ───────────────────────────────────────────────

    def _display_k(self) -> int:
        """Total 90° CCW turns between a result array and what is on screen:
        the layout's base rot90 (equispaced draws np.rot90) + the Rotate button."""
        base = 1 if self._result.get("scan_type") == SCAN_EQUISPACED else 0
        return (base + self._rotation) % 4

    def _sync_ref_orient_combo(self) -> None:
        """Relabel the combo for the current screen rotation: the Auto entry
        names what auto picked, and a manual selection keeps describing the SAME
        (unchanged) reference — a screen-space flip relabels when rotated."""
        a = self._result.get("reference_orientation")
        if a is None:
            return
        screen = _array_to_screen_orient(a, self._display_k())
        combo = self._combo_ref_orient
        combo.setItemText(0, f"Auto ({_ORIENT_LABELS[screen].lower()})")
        if combo.currentData() != "auto":
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(screen))
            combo.blockSignals(False)

    def _ref_orient_note(self):
        """Short 'Ref …' line for the metrics panel, or None if untouched."""
        a = self._result.get("reference_orientation")
        if not a or a == "none":
            return None
        lbl = _ORIENT_LABELS[_array_to_screen_orient(a, self._display_k())]
        how = "auto" if self._result.get("reference_orientation_auto") else "manual"
        note = f"Ref oriented: {lbl.lower()} ({how})"
        if self._result.get("reference_orientation_ambiguous"):
            note += "\n⚠ close call — check visually"
        return note

    def _metrics_blocked(self) -> bool:
        """The reference is a different acquisition and 'Score anyway' is off."""
        return (bool(self._result.get("reference_blocked_reason"))
                and not self._result.get("_score_anyway"))

    def _toggle_score_anyway(self) -> None:
        self._result["_score_anyway"] = self._btn_score_anyway.isChecked()
        self._on_ref_orient_changed()

    def _on_ref_orient_changed(self, _index=None) -> None:
        """Re-orient the reference (auto or forced), re-score, and redraw."""
        if self._eq_base is None:
            return
        r = self._result
        data = self._combo_ref_orient.currentData()
        orient = data if data == "auto" else _screen_to_array_orient(data, self._display_k())
        from processing.reference import compare_to_reference
        try:
            cmp = compare_to_reference(
                self._eq_base["recon"], r["reference_image_raw"],
                recon_name=r.get("recon_name", ""),
                reference_name=r.get("reference_name", ""),
                orientation=orient,
                recon_dat_path=r.get("dat_path"),
                reference_dat_path=r.get("reference_dat_path"),
                slice_idx=r.get("reference_slice_idx"),
                score_anyway=bool(r.get("_score_anyway")),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Reference orientation",
                                f"Could not re-orient the reference:\n{exc}")
            return
        for key in ("reference_image", "reference_orientation",
                    "reference_orientation_auto",
                    "reference_orientation_ambiguous", "reference_warning",
                    "reference_blocked_reason",
                    "ssim", "nmse", "psnr", "rmse"):
            r[key] = cmp[key]
        self._eq_base["ref"] = np.asarray(cmp["reference_image"]).copy()
        self._sync_ref_orient_combo()

        if r.get("scan_type") == SCAN_POISSON:
            self._apply_poisson_masked()            # recomputes ROI metrics + redraws
        elif self._eq_center_on or self._eq_crop_on or self._eq_masked_on:
            self._apply_eq_transforms()
        else:
            self._draw()

    def _apply_rotation(self) -> None:
        """
        Rotate all rendered content images by ``self._rotation`` × 90°.

        Applied post-hoc to every AxesImage so it works uniformly across all
        recon layouts without touching each draw path. Colorbars use a QuadMesh
        (not an AxesImage) and any thin gradient image is skipped by the size
        guard. W/L is unaffected — percentiles are rotation-invariant.
        """
        if not self._rotation:
            return
        k = self._rotation
        for ax in self._fig.axes:
            for im in list(ax.images):
                arr = im.get_array()
                if arr is None:
                    continue
                a = np.asarray(arr)
                if a.ndim < 2 or min(a.shape[0], a.shape[1]) < 8:
                    continue                      # skip colorbars / tiny gradients
                rot = np.rot90(a, k)
                im.set_data(rot)
                h, w = rot.shape[0], rot.shape[1]
                im.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
                ax.set_xlim(-0.5, w - 0.5)
                ax.set_ylim(h - 0.5, -0.5)

    def _add_cbar(self, ax, im):
        """Attach a compact intensity colorbar next to an image axis.

        The bar is linked to the image's norm, so it follows any later
        ``set_clim`` from the W/L and brightness sliders automatically.
        """
        cbar = self._fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        return cbar

    def _draw_nufft(self) -> None:
        r = self._result

        combined  = r.get("combined_img")
        kspace_c  = r.get("kspace_combined")
        coil_imgs = r.get("imgs_per_coil")
        sens      = r.get("sens_maps")
        traj      = r.get("traj_scaled")
        recon_mode = r.get("recon_mode", "pics")   # "pics" | "zerofill"

        combined_title = "Combined Image (Zero-Fill RSS)" if recon_mode == "zerofill" else "Combined Image (PICS)"

        # Layout: 3 columns, 2 rows
        axes = self._fig.subplots(2, 3)

        # Row 0
        if combined is not None:
            img_data = _norm(combined)
            im = axes[0, 0].imshow(img_data, cmap="gray", aspect="auto")
            axes[0, 0].set_title(combined_title)
            self._add_cbar(axes[0, 0], im)
            self._axes_images.append((axes[0, 0], im, img_data, True))

            # ROI anatomy-mask outline (masked-metrics). Drawn pre-rotated by the
            # tab's Rotate button so it stays aligned with the image (the combined
            # image has no base rotation here, so total k = self._rotation).
            mask = r.get("_mask")
            if mask is not None and np.shape(mask) == np.shape(img_data):
                axes[0, 0].contour(
                    np.rot90(np.asarray(mask).astype(float), self._rotation),
                    levels=[0.5], colors="lime", linewidths=0.8)

        if kspace_c is not None:
            kd = np.log(np.abs(np.squeeze(kspace_c)) + 1e-9)
            im2 = axes[0, 1].imshow(kd, cmap="gray", aspect="auto")
            axes[0, 1].set_title("K-space (log)")
            self._add_cbar(axes[0, 1], im2)
            self._axes_images.append((axes[0, 1], im2, kd, False))

        if traj is not None:
            kx = traj[:, 0] if traj.ndim == 2 else traj[0]
            ky = traj[:, 1] if traj.ndim == 2 else traj[1]
            # Negate ky so the trajectory plot matches the imshow y-axis convention
            # (imshow rows increase downward; scatter uses y-up by default).
            axes[0, 2].scatter(kx, -ky, s=0.5, alpha=0.4, c="royalblue")
            # axes[0, 2].scatter(kx, ky, s=0.5, alpha=0.4, c="royalblue")

            axes[0, 2].set_title("Trajectory")
            axes[0, 2].set_xlabel("kx")
            axes[0, 2].set_ylabel("ky  (display: −ky)")
            axes[0, 2].set_aspect("equal")

        # Row 1 — fully-sampled reference comparison (if available), else coils
        reference = r.get("reference_image")
        if reference is not None and combined is not None:
            ref_img = np.abs(np.squeeze(reference)).astype(float)
            rec_img = np.abs(np.squeeze(combined)).astype(float)
            # Match the (arbitrary-scale) NUFFT recon to the reference before
            # differencing, mirroring the metric computation.
            r_scale = float(np.percentile(rec_img.ravel(), 95)) + 1e-12
            f_scale = float(np.percentile(ref_img.ravel(), 95)) + 1e-12
            rec_img = rec_img * (f_scale / r_scale)
            scale = f_scale
            g_n, r_n = rec_img / scale, ref_img / scale

            im_r = axes[1, 0].imshow(_norm(ref_img), cmap="gray", aspect="auto")
            axes[1, 0].set_title("Fully-Sampled Reference")
            self._add_cbar(axes[1, 0], im_r)
            self._axes_images.append((axes[1, 0], im_r, _norm(ref_img), True))

            diff = np.abs(g_n - r_n)
            im_d = axes[1, 1].imshow(diff, cmap="gray", aspect="auto", vmin=0.0, vmax=0.1)
            axes[1, 1].set_title("Absolute Difference")
            self._add_cbar(axes[1, 1], im_d)

            axes[1, 2].axis("off")
            lines = []
            if r.get("ssim") is not None:
                lines.append(f"SSIM = {r['ssim']:.4f}")
            if r.get("nmse") is not None:
                lines.append(f"NMSE = {r['nmse']:.4f}")
            if r.get("psnr") is not None:
                lines.append(f"PSNR = {r['psnr']:.2f} dB")
            txt = "\n".join(lines) or "no metrics"
            if self._metrics_blocked():
                txt = ("Metrics withheld:\nthe reference is a different\n"
                       "acquisition (see the log).\n'Score anyway' overrides.")

            # ROI (anatomy-mask) metrics, when the Masked-Metrics toggle is on.
            if r.get("_mask") is not None:
                mp = r.get("_mask_params", {})
                m_ssim = r.get("masked_ssim"); m_nmse = r.get("masked_nmse")
                m_psnr = r.get("masked_psnr")
                mlines = [
                    "",
                    f"ROI / anatomy mask",
                    f"[thr={mp.get('threshold', '?')}, dil={mp.get('dilation', '?')}]",
                    f"SSIM = {m_ssim:.4f}" if m_ssim is not None else "SSIM = —",
                    f"NMSE = {m_nmse:.4f}" if m_nmse is not None else "NMSE = —",
                    f"PSNR = {m_psnr:.2f} dB" if m_psnr is not None else "PSNR = —",
                ]
                txt = "Whole image\n" + txt + "\n" + "\n".join(mlines)
            elif r.get("_mask_error"):
                txt += f"\n\nMasked: error\n{r.get('_mask_error')}"

            _onote = self._ref_orient_note()
            if _onote:
                txt += f"\n\n{_onote}"
            if r.get("reference_warning") and not self._metrics_blocked():
                txt += "\n\n⚠ reference mismatch:\nmetrics reflect sequence/\nFOV differences too."
            axes[1, 2].text(0.5, 0.5, txt, ha="center", va="center",
                            transform=axes[1, 2].transAxes, fontsize=9)
        elif sens is not None:
            arr = np.squeeze(sens)
            n = min(arr.shape[-1], 3) if arr.ndim == 3 else 1
            for c in range(n):
                img_s = _norm(arr[:, :, c] if arr.ndim == 3 else arr)
                im_s = axes[1, c].imshow(img_s, cmap="viridis", aspect="auto")
                axes[1, c].set_title(f"Sensitivity Map — Coil {c}")
                self._add_cbar(axes[1, c], im_s)
                self._axes_images.append((axes[1, c], im_s, img_s, False))
            for c in range(n, 3):
                axes[1, c].axis("off")
        elif coil_imgs is not None:
            arr = np.squeeze(coil_imgs)
            n = min(arr.shape[-1], 3) if arr.ndim == 3 else 1
            for c in range(n):
                img_c = np.rot90(_norm(arr[:, :, c] if arr.ndim == 3 else arr), k=3)
                im_c = axes[1, c].imshow(img_c, cmap="gray", aspect="auto")
                axes[1, c].set_title(f"Coil {c} (adj. NUFFT)")
                self._add_cbar(axes[1, c], im_c)
                self._axes_images.append((axes[1, c], im_c, img_c, False))
            for c in range(n, 3):
                axes[1, c].axis("off")

        for ax in axes.ravel():
            ax.axis("off") if not ax.has_data() else None

    def _draw_fullysamp(self) -> None:
        r = self._result
        image_rss      = r.get("image_rss")
        image_combined = r.get("image_combined")   # None for old results
        ksp_rss        = r.get("kspace_rss")        # pre-computed 2-D RSS magnitude
        sens_maps      = r.get("sens_maps")         # None when RSS mode
        combination    = r.get("combination", "rss")

        has_sense = (combination == "sense" and image_combined is not None
                     and sens_maps is not None)

        if has_sense:
            # 2-row layout: [SENSE img | RSS img | K-space] / [Sens coil 0..2]
            axes = self._fig.subplots(2, 3)
            axes_row0 = axes[0]
            axes_row1 = axes[1]
        else:
            # single row
            axes_row0 = self._fig.subplots(1, 3)
            axes_row1 = None

        # ── Row 0: main images ────────────────────────────────────────────────
        if has_sense and image_combined is not None:
            d = np.rot90(_norm(image_combined))
            im = axes_row0[0].imshow(d, cmap="gray", aspect="auto")
            axes_row0[0].set_title("SENSE Combined (ESPIRiT)")
            self._add_cbar(axes_row0[0], im)
            self._axes_images.append((axes_row0[0], im, d, True))
        elif image_rss is not None:
            d = np.rot90(_norm(image_rss))
            im = axes_row0[0].imshow(d, cmap="gray", aspect="auto")
            axes_row0[0].set_title("RSS Image")
            self._add_cbar(axes_row0[0], im)
            self._axes_images.append((axes_row0[0], im, d, True))

        # When SENSE is used also show RSS for comparison
        if has_sense and image_rss is not None:
            d2 = np.rot90(_norm(image_rss))
            im2 = axes_row0[1].imshow(d2, cmap="gray", aspect="auto")
            axes_row0[1].set_title("RSS Image (reference)")
            self._add_cbar(axes_row0[1], im2)
            self._axes_images.append((axes_row0[1], im2, d2, True))
        else:
            axes_row0[1].axis("off")

        if ksp_rss is not None:
            # ksp_rss is already a 2-D float32 RSS magnitude
            kd = np.rot90(np.log(ksp_rss + 1e-9))
            im3 = axes_row0[2].imshow(kd, cmap="gray", aspect="auto")
            axes_row0[2].set_title("K-space (log RSS)")
            self._add_cbar(axes_row0[2], im3)
            self._axes_images.append((axes_row0[2], im3, kd, False))

        # ── Row 1: first 3 sensitivity maps (SENSE only) ──────────────────────
        if has_sense and axes_row1 is not None:
            arr = np.squeeze(sens_maps)   # (Nread, Nphase, Ncoils)
            n = min(arr.shape[-1], 3) if arr.ndim == 3 else 1
            for c in range(n):
                img_s = np.rot90(_norm(arr[:, :, c] if arr.ndim == 3 else arr))
                im_s  = axes_row1[c].imshow(img_s, cmap="viridis", aspect="auto")
                axes_row1[c].set_title(f"Sensitivity Map — Coil {c}")
                self._add_cbar(axes_row1[c], im_s)
                self._axes_images.append((axes_row1[c], im_s, img_s, False))
            for c in range(n, 3):
                axes_row1[c].axis("off")

        for ax in np.array(axes_row0).ravel():
            ax.axis("off") if not ax.has_data() else None

    def _draw_equispaced(self) -> None:
        r = self._result
        method      = r.get("method", "grappa")
        # Primary reconstructed image: GRAPPA or SENSE/CS-SENSE
        primary_img = r.get("grappa_image")
        zf_img      = r.get("zf_image")
        ref_img     = r.get("reference_image")
        recon_ksp   = r.get("grappa_kspace_rss")        # pre-computed 2-D RSS
        under_ksp   = r.get("undersampled_kspace_rss")  # pre-computed 2-D RSS
        rmse        = r.get("rmse")
        psnr        = r.get("psnr")
        ssim_val    = r.get("ssim")
        actual_r    = r.get("actual_r", 4)

        method_label = _method_display_label(r, method)
        ksp_title    = f"{method_label} K-space (log)" if method == "grappa" else "Recon K-space (log)"

        def n01(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-12)

        # Shared normalization: when a reference exists, normalize BOTH the
        # reconstruction and the reference by the reference's 95th percentile.
        # This puts them on one common, outlier-robust intensity scale, so the
        # difference map reflects true intensity error rather than each image's
        # own min-max stretch (which would hide global scale mismatches and
        # make a dim recon look as bright as the reference).
        if ref_img is not None:
            ref_p95 = float(np.percentile(np.abs(ref_img).ravel(), 95)) + 1e-12
            def n01_img(x):
                return x / ref_p95
        else:
            n01_img = n01

        axes = self._fig.subplots(2, 3)

        ssim = r.get("ssim")

        if primary_img is not None:
            g = n01_img(primary_img)
            gd = np.rot90(g)
            im = axes[0, 0].imshow(gd, cmap="gray", aspect="auto")
            lbl = f"{method_label} (R={actual_r:.2f}×)"
            if rmse:
                lbl += f"\nRMSE={rmse:.4f}  PSNR={psnr:.1f}dB"
                if ssim is not None:
                    lbl += f"  SSIM={ssim:.4f}"
            axes[0, 0].set_title(lbl, fontsize=9)
            self._add_cbar(axes[0, 0], im)
            self._axes_images.append((axes[0, 0], im, gd, True))

            # Outline the anatomy mask used for masked metrics (if active). The
            # primary image is drawn as np.rot90(g) (base k=1) and _apply_rotation
            # then adds self._rotation, so pre-rotate the mask by (1 + rotation)
            # to keep the outline aligned when the Rotate button is used.
            mask = r.get("_mask")
            if mask is not None and np.shape(mask) == g.shape:
                axes[0, 0].contour(
                    np.rot90(np.asarray(mask).astype(float), 1 + self._rotation),
                    levels=[0.5], colors="lime", linewidths=0.8)

        if ref_img is not None:
            rn = n01_img(ref_img)
            rnd = np.rot90(rn)
            im2 = axes[0, 1].imshow(rnd, cmap="gray", aspect="auto")
            axes[0, 1].set_title("Fully-Sampled Reference")
            self._add_cbar(axes[0, 1], im2)
            self._axes_images.append((axes[0, 1], im2, rnd, True))

            if primary_img is not None:
                # True absolute error on the shared reference scale. Displayed
                # with vmax=0.1 so a 10% intensity error saturates to white
                # (same visual sensitivity as the old "×10" trick), while the
                # colorbar now reads real error values instead of 10× inflated.
                diff = np.rot90(np.abs(g - rn))
                im3 = axes[0, 2].imshow(diff, cmap="gray", aspect="auto",
                                        vmin=0, vmax=0.1)
                axes[0, 2].set_title("Absolute Difference")
                self._add_cbar(axes[0, 2], im3)
                self._axes_images.append((axes[0, 2], im3, diff, False))
        else:
            if zf_img is not None:
                zn = n01_img(zf_img)
                znd = np.rot90(zn)
                im2 = axes[0, 1].imshow(znd, cmap="gray", aspect="auto")
                axes[0, 1].set_title("Zero-Fill")
                self._add_cbar(axes[0, 1], im2)
                self._axes_images.append((axes[0, 1], im2, znd, True))
            axes[0, 2].axis("off")

        if under_ksp is not None:
            kd = np.rot90(np.log(under_ksp + 1e-9))
            im4 = axes[1, 0].imshow(kd, cmap="gray", aspect="auto")
            axes[1, 0].set_title("Undersampled K-space (log)")
            self._add_cbar(axes[1, 0], im4)
            self._axes_images.append((axes[1, 0], im4, kd, False))

        if recon_ksp is not None:
            kg = np.rot90(np.log(recon_ksp + 1e-9))
            im5 = axes[1, 1].imshow(kg, cmap="gray", aspect="auto")
            axes[1, 1].set_title(ksp_title)
            self._add_cbar(axes[1, 1], im5)
            self._axes_images.append((axes[1, 1], im5, kg, False))

        axes[1, 2].axis("off")
        if rmse is None and self._metrics_blocked():
            axes[1, 2].text(
                0.1, 0.6,
                "Metrics withheld\n\nThe reference is a different\nacquisition. "
                "'Score anyway'\noverrides (labelled not valid).",
                transform=axes[1, 2].transAxes, fontsize=10, va="center",
            )
        if rmse is not None:
            metric_text = f"Metrics (whole image)\n\nRMSE : {rmse:.4f}\nPSNR : {psnr:.2f} dB"
            if ssim is not None:
                metric_text += f"\nSSIM : {ssim:.4f}"

            # Masked metrics (anatomy region only), when active.
            if r.get("_mask") is not None:
                mp = r.get("_mask_params", {})
                metric_text += (
                    f"\n\nMasked (anatomy only)"
                    f"\n[thr={mp.get('threshold', '?')}, dil={mp.get('dilation', '?')}]"
                )
                m_nmse, m_psnr, m_ssim = (
                    r.get("masked_nmse"), r.get("masked_psnr"), r.get("masked_ssim"))
                metric_text += (
                    f"\nNMSE : {m_nmse:.4f}" if m_nmse is not None else "\nNMSE : —")
                metric_text += (
                    f"\nPSNR : {m_psnr:.2f} dB" if m_psnr is not None else "\nPSNR : —")
                metric_text += (
                    f"\nSSIM : {m_ssim:.4f}" if m_ssim is not None else "\nSSIM : —")
            elif r.get("_mask_error"):
                metric_text += f"\n\nMasked: error\n{r.get('_mask_error')}"

            align_note = r.get("_eq_align_note")
            if align_note:
                metric_text += f"\n\n[aligned: {align_note}]"
            _onote = self._ref_orient_note()
            if _onote:
                metric_text += f"\n\n{_onote}"
            axes[1, 2].text(
                0.1, 0.5,
                metric_text,
                transform=axes[1, 2].transAxes, fontsize=11, va="center",
            )

        warn = r.get("reference_warning")
        if warn:
            import textwrap
            wrapped = "\n".join(textwrap.wrap("\u26a0 " + warn, width=46))
            axes[1, 2].text(
                0.05, 0.02, wrapped,
                transform=axes[1, 2].transAxes, fontsize=6.5, va="bottom",
                color="crimson",
            )

        for ax in axes.ravel():
            ax.axis("off") if not ax.has_data() else None


    # ─── Equispaced post-hoc alignment (Center / Crop) ────────────────────────

    def _primary_image_key(self) -> str:
        """Result-dict key holding the primary reconstructed image for this tab
        (``combined_img`` for the NUFFT/Poisson layout, ``grappa_image`` else)."""
        return "combined_img" if self._result.get("scan_type") == SCAN_POISSON else "grappa_image"

    def _capture_eq_base(self) -> None:
        """Store pristine copies so the toggles can be (re)composed freely."""
        r = self._result
        und = r.get("undersampled_kspace_rss")
        rks = r.get("grappa_kspace_rss")
        self._eq_base = {
            "recon":     np.asarray(r.get(self._primary_image_key())).copy(),
            "ref":       np.asarray(r.get("reference_image")).copy(),
            "under_ksp": None if und is None else np.asarray(und).copy(),
            "recon_ksp": None if rks is None else np.asarray(rks).copy(),
        }
        # Shared with the result dict so exports can offer the untransformed
        # reconstruction while Center/Crop toggles are active. Never mutated
        # in place (the transforms always work on fresh .copy()s).
        r["_pristine_recon"] = self._eq_base["recon"]

    def _toggle_eq_center(self) -> None:
        self._eq_center_on = self._btn_center.isChecked()
        self._apply_eq_transforms()

    def _toggle_eq_crop(self) -> None:
        self._eq_crop_on = self._btn_crop.isChecked()
        self._apply_eq_transforms()

    def _toggle_eq_masked(self) -> None:
        self._eq_masked_on = self._btn_masked.isChecked()
        self._apply_masked_transforms()

    def _on_mask_param_changed(self, _value=None) -> None:
        # Re-evaluate only when masked metrics are currently active.
        if self._eq_masked_on:
            self._apply_masked_transforms()

    def _apply_masked_transforms(self) -> None:
        """Dispatch the masked-metrics recompute. Poisson has no Center/Crop, so
        it uses a dedicated ROI-only path; equispaced composes with its toggles."""
        if self._result.get("scan_type") == SCAN_POISSON:
            self._apply_poisson_masked()
        else:
            self._apply_eq_transforms()

    def _apply_poisson_masked(self) -> None:
        """Compute (or clear) ROI anatomy-mask metrics for a Poisson result and
        redraw — mirrors the equispaced masked-metrics feature, without the
        Center/Crop alignment (the Cartesian recon is already reference-aligned)."""
        if self._eq_base is None:
            return
        recon = self._eq_base["recon"]
        ref   = self._eq_base["ref"]
        if self._eq_masked_on and not self._metrics_blocked():
            thr = float(self._spin_thr.value())
            dil = int(self._spin_dil.value())
            try:
                mask, m_nmse, m_psnr, m_ssim = _masked_metrics(recon, ref, thr, dil)
            except Exception as exc:
                self._result["_mask_error"] = str(exc)
                for k in ("_mask", "masked_nmse", "masked_psnr", "masked_ssim",
                          "_mask_params"):
                    self._result.pop(k, None)
            else:
                self._result.pop("_mask_error", None)
                self._result["_mask"]        = mask
                self._result["masked_nmse"]  = m_nmse
                self._result["masked_psnr"]  = m_psnr
                self._result["masked_ssim"]  = m_ssim
                self._result["_mask_params"] = {"threshold": thr, "dilation": dil}
        else:
            for k in ("_mask", "masked_nmse", "masked_psnr", "masked_ssim",
                      "_mask_params", "_mask_error"):
                self._result.pop(k, None)
        self._draw()

    def _apply_eq_transforms(self) -> None:
        """Rebuild working images from the pristine base applying the active
        toggles (center → crop), recompute metrics, and redraw."""
        if self._eq_base is None:
            return
        recon = self._eq_base["recon"].copy()
        ref   = self._eq_base["ref"].copy()
        under = self._eq_base["under_ksp"]
        rks   = self._eq_base["recon_ksp"]
        under = None if under is None else under.copy()
        rks   = None if rks is None else rks.copy()

        # 2.1 Center: roll DC to the array centre (k-space) and centre the
        #     image content (so recon & reference are aligned before scoring).
        if self._eq_center_on:
            recon = self._center_image(recon)
            ref   = self._center_image(ref)
            if under is not None:
                under = self._center_ksp(under)
            if rks is not None:
                rks = self._center_ksp(rks)

        # 2.2 Crop: tight bounding box shared by recon & reference → equal size.
        if self._eq_crop_on:
            recon, ref = self._crop_bbox(recon, ref)

        blocked = self._metrics_blocked()
        rmse, psnr, ssim = (None, None, None) if blocked else self._calc_metrics(recon, ref)

        self._result["grappa_image"]    = recon
        self._result["reference_image"] = ref
        if under is not None:
            self._result["undersampled_kspace_rss"] = under
        if rks is not None:
            self._result["grappa_kspace_rss"] = rks
        self._result["rmse"], self._result["psnr"], self._result["ssim"] = rmse, psnr, ssim

        # Masked metrics (anatomy region only), computed on the current recon/ref.
        if self._eq_masked_on and not blocked:
            thr = float(self._spin_thr.value())
            dil = int(self._spin_dil.value())
            try:
                mask, m_nmse, m_psnr, m_ssim = _masked_metrics(recon, ref, thr, dil)
            except Exception as exc:
                mask, m_nmse, m_psnr, m_ssim = None, None, None, None
                self._result["_mask_error"] = str(exc)
            else:
                self._result.pop("_mask_error", None)
            self._result["_mask"]        = mask
            self._result["masked_nmse"]  = m_nmse
            self._result["masked_psnr"]  = m_psnr
            self._result["masked_ssim"]  = m_ssim
            self._result["_mask_params"] = {"threshold": thr, "dilation": dil}
        else:
            for k in ("_mask", "masked_nmse", "masked_psnr", "masked_ssim",
                      "_mask_params", "_mask_error"):
                self._result.pop(k, None)

        active = []
        if self._eq_center_on:
            active.append("centered")
        if self._eq_crop_on:
            active.append("cropped")
        if self._eq_masked_on:
            active.append("masked")
        self._result["_eq_align_note"] = " + ".join(active) if active else None
        # Center/Crop replace the image arrays in the result dict; flag it so
        # exports can ask whether to save the modified or the raw recon.
        self._result["_view_modified"] = bool(self._eq_center_on or self._eq_crop_on)

        self._draw()

    @staticmethod
    def _center_image(img: np.ndarray) -> np.ndarray:
        from scipy import ndimage
        a = np.abs(img).astype(float)
        if a.max() <= 0:
            return img
        mask = a > 0.1 * a.max()
        if not mask.any():
            return img
        cy, cx = ndimage.center_of_mass(mask)
        sy = int(round(img.shape[0] / 2.0 - cy))
        sx = int(round(img.shape[1] / 2.0 - cx))
        return np.roll(np.roll(img, sy, axis=0), sx, axis=1)

    @staticmethod
    def _center_ksp(k: np.ndarray) -> np.ndarray:
        p = np.unravel_index(int(np.argmax(k)), k.shape)
        sy = k.shape[0] // 2 - p[0]
        sx = k.shape[1] // 2 - p[1]
        return np.roll(np.roll(k, sy, axis=0), sx, axis=1)

    @staticmethod
    def _crop_bbox(recon: np.ndarray, ref: np.ndarray):
        """Crop each image to its OWN tight bounding box, then resize the
        reference crop to the recon-crop shape so the two are comparable."""
        from scipy import ndimage

        def n01(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-12)

        def tight(img: np.ndarray) -> np.ndarray:
            a = n01(np.abs(img).astype(float))
            # Smooth so isolated background noise pixels don't loosen the box.
            sm = ndimage.uniform_filter(a, size=5)
            m = sm > 0.10
            if not m.any():
                return img
            # Keep only the largest connected component (the brain), so stray
            # bright specks at the edges don't inflate the bounding box.
            lbl, n = ndimage.label(m)
            if n > 1:
                sizes = ndimage.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
                m = lbl == (int(np.argmax(sizes)) + 1)
            rows = np.any(m, axis=1)
            cols = np.any(m, axis=0)
            r0, r1 = np.where(rows)[0][[0, -1]]
            c0, c1 = np.where(cols)[0][[0, -1]]
            return img[r0:r1 + 1, c0:c1 + 1]

        rc = tight(recon)
        fc = tight(ref)
        if rc.shape != fc.shape:
            from scipy.ndimage import zoom
            zf = (rc.shape[0] / fc.shape[0], rc.shape[1] / fc.shape[1])
            fc = zoom(fc, zf, order=1)
        return rc, fc


    @staticmethod
    def _calc_metrics(recon: np.ndarray, ref: np.ndarray):
        from processing.imaging import metrics_vs_reference

        if recon.shape != ref.shape:
            from scipy.ndimage import zoom
            zf = (recon.shape[0] / ref.shape[0], recon.shape[1] / ref.shape[1])
            ref = zoom(ref, zf, order=1)
        m = metrics_vs_reference(recon, ref)
        return m["rmse"], m["psnr"], m["ssim"]

    def _draw_retro(self) -> None:
        """Display layout for retrospective undersampling simulator results."""
        r               = self._result
        pe_mask         = r.get("pe_mask")
        fs_img          = r.get("fully_sampled_image")
        zf_img          = r.get("zf_image")
        recon_img       = r.get("recon_image")
        diff_img        = r.get("diff_image")
        fs_ksp          = r.get("fs_kspace_rss")
        under_ksp       = r.get("undersampled_kspace_rss")
        recon_ksp       = r.get("recon_kspace_rss")
        actual_r        = r.get("actual_r", 0.0)
        mask_type       = r.get("mask_type", "equispaced")
        swap_pe_read    = r.get("swap_pe_read", False)
        acs_start       = r.get("acs_start")
        acs_end         = r.get("acs_end")
        rmse            = r.get("rmse")
        psnr            = r.get("psnr")
        ssim_val        = r.get("ssim")
        method          = r.get("method", "none")
        params_dict     = r.get("params", {})

        axis_label  = "RO (L-R)" if swap_pe_read else "PE (A-P)"
        axis_ylabel = "RO index" if swap_pe_read else "PE index"

        def n01(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-12)

        # Shared normalization by the fully-sampled (reference) 95th
        # percentile, so the difference map reflects true intensity error
        # rather than each image's own min-max stretch (matches
        # _draw_equispaced).
        if fs_img is not None:
            fs_p95 = float(np.percentile(np.abs(fs_img).ravel(), 95)) + 1e-12
            def n01_img(x):
                return x / fs_p95
        else:
            n01_img = n01

        axes = self._fig.subplots(3, 3)

        # ── Row 0: Fully-Sampled | Zero-Filled | PE Mask (2-D image) ─────────
        if fs_img is not None:
            d = np.rot90(n01_img(fs_img))
            im = axes[0, 0].imshow(d, cmap="gray", aspect="auto")
            axes[0, 0].set_title("Fully-Sampled (Reference)")
            self._add_cbar(axes[0, 0], im)
            self._axes_images.append((axes[0, 0], im, d, True))

        if zf_img is not None:
            d2 = np.rot90(n01_img(zf_img))
            im2 = axes[0, 1].imshow(d2, cmap="gray", aspect="auto")
            axes[0, 1].set_title("Zero-Filled")
            self._add_cbar(axes[0, 1], im2)
            self._axes_images.append((axes[0, 1], im2, d2, True))

        # PE mask as a 2-D image: each row = one sampled line, white = sampled
        if pe_mask is not None:
            n = len(pe_mask)
            width = max(1, n // 4)
            mask_2d = np.tile(pe_mask.reshape(-1, 1).astype(float), (1, width))
            axes[0, 2].imshow(mask_2d, cmap="gray", aspect="auto", origin="upper",
                              vmin=0, vmax=1)
            if acs_start is not None and acs_end is not None:
                axes[0, 2].axhspan(acs_start - 0.5, acs_end - 0.5, alpha=0.35,
                                   color="orange", label="ACS")
                axes[0, 2].legend(fontsize=7, loc="upper right")
            axes[0, 2].set_title(
                f"{axis_label} Mask — {mask_type.upper()}  R={actual_r:.2f}×"
                f"  ({pe_mask.sum()}/{n})"
            )
            axes[0, 2].set_xlabel("(replicated)")
            axes[0, 2].set_ylabel(axis_ylabel)

        # ── Row 1: Reconstructed | Difference | Metrics ───────────────────────
        method_label = _method_display_label(r, method)

        if recon_img is not None and method != "none":
            d3 = np.rot90(n01_img(recon_img))
            im3 = axes[1, 0].imshow(d3, cmap="gray", aspect="auto")
            lbl = f"{method_label} (R={actual_r:.2f}×)"
            if rmse is not None:
                lbl += f"\nRMSE={rmse:.4f}  PSNR={psnr:.1f}dB"
                if ssim_val is not None:
                    lbl += f"  SSIM={ssim_val:.4f}"
            axes[1, 0].set_title(lbl, fontsize=9)
            self._add_cbar(axes[1, 0], im3)
            self._axes_images.append((axes[1, 0], im3, d3, True))
        else:
            axes[1, 0].axis("off")

        # Recompute the difference on the shared reference scale when possible;
        # fall back to the precomputed diff_image if recon/reference shapes
        # don't line up.
        if (recon_img is not None and fs_img is not None
                and np.shape(recon_img) == np.shape(fs_img) and method != "none"):
            d4 = np.rot90(np.abs(n01_img(recon_img) - n01_img(fs_img)))
            diff_vmax = 0.1
        elif diff_img is not None and method != "none":
            d4 = np.rot90(diff_img)
            diff_vmax = None
        else:
            d4 = None

        if d4 is not None:
            im4 = axes[1, 1].imshow(d4, cmap="gray", aspect="auto",
                                    vmin=0, vmax=diff_vmax)
            axes[1, 1].set_title("Absolute Difference")
            self._add_cbar(axes[1, 1], im4)
            self._axes_images.append((axes[1, 1], im4, d4, False))
        else:
            axes[1, 1].axis("off")

        # Metrics text panel
        axes[1, 2].axis("off")
        info_lines = ["Simulation Parameters"]
        info_lines.append(f"Mask type  : {mask_type.upper()}")
        info_lines.append(f"Undersamp. : {axis_label}")
        info_lines.append(f"Target R   : {params_dict.get('R', '?')}")
        info_lines.append(f"Actual R   : {actual_r:.2f}×")
        info_lines.append(f"ACS lines  : {params_dict.get('acs_lines', '?')}")
        sampled = pe_mask.sum() if pe_mask is not None else "?"
        total   = len(pe_mask)  if pe_mask is not None else "?"
        info_lines.append(f"Sampled    : {sampled}/{total}")
        if mask_type == "vd":
            info_lines.append(f"Power decay: {params_dict.get('power_decay', '?')}")
        info_lines.append(f"Recon      : {method_label}")
        if method in ("grappa", "ggrappa"):
            ks = params_dict.get('kernel_size', '?')
            ks_str = f"{ks[0]}×{ks[1]}" if isinstance(ks, (list, tuple)) else str(ks)
            info_lines.append(f"  Kernel    : {ks_str}")
            info_lines.append(f"  λ GRAPPA  : {params_dict.get('grappa_lambda', '?')}")
        elif method in ("sense", "sense_cs", "espirit", "espirit_l1",
                        "espirit_l2", "espirit_tv", "espirit_l1wav"):
            info_lines.append(f"  Iterations: {params_dict.get('sense_max_iter', '?')}")
            if method in ("sense_cs", "espirit_l1", "espirit_l2", "espirit_tv", "espirit_l1wav"):
                info_lines.append(f"  λ L1      : {params_dict.get('sense_lambda', '?')}")
            if method in ("espirit", "espirit_l1", "espirit_l2", "espirit_tv", "espirit_l1wav"):
                info_lines.append(f"  Map sets  : {params_dict.get('espirit_num_maps', '?')}")
        if rmse is not None:
            info_lines.append(f"\nMetrics")
            info_lines.append(f"RMSE : {rmse:.4f}")
            info_lines.append(f"PSNR : {psnr:.2f} dB")
            if ssim_val is not None:
                info_lines.append(f"SSIM : {ssim_val:.4f}")
        axes[1, 2].text(
            0.05, 0.95,
            "\n".join(info_lines),
            transform=axes[1, 2].transAxes,
            fontsize=5, va="top",
        )

        # ── Row 2: FS k-space | Undersampled k-space | Reconstructed k-space ─
        if fs_ksp is not None:
            kd = np.rot90(np.log(fs_ksp + 1e-9))
            im5 = axes[2, 0].imshow(kd, cmap="gray", aspect="auto")
            axes[2, 0].set_title("FS K-space (log)")
            self._add_cbar(axes[2, 0], im5)
            self._axes_images.append((axes[2, 0], im5, kd, False))

        if under_ksp is not None:
            kd2 = np.rot90(np.log(under_ksp + 1e-9))
            im6 = axes[2, 1].imshow(kd2, cmap="gray", aspect="auto")
            axes[2, 1].set_title("Undersampled K-space (log)")
            self._add_cbar(axes[2, 1], im6)
            self._axes_images.append((axes[2, 1], im6, kd2, False))

        if recon_ksp is not None:
            kd3 = np.rot90(np.log(recon_ksp + 1e-9))
            im7 = axes[2, 2].imshow(kd3, cmap="gray", aspect="auto")
            axes[2, 2].set_title(f"{method_label} K-space (log)")
            self._add_cbar(axes[2, 2], im7)
            self._axes_images.append((axes[2, 2], im7, kd3, False))
        else:
            axes[2, 2].axis("off")

        for ax in axes.ravel():
            ax.axis("off") if not ax.has_data() else None

    # ─── W/L update ──────────────────────────────────────────────────────────

    def _update_clim(self) -> None:
        vmin_pct  = self._slider_min.value()
        vmax_pct  = self._slider_max.value()
        brightness = self._slider_brightness.value() / 100.0
        for (ax, im, data, is_main) in self._axes_images:
            # W/L and brightness sliders only retouch the main reconstructed
            # images. Difference, mask and k-space panels keep the fixed
            # display range they were drawn with.
            if not is_main:
                continue
            vmin = np.percentile(data, vmin_pct)
            vmax = np.percentile(data, vmax_pct)
            vmax = vmax / brightness
            vmax = max(vmax, vmin + 1e-6)
            im.set_clim(vmin, vmax)
        self._canvas.draw_idle()

    def get_display_settings(self) -> dict:
        """Return current W/L, brightness and rotation settings for this tab."""
        return {
            "wl_min":    self._slider_min.value(),
            "wl_max":    self._slider_max.value(),
            "brightness": self._slider_brightness.value() / 100.0,
            "rotation":  self._rotation,
        }

    # ─── Export ──────────────────────────────────────────────────────────────

    def _export_png(self) -> None:
        # ── Step 1: let user choose which subplots to include ───────────────
        if self._axes_images:
            titles = [
                ax.get_title() or f"Image {i + 1}"
                for i, (ax, im, data, is_main) in enumerate(self._axes_images)
            ]
            dlg = _ImageSelectDialog(titles, self)
            if dlg.exec_() != QDialog.Accepted:
                return
            selected = dlg.selected_indices()
            if not selected:
                QMessageBox.warning(self, "No selection", "No images selected for export.")
                return
        else:
            selected = []

        # ── Step 2: choose save path ─────────────────────────────────────────
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Figure", f"{self._title}.png", "PNG (*.png)"
        )
        if not path:
            return

        # ── Step 3: export ───────────────────────────────────────────────────
        vmin_pct   = self._slider_min.value()
        vmax_pct   = self._slider_max.value()
        brightness = self._slider_brightness.value() / 100.0

        if not self._axes_images or selected == list(range(len(self._axes_images))):
            # All images selected → save the current figure as-is
            self._fig.savefig(path, dpi=150, bbox_inches="tight")
        else:
            # Subset selected → build a new figure containing only those images
            n     = len(selected)
            ncols = min(n, 3)
            nrows = int(np.ceil(n / ncols))

            export_fig, export_axes = plt.subplots(
                nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
            )
            flat = export_axes.ravel()

            for out_idx, src_idx in enumerate(selected):
                ax_src, im_src, data, is_main = self._axes_images[src_idx]
                ax_dst = flat[out_idx]
                im_dst = ax_dst.imshow(data, cmap=im_src.get_cmap(), aspect="auto")
                ax_dst.set_title(ax_src.get_title(), fontsize=9)
                ax_dst.axis("off")
                if is_main:
                    vmin = np.percentile(data, vmin_pct)
                    vmax = np.percentile(data, vmax_pct)
                    vmax = vmax / brightness
                    vmax = max(vmax, vmin + 1e-6)
                else:
                    # Preserve the fixed display range of the source panel
                    vmin, vmax = im_src.get_clim()
                im_dst.set_clim(vmin, vmax)
                cbar = export_fig.colorbar(im_dst, ax=ax_dst,
                                           fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=7)

            for ax in flat[n:]:
                ax.axis("off")

            export_fig.suptitle(self._title, fontsize=11, fontweight="bold")
            export_fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(export_fig)

        QMessageBox.information(self, "Exported", f"Saved to:\n{path}")

    # ─── All coils viewer ────────────────────────────────────────────────────

    def show_all_coils(self) -> bool:
        """Open a standalone window with all per-coil images.

        Returns False when this tab has no per-coil data (so the caller can
        tell the user instead of silently doing nothing).
        """
        arr = self._result.get("imgs_per_coil")
        if arr is None:
            return False
        arr = np.squeeze(arr)
        if arr.ndim != 3:
            return False
        # Close the previous popout for this tab — pyplot-managed figures live
        # in a global registry and would otherwise accumulate on every click.
        prev = getattr(self, "_coil_fig", None)
        if prev is not None:
            plt.close(prev)
        nc = arr.shape[-1]
        ncols = int(np.ceil(np.sqrt(nc)))
        nrows = int(np.ceil(nc / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        fig.suptitle(f"All coils — {self._title}", fontsize=10)
        axes = np.array(axes).ravel()
        for c in range(nc):
            axes[c].imshow(np.rot90(_norm(arr[:, :, c]), k=3), cmap="gray", aspect="auto")
            axes[c].set_title(f"Coil {c}", fontsize=7)
            axes[c].axis("off")
        for c in range(nc, len(axes)):
            axes[c].axis("off")
        plt.tight_layout()
        self._coil_fig = fig
        fig.canvas.mpl_connect(
            "close_event", lambda _e: setattr(self, "_coil_fig", None)
        )
        plt.show()
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Main results viewer (tabbed container)
# ─────────────────────────────────────────────────────────────────────────────

class ResultsViewer(QTabWidget):
    """
    Tabbed widget: each tab is a ReconTab for one reconstruction result.
    """

    # Emitted with the result dict of a tab the user closed, so the owner
    # (MainWindow) can drop it from its results list too.
    result_closed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self._on_tab_close_requested)
        self.setMovable(True)

    def add_result(self, result: Dict[str, Any], title: str) -> None:
        """Add a new result tab."""
        tab = ReconTab(result, title)
        self.addTab(tab, title[:40])   # truncate long labels
        self.setCurrentIndex(self.count() - 1)

    def _on_tab_close_requested(self, index: int) -> None:
        tab = self.widget(index)
        self.removeTab(index)
        if isinstance(tab, ReconTab):
            self.result_closed.emit(tab._result)
            tab.deleteLater()   # removeTab alone would leak the widget + Figure

    def clear_all(self) -> None:
        """Remove all tabs."""
        while self.count() > 0:
            tab = self.widget(0)
            self.removeTab(0)
            if tab is not None:
                tab.deleteLater()

    def get_display_settings_map(self) -> dict:
        """Return {tab_title: display_settings_dict} for all open ReconTabs."""
        result = {}
        for i in range(self.count()):
            tab = self.widget(i)
            if isinstance(tab, ReconTab):
                result[tab._title] = tab.get_display_settings()
        return result

    def show_all_coils_current(self) -> bool:
        """All-coils view for the visible tab; False when there is nothing to show."""
        tab = self.currentWidget()
        if isinstance(tab, ReconTab):
            return tab.show_all_coils()
        return False
