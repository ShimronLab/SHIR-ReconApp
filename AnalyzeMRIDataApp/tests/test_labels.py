"""Label generation, uniqueness, and folder grouping (ui.main_window helpers)."""

import types

from ui.main_window import MainWindow, _make_label


def test_make_label_variants():
    assert _make_label({"folder_key": "SE_1mm", "scan_type": "BAND",
                        "band_idx": 3, "dat_path": "x/band_se_3.dat"}) == "SE_1mm — Band 3"
    assert _make_label({"folder_key": "SE_1mm", "scan_type": "EQUISPACED",
                        "dat_path": "x/equispaced_R4_acs24.dat"}) == "SE_1mm — Equispaced"
    assert _make_label({"folder_key": "F", "scan_type": "UNKNOWN",
                        "dat_path": "x/mystery_scan.dat"}) == "F — mystery_scan"
    # Per-slice suffix
    lbl = _make_label({"folder_key": "F", "scan_type": "FULLY_SAMPLED",
                       "dat_path": "x/fully_sampled.dat",
                       "slice_idx": 2, "n_slices": 5})
    assert lbl.endswith("slice 3/5")


def test_unique_label_suffixing():
    """Duplicate labels get ' (2)', ' (3)' suffixes (display-map/.npy collisions)."""
    fake = types.SimpleNamespace(_results=[])
    ul = MainWindow._unique_label
    assert ul(fake, "A — Equispaced") == "A — Equispaced"
    fake._results.append({"label": "A — Equispaced"})
    assert ul(fake, "A — Equispaced") == "A — Equispaced (2)"
    fake._results.append({"label": "A — Equispaced (2)"})
    assert ul(fake, "A — Equispaced") == "A — Equispaced (3)"


def test_group_by_folder_uses_folder_key():
    results = [
        {"label": "a", "folder_key": "SE_1mm", "dat_path": "/x/dat_files/a.dat"},
        {"label": "b", "folder_key": "SE_1mm", "dat_path": "/x/dat_files/b.dat"},
        {"label": "c", "folder_key": "SE_3mm", "dat_path": "/y/dat_files/c.dat"},
        # legacy result without folder_key → parent.parent fallback
        {"label": "d", "dat_path": "/scans/OLD/dat_files/d.dat"},
    ]
    groups = MainWindow._group_by_folder(results)
    assert set(groups) == {"SE_1mm", "SE_3mm", "OLD"}
    assert [r["label"] for r in groups["SE_1mm"]] == ["a", "b"]
