"""End-to-end reconstruction on the real phantom scan folder.

Marked `realdata` — skipped automatically when the sample folder is absent
(see conftest.SAMPLE_SCAN_DIR). Uses only the engine-free pipelines
(fully-sampled RSS and pygrappa GRAPPA) so it needs neither BART nor a forced
sigpy path, and picks the smaller custom .dat files by glob so exact Siemens
MID numbers don't matter.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.realdata


def _first(sample_scan_dir, pattern):
    matches = sorted((sample_scan_dir / "dat_files").glob(pattern))
    if not matches:
        pytest.skip(f"no file matching {pattern} in the sample folder")
    return str(matches[0])


def test_fully_sampled_rss(sample_scan_dir):
    from processing.fullysamp_recon import reconstruct_fullysamp

    dat = _first(sample_scan_dir, "*band_se_fully_sampled*.dat")
    result = reconstruct_fullysamp(dat, use_sense=False, tse=False)

    img = result["image_rss"]
    assert img.ndim == 2 and min(img.shape) >= 64
    assert np.all(np.isfinite(img))
    assert img.max() > 0
    assert result["combination"] == "rss"


def test_equispaced_grappa_with_reference(sample_scan_dir):
    from processing.equispaced_recon import reconstruct_equispaced

    dat = _first(sample_scan_dir, "*equispaced_R4*.dat")
    ref = _first(sample_scan_dir, "*band_se_fully_sampled*.dat")
    json_matches = sorted(sample_scan_dir.glob("Undersampled4equispace*.json"))
    json_path = str(json_matches[0]) if json_matches else None

    result = reconstruct_equispaced(
        dat, method="grappa", json_path=json_path,
        acs_lines=24, r_factor=4, kernel_size=(5, 5), lamda=0.01,
        reference_dat_path=ref, tse=False,
    )

    img = result["grappa_image"]
    assert img.ndim == 2 and np.all(np.isfinite(img)) and img.max() > 0
    # Undersampled at ~R4 (ACS lowers the effective factor).
    assert 2.0 < result["actual_r"] <= 4.5
    assert result["method"] == "grappa"
    # A reference was supplied → finite comparison metrics.
    assert result["reference_image"] is not None
    assert np.isfinite(result["rmse"])


def test_pulseq_oversampled_readout_is_cropped(sample_scan_dir):
    """Pulseq scan acquired with 640 readout samples for a 320 matrix. Its header
    claims no oversampling (interpreter placeholder), so the factor must be
    inferred from the data and the image must come out 320 x 320."""
    from processing.fullysamp_recon import reconstruct_fullysamp

    folder = (sample_scan_dir.parent.parent / "subject_10000_07_06_26"
              / "200012_oversampling" / "dat_files")
    matches = sorted(folder.glob("*band_se_fully_sampled*.dat"))
    if not matches:
        pytest.skip(f"oversampled Pulseq scan not available in {folder}")

    result = reconstruct_fullysamp(str(matches[0]), use_sense=False, tse=False)
    assert result["image_rss"].shape == (320, 320)
