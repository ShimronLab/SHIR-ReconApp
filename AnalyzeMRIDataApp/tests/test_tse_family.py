"""Sequence family from the TWIX header, and the two decisions that depend on it:
TSE line re-ordering (Pulseq only) and the readout oversampling factor
(trusted from the header for Siemens sequences only)."""

import numpy as np
import pytest

import processing.utils as utils
from processing.tse_reorder import detect_tse_params

SIEMENS_TSE = {"Config": {"SequenceFileName": "%SiemensSeq%\\tse"},
               "Meas": {"flReadoutOSFactor": 2.0}}
PULSEQ = {"Config": {"SequenceFileName": "%CustomerSeq%\\pulseq"},
          "Meas": {"flReadoutOSFactor": 1.0}}       # placeholder, whatever the .seq did


@pytest.fixture
def header(monkeypatch):
    """header(hdr) makes every read_twix_header call return `hdr`."""
    def _set(hdr):
        monkeypatch.setattr(utils, "read_twix_header", lambda _path: hdr)
    return _set


@pytest.mark.parametrize("hdr,family", [
    (SIEMENS_TSE, "siemens"),
    (PULSEQ, "pulseq"),
    ({"Dicom": {"tSequenceFileName": "%SiemensSeq%\\se"}}, "siemens"),   # fallback key
    ({"Config": {}}, None),                                              # no name
    (None, None),                                                        # unreadable
])
def test_sequence_family(header, hdr, family):
    header(hdr)
    assert utils.sequence_family("scan.dat") == family


def test_unreadable_file_has_no_family():
    assert utils.sequence_family("/no/such/file.dat") is None


@pytest.mark.parametrize("tse", [None, True])
def test_siemens_tse_is_never_reordered(header, tse):
    """Product TSE lines are already in true PE order: re-ordering them with the
    Pulseq echo plan would scramble the image — even if TSE is forced on."""
    header(SIEMENS_TSE)
    assert detect_tse_params("meas_t1_tse_cor_384.dat", tse=tse) is None


@pytest.mark.parametrize("hdr", [PULSEQ, None])
def test_pulseq_or_unknown_tse_keeps_filename_rule(header, hdr):
    header(hdr)
    params = detect_tse_params("meas_fully_sampled_tse_t2.dat")
    assert params is not None and params["necho"] == 16


@pytest.mark.parametrize("hdr,expected", [
    (SIEMENS_TSE, 2),
    ({"Config": {"SequenceFileName": "%SiemensSeq%\\se"}, "Meas": {"flReadoutOSFactor": 1.0}}, 1),
    ({"Config": {"SequenceFileName": "%SiemensSeq%\\se"}, "Dicom": {"flReadoutOSFactor": 2.0}}, 2),
    ({"Config": {"SequenceFileName": "%SiemensSeq%\\se"}}, None),   # no OS field
    (PULSEQ, None),                                                 # placeholder ignored
    (None, None),
])
def test_readout_os_factor_from_header(header, hdr, expected):
    header(hdr)
    assert utils.readout_os_factor_from_header("scan.dat") == expected


def test_pulseq_oversampled_readout_still_cropped(header):
    """Pulseq header says OS=1 while the data has 640 samples for 320 lines:
    the factor must come from the data (640/320 = 2), not the placeholder."""
    header(PULSEQ)
    k = np.ones((320, 2, 640), dtype=np.complex64)
    out = utils.crop_readout_oversampling(k, expected_nx=320, readout_axis=2,
                                          dat_path="scan.dat")
    assert out.shape == (320, 2, 320)


def test_siemens_header_factor_beats_line_count(header):
    """Siemens: 768 samples with only 235 PE lines (partial Fourier) is 2x OS,
    not 'factor 3' from 768/235."""
    header(SIEMENS_TSE)
    k = np.ones((235, 2, 768), dtype=np.complex64)
    out = utils.crop_readout_oversampling(k, expected_nx=235, readout_axis=2,
                                          dat_path="scan.dat")
    assert out.shape == (235, 2, 384)
