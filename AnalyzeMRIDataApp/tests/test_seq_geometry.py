"""Finding the played Pulseq .seq by its signature, and the shadow-mode
comparison of the legacy line placement against it (processing.seq_geometry)."""

import numpy as np
import pytest

import processing.seq_geometry as sg
import processing.utils as utils

SIG = "af50264825562a207cc9cde76ef93e19"


def write_seq(path, signature):
    path.write_text("[VERSION]\nmajor 1\n[DEFINITIONS]\nTotalDuration 1\n\n"
                    f"[SIGNATURE]\nType md5\nHash {signature}\n")
    return path


@pytest.fixture
def pulseq_scan(tmp_path, monkeypatch):
    """A Pulseq scan in <scan>/dat_files/ whose header names tse.seq + SIG."""
    dat_dir = tmp_path / "scan" / "dat_files"
    dat_dir.mkdir(parents=True)
    dat = dat_dir / "meas_tse.dat"
    dat.write_bytes(b"")
    header = {"Config": {"SequenceFileName": "%CustomerSeq%\\pulseq"},
              "Meas": {"tFree": "Saar/tse.seq"}, "Dicom": {"tSequenceVariant": SIG}}
    monkeypatch.setattr(utils, "read_twix_header", lambda _p: header)
    sg._signature_index.cache_clear()
    return dat


def test_signature_read_from_file_tail(tmp_path):
    assert sg.seq_signature(str(write_seq(tmp_path / "a.seq", SIG))) == SIG
    (tmp_path / "b.seq").write_text("no signature here")
    assert sg.seq_signature(str(tmp_path / "b.seq")) is None


def test_played_seq_found_by_signature_in_seq_files(pulseq_scan):
    seq_dir = pulseq_scan.parent.parent / "seq_files"
    seq_dir.mkdir()
    write_seq(seq_dir / "renamed_on_disk.seq", SIG)          # name may differ, signature decides
    info = sg.played_seq(str(pulseq_scan))
    assert info["path"].endswith("renamed_on_disk.seq") and info["signature"] == SIG


def test_same_name_with_other_signature_is_not_used(pulseq_scan):
    seq_dir = pulseq_scan.parent.parent / "seq_files"
    seq_dir.mkdir()
    write_seq(seq_dir / "tse.seq", "0" * 32)                 # regenerated after the scan
    info = sg.played_seq(str(pulseq_scan))
    assert info["path"] is None and "not found" in info["note"]


def test_library_folder_is_searched(pulseq_scan, tmp_path):
    lib = tmp_path / "library" / "deep"
    lib.mkdir(parents=True)
    write_seq(lib / "x.seq", SIG)
    sg.set_library_dir(str(tmp_path / "library"))
    try:
        assert sg.played_seq(str(pulseq_scan))["path"].endswith("x.seq")
    finally:
        sg.set_library_dir(None)


@pytest.mark.parametrize("legacy,verdict", [
    (np.arange(8), "matches the played .seq exactly"),
    (7 - np.arange(8) - 1 + 1, "REVERSED phase-encode direction"),
    (np.roll(np.arange(8), 1), "same lines, different ORDER"),
    (np.arange(8) - 1, "shifted by +1 line(s)"),
    (np.arange(7), "line count differs"),
])
def test_compare_with_legacy_verdicts(monkeypatch, legacy, verdict):
    monkeypatch.setattr(sg, "played_seq", lambda _p: {"path": "x.seq", "signature": SIG,
                                                      "name": "x.seq", "note": ""})
    monkeypatch.setattr(sg, "seq_line_geometry", lambda _p: {
        "pe_index": np.arange(8) - 4, "readout_os": 1.0, "fov_phase_mm": 256.0})
    logs = []
    out = sg.compare_with_legacy("scan.dat", legacy, 8, logs.append)
    assert verdict in out and logs and logs[0].startswith("  [SEQ]")


class _FakeSeq:
    """Just enough of pypulseq.Sequence for seq_line_geometry: Cartesian lines
    at the given phase-encode k values (in units of Δk), 8 samples each."""
    def __init__(self, pe_units, d=4.0):
        self._pe = np.asarray(pe_units, float) * d
        self.block_events = list(range(len(self._pe)))

    def read(self, _path):
        pass

    def calculate_kspace(self):
        kx = np.tile(np.arange(-4, 4) * 2.0, len(self._pe))
        ky = np.repeat(self._pe, 8)
        return (np.stack([kx, ky, np.zeros_like(kx)]),)

    def get_block(self, _b):
        class _B:
            class adc:
                num_samples = 8
        return _B


@pytest.mark.parametrize("pe_units,half,expected", [
    (np.arange(-4, 4), False, np.arange(-4, 4)),                     # k = 0 sampled
    (np.arange(3.5, -4, -1), True, np.arange(3, -5, -1)),            # ±½…±3½, top-down
])
def test_seq_line_geometry_integer_and_half_line_grids(monkeypatch, pe_units, half, expected):
    import sys, types
    fake = types.SimpleNamespace(Sequence=lambda: _FakeSeq(pe_units))
    monkeypatch.setitem(sys.modules, "pypulseq", fake)
    monkeypatch.setattr(sg, "seq_signature", lambda _p: SIG)
    sg.seq_line_geometry.cache_clear()
    geo = sg.seq_line_geometry("x.seq")
    sg.seq_line_geometry.cache_clear()
    assert geo["pe_half_line"] is half
    np.testing.assert_array_equal(geo["pe_index"], expected)
    assert geo["n_pe_full"] == 8


def test_shadow_check_never_raises(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("broken .seq")
    monkeypatch.setattr(sg, "compare_with_legacy", boom)
    logs = []
    sg.shadow_check("scan.dat", np.arange(4), 4, logs.append)
    assert "cross-check skipped" in logs[0]


def test_non_pulseq_scan_has_no_played_seq(monkeypatch):
    monkeypatch.setattr(utils, "read_twix_header",
                        lambda _p: {"Config": {"SequenceFileName": "%SiemensSeq%\\se"}})
    assert sg.played_seq("x.dat")["path"] is None
