"""Lock the wire-format invariants of processing.constants."""

import numpy as np

from processing import constants
from processing.utils import ScanType


def test_scan_constants_match_enum_names():
    """SCAN_* values are the exact strings carried in job/result dicts —
    they MUST equal ScanType.<X>.name forever."""
    for member in ScanType:
        assert getattr(constants, f"SCAN_{member.name}") == member.name


def test_method_labels_cover_ui_methods():
    """Every method key offered by the params panel has a display label."""
    ui_method_keys = {
        "grappa", "grappa_sense", "sense", "sense_cs",
        "espirit", "espirit_l1", "espirit_l2", "espirit_tv", "espirit_l1wav",
        "none", "ggrappa",
    }
    assert ui_method_keys <= set(constants.METHOD_LABELS)


def test_method_label_engine_suffix():
    assert constants.method_label("sense") == "Pure SENSE"
    assert constants.method_label("sense", engine="BART") == "Pure SENSE (BART)"
    assert constants.method_label("unknown_key") == "unknown_key"


def test_method_display_label_uses_engine_only_for_bart_family():
    r = {"recon_engine": "sigpy"}
    assert constants.method_display_label(r, "espirit_l1") == "ESPIRiT+L1 (sigpy)"
    assert constants.method_display_label(r, "grappa") == "GRAPPA"   # pygrappa — no engine
    assert constants.method_display_label({}, "sense") == "Pure SENSE"


def test_jobaborted_is_baseexception_not_exception():
    assert issubclass(constants.JobAborted, BaseException)
    assert not issubclass(constants.JobAborted, Exception)
