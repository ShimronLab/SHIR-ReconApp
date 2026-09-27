"""PE-mask builders in equispaced_recon: analytical + JSON-derived."""

import json

import numpy as np
import pytest

from processing.equispaced_recon import (
    _build_grappa_mask_computed,
    _build_pe_mask_from_json,
)


def _quiet(_msg):
    pass


def test_computed_mask_matches_spec():
    n, R, acs = 256, 4, 24
    mask, idx, acs_start, acs_end = _build_grappa_mask_computed(n, R, acs, _quiet)
    c = n // 2
    assert mask[c]                                   # centre sampled
    assert mask[acs_start:acs_end].all()             # ACS block fully sampled
    assert acs_end - acs_start == acs
    right = idx[idx >= c + acs]
    assert np.all(np.diff(right) == R)               # equispaced step outside ACS


def _write_equispaced_json(path, n_phase, R, acs, n_read=128, delta=1.0):
    """Synthesize an Undersampled JSON matching the analytical R/ACS mask.

    Stored as {"0": [[kx...],[ky...]]} (coord_dim, N_samples) — the format
    load_trajectory expects. ky carries the PE pattern; kx is the dense readout.
    n_read must exceed the number of acquired PE lines so the builder's
    "fewer-unique-values = PE axis" heuristic correctly identifies ky as PE.
    """
    mask, idx, _, _ = _build_grappa_mask_computed(n_phase, R, acs, _quiet)
    c = n_phase // 2
    kx_all, ky_all = [], []
    read_line = np.linspace(-0.5, 0.5, n_read, endpoint=False)
    for pe in idx:
        ky = (int(pe) - c) * delta
        kx_all.extend(read_line.tolist())
        ky_all.extend([ky] * n_read)
    with open(path, "w") as f:
        json.dump({"0": [kx_all, ky_all]}, f)
    return mask


def test_json_mask_matches_analytical(tmp_path):
    n, R, acs = 256, 4, 24
    jp = tmp_path / "Undersampled4equispaces.json"
    expected = _write_equispaced_json(jp, n, R, acs)

    mask = _build_pe_mask_from_json(str(jp), n, _quiet)
    assert mask is not None
    assert mask.dtype == bool
    assert np.array_equal(mask, expected)


def test_json_mask_bad_file_returns_none(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json ]")
    assert _build_pe_mask_from_json(str(bad), 256, _quiet) is None
