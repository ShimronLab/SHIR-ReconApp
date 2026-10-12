"""
Parity report: for every Pulseq scan, does the played .seq agree with the
geometry SHIR derives today (line count, line placement and order, readout
oversampling)?

  python tools/seq_parity.py [--pypulseq DIR] [--out report.json] [--filter REGEX]

processing.seq_geometry runs in shadow mode until this report shows every
scan matching; only then should Pulseq geometry come from the .seq (switching
would otherwise change existing results).
"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import argparse
import json
import re
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
for p in (str(APP_DIR), str(APP_DIR.parent / "read_siemens_scans"), str(Path(__file__).parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from golden import FOLDERS, SCANS            # the same scan folders as the golden set


def legacy_fully_sampled(dat):
    """Rows the fully-sampled legacy path puts each acquired line on."""
    from read_siemens_scans.twix_reading import read_raw_twix_data
    from processing.tse_reorder import detect_tse_params, tse_acq_to_line
    from processing.utils import select_slice
    k, _ = read_raw_twix_data(dat, fix_oversampling=False)
    k, _ = select_slice(k)
    n_acq, _, ncol = k.shape
    tse = detect_tse_params(dat)
    rows = np.arange(n_acq)
    if tse is not None and n_acq % tse["necho"] == 0:
        plan = tse_acq_to_line(n_acq, tse["necho"], tse["TE1"], tse["TEeff"])
        if len(plan) == n_acq and np.unique(plan).size == n_acq:
            rows = plan
    return rows, n_acq, ncol, round(ncol / n_acq)


def legacy_equispaced(dat, ref, json_path):
    """Rows the equispaced legacy geometry puts each acquired line on."""
    from processing.equispaced_recon import _legacy_geometry
    from processing.tse_reorder import detect_tse_params, tse_acq_to_line
    from read_siemens_scans.twix_reading import read_raw_twix_data
    raw, _ = read_raw_twix_data(dat, fix_oversampling=False)
    ncol = raw.shape[-1]
    (full_kspace, _calib, pe_mask, _a, _b, _r, n_phase, _sl, _rc, _mm, _rf,
     tse) = _legacy_geometry(
        dat, json_path=json_path, mask_method="analytical", acs_lines=24, r_factor=4,
        tse=None, necho=None, te1=None, teeff=None, reference_dat_path=ref,
        slice_selector_cb=None, mask_fallback_cb=lambda: "analytical", log=lambda m: None)
    rows = np.where(pe_mask)[0]
    tse = detect_tse_params(dat)
    if tse is not None and n_phase % tse["necho"] == 0:
        plan = tse_acq_to_line(n_phase, tse["necho"], tse["TE1"], tse["TEeff"], pe_mask=pe_mask)
        if len(plan) == rows.size:
            rows = plan
    return rows, n_phase, ncol, round(ncol / full_kspace.shape[0]) if full_kspace.shape[0] else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pypulseq", help="extra folder holding pypulseq (if not installed)")
    ap.add_argument("--out", help="write the report as JSON")
    ap.add_argument("--filter", help="regex on '<folder>/<file>'")
    args = ap.parse_args()
    if args.pypulseq:
        sys.path.insert(0, args.pypulseq)

    from processing.seq_geometry import played_seq, seq_line_geometry
    from processing.utils import ScanType, auto_detect_scan_type, sequence_family
    from ui.folder_panel import _find_equispaced_json
    from ui.main_window import _attach_references

    report = []
    for rel in FOLDERS:
        folder = SCANS / rel
        jobs = [{"dat_path": str(f), "scan_type": auto_detect_scan_type(f.name).name}
                for f in sorted(folder.glob("*.dat"))]
        jobs = _attach_references(jobs)
        for job in jobs:
            dat, stype = job["dat_path"], job["scan_type"]
            name = f"{rel.replace('/dat_files', '')}/{Path(dat).name}"
            if stype not in ("FULLY_SAMPLED", "EQUISPACED") or sequence_family(dat) != "pulseq":
                continue
            if args.filter and not re.search(args.filter, name):
                continue
            row = {"scan": name, "type": stype}
            info = played_seq(dat)
            row["seq"] = Path(info["path"]).name if info["path"] else None
            row["seq_note"] = info["note"] or "exact signature match"
            if not info["path"]:
                report.append(row)
                print(f"NO-SEQ   {name}: {row['seq_note']} ({info['name']})", flush=True)
                continue
            geo = seq_line_geometry(info["path"])
            if stype == "FULLY_SAMPLED":
                rows, n_phase, ncol, legacy_os = legacy_fully_sampled(dat)
            else:
                rows, n_phase, ncol, legacy_os = legacy_equispaced(
                    dat, job.get("reference_dat_path"),
                    _find_equispaced_json(Path(dat).parent, Path(dat).name))
            seq_rows = geo["pe_index"] + n_phase // 2
            # A phase-encode direction reversed relative to the legacy placement
            # shows up as seq_rows == (n_phase - 1 - rows) + constant shift.
            shift = (seq_rows - (n_phase - 1 - rows)) if seq_rows.size == rows.size else None
            if seq_rows.size != rows.size:
                verdict = f"LINE COUNT .seq {seq_rows.size} vs legacy {rows.size}"
            elif np.array_equal(seq_rows, rows):
                verdict = "MATCH"
            elif shift is not None and np.unique(shift).size == 1:
                verdict = (f"REVERSED PE DIRECTION (mirror image, {int(shift[0]):+d}-line shift)")
            elif np.array_equal(np.sort(seq_rows), np.sort(rows)):
                verdict = "SAME LINES, DIFFERENT ORDER"
            else:
                verdict = f"{int(np.sum(seq_rows != rows))}/{rows.size} LINES DIFFER"
            os_ok = (legacy_os is not None and geo["readout_os"] is not None
                     and round(geo["readout_os"]) == legacy_os)
            row.update(verdict=verdict, n_phase_legacy=n_phase, n_pe_full_seq=geo["n_pe_full"],
                       readout_os_seq=round(geo["readout_os"], 3), readout_os_legacy=legacy_os,
                       samples_per_adc_seq=geo["samples_per_adc"], ncol=ncol,
                       fov_phase_mm=round(geo["fov_phase_mm"], 1), swapped=geo["swapped"])
            report.append(row)
            flag = "OK  " if verdict == "MATCH" and os_ok and geo["samples_per_adc"] == ncol else "DIFF"
            print(f"{flag}     {name}: {verdict}; readout OS .seq {geo['readout_os']:.2f} vs legacy "
                  f"{legacy_os}; samples {geo['samples_per_adc']} vs {ncol}; grid .seq "
                  f"{geo['n_pe_full']} vs legacy {n_phase}; FOV {geo['fov_phase_mm']:.0f} mm",
                  flush=True)

    counts = {}
    for r in report:
        key = "no .seq" if not r.get("seq") else ("match" if r["verdict"] == "MATCH" else "differs")
        counts[key] = counts.get(key, 0) + 1
    print("summary:", counts, flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
