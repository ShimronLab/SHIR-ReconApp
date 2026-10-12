"""
Print what SHIR reads from each TWIX file's own metadata: sequence family and,
for Siemens files, the full acquisition fact sheet (grid, partial Fourier,
acceleration, oversampling, calibration lines, pixel size) with sources.

  python tools/acq_report.py <folder or .dat> [...]   [--table]

Use it to check new data before reconstructing it. Pulseq files are listed
with their played .seq and signature only: their protocol fields are
placeholders, so geometry comes from the data / .seq instead.
"""

import argparse
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
for p in (str(APP_DIR), str(APP_DIR.parent / "read_siemens_scans")):
    if p not in sys.path:
        sys.path.insert(0, p)

from processing.acquisition import (AcquisitionConflict, UnsupportedAcquisition,  # noqa: E402
                                    acquire_cartesian)
from processing.utils import read_twix_header, sequence_family                    # noqa: E402


def report(dat: Path, show_table: bool) -> None:
    family = sequence_family(str(dat))
    print(f"== {dat.name}  [{family or 'unreadable header'}]")
    if family == "pulseq":
        hdr = read_twix_header(str(dat)) or {}
        played = (hdr.get("Meas", {}) or {}).get("tFree", "?")
        sig = (hdr.get("Dicom", {}) or {}).get("tSequenceVariant", "?")
        print(f"   played .seq: {played}   signature: {sig}")
        return
    if family != "siemens":
        return
    try:
        asm = acquire_cartesian(str(dat), slice_idx=None)
    except (UnsupportedAcquisition, AcquisitionConflict) as exc:
        print(f"   REFUSED: {exc}")
        return
    pixel = ("×".join(f"{v:.3f}" for v in asm.pixel_spacing_mm) + " mm"
             if asm.pixel_spacing_mm else "unknown")
    print(f"   {asm.facts.summary()}")
    print(f"   grid {asm.kspace.shape[0]}×{len(asm.line_category)} → image "
          f"{asm.kspace.shape[0]}×{asm.n_pe_image}, pixel {pixel}, "
          f"calibration rows {asm.calib_range}")
    for w in asm.facts.warnings:
        print(f"   WARNING: {w}")
    if show_table:
        for name, value, src, detail in asm.facts.table():
            print(f"     {name:20s} {str(value):24s} {src:20s} {detail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="folders (searched recursively) or .dat files")
    ap.add_argument("--table", action="store_true", help="print every fact with its source")
    args = ap.parse_args()
    for p in map(Path, args.paths):
        files = sorted(p.rglob("*.dat")) if p.is_dir() else [p]
        for dat in files:
            report(dat, args.table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
