"""
Golden-output regression on real scans.

Runs real TWIX scans through ReconWorker exactly as the GUI does and stores the
outputs, so another app version can be checked against them:

  capture  build the case list from the scan folders (with the same helpers the
           folder panel and run button use), run every case, save arrays and
           index.json.
  compare  replay the stored cases (same jobs, same parameters) with another
           app checkout and diff the outputs.

Run under WSL in the mri_toolkit env, for example:

  python tools/golden.py capture --app-root <baseline checkout> \
      --out <golden dir>/fdad7a9
  python tools/golden.py compare --app-root <repo root> \
      --baseline <golden dir>/fdad7a9 --fast

Cases from Pulseq scans must match the baseline (exactly, or within --rtol).
Cases from Siemens product scans are reported but not failed: their geometry is
expected to change once it is read from the header.
"""

import os

# Before numpy / Qt are imported: headless, and single-threaded so BART and
# BLAS give reproducible results.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MRI_BART_BIN", "/opt/bart/bart")
os.environ.setdefault("TOOLBOX_PATH", "/opt/bart")
os.environ["PATH"] = "/opt/bart:" + os.environ.get("PATH", "")

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SCANS = Path("/path/to/"
             "MRI_Acquisition_Reconstruction_Toolkit/scans")

# dat_files folders to cover ("demo" duplicates SE_1mm and is left out).
FOLDERS = [
    "18_5_26/BrainPhantom_scan_200011/dat_files",
    "18_5_26/subject_100000_scan_200010/dat_files",
    "19_7_26/subject_100000_scan_200016/dat_files",
    "1_7_26/Phantom/Water_Ball/SE/dat_files",
    "1_7_26/Phantom/Water_Ball/TSE/dat_files",
    "1_7_26/Phantom/Water_With_Oil/SE/dat_files",
    "1_7_26/Phantom/Water_With_Oil/SE_TSE/dat_files",
    "1_7_26/Phantom/Water_With_Oil/TSE/dat_files",
    "SE_1mm/dat_files",
    "SE_3mm/dat_files",
    "subject_100000_scan_200009/A_P/dat_files",
    "subject_100000_scan_200009/L_R/dat_files",
    "subject_100008/SE/dat_files",
    "subject_100008/TSE/dat_files",
    "subject_10000_07_06_26/200012_oversampling/dat_files",
    "subject_10000_07_06_26/200013_regular_sampling/dat_files",
    "TSE_1mm/dat_files",
    "TSE_3mm/dat_files",
]

# Undersampling-simulator cases: fully-sampled file (glob within the folder)
# and the simulator variants to run on it.
_GRAPPA = lambda r, fast=False: {"name": f"retro_equi_R{r}_grappa", "fast": fast,
                                 "engines": ["sigpy"],
                                 "params": {"sim_mask_type": "equispaced", "sim_R": float(r),
                                            "sim_recon_method": "grappa"}}
_SENSE_R4 = {"name": "retro_equi_R4_sense", "fast": False, "engines": ["sigpy", "bart"],
             "params": {"sim_mask_type": "equispaced", "sim_R": 4.0,
                        "sim_recon_method": "sense"}}
_VD_R4 = {"name": "retro_vd_R4_none_seed0", "fast": False, "engines": ["sigpy"],
          "params": {"sim_mask_type": "vd", "sim_R": 4.0, "sim_recon_method": "none",
                     "sim_rng_seed": 0}}
RETRO = [
    ("19_7_26/subject_100000_scan_200016/dat_files", "*_fully_sampled_tse_t2.dat",
     [_GRAPPA(2), _GRAPPA(3), _GRAPPA(4, fast=True), _SENSE_R4, _VD_R4]),
    ("SE_3mm/dat_files", "*fully_sampled*.dat", [_GRAPPA(4), _SENSE_R4]),
    ("subject_100008/SE/dat_files", "*fully_sampled*.dat", [_GRAPPA(4), _SENSE_R4]),
    ("subject_10000_07_06_26/200012_oversampling/dat_files", "*fully_sampled*.dat",
     [_GRAPPA(4, fast=True)]),
    ("18_5_26/subject_100000_scan_200010/dat_files", "*_band_se_fully_sampled*.dat",
     [_GRAPPA(4)]),
    ("1_7_26/Phantom/Water_Ball/TSE/dat_files", "*fully_sampled*.dat", [_GRAPPA(2)]),
]

# Per-coil arrays are large; only their hash and summary statistics are kept.
PER_COIL_KEYS = {"sens_maps", "imgs_per_coil"}
# Keys that legitimately change with labels/paths and are not compared.
IGNORED_KEYS = {"label", "dat_path", "folder_key", "recon_name", "reference_name"}


# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

_QT_APP = None   # module-level so the QApplication is never garbage-collected


def setup_app(app_root: Path) -> None:
    """Put the given checkout first on sys.path and start a headless Qt app."""
    global _QT_APP
    app_dir = app_root / "AnalyzeMRIDataApp"
    for p in (str(app_root / "read_siemens_scans"), str(app_dir)):
        sys.path.insert(0, p)
    from PyQt5.QtCore import QSettings
    from PyQt5.QtWidgets import QApplication
    # Never read or write the user's real settings.
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, tempfile.mkdtemp())
    _QT_APP = QApplication.instance() or QApplication([])
    import processing
    loaded_from = Path(processing.__file__).resolve().parent.parent
    if loaded_from != app_dir.resolve():
        raise RuntimeError(f"processing imported from {loaded_from}, expected {app_dir}")


def default_params() -> dict:
    from ui.params_panel import ParamsPanel
    return ParamsPanel().get_params()


def git_rev(app_root: Path) -> str:
    def git(*args):
        return subprocess.run(["git", "-C", str(app_root), *args],
                              capture_output=True, text=True).stdout.strip()
    rev = git("rev-parse", "--short", "HEAD")
    if rev:
        return rev + ("+dirty" if git("status", "--porcelain", "--untracked-files=no") else "")
    # A worktree created from Windows stores a Windows gitdir path that WSL's
    # git cannot follow; read the detached HEAD directly instead.
    try:
        gitdir = (app_root / ".git").read_text().split("gitdir:", 1)[1].strip()
        gitdir = re.sub(r"^([A-Za-z]):", lambda m: f"/mnt/{m.group(1).lower()}", gitdir)
        head = (Path(gitdir) / "HEAD").read_text().strip()
        return head[:7] + " (worktree)"
    except (OSError, IndexError):
        return "unknown"


def versions() -> dict:
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for pkg in ("numpy", "scipy", "sigpy", "pygrappa", "twixtools", "scikit-image"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    try:
        out["bart"] = subprocess.run([os.environ["MRI_BART_BIN"], "version"],
                                     capture_output=True, text=True).stdout.strip()
    except OSError:
        out["bart"] = None
    return out


def sequence_family(dat_path: str) -> str:
    """'pulseq' / 'product' from the header's sequence file name (independent of app code)."""
    import twixtools
    try:
        hdr = twixtools.read_twix(dat_path, parse_data=False, verbose=False)[-1]["hdr"]
        name = str(hdr["Config"]["SequenceFileName"]).lower()
    except Exception:
        return "unknown"
    return "pulseq" if "pulseq" in name else "product"


# ─────────────────────────────────────────────────────────────────────────────
# Case list
# ─────────────────────────────────────────────────────────────────────────────

def build_cases() -> list:
    """Jobs built the way the folder panel + Run button build them."""
    from processing.utils import (ScanType, auto_detect_scan_type, extract_band_index,
                                  find_generic_traj_json)
    from ui.folder_panel import _find_equispaced_json, _find_json
    from ui.main_window import _attach_references

    cases = []
    for rel in FOLDERS:
        folder = SCANS / rel
        folder_key = rel.replace("/dat_files", "")
        jobs = []
        for f in sorted(folder.glob("*.dat")):
            stype = auto_detect_scan_type(f.name)
            bidx = extract_band_index(f.name)
            if stype == ScanType.BAND and bidx != 0:
                continue                                   # band 0 stands for all bands
            if stype == ScanType.EQUISPACED:
                json_path = _find_equispaced_json(folder, f.name)
            elif stype == ScanType.UNKNOWN:
                json_path = find_generic_traj_json(str(f))
            else:
                json_path = _find_json(folder, stype)
            jobs.append({"dat_path": str(f), "scan_type": stype.name,
                         "_detected_type": stype.name, "json_path": json_path,
                         "band_idx": bidx, "folder_key": folder_key})
        jobs = _attach_references(jobs)

        equi = [j for j in jobs if j["scan_type"] == "EQUISPACED"]
        sense_job = next((j for j in equi if "_R4_" in Path(j["dat_path"]).name), equi[0] if equi else None)
        for job in jobs:
            stype = job["scan_type"]
            both = stype in ("POISSON", "BAND")            # engine-dependent recons
            fast = stype == "FULLY_SAMPLED" or (stype == "EQUISPACED" and job is (equi[0] if equi else None))
            add_case(cases, job, "default", {}, ["sigpy", "bart"] if both else ["sigpy"], fast)
            if job is sense_job:
                add_case(cases, job, "equi_sense", {"equi_method": "sense"}, ["sigpy", "bart"], False)

    for rel, pattern, variants in RETRO:
        folder = SCANS / rel
        matches = sorted(folder.glob(pattern))
        if not matches:
            print(f"  [skip] no {pattern} in {rel}")
            continue
        job = {"dat_path": str(matches[0]), "scan_type": "RETRO_UNDERSAMPLE",
               "_detected_type": "FULLY_SAMPLED", "json_path": None, "band_idx": None,
               "folder_key": rel.replace("/dat_files", "")}
        for v in variants:
            add_case(cases, job, v["name"], v["params"], v["engines"], v["fast"])
    return cases


def add_case(cases, job, variant, params, engines, fast):
    family = sequence_family(job["dat_path"])
    for engine in engines:
        stem = re.sub(r"^meas_MID\d+_FID\d+_", "", Path(job["dat_path"]).stem)
        cases.append({
            "id": f"{job['folder_key']}|{stem}|{variant}|{engine}",
            "job": dict(job), "params": dict(params), "engine": engine,
            "fast": bool(fast and engine == "sigpy"), "family": family,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Running and summarising one case
# ─────────────────────────────────────────────────────────────────────────────

def run_case(case: dict, base_params: dict):
    """Run one case through ReconWorker synchronously; return (result, error, logs)."""
    from ui.main_window import ReconWorker
    params = dict(base_params)
    params.update(case["params"])
    params["recon_backend"] = case["engine"]
    worker = ReconWorker(dict(case["job"]), params)
    out, logs = {}, []
    worker.progress_msg.connect(logs.append)
    worker.result_ready.connect(lambda r: out.setdefault("result", r))
    worker.error_signal.connect(lambda m: out.setdefault("error", m))
    # Unattended answers to the dialogs the GUI would show.
    worker.slice_selection_needed.connect(lambda n: worker.set_slice_choice(n // 2))
    worker.mask_fallback_needed.connect(lambda: worker.set_mask_choice("analytical"))
    worker.run()
    return out.get("result"), out.get("error"), logs


def _sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()[:20]


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return {"__array__": list(v.shape), "dtype": v.dtype.str, "sha": _sha(v)}
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return repr(v)


def summarise(result: dict):
    """Split a result dict into stored arrays and JSON metadata."""
    arrays, meta = {}, {"arrays": {}, "per_coil": {}, "scalars": {}}
    for key, val in result.items():
        if key in IGNORED_KEYS:
            continue
        if isinstance(val, np.ndarray) and val.ndim >= 1:
            if key in PER_COIL_KEYS:
                mag = np.abs(val).astype(np.float64)
                meta["per_coil"][key] = {"shape": list(val.shape), "sha": _sha(val),
                                         "mean": float(mag.mean()), "std": float(mag.std())}
            else:
                arrays[key] = val
                meta["arrays"][key] = {"shape": list(val.shape), "dtype": val.dtype.str,
                                       "sha": _sha(val)}
        else:
            meta["scalars"][key] = _jsonable(val)
    return arrays, meta


# ─────────────────────────────────────────────────────────────────────────────
# capture / compare
# ─────────────────────────────────────────────────────────────────────────────

def cmd_capture(args) -> int:
    app_root = Path(args.app_root).resolve()
    setup_app(app_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base_params = default_params()
    cases = build_cases()
    if args.filter:
        cases = [c for c in cases if re.search(args.filter, c["id"])]
    print(f"capture: {len(cases)} cases from {app_root} ({git_rev(app_root)})", flush=True)

    index = {"app_root": str(app_root), "git_rev": git_rev(app_root), "versions": versions(),
             "created": time.strftime("%Y-%m-%d %H:%M:%S"), "base_params": _jsonable(base_params),
             "cases": []}
    for i, case in enumerate(cases):
        t0 = time.time()
        result, error, logs = run_case(case, base_params)
        entry = dict(case, file=f"case_{i:03d}.npz", logs=logs,
                     seconds=round(time.time() - t0, 1))
        if result is None:
            entry["error"] = (error or "no result").strip().splitlines()[0]
        else:
            arrays, meta = summarise(result)
            entry.update(meta)
            np.savez_compressed(out / entry["file"], **arrays)
        index["cases"].append(entry)
        status = "ERROR " + entry["error"] if "error" in entry else "ok"
        print(f"[{i + 1}/{len(cases)}] {case['id']}  {status}  ({entry['seconds']} s)", flush=True)
        (out / "index.json").write_text(json.dumps(index, indent=1))
    return 0


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def compare_case(entry: dict, base_dir: Path, result, error, rtol: float):
    """Return (status, details) for one replayed case."""
    if "error" in entry or result is None:
        now_err = None if result is not None else (error or "no result").strip().splitlines()[0]
        if "error" in entry and now_err == entry["error"]:
            return "SAME-ERROR", entry["error"]
        return "FAIL", f"error changed: baseline={entry.get('error')!r} now={now_err!r}"

    arrays, meta = summarise(result)
    problems, worst = [], 0.0
    stored = np.load(base_dir / entry["file"])
    for key, info in entry["arrays"].items():
        if key not in arrays:
            problems.append(f"{key} missing")
            continue
        if list(arrays[key].shape) != info["shape"]:
            problems.append(f"{key} shape {info['shape']}→{list(arrays[key].shape)}")
            continue
        if meta["arrays"][key]["sha"] != info["sha"]:
            err = _rel_err(arrays[key], stored[key])
            worst = max(worst, err)
            if err > rtol:
                problems.append(f"{key} rel.err {err:.2e}")
    for key, info in entry["per_coil"].items():
        now = meta["per_coil"].get(key)
        if now is None or now["shape"] != info["shape"]:
            problems.append(f"{key} per-coil shape changed")
        elif now["sha"] != info["sha"]:
            err = abs(now["mean"] - info["mean"]) / max(abs(info["mean"]), 1e-30)
            worst = max(worst, err)
            if err > max(rtol, 1e-12):
                problems.append(f"{key} per-coil mean rel.err {err:.2e}")
    for key, val in entry["scalars"].items():
        now = meta["scalars"].get(key, "<missing>")
        if isinstance(val, float) and isinstance(now, float):
            if abs(now - val) > rtol * max(abs(val), 1e-12) and not (np.isnan(val) and np.isnan(now)):
                problems.append(f"{key} {val:.6g}→{now:.6g}")
        elif _norm_scalar(now) != _norm_scalar(val):
            problems.append(f"{key} {str(val)[:40]}→{str(now)[:40]}")
    added = sorted(set(meta["arrays"]) - set(entry["arrays"]))
    if problems:
        return "FAIL", "; ".join(problems)
    if worst:
        return "PASS~", f"max rel.err {worst:.2e}" + (f"; added {added}" if added else "")
    return "PASS", f"added {added}" if added else ""


def _norm_scalar(v):
    """Compare nested scalars/dicts ignoring dict ordering."""
    return json.dumps(v, sort_keys=True, default=str)


def cmd_compare(args) -> int:
    app_root = Path(args.app_root).resolve()
    setup_app(app_root)
    base_dir = Path(args.baseline)
    index = json.loads((base_dir / "index.json").read_text())
    entries = index["cases"]
    if args.fast:
        entries = [e for e in entries if e["fast"]]
    if args.filter:
        entries = [e for e in entries if re.search(args.filter, e["id"])]
    base_params = default_params()
    base_params.update(index["base_params"])      # same parameters as the capture
    print(f"compare: {len(entries)} cases, {app_root} ({git_rev(app_root)}) "
          f"vs baseline {index['git_rev']}", flush=True)

    report, failed = [], 0
    for i, entry in enumerate(entries):
        result, error, logs = run_case(entry, base_params)
        status, details = compare_case(entry, base_dir, result, error, args.rtol)
        if status == "FAIL" and entry["family"] != "pulseq":
            status = "CHANGED"                    # product geometry is expected to change
        failed += status == "FAIL"
        seen = set(entry["logs"])
        new_logs = [m for m in logs if m not in seen]
        report.append({"id": entry["id"], "family": entry["family"], "status": status,
                       "details": details, "new_log_lines": new_logs[:20]})
        print(f"[{i + 1}/{len(entries)}] {status:10s} {entry['id']}  {details}", flush=True)

    counts = {s: sum(r["status"] == s for r in report) for s in
              ("PASS", "PASS~", "SAME-ERROR", "CHANGED", "FAIL")}
    print("summary:", ", ".join(f"{k}={v}" for k, v in counts.items()), flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps({"baseline": index["git_rev"],
                                                 "current": git_rev(app_root),
                                                 "counts": counts, "cases": report}, indent=1))
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--app-root", required=True, help="repo root of the checkout to run")
    cap.add_argument("--out", required=True)
    cap.add_argument("--filter", help="regex on case id")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--app-root", required=True)
    cmp_.add_argument("--baseline", required=True)
    cmp_.add_argument("--fast", action="store_true", help="only the fast subset")
    cmp_.add_argument("--filter", help="regex on case id")
    cmp_.add_argument("--rtol", type=float, default=0.0,
                      help="relative tolerance for arrays/floats (0 = exact)")
    cmp_.add_argument("--report", help="write a JSON report here")
    args = ap.parse_args()
    return cmd_capture(args) if args.cmd == "capture" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
