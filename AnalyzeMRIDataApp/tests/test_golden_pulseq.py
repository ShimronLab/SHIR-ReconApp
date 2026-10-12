"""Golden-output regression: real Pulseq scans must reconstruct exactly as in
the captured baseline (see tools/golden.py).

Skipped unless SHIR_GOLDEN_DIR points at a baseline captured with
`tools/golden.py capture`. Runs the fast subset by default; set
SHIR_GOLDEN_FULL=1 to replay every case (slow: real TWIX files, both engines).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.golden, pytest.mark.realdata]

GOLDEN_DIR = os.environ.get("SHIR_GOLDEN_DIR")
REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "AnalyzeMRIDataApp" / "tools" / "golden.py"


@pytest.mark.skipif(not GOLDEN_DIR or not (Path(GOLDEN_DIR) / "index.json").is_file(),
                    reason="SHIR_GOLDEN_DIR not set to a captured baseline")
def test_pulseq_outputs_match_golden(tmp_path):
    cmd = [sys.executable, str(TOOL), "compare", "--app-root", str(REPO_ROOT),
           "--baseline", GOLDEN_DIR, "--report", str(tmp_path / "report.json")]
    if os.environ.get("SHIR_GOLDEN_FULL") != "1":
        cmd.append("--fast")
    rtol = os.environ.get("SHIR_GOLDEN_RTOL")
    if rtol:
        cmd += ["--rtol", rtol]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    summary = [line for line in proc.stdout.splitlines()
               if line.startswith("summary:") or " FAIL " in line]
    assert proc.returncode == 0, "\n".join(summary) or proc.stdout[-3000:] + proc.stderr[-3000:]
