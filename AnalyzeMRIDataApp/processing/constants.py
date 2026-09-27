"""
Shared constants and control-flow exceptions for the processing + UI layers.

The scan-type and method strings below are the *wire format* of the app: they
travel inside plain job/result dicts through Qt signals, appear in saved file
names and PDFs, and must therefore never change value. Using these named
constants (instead of re-typing the literals in every module) only removes the
risk of a typo drifting one call site out of sync.
"""


class JobAborted(BaseException):
    """
    Raised inside a reconstruction job when the user pressed Stop.

    Derives from BaseException (NOT Exception) on purpose: the processing
    modules contain many broad ``except Exception`` fallback blocks that would
    otherwise swallow the abort and convert it into a degraded reconstruction.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Scan types — MUST equal processing.utils.ScanType.<X>.name (unit-tested).
# ─────────────────────────────────────────────────────────────────────────────

SCAN_BAND              = "BAND"
SCAN_FULLY_SAMPLED     = "FULLY_SAMPLED"
SCAN_EQUISPACED        = "EQUISPACED"
SCAN_POISSON           = "POISSON"
SCAN_RETRO_UNDERSAMPLE = "RETRO_UNDERSAMPLE"
SCAN_UNKNOWN           = "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction methods (equispaced pipeline + retro simulator)
# ─────────────────────────────────────────────────────────────────────────────

# Canonical display names, engine-neutral (the engine actually used — BART or
# sigpy — is decided per job by processing.recon_backend, so hardcoding
# "(BART)" into a method name would often be wrong).
METHOD_LABELS = {
    "none":          "ZF only",
    "grappa":        "GRAPPA",
    "ggrappa":       "gGRAPPA",
    "grappa_sense":  "GRAPPA+SENSE",
    "sense":         "Pure SENSE",
    "sense_cs":      "CS-SENSE",
    "espirit":       "ESPIRiT",
    "espirit_l1":    "ESPIRiT+L1",
    "espirit_l2":    "ESPIRiT+L2",
    "espirit_tv":    "ESPIRiT+TV",
    "espirit_l1wav": "ESPIRiT+L1-Wav",
}


def method_label(method: str, engine: str = None) -> str:
    """Display name for a method key, optionally suffixed with the engine
    that actually ran it (e.g. ``"ESPIRiT+L1 (sigpy)"``)."""
    label = METHOD_LABELS.get(method, method)
    return f"{label} ({engine})" if engine else label


def method_display_label(result: dict, method: str) -> str:
    """Method name for a result, engine-suffixed for the ESPIRiT/SENSE family
    (those run on the BART⇄sigpy engine recorded in ``result["recon_engine"]``)."""
    engine = result.get("recon_engine") if method in BART_FAMILY_METHODS else None
    return method_label(method, engine=engine)


# Methods that run on the BART⇄sigpy engine (ESPIRiT maps + PICS).
BART_FAMILY_METHODS = (
    "sense", "sense_cs",
    "espirit", "espirit_l1", "espirit_l2", "espirit_tv", "espirit_l1wav",
)

# Methods that take a regularisation weight λ.
REGULARIZED_METHODS = (
    "sense_cs", "espirit_l1", "espirit_l2", "espirit_tv", "espirit_l1wav",
)
