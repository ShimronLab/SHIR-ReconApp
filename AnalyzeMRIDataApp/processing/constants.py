"""
Shared constants and control-flow exceptions for the processing + UI layers.
"""


class JobAborted(BaseException):
    """
    Raised inside a reconstruction job when the user pressed Stop.

    Derives from BaseException (NOT Exception) on purpose: the processing
    modules contain many broad ``except Exception`` fallback blocks that would
    otherwise swallow the abort and convert it into a degraded reconstruction.
    """
