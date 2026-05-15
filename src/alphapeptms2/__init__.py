"""Compatibility wrappers for the bundled alphapeptms2 package."""

from . import constants, core, result, spectrum
from alpha2rescore.alphapeptms2 import (
    ProcessingResult,
    calculate_correlations,
    correlate,
    predict_batch,
    predict_single,
)

__version__ = "0.1.0"
__all__ = [
    "constants",
    "core",
    "result",
    "spectrum",
    "predict_single",
    "predict_batch",
    "correlate",
    "ProcessingResult",
    "calculate_correlations",
]
