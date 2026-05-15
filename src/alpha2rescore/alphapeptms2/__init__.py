"""alphapeptms2 - AlphaPeptDeep MS2 prediction with MS2PIP-compatible API."""

from . import constants, core, result, spectrum
from .core import correlate, predict_batch, predict_single
from .result import ProcessingResult, calculate_correlations

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
