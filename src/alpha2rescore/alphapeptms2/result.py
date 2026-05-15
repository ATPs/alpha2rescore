"""ProcessingResult definition mirroring MS2PIP."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from psm_utils import PSM
from pydantic import BaseModel, ConfigDict


class ProcessingResult(BaseModel):
    """Result of processing a single PSM, aligned to ms2pip.result.ProcessingResult.

    theoretical_mz and predicted_intensity are dicts keyed by ion_type ("b", "y").
    Each value is a 2D ndarray of shape (n_fragment_positions, n_charges) where
    n_charges=3 for z1, z2, z3 (z3 is zero-filled for AlphaPeptDeep).
    """

    psm_index: int
    psm: Optional[PSM] = None
    theoretical_mz: Optional[Dict[str, np.ndarray]] = None
    predicted_intensity: Optional[Dict[str, np.ndarray]] = None
    observed_intensity: Optional[Dict[str, np.ndarray]] = None
    correlation: Optional[float] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


def calculate_correlations(results: list[ProcessingResult]) -> None:
    """Calculate Pearson correlations between predicted and observed intensities."""
    for result in results:
        if (
            result.predicted_intensity is None
            or result.observed_intensity is None
            or result.theoretical_mz is None
        ):
            result.correlation = None
            continue
        ion_types = [
            key
            for key in result.predicted_intensity
            if key in result.observed_intensity and key in result.theoretical_mz
        ]
        if not ion_types:
            result.correlation = 0.0
            continue
        pred = np.concatenate([result.predicted_intensity[k].flatten() for k in ion_types])
        obs = np.concatenate([result.observed_intensity[k].flatten() for k in ion_types])
        mz = np.concatenate([result.theoretical_mz[k].flatten() for k in ion_types])
        mask = (mz > 0) & np.isfinite(pred) & np.isfinite(obs)
        if mask.sum() < 2:
            result.correlation = 0.0
        elif np.std(pred[mask]) == 0 or np.std(obs[mask]) == 0:
            result.correlation = 0.0
        else:
            result.correlation = float(np.corrcoef(pred[mask], obs[mask])[0, 1])
