"""AlphaPept spectral feature calculation."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata


FEATURE_NAMES = [
    "spec_pearson_norm",
    "ionb_pearson_norm",
    "iony_pearson_norm",
    "spec_mse_norm",
    "ionb_mse_norm",
    "iony_mse_norm",
    "min_abs_diff_norm",
    "max_abs_diff_norm",
    "abs_diff_Q1_norm",
    "abs_diff_Q2_norm",
    "abs_diff_Q3_norm",
    "mean_abs_diff_norm",
    "std_abs_diff_norm",
    "ionb_min_abs_diff_norm",
    "ionb_max_abs_diff_norm",
    "ionb_abs_diff_Q1_norm",
    "ionb_abs_diff_Q2_norm",
    "ionb_abs_diff_Q3_norm",
    "ionb_mean_abs_diff_norm",
    "ionb_std_abs_diff_norm",
    "iony_min_abs_diff_norm",
    "iony_max_abs_diff_norm",
    "iony_abs_diff_Q1_norm",
    "iony_abs_diff_Q2_norm",
    "iony_abs_diff_Q3_norm",
    "iony_mean_abs_diff_norm",
    "iony_std_abs_diff_norm",
    "dotprod_norm",
    "dotprod_ionb_norm",
    "dotprod_iony_norm",
    "cos_norm",
    "cos_ionb_norm",
    "cos_iony_norm",
    "spec_pearson",
    "ionb_pearson",
    "iony_pearson",
    "spec_spearman",
    "ionb_spearman",
    "iony_spearman",
    "spec_mse",
    "ionb_mse",
    "iony_mse",
    "min_abs_diff_iontype",
    "max_abs_diff_iontype",
    "min_abs_diff",
    "max_abs_diff",
    "abs_diff_Q1",
    "abs_diff_Q2",
    "abs_diff_Q3",
    "mean_abs_diff",
    "std_abs_diff",
    "ionb_min_abs_diff",
    "ionb_max_abs_diff",
    "ionb_abs_diff_Q1",
    "ionb_abs_diff_Q2",
    "ionb_abs_diff_Q3",
    "ionb_mean_abs_diff",
    "ionb_std_abs_diff",
    "iony_min_abs_diff",
    "iony_max_abs_diff",
    "iony_abs_diff_Q1",
    "iony_abs_diff_Q2",
    "iony_abs_diff_Q3",
    "iony_mean_abs_diff",
    "iony_std_abs_diff",
    "dotprod",
    "dotprod_ionb",
    "dotprod_iony",
    "cos",
    "cos_ionb",
    "cos_iony",
]


@dataclass(slots=True)
class SpectrumRecord:
    """Observed spectrum used for peak matching."""

    scan_number: int
    retention_time: float
    mz_array: np.ndarray
    intensity_array: np.ndarray


@dataclass(slots=True)
class PredictedSpectrum:
    """Predicted AlphaPept spectrum in normalized b/y matrix form."""

    b_mz: np.ndarray
    b_intensity: np.ndarray
    y_mz: np.ndarray
    y_intensity: np.ndarray


def empty_feature_dict() -> dict[str, float]:
    return {name: 0.0 for name in FEATURE_NAMES}


def _mse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((x - y) ** 2))


def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    denominator = np.linalg.norm(x, 2) * np.linalg.norm(y, 2)
    if denominator == 0:
        return 0.0
    return float(np.dot(x, y) / denominator)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = rankdata(x, method="average")
    y_rank = rankdata(y, method="average")
    return float(np.corrcoef(x_rank, y_rank)[0][1])


def _zero_nan(values: list[float]) -> list[float]:
    return [0.0 if np.isnan(value) else float(value) for value in values]


def _summary_stats(values: np.ndarray) -> list[float]:
    if len(values) == 0:
        return [0.0] * 7
    q1, q2, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return [
        float(np.min(values)),
        float(np.max(values)),
        float(q1),
        float(q2),
        float(q3),
        float(np.mean(values)),
        float(np.std(values)),
    ]


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0][1])


def _safe_dot(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return 0.0
    return float(np.dot(x, y))


def _safe_min_compare(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return 0.0
    return 0.0 if np.min(x) <= np.min(y) else 1.0


def _safe_max_compare(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return 0.0
    return 0.0 if np.max(x) >= np.max(y) else 1.0


def match_peaks(
    theoretical_mz: np.ndarray,
    observed_mz: np.ndarray,
    observed_intensity: np.ndarray,
    ms2_tolerance: float = 0.02,
) -> np.ndarray:
    """Vectorized m/z matching copied from alphapeptms2."""
    observed_mz = np.asarray(observed_mz, dtype=np.float32)
    observed_intensity = np.asarray(observed_intensity, dtype=np.float32)
    theoretical_mz = np.asarray(theoretical_mz, dtype=np.float32)

    matched = np.zeros(len(theoretical_mz), dtype=np.float32)
    valid = theoretical_mz > 0
    if not np.any(valid):
        return matched

    valid_theoretical = theoretical_mz[valid]
    indices = np.searchsorted(observed_mz, valid_theoretical)
    left_indices = np.clip(indices - 1, 0, len(observed_mz) - 1)
    right_indices = np.clip(indices, 0, len(observed_mz) - 1)

    left_dist = valid_theoretical - observed_mz[left_indices]
    right_dist = observed_mz[right_indices] - valid_theoretical

    use_left = (indices > 0) & (left_dist <= right_dist) & (left_dist < ms2_tolerance)
    use_right = (
        (indices < len(observed_mz))
        & (right_dist < left_dist)
        & (right_dist < ms2_tolerance)
    )
    chosen_indices = np.where(use_left, left_indices, np.where(use_right, right_indices, -1))
    hit_mask = chosen_indices >= 0
    if np.any(hit_mask):
        matched_values = np.log2(observed_intensity[chosen_indices[hit_mask]] + 0.001)
        valid_indices = np.flatnonzero(valid)
        matched[valid_indices[hit_mask]] = matched_values
    return matched


def calculate_feature_dict(
    predicted: PredictedSpectrum,
    observed: SpectrumRecord,
    ms2_tolerance: float,
) -> dict[str, float]:
    """Calculate the 71 AlphaPept/MS2PIP-style spectral features."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        observed_b = match_peaks(
            predicted.b_mz.reshape(-1),
            observed.mz_array,
            observed.intensity_array,
            ms2_tolerance=ms2_tolerance,
        ).reshape(predicted.b_mz.shape)
        observed_y = match_peaks(
            predicted.y_mz.reshape(-1),
            observed.mz_array,
            observed.intensity_array,
            ms2_tolerance=ms2_tolerance,
        ).reshape(predicted.y_mz.shape)

        target_b = predicted.b_intensity.reshape(-1)
        target_y = predicted.y_intensity.reshape(-1)
        mz_b = predicted.b_mz.reshape(-1)
        mz_y = predicted.y_mz.reshape(-1)
        valid_b = mz_b > 0
        valid_y = mz_y > 0
        target_b = target_b[valid_b]
        target_y = target_y[valid_y]
        target_all = np.concatenate([target_b, target_y])
        prediction_b = observed_b.reshape(-1)[valid_b]
        prediction_y = observed_y.reshape(-1)[valid_y]
        prediction_all = np.concatenate([prediction_b, prediction_y])

        if len(target_all) == 0:
            return empty_feature_dict()

        target_b_unlog = 2**target_b - 0.001
        target_y_unlog = 2**target_y - 0.001
        target_all_unlog = 2**target_all - 0.001
        prediction_b_unlog = 2**prediction_b - 0.001
        prediction_y_unlog = 2**prediction_y - 0.001
        prediction_all_unlog = 2**prediction_all - 0.001

        abs_diff_b = np.abs(target_b - prediction_b)
        abs_diff_y = np.abs(target_y - prediction_y)
        abs_diff_all = np.abs(target_all - prediction_all)
        abs_diff_b_unlog = np.abs(target_b_unlog - prediction_b_unlog)
        abs_diff_y_unlog = np.abs(target_y_unlog - prediction_y_unlog)
        abs_diff_all_unlog = np.abs(target_all_unlog - prediction_all_unlog)
        diff_stats_all = _summary_stats(abs_diff_all)
        diff_stats_b = _summary_stats(abs_diff_b)
        diff_stats_y = _summary_stats(abs_diff_y)
        diff_stats_all_unlog = _summary_stats(abs_diff_all_unlog)
        diff_stats_b_unlog = _summary_stats(abs_diff_b_unlog)
        diff_stats_y_unlog = _summary_stats(abs_diff_y_unlog)

        feature_values = [
            _safe_pearson(target_all, prediction_all),
            _safe_pearson(target_b, prediction_b),
            _safe_pearson(target_y, prediction_y),
            _mse(target_all, prediction_all),
            _mse(target_b, prediction_b),
            _mse(target_y, prediction_y),
            *diff_stats_all,
            *diff_stats_b,
            *diff_stats_y,
            _safe_dot(target_all, prediction_all),
            _safe_dot(target_b, prediction_b),
            _safe_dot(target_y, prediction_y),
            _cosine_similarity(target_all, prediction_all),
            _cosine_similarity(target_b, prediction_b),
            _cosine_similarity(target_y, prediction_y),
            _safe_pearson(target_all_unlog, prediction_all_unlog),
            _safe_pearson(target_b_unlog, prediction_b_unlog),
            _safe_pearson(target_y_unlog, prediction_y_unlog),
            _spearman(target_all_unlog, prediction_all_unlog),
            _spearman(target_b_unlog, prediction_b_unlog),
            _spearman(target_y_unlog, prediction_y_unlog),
            _mse(target_all_unlog, prediction_all_unlog),
            _mse(target_b_unlog, prediction_b_unlog),
            _mse(target_y_unlog, prediction_y_unlog),
            _safe_min_compare(abs_diff_b_unlog, abs_diff_y_unlog),
            _safe_max_compare(abs_diff_b_unlog, abs_diff_y_unlog),
            *diff_stats_all_unlog,
            *diff_stats_b_unlog,
            *diff_stats_y_unlog,
            _safe_dot(target_all_unlog, prediction_all_unlog),
            _safe_dot(target_b_unlog, prediction_b_unlog),
            _safe_dot(target_y_unlog, prediction_y_unlog),
            _cosine_similarity(target_all_unlog, prediction_all_unlog),
            _cosine_similarity(target_b_unlog, prediction_b_unlog),
            _cosine_similarity(target_y_unlog, prediction_y_unlog),
        ]

    return dict(zip(FEATURE_NAMES, _zero_nan(feature_values)))


def finite_feature_values(row: pd.Series, columns: Iterable[str]) -> list[str] | None:
    """Format a row of finite features for pin writing."""
    values: list[str] = []
    for column in columns:
        value = row.get(column)
        if value is None or pd.isna(value):
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        values.append(f"{value:.10g}")
    return values
