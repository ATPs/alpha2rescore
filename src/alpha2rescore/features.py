"""AlphaPept spectral feature calculation."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

try:
    from numba import njit as _njit
except Exception:  # pragma: no cover - optional acceleration dependency
    _njit = None


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
    if len(x) == 0 or len(y) == 0:
        return 0.0
    return float(np.mean((x - y) ** 2))


def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return 0.0
    denominator = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    if denominator == 0:
        return 0.0
    return float(np.dot(x, y) / denominator)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return 0.0
    x_rank = rankdata(x, method="average")
    y_rank = rankdata(y, method="average")
    return _safe_pearson(x_rank, y_rank)


def _zero_nan(values: list[float]) -> list[float]:
    return [0.0 if np.isnan(value) else float(value) for value in values]


def _summary_stats(values: np.ndarray) -> list[float]:
    if len(values) == 0:
        return [0.0] * 7
    sorted_values = np.sort(values)
    q1 = _linear_quantile_sorted(sorted_values, 0.25)
    q2 = _linear_quantile_sorted(sorted_values, 0.5)
    q3 = _linear_quantile_sorted(sorted_values, 0.75)
    return [
        float(sorted_values[0]),
        float(sorted_values[-1]),
        float(q1),
        float(q2),
        float(q3),
        float(np.mean(values)),
        float(np.std(values)),
    ]


def _linear_quantile_sorted(sorted_values: np.ndarray, quantile: float) -> float:
    n = len(sorted_values)
    if n == 0:
        return 0.0
    position = (n - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return 0.0
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    n = float(len(x64))
    sum_x = float(np.sum(x64))
    sum_y = float(np.sum(y64))
    sum_xx = float(np.dot(x64, x64))
    sum_yy = float(np.dot(y64, y64))
    sum_xy = float(np.dot(x64, y64))
    cov = sum_xy - (sum_x * sum_y / n)
    var_x = sum_xx - (sum_x * sum_x / n)
    var_y = sum_yy - (sum_y * sum_y / n)
    denominator = math.sqrt(max(var_x * var_y, 0.0))
    if denominator == 0.0:
        return 0.0
    return float(cov / denominator)


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


if _njit is not None:

    @_njit(cache=True)
    def _linear_quantile_sorted_numba(sorted_values: np.ndarray, quantile: float) -> float:
        n = sorted_values.shape[0]
        if n == 0:
            return 0.0
        position = (n - 1) * quantile
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return float(sorted_values[lower])
        weight = position - lower
        return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


    @_njit(cache=True)
    def _fill_summary_stats_numba(values: np.ndarray, out: np.ndarray, offset: int) -> None:
        n = values.shape[0]
        if n == 0:
            for i in range(7):
                out[offset + i] = 0.0
            return

        sorted_values = np.sort(values)
        total = 0.0
        for i in range(n):
            total += values[i]
        mean = total / n

        variance = 0.0
        for i in range(n):
            delta = values[i] - mean
            variance += delta * delta
        std = math.sqrt(variance / n)

        out[offset] = sorted_values[0]
        out[offset + 1] = sorted_values[n - 1]
        out[offset + 2] = _linear_quantile_sorted_numba(sorted_values, 0.25)
        out[offset + 3] = _linear_quantile_sorted_numba(sorted_values, 0.5)
        out[offset + 4] = _linear_quantile_sorted_numba(sorted_values, 0.75)
        out[offset + 5] = mean
        out[offset + 6] = std


    @_njit(cache=True)
    def _pearson_numba(x: np.ndarray, y: np.ndarray) -> float:
        n = x.shape[0]
        if n == 0:
            return 0.0
        sum_x = 0.0
        sum_y = 0.0
        sum_xx = 0.0
        sum_yy = 0.0
        sum_xy = 0.0
        for i in range(n):
            x_value = x[i]
            y_value = y[i]
            sum_x += x_value
            sum_y += y_value
            sum_xx += x_value * x_value
            sum_yy += y_value * y_value
            sum_xy += x_value * y_value
        cov = sum_xy - (sum_x * sum_y / n)
        var_x = sum_xx - (sum_x * sum_x / n)
        var_y = sum_yy - (sum_y * sum_y / n)
        denominator = var_x * var_y
        if denominator <= 0.0:
            return 0.0
        value = cov / math.sqrt(denominator)
        if not math.isfinite(value):
            return 0.0
        return value


    @_njit(cache=True)
    def _mse_numba(x: np.ndarray, y: np.ndarray) -> float:
        n = x.shape[0]
        if n == 0:
            return 0.0
        total = 0.0
        for i in range(n):
            diff = x[i] - y[i]
            total += diff * diff
        return total / n


    @_njit(cache=True)
    def _dot_numba(x: np.ndarray, y: np.ndarray) -> float:
        total = 0.0
        for i in range(x.shape[0]):
            total += x[i] * y[i]
        return total


    @_njit(cache=True)
    def _cosine_numba(x: np.ndarray, y: np.ndarray) -> float:
        n = x.shape[0]
        if n == 0:
            return 0.0
        dot = 0.0
        x_norm = 0.0
        y_norm = 0.0
        for i in range(n):
            x_value = x[i]
            y_value = y[i]
            dot += x_value * y_value
            x_norm += x_value * x_value
            y_norm += y_value * y_value
        denominator = math.sqrt(x_norm * y_norm)
        if denominator == 0.0:
            return 0.0
        value = dot / denominator
        if not math.isfinite(value):
            return 0.0
        return value


    @_njit(cache=True)
    def _fill_average_ranks_numba(values: np.ndarray, ranks: np.ndarray) -> None:
        n = values.shape[0]
        order = np.argsort(values)
        i = 0
        while i < n:
            j = i + 1
            value = values[order[i]]
            while j < n and values[order[j]] == value:
                j += 1
            rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[order[k]] = rank
            i = j


    @_njit(cache=True)
    def _spearman_numba(x: np.ndarray, y: np.ndarray) -> float:
        n = x.shape[0]
        if n == 0:
            return 0.0
        x_rank = np.empty(n, dtype=np.float64)
        y_rank = np.empty(n, dtype=np.float64)
        _fill_average_ranks_numba(x, x_rank)
        _fill_average_ranks_numba(y, y_rank)
        return _pearson_numba(x_rank, y_rank)


    @_njit(cache=True)
    def _match_peaks_numba(
        theoretical_mz: np.ndarray,
        observed_mz: np.ndarray,
        observed_intensity: np.ndarray,
        ms2_tolerance: float,
        matched: np.ndarray,
    ) -> None:
        n_observed = observed_mz.shape[0]
        for i in range(theoretical_mz.shape[0]):
            matched[i] = 0.0
            mz = theoretical_mz[i]
            if mz <= 0.0 or n_observed == 0:
                continue

            left = 0
            right = n_observed
            while left < right:
                middle = (left + right) // 2
                if observed_mz[middle] < mz:
                    left = middle + 1
                else:
                    right = middle
            index = left

            left_index = index - 1
            if left_index < 0:
                left_index = 0
            right_index = index
            if right_index >= n_observed:
                right_index = n_observed - 1

            left_dist = mz - observed_mz[left_index]
            right_dist = observed_mz[right_index] - mz

            chosen_index = -1
            if index > 0 and left_dist <= right_dist and left_dist < ms2_tolerance:
                chosen_index = left_index
            elif index < n_observed and right_dist < left_dist and right_dist < ms2_tolerance:
                chosen_index = right_index

            if chosen_index >= 0:
                matched[i] = math.log2(observed_intensity[chosen_index] + 0.001)


    @_njit(cache=True)
    def _valid_count_numba(mz_values: np.ndarray) -> int:
        count = 0
        for i in range(mz_values.shape[0]):
            if mz_values[i] > 0.0:
                count += 1
        return count


    @_njit(cache=True)
    def _collect_valid_numba(
        mz_values: np.ndarray,
        target_values: np.ndarray,
        matched_values: np.ndarray,
        target_out: np.ndarray,
        prediction_out: np.ndarray,
    ) -> None:
        out_index = 0
        for i in range(mz_values.shape[0]):
            if mz_values[i] > 0.0:
                target_out[out_index] = target_values[i]
                prediction_out[out_index] = matched_values[i]
                out_index += 1


    @_njit(cache=True)
    def _concat_numba(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        out = np.empty(left.shape[0] + right.shape[0], dtype=np.float64)
        for i in range(left.shape[0]):
            out[i] = left[i]
        offset = left.shape[0]
        for i in range(right.shape[0]):
            out[offset + i] = right[i]
        return out


    @_njit(cache=True)
    def _unlog_numba(values: np.ndarray) -> np.ndarray:
        out = np.empty(values.shape[0], dtype=np.float64)
        for i in range(values.shape[0]):
            out[i] = (2.0 ** values[i]) - 0.001
        return out


    @_njit(cache=True)
    def _abs_diff_numba(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.empty(x.shape[0], dtype=np.float64)
        for i in range(x.shape[0]):
            diff = x[i] - y[i]
            if diff < 0.0:
                diff = -diff
            out[i] = diff
        return out


    @_njit(cache=True)
    def _min_compare_numba(x: np.ndarray, y: np.ndarray) -> float:
        if x.shape[0] == 0 or y.shape[0] == 0:
            return 0.0
        x_min = x[0]
        y_min = y[0]
        for i in range(1, x.shape[0]):
            if x[i] < x_min:
                x_min = x[i]
        for i in range(1, y.shape[0]):
            if y[i] < y_min:
                y_min = y[i]
        if x_min <= y_min:
            return 0.0
        return 1.0


    @_njit(cache=True)
    def _max_compare_numba(x: np.ndarray, y: np.ndarray) -> float:
        if x.shape[0] == 0 or y.shape[0] == 0:
            return 0.0
        x_max = x[0]
        y_max = y[0]
        for i in range(1, x.shape[0]):
            if x[i] > x_max:
                x_max = x[i]
        for i in range(1, y.shape[0]):
            if y[i] > y_max:
                y_max = y[i]
        if x_max >= y_max:
            return 0.0
        return 1.0


    @_njit(cache=True)
    def _calculate_feature_values_numba(
        b_mz: np.ndarray,
        b_intensity: np.ndarray,
        y_mz: np.ndarray,
        y_intensity: np.ndarray,
        observed_mz: np.ndarray,
        observed_intensity: np.ndarray,
        ms2_tolerance: float,
    ) -> np.ndarray:
        values = np.zeros(71, dtype=np.float64)
        matched_b = np.empty(b_mz.shape[0], dtype=np.float64)
        matched_y = np.empty(y_mz.shape[0], dtype=np.float64)
        _match_peaks_numba(b_mz, observed_mz, observed_intensity, ms2_tolerance, matched_b)
        _match_peaks_numba(y_mz, observed_mz, observed_intensity, ms2_tolerance, matched_y)

        n_b = _valid_count_numba(b_mz)
        n_y = _valid_count_numba(y_mz)
        if n_b + n_y == 0:
            return values

        target_b = np.empty(n_b, dtype=np.float64)
        target_y = np.empty(n_y, dtype=np.float64)
        prediction_b = np.empty(n_b, dtype=np.float64)
        prediction_y = np.empty(n_y, dtype=np.float64)
        _collect_valid_numba(b_mz, b_intensity, matched_b, target_b, prediction_b)
        _collect_valid_numba(y_mz, y_intensity, matched_y, target_y, prediction_y)

        target_all = _concat_numba(target_b, target_y)
        prediction_all = _concat_numba(prediction_b, prediction_y)

        target_b_unlog = _unlog_numba(target_b)
        target_y_unlog = _unlog_numba(target_y)
        target_all_unlog = _unlog_numba(target_all)
        prediction_b_unlog = _unlog_numba(prediction_b)
        prediction_y_unlog = _unlog_numba(prediction_y)
        prediction_all_unlog = _unlog_numba(prediction_all)

        abs_diff_b = _abs_diff_numba(target_b, prediction_b)
        abs_diff_y = _abs_diff_numba(target_y, prediction_y)
        abs_diff_all = _abs_diff_numba(target_all, prediction_all)
        abs_diff_b_unlog = _abs_diff_numba(target_b_unlog, prediction_b_unlog)
        abs_diff_y_unlog = _abs_diff_numba(target_y_unlog, prediction_y_unlog)
        abs_diff_all_unlog = _abs_diff_numba(target_all_unlog, prediction_all_unlog)

        offset = 0
        values[offset] = _pearson_numba(target_all, prediction_all)
        values[offset + 1] = _pearson_numba(target_b, prediction_b)
        values[offset + 2] = _pearson_numba(target_y, prediction_y)
        values[offset + 3] = _mse_numba(target_all, prediction_all)
        values[offset + 4] = _mse_numba(target_b, prediction_b)
        values[offset + 5] = _mse_numba(target_y, prediction_y)
        offset = 6
        _fill_summary_stats_numba(abs_diff_all, values, offset)
        offset += 7
        _fill_summary_stats_numba(abs_diff_b, values, offset)
        offset += 7
        _fill_summary_stats_numba(abs_diff_y, values, offset)
        offset += 7
        values[offset] = _dot_numba(target_all, prediction_all)
        values[offset + 1] = _dot_numba(target_b, prediction_b)
        values[offset + 2] = _dot_numba(target_y, prediction_y)
        values[offset + 3] = _cosine_numba(target_all, prediction_all)
        values[offset + 4] = _cosine_numba(target_b, prediction_b)
        values[offset + 5] = _cosine_numba(target_y, prediction_y)
        offset += 6
        values[offset] = _pearson_numba(target_all_unlog, prediction_all_unlog)
        values[offset + 1] = _pearson_numba(target_b_unlog, prediction_b_unlog)
        values[offset + 2] = _pearson_numba(target_y_unlog, prediction_y_unlog)
        values[offset + 3] = _spearman_numba(target_all_unlog, prediction_all_unlog)
        values[offset + 4] = _spearman_numba(target_b_unlog, prediction_b_unlog)
        values[offset + 5] = _spearman_numba(target_y_unlog, prediction_y_unlog)
        values[offset + 6] = _mse_numba(target_all_unlog, prediction_all_unlog)
        values[offset + 7] = _mse_numba(target_b_unlog, prediction_b_unlog)
        values[offset + 8] = _mse_numba(target_y_unlog, prediction_y_unlog)
        values[offset + 9] = _min_compare_numba(abs_diff_b_unlog, abs_diff_y_unlog)
        values[offset + 10] = _max_compare_numba(abs_diff_b_unlog, abs_diff_y_unlog)
        offset += 11
        _fill_summary_stats_numba(abs_diff_all_unlog, values, offset)
        offset += 7
        _fill_summary_stats_numba(abs_diff_b_unlog, values, offset)
        offset += 7
        _fill_summary_stats_numba(abs_diff_y_unlog, values, offset)
        offset += 7
        values[offset] = _dot_numba(target_all_unlog, prediction_all_unlog)
        values[offset + 1] = _dot_numba(target_b_unlog, prediction_b_unlog)
        values[offset + 2] = _dot_numba(target_y_unlog, prediction_y_unlog)
        values[offset + 3] = _cosine_numba(target_all_unlog, prediction_all_unlog)
        values[offset + 4] = _cosine_numba(target_b_unlog, prediction_b_unlog)
        values[offset + 5] = _cosine_numba(target_y_unlog, prediction_y_unlog)
        return values

else:
    _calculate_feature_values_numba = None


def _calculate_feature_values_numpy(
    b_mz: np.ndarray,
    b_intensity: np.ndarray,
    y_mz: np.ndarray,
    y_intensity: np.ndarray,
    observed_mz: np.ndarray,
    observed_intensity: np.ndarray,
    ms2_tolerance: float,
) -> list[float]:
    observed_b = match_peaks(
        b_mz,
        observed_mz,
        observed_intensity,
        ms2_tolerance=ms2_tolerance,
    )
    observed_y = match_peaks(
        y_mz,
        observed_mz,
        observed_intensity,
        ms2_tolerance=ms2_tolerance,
    )

    valid_b = b_mz > 0
    valid_y = y_mz > 0
    target_b = b_intensity[valid_b]
    target_y = y_intensity[valid_y]
    target_all = np.concatenate([target_b, target_y])
    prediction_b = observed_b[valid_b]
    prediction_y = observed_y[valid_y]
    prediction_all = np.concatenate([prediction_b, prediction_y])

    if len(target_all) == 0:
        return [0.0] * len(FEATURE_NAMES)

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

    return [
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


def calculate_feature_dict(
    predicted: PredictedSpectrum,
    observed: SpectrumRecord,
    ms2_tolerance: float,
) -> dict[str, float]:
    """Calculate the 71 AlphaPept/MS2PIP-style spectral features."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        b_mz = np.asarray(predicted.b_mz, dtype=np.float32).reshape(-1)
        b_intensity = np.asarray(predicted.b_intensity, dtype=np.float32).reshape(-1)
        y_mz = np.asarray(predicted.y_mz, dtype=np.float32).reshape(-1)
        y_intensity = np.asarray(predicted.y_intensity, dtype=np.float32).reshape(-1)
        observed_mz = np.asarray(observed.mz_array, dtype=np.float32)
        observed_intensity = np.asarray(observed.intensity_array, dtype=np.float32)

        if _calculate_feature_values_numba is not None:
            feature_values = _calculate_feature_values_numba(
                b_mz,
                b_intensity,
                y_mz,
                y_intensity,
                observed_mz,
                observed_intensity,
                float(ms2_tolerance),
            )
        else:
            feature_values = _calculate_feature_values_numpy(
                b_mz,
                b_intensity,
                y_mz,
                y_intensity,
                observed_mz,
                observed_intensity,
                float(ms2_tolerance),
            )

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
