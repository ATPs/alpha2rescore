"""DeepLC calibration persistence and feature generation."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from psm_utils import PSM, PSMList

from .config import Alpha2RescoreConfig
from .peptides import pin_peptide_to_unimod_proforma


DEEPLC_BASE_FEATURES = [
    "observed_retention_time",
    "predicted_retention_time",
    "rt_diff",
]
DEEPLC_FEATURES = DEEPLC_BASE_FEATURES + [
    "observed_retention_time_best",
    "predicted_retention_time_best",
    "rt_diff_best",
]


def save_deeplc_calibration(idn: str, cache_dir: Path, state: dict) -> Path:
    """Persist a small DeepLC calibration state for one idn."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{idn}.deeplc_calibration.pkl"
    with path.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_deeplc_calibration(idn: str, cache_dir: Path) -> dict | None:
    """Load a previously saved DeepLC calibration state."""
    path = cache_dir / f"{idn}.deeplc_calibration.pkl"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def _make_psm_list(df: pd.DataFrame) -> PSMList:
    psms = []
    for row in df.itertuples(index=False):
        psms.append(
            PSM(
                peptidoform=pin_peptide_to_unimod_proforma(row.Peptide, int(row.charge)),
                spectrum_id=str(row.SpecId),
                retention_time=float(row.observed_retention_time),
            )
        )
    return PSMList(psm_list=psms)


def _build_predictor(config: Alpha2RescoreConfig) -> DeepLC:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from deeplc import DeepLC

    return DeepLC(
        n_jobs=config.deeplc_processes,
        verbose=False,
        config_file=None,
        deeplc_retrain=False,
    )


def _calibrate_predictor(
    predictor: DeepLC,
    calibration_df: pd.DataFrame,
    config: Alpha2RescoreConfig,
) -> dict:
    targets = calibration_df[
        (calibration_df["Label"] == 1) & (calibration_df["observed_retention_time"] > 0)
    ].copy()
    if targets.empty:
        raise ValueError("DeepLC calibration requires at least one target PSM")
    n_calibration = max(1, round(len(targets) * config.deeplc_calibration_fraction))
    selected = targets.nlargest(n_calibration, "Xcorr")
    calibration_psms = _make_psm_list(selected)
    predictor.calibrate_preds(calibration_psms)
    return {
        "model": predictor.model,
        "calibrate_dict": predictor.calibrate_dict,
        "calibrate_min": predictor.calibrate_min,
        "calibrate_max": predictor.calibrate_max,
    }


def _apply_calibration(predictor: DeepLC, state: dict) -> None:
    predictor.model = state["model"]
    predictor.calibrate_dict = state["calibrate_dict"]
    predictor.calibrate_min = state["calibrate_min"]
    predictor.calibrate_max = state["calibrate_max"]


def build_deeplc_base_features(
    pin_df: pd.DataFrame,
    pending_df: pd.DataFrame,
    config: Alpha2RescoreConfig,
) -> pd.DataFrame:
    """Predict base DeepLC features for pending PSM rows."""
    if pending_df.empty:
        return pd.DataFrame(columns=["psm_key", *DEEPLC_BASE_FEATURES])

    predictor = _build_predictor(config)
    calibration_state = None if config.recalibrate_deeplc else load_deeplc_calibration(
        config.idn, config.cache_dir
    )
    if calibration_state is None:
        calibration_state = _calibrate_predictor(predictor, pin_df, config)
        save_deeplc_calibration(config.idn, config.cache_dir, calibration_state)
    else:
        _apply_calibration(predictor, calibration_state)

    psm_list = _make_psm_list(pending_df)
    predictions = np.asarray(predictor.make_preds(psm_list), dtype=np.float32)
    observations = pending_df["observed_retention_time"].to_numpy(dtype=np.float32, copy=False)
    diffs = np.abs(predictions - observations)

    return pd.DataFrame(
        {
            "psm_key": pending_df["psm_key"].tolist(),
            "observed_retention_time": observations,
            "predicted_retention_time": predictions,
            "rt_diff": diffs,
        }
    )


def finalize_deeplc_features(
    current_rows: pd.DataFrame,
    deeplc_base_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add best-per-peptide DeepLC variants to per-PSM base features."""
    merged = current_rows.loc[:, ["psm_key", "Peptide"]].merge(
        deeplc_base_df,
        on="psm_key",
        how="left",
    )
    for column in DEEPLC_BASE_FEATURES:
        merged[column] = merged[column].fillna(0.0)

    best_rows = merged.groupby("Peptide", sort=False)["rt_diff"].idxmin()
    best = merged.loc[
        best_rows,
        [
            "Peptide",
            "observed_retention_time",
            "predicted_retention_time",
            "rt_diff",
        ],
    ].rename(
        columns={
            "observed_retention_time": "observed_retention_time_best",
            "predicted_retention_time": "predicted_retention_time_best",
            "rt_diff": "rt_diff_best",
        }
    )
    merged = merged.merge(best, on="Peptide", how="left")
    return merged.loc[:, ["psm_key", *DEEPLC_FEATURES]]
