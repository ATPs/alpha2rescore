from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alpha2rescore.config import Alpha2RescoreConfig
from alpha2rescore.deeplc_module import build_deeplc_base_features


class _FakePredictor:
    def make_preds(self, psm_list):
        return np.array([[10.5], [20.5]], dtype=np.float32)


def test_build_deeplc_base_features_flattens_column_vector_predictions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pin_df = pd.DataFrame(
        {
            "SpecId": ["spec-1", "spec-2"],
            "Label": [1, 1],
            "Peptide": ["PEPTIDE", "PEPTIDER"],
            "charge": [2, 3],
            "observed_retention_time": [10.0, 21.0],
            "Xcorr": [3.0, 2.5],
            "psm_key": ["k1", "k2"],
        }
    )
    pending_df = pin_df.copy()
    config = Alpha2RescoreConfig(
        pin_parquet=tmp_path / "input.pin",
        mgf_parquet=tmp_path / "input.mgf",
        out_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        idn="1554451",
    )

    monkeypatch.setattr("alpha2rescore.deeplc_module._build_predictor", lambda config: _FakePredictor())
    monkeypatch.setattr(
        "alpha2rescore.deeplc_module._calibrate_predictor",
        lambda predictor, calibration_df, config: {"model": None},
    )

    features = build_deeplc_base_features(pin_df, pending_df, config)

    assert features["predicted_retention_time"].tolist() == [10.5, 20.5]
    assert features["rt_diff"].tolist() == [0.5, 0.5]
    assert features["psm_key"].tolist() == ["k1", "k2"]
