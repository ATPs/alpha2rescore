from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alpha2rescore.features import PredictedSpectrum, SpectrumRecord, calculate_feature_dict
from alpha2rescore.io import insert_feature_columns


def test_insert_feature_columns_before_peptide_and_proteins() -> None:
    pin_df = pd.DataFrame(
        {
            "SpecId": ["a"],
            "Label": [1],
            "ScanNr": [10],
            "Peptide": ["PEPTIDE"],
            "Proteins": ["P1"],
        }
    )
    feature_df = pd.DataFrame({"spec_pearson_norm": [0.1], "rt_diff": [1.2]})
    out = insert_feature_columns(pin_df, feature_df, ["spec_pearson_norm", "rt_diff"])
    assert list(out.columns) == [
        "SpecId",
        "Label",
        "ScanNr",
        "spec_pearson_norm",
        "rt_diff",
        "Peptide",
        "Proteins",
    ]


def test_calculate_feature_dict_returns_expected_columns() -> None:
    predicted = PredictedSpectrum(
        b_mz=np.array([[100.0, 0.0, 0.0], [200.0, 0.0, 0.0]], dtype=np.float32),
        b_intensity=np.log2(np.array([[0.5, 0.001, 0.001], [0.2, 0.001, 0.001]], dtype=np.float32)),
        y_mz=np.array([[300.0, 0.0, 0.0], [400.0, 0.0, 0.0]], dtype=np.float32),
        y_intensity=np.log2(np.array([[0.4, 0.001, 0.001], [0.1, 0.001, 0.001]], dtype=np.float32)),
    )
    observed = SpectrumRecord(
        scan_number=10,
        retention_time=12.3,
        mz_array=np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
        intensity_array=np.array([50.0, 20.0, 40.0, 10.0], dtype=np.float32),
    )
    features = calculate_feature_dict(predicted, observed, ms2_tolerance=0.02)
    assert "spec_pearson_norm" in features
    assert "cos_iony" in features
    assert len(features) == 71


def test_mzduck_scan_mapping_shape() -> None:
    record = SpectrumRecord(
        scan_number=5,
        retention_time=4.2,
        mz_array=np.array([200.0, 100.0], dtype=np.float32),
        intensity_array=np.array([2.0, 1.0], dtype=np.float32),
    )
    assert record.scan_number == 5
    assert record.mz_array.shape == (2,)
