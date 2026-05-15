from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from alpha2rescore.cli import resolve_paths
from alpha2rescore.features import PredictedSpectrum, SpectrumRecord, calculate_feature_dict
from alpha2rescore.io import insert_feature_columns, load_spectra, read_pin_table


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
    assert np.isclose(features["spec_pearson_norm"], -0.3138841974465237)
    assert np.isclose(features["spec_spearman"], -0.2)
    assert np.isclose(features["min_abs_diff_iontype"], 0.0)
    assert np.isclose(features["max_abs_diff_iontype"], 1.0)
    assert np.isclose(features["cos"], 0.6910891532897949)


def test_mzduck_scan_mapping_shape() -> None:
    record = SpectrumRecord(
        scan_number=5,
        retention_time=4.2,
        mz_array=np.array([200.0, 100.0], dtype=np.float32),
        intensity_array=np.array([2.0, 1.0], dtype=np.float32),
    )
    assert record.scan_number == 5
    assert record.mz_array.shape == (2,)


def test_read_pin_table_supports_tsv(tmp_path: Path) -> None:
    pin_path = tmp_path / "1554451.pin.tsv"
    pin_path.write_text(
        "SpecId\tLabel\tScanNr\tPeptide\tProteins\tCharge2\n"
        "scan=42\t1\t42\tPEPTIDE\tP1\t1\n",
        encoding="utf-8",
    )

    df = read_pin_table(pin_path)

    assert list(df.columns[:5]) == ["SpecId", "Label", "ScanNr", "Peptide", "Proteins"]
    assert df.loc[0, "SpecId"] == "scan=42"
    assert int(df.loc[0, "ScanNr"]) == 42


def test_load_spectra_supports_text_mgf_with_scan_and_title_fallback(tmp_path: Path) -> None:
    mgf_path = tmp_path / "1554451.mgf"
    mgf_path.write_text(
        "\n".join(
            [
                "BEGIN IONS",
                "TITLE=controllerType=0 controllerNumber=1 scan=42",
                "SCANS=42",
                "RTINSECONDS=12.5",
                "100.0 10.0",
                "50.0 5.0",
                "END IONS",
                "BEGIN IONS",
                "TITLE=run.43.43.2",
                "RTINSECONDS=13.5",
                "150.0 15.0",
                "75.0 7.5",
                "END IONS",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    spectra = load_spectra(mgf_path)

    assert sorted(spectra) == [42, 43]
    assert spectra[42].retention_time == 12.5
    assert spectra[42].mz_array.tolist() == [50.0, 100.0]
    assert spectra[43].retention_time == 13.5
    assert spectra[43].intensity_array.tolist() == [7.5, 15.0]


def test_resolve_paths_supports_text_formats_in_directories(tmp_path: Path) -> None:
    pin_dir = tmp_path / "pins"
    mgf_dir = tmp_path / "mgf"
    pin_dir.mkdir()
    mgf_dir.mkdir()
    pin_path = pin_dir / "1554451.pin"
    mgf_path = mgf_dir / "1554451.mgf"
    pin_path.write_text("", encoding="utf-8")
    mgf_path.write_text("", encoding="utf-8")

    args = Namespace(
        pin_file=None,
        spectrum_file=None,
        pin_dir=str(pin_dir),
        mgf_dir=str(mgf_dir),
        idn="1554451",
    )

    resolved_idn, resolved_pin, resolved_mgf = resolve_paths(args)

    assert resolved_idn == "1554451"
    assert resolved_pin == pin_path
    assert resolved_mgf == mgf_path
