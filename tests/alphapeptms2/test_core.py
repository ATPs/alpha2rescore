"""Tests for MS2PIP-compatible public API behavior."""

from pathlib import Path

import pandas as pd
from psm_utils import PSM, PSMList

from alphapeptms2 import core


class FakeModelManager:
    """Small AlphaPeptDeep stand-in that intentionally reorders precursors."""

    def predict_all(self, precursor_df, **kwargs):
        precursor_df = precursor_df.copy()
        precursor_df["_sort_len"] = precursor_df["sequence"].str.len()
        precursor_df = precursor_df.sort_values("_sort_len").drop(columns=["_sort_len"])

        precursor_rows = []
        mz_rows = []
        intensity_rows = []
        for _, row in precursor_df.iterrows():
            row = row.copy()
            start = len(mz_rows)
            n_positions = len(row["sequence"]) - 1
            for pos in range(n_positions):
                mz_rows.append(
                    {
                        "b_z1": 100.0 + pos,
                        "b_z2": 200.0 + pos,
                        "y_z1": 300.0 + pos,
                        "y_z2": 400.0 + pos,
                    }
                )
                intensity_rows.append(
                    {
                        "b_z1": 10.0 + pos,
                        "b_z2": 20.0 + pos,
                        "y_z1": 30.0 + pos,
                        "y_z2": 40.0 + pos,
                    }
                )
            row["frag_start_idx"] = start
            row["frag_stop_idx"] = len(mz_rows)
            precursor_rows.append(row)

        return {
            "precursor_df": pd.DataFrame(precursor_rows),
            "fragment_mz_df": pd.DataFrame(mz_rows),
            "fragment_intensity_df": pd.DataFrame(intensity_rows),
        }


def test_read_psm_input_uses_infer_for_default_filetype(monkeypatch):
    captured = {}

    def fake_read_file(path, filetype):
        captured["path"] = path
        captured["filetype"] = filetype
        return PSMList(psm_list=[])

    monkeypatch.setattr(core, "psm_read_file", fake_read_file)

    result = core._read_psm_input("input.psms", None)

    assert len(result) == 0
    assert captured == {"path": Path("input.psms"), "filetype": "infer"}


def test_predict_batch_returns_results_in_input_order(monkeypatch):
    monkeypatch.setattr(core, "_get_model_mgr", lambda **kwargs: FakeModelManager())
    psms = PSMList(
        psm_list=[
            PSM(peptidoform="PGAQANPYSR/3", spectrum_id="scan=long"),
            PSM(peptidoform="PEPTIDEK/2", spectrum_id="scan=short"),
        ]
    )

    results = core.predict_batch(psms, device="cpu")

    assert [result.psm_index for result in results] == [0, 1]
    assert [result.psm.spectrum_id for result in results] == ["scan=long", "scan=short"]


def test_correlate_normalizes_psm_and_mgf_ids_and_sets_correlation(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(core, "_get_model_mgr", lambda **kwargs: FakeModelManager())
    spectrum_file = tmp_path / "spectra.mgf"
    spectrum_file.write_text(
        "\n".join(
            [
                "BEGIN IONS",
                "TITLE=scan=short",
                "PEPMASS=500.2",
                "CHARGE=2+",
                "100.0 1000.0",
                "300.0 2000.0",
                "END IONS",
                "",
            ]
        )
    )
    psms = PSMList(psm_list=[PSM(peptidoform="PEPTIDEK/2", spectrum_id="scan=short")])

    results = core.correlate(
        psms,
        spectrum_file,
        device="cpu",
        spectrum_id_pattern=r"scan=.*",
        ms2_tolerance=0.02,
    )

    assert len(results) == 1
    assert results[0].observed_intensity is not None
    assert results[0].correlation is not None
