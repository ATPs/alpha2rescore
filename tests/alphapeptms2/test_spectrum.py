"""Tests for spectrum matching helpers."""

import numpy as np
import pytest

from alphapeptms2.spectrum import load_spectrum_index, match_peaks


def test_match_peaks_closest_within_tolerance_and_ignores_zero_mz():
    matched = match_peaks(
        np.array([0.0, 100.01, 200.0, 300.0], dtype=np.float32),
        np.array([99.99, 100.02, 200.05], dtype=np.float32),
        np.array([10.0, 20.0, 30.0], dtype=np.float32),
        ms2_tolerance=0.03,
    )

    assert matched[0] == 0.0
    assert np.isclose(matched[1], np.log2(20.001))
    assert matched[2] == 0.0
    assert matched[3] == 0.0


def test_load_spectrum_index_sanitizes_bad_lines_and_warns_on_duplicates(tmp_path):
    spectrum_file = tmp_path / "broken.mgf"
    spectrum_file.write_text(
        "\n".join(
            [
                "BEGIN IONS",
                "TITLE=scan=1",
                "PEPMASS=500.2",
                "CHARGE=2+",
                "100.0 10.0",
                "MALFORMED PEAK",
                "END IONS",
                "BEGIN IONS",
                "TITLE=scan=1",
                "PEPMASS=600.2",
                "CHARGE=3+",
                "200.0 20.0",
                "END IONS",
                "",
            ]
        )
    )

    with pytest.warns(RuntimeWarning):
        spectrum_index = load_spectrum_index(spectrum_file)

    assert list(spectrum_index) == ["scan=1"]
    assert spectrum_index["scan=1"].precursor_mz == 600.2
    assert spectrum_index["scan=1"].precursor_charge == 3
