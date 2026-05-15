from __future__ import annotations

import pandas as pd
import pytest

from alpha2rescore.peptides import extract_charge, parse_pin_peptide


def test_parse_pin_peptide_internal_mod() -> None:
    parsed = parse_pin_peptide("LHWLVM[U:35]RK")
    assert parsed.sequence == "LHWLVMRK"
    assert parsed.var_mod_sites_unimod == "5:35"


def test_parse_pin_peptide_nterm_mod() -> None:
    parsed = parse_pin_peptide("[U:1]-M[U:35]LQFLLEVNK")
    assert parsed.sequence == "MLQFLLEVNK"
    assert parsed.var_mod_sites_unimod == "0:35;10:1"


def test_parse_pin_peptide_pyro_mod() -> None:
    parsed = parse_pin_peptide("Q[U:28]AVKLVKANK")
    assert parsed.sequence == "QAVKLVKANK"
    assert parsed.var_mod_sites_unimod == "0:28"


def test_extract_charge() -> None:
    row = pd.Series(
        {
            "SpecId": "1554451_10_2_1",
            "Charge1": 0,
            "Charge2": 1,
            "Charge3": 0,
            "Charge4": 0,
            "Charge5": 0,
            "Charge6": 0,
        }
    )
    assert extract_charge(row) == 2


def test_extract_charge_raises_on_ambiguous() -> None:
    row = pd.Series({"SpecId": "x", "Charge1": 1, "Charge2": 1})
    with pytest.raises(ValueError):
        extract_charge(row)
