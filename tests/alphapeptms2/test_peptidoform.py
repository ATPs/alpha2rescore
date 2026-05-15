"""Tests for ProForma to AlphaPeptDeep conversion."""

import pytest

from alphapeptms2._utils.peptidoform import peptidoform_to_row


def test_peptidoform_to_row_residue_modification():
    assert peptidoform_to_row("RNVIM[Oxidation]DKVAK/2") == {
        "sequence": "RNVIMDKVAK",
        "mods": "Oxidation@M",
        "mod_sites": "5",
        "charge": 2,
    }


def test_peptidoform_to_row_terminal_modifications():
    assert peptidoform_to_row("[Acetyl]-PEPTIDEK/2") == {
        "sequence": "PEPTIDEK",
        "mods": "Acetyl@Any_N-term",
        "mod_sites": "0",
        "charge": 2,
    }
    assert peptidoform_to_row("PEPTIDEK-[Amidated]/2") == {
        "sequence": "PEPTIDEK",
        "mods": "Amidated@Any_C-term",
        "mod_sites": "-1",
        "charge": 2,
    }
    assert peptidoform_to_row("[UNIMOD:1]-PEPTIDEK/2") == {
        "sequence": "PEPTIDEK",
        "mods": "Acetyl@Any_N-term",
        "mod_sites": "0",
        "charge": 2,
    }


def test_peptidoform_to_row_pyro_glu_normalization():
    assert peptidoform_to_row("[Gln->pyro-Glu]-QPEPTIDE/2") == {
        "sequence": "QPEPTIDE",
        "mods": "Gln->pyro-Glu@Q^Any_N-term",
        "mod_sites": "0",
        "charge": 2,
    }
    assert peptidoform_to_row("Q[pyro-Glu]PEPTIDE/2") == {
        "sequence": "QPEPTIDE",
        "mods": "Gln->pyro-Glu@Q^Any_N-term",
        "mod_sites": "0",
        "charge": 2,
    }
    assert peptidoform_to_row("Q[UNIMOD:28]PEPTIDE/2") == {
        "sequence": "QPEPTIDE",
        "mods": "Gln->pyro-Glu@Q^Any_N-term",
        "mod_sites": "0",
        "charge": 2,
    }


def test_peptidoform_to_row_requires_charge():
    with pytest.raises(ValueError, match="precursor charge"):
        peptidoform_to_row("PEPTIDEK")


def test_peptidoform_to_row_unimod_residue_and_terminal_specific_mapping():
    assert peptidoform_to_row("PEPTIDEK[UNIMOD:35]/2") == {
        "sequence": "PEPTIDEK",
        "mods": "Oxidation@K",
        "mod_sites": "8",
        "charge": 2,
    }
    assert peptidoform_to_row("PEPTIDEG[UNIMOD:35]/2") == {
        "sequence": "PEPTIDEG",
        "mods": "Oxidation@G^Any_C-term",
        "mod_sites": "-1",
        "charge": 2,
    }
