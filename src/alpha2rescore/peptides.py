"""Peptide parsing and normalization helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


UNIMOD_TOKEN_RE = re.compile(r"\[U:(\d+)\]")
CHARGE_COLUMNS = tuple(f"Charge{i}" for i in range(1, 7))


@dataclass(frozen=True, slots=True)
class ParsedPeptide:
    """Normalized peptide fields used across lookup and feature generation."""

    original: str
    sequence: str
    var_mod_sites_unimod: str


def extract_charge(row: pd.Series) -> int:
    """Extract precursor charge from Comet PIN one-hot charge columns."""
    active = [
        index + 1
        for index, column in enumerate(CHARGE_COLUMNS)
        if int(row.get(column, 0) or 0) == 1
    ]
    if len(active) != 1:
        raise ValueError(
            f"Expected exactly one active charge column for SpecId={row.get('SpecId')}, "
            f"got {active}"
        )
    return active[0]


def parse_pin_peptide(peptide: str) -> ParsedPeptide:
    """Parse a `[U:id]` peptide string into sequence and DB var_mod field text."""
    text = str(peptide).strip()
    if not text:
        raise ValueError("Peptide cannot be empty")

    prefix_mods: list[int] = []
    suffix_mods: list[int] = []
    residue_mods: list[tuple[int, int]] = []
    sequence_chars: list[str] = []

    index = 0
    while True:
        match = UNIMOD_TOKEN_RE.match(text, index)
        if not match:
            break
        prefix_mods.append(int(match.group(1)))
        index = match.end()
        if index >= len(text) or text[index] != "-":
            raise ValueError(f"Expected '-' after N-term mod in peptide {peptide!r}")
        index += 1

    residue_index = 0
    while index < len(text):
        if text[index] == "-":
            index += 1
            match = UNIMOD_TOKEN_RE.match(text, index)
            if not match:
                raise ValueError(f"Unexpected peptide suffix near position {index} in {peptide!r}")
            suffix_mods.append(int(match.group(1)))
            index = match.end()
            continue

        aa = text[index]
        if not aa.isalpha() or len(aa) != 1 or not aa.isupper():
            raise ValueError(f"Unexpected residue token {aa!r} in peptide {peptide!r}")
        sequence_chars.append(aa)
        index += 1

        while True:
            match = UNIMOD_TOKEN_RE.match(text, index)
            if not match:
                break
            residue_mods.append((residue_index, int(match.group(1))))
            index = match.end()
        residue_index += 1

    sequence = "".join(sequence_chars)
    if not sequence:
        raise ValueError(f"Could not parse peptide sequence from {peptide!r}")

    mod_pairs = [(len(sequence), mod_id) for mod_id in prefix_mods]
    mod_pairs.extend(residue_mods)
    mod_pairs.extend((-1, mod_id) for mod_id in suffix_mods)
    mod_pairs.sort(key=lambda item: (len(sequence) + 1) if item[0] == -1 else item[0])
    mod_tokens = [f"{position}:{mod_id}" for position, mod_id in mod_pairs]

    return ParsedPeptide(
        original=text,
        sequence=sequence,
        var_mod_sites_unimod=";".join(mod_tokens),
    )


def pin_peptide_to_unimod_proforma(peptide: str, charge: int) -> str:
    """Convert a `[U:id]` pin peptide into a psm_utils-compatible UNIMOD ProForma string."""
    converted = str(peptide).replace("[U:", "[UNIMOD:")
    return f"{converted}/{charge}"


def make_psm_key(idn: str, spec_id: object, peptide: object, charge: int, label: object) -> str:
    """Build the stable per-PSM cache key."""
    return f"{idn}|{spec_id}|{peptide}|{charge}|{label}"


def make_precursor_key(
    label: int,
    sequence: str,
    var_mod_sites_unimod: str,
    charge: int,
) -> str:
    """Build the stable precursor cache key."""
    return f"{label}|{sequence}|{var_mod_sites_unimod}|{charge}"
