"""Parquet I/O, cache I/O, and PIN writing helpers."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import SpectrumRecord


PIN_REQUIRED_COLUMNS = [
    "SpecId",
    "Label",
    "ScanNr",
    "Peptide",
    "Proteins",
]

MZDUCK_COLUMNS = ["scan_number", "rt_seconds", "mz_array", "intensity_array"]


def read_pin_parquet(path: Path, max_psms: int | None = None) -> pd.DataFrame:
    """Read a Comet pin parquet file."""
    df = pd.read_parquet(path)
    missing = [column for column in PIN_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"PIN parquet is missing required columns: {missing}")
    if max_psms is not None:
        df = df.head(max_psms).copy()
    else:
        df = df.copy()
    return df.reset_index(drop=True)


def load_mzduck_spectra(path: Path) -> dict[int, SpectrumRecord]:
    """Load mzDuck mgf.parquet into a ScanNr-indexed spectrum map."""
    df = pd.read_parquet(path, columns=MZDUCK_COLUMNS)
    spectra: dict[int, SpectrumRecord] = {}
    for row in df.itertuples(index=False):
        mz_array = np.asarray(row.mz_array, dtype=np.float32)
        intensity_array = np.asarray(row.intensity_array, dtype=np.float32)
        sort_order = np.argsort(mz_array)
        spectra[int(row.scan_number)] = SpectrumRecord(
            scan_number=int(row.scan_number),
            retention_time=float(row.rt_seconds),
            mz_array=mz_array[sort_order],
            intensity_array=intensity_array[sort_order],
        )
    return spectra


def read_cache_parquet(path: Path, key_column: str | None = None) -> pd.DataFrame:
    """Load a parquet cache or return an empty DataFrame."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if key_column and key_column in df.columns:
        df = df.drop_duplicates(subset=[key_column], keep="last").reset_index(drop=True)
    return df


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write parquet via a temporary file and atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(temp_path, index=False)
    os.replace(temp_path, path)


def read_json(path: Path, default: Any) -> Any:
    """Read a JSON file or return a default value."""
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(data: Any, path: Path) -> None:
    """Write JSON via a temporary file and atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    os.replace(temp_path, path)


def format_pin_value(value: Any) -> str:
    """Format a scalar value for PIN text output."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.10g}"
    return str(value)


def write_pin_gz(
    full_df: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
) -> None:
    """Write a gzipped Percolator PIN text file."""
    header = list(full_df.columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in full_df.itertuples(index=False, name=None):
            handle.write("\t".join(format_pin_value(value) for value in row) + "\n")


def insert_feature_columns(
    pin_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Insert feature columns before Peptide and Proteins."""
    merged = pin_df.reset_index(drop=True).copy()
    features = feature_df.reset_index(drop=True).loc[:, feature_columns].copy()

    peptide_index = merged.columns.get_loc("Peptide")
    left = merged.iloc[:, :peptide_index]
    right = merged.iloc[:, peptide_index:]
    return pd.concat([left, features, right], axis=1)
