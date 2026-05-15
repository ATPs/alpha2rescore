"""PIN/spectrum I/O, cache I/O, and PIN writing helpers."""

from __future__ import annotations

import gzip
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyteomics import mgf as pyteomics_mgf

from .features import SpectrumRecord


PIN_REQUIRED_COLUMNS = [
    "SpecId",
    "Label",
    "ScanNr",
    "Peptide",
    "Proteins",
]

MZDUCK_COLUMNS = ["scan_number", "rt_seconds", "mz_array", "intensity_array"]
TITLE_SCAN_PATTERNS = [
    re.compile(r"(?:^|[^\d])scan(?:s)?[ =:]+(\d+)(?:[^\d]|$)", re.IGNORECASE),
    re.compile(r"^(\d+)$"),
]


def _strip_gzip_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".gz"):
        return name[:-3]
    return name


def _validate_pin_columns(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = [column for column in PIN_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"PIN file is missing required columns: {missing} ({path})")
    return df


def read_pin_table(path: Path, max_psms: int | None = None) -> pd.DataFrame:
    """Read a Comet PIN file from parquet or tab-separated text."""
    normalized_name = _strip_gzip_suffix(path)
    if normalized_name.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep="\t", compression="infer", low_memory=False)
    df = _validate_pin_columns(df, path)
    if max_psms is not None:
        df = df.head(max_psms).copy()
    else:
        df = df.copy()
    return df.reset_index(drop=True)


def _build_spectrum_record(
    scan_number: int,
    retention_time: float,
    mz_array,
    intensity_array,
) -> SpectrumRecord:
    mz_array = np.asarray(mz_array, dtype=np.float32)
    intensity_array = np.asarray(intensity_array, dtype=np.float32)
    sort_order = np.argsort(mz_array)
    return SpectrumRecord(
        scan_number=int(scan_number),
        retention_time=float(retention_time),
        mz_array=mz_array[sort_order],
        intensity_array=intensity_array[sort_order],
    )


def _scan_number_from_title(title: str) -> int | None:
    cleaned = title.strip()
    if not cleaned:
        return None
    for pattern in TITLE_SCAN_PATTERNS:
        match = pattern.search(cleaned)
        if match is not None:
            return int(match.group(1))

    parts = cleaned.split(".")
    if len(parts) >= 3 and parts[1].isdigit():
        if parts[2].isdigit() and parts[1] == parts[2]:
            return int(parts[1])
        if len(parts) >= 4 and parts[2].isdigit():
            return int(parts[1])

    numeric_tokens = re.findall(r"\d+", cleaned)
    if len(numeric_tokens) == 1:
        return int(numeric_tokens[0])
    return None


def _scan_number_from_mgf_params(params: dict[str, object]) -> int:
    scans = params.get("scans")
    if scans is not None and str(scans).strip():
        scan_text = str(scans).split(",", 1)[0].strip()
        if scan_text.isdigit():
            return int(scan_text)

    title = str(params.get("title", "") or "")
    scan_number = _scan_number_from_title(title)
    if scan_number is not None:
        return scan_number

    raise ValueError(
        "Could not derive scan number from MGF entry. Expected SCANS=... or a TITLE "
        f"containing a parseable scan number. params={params}"
    )


def _load_mzduck_parquet_spectra(path: Path) -> dict[int, SpectrumRecord]:
    """Load mzDuck mgf.parquet into a ScanNr-indexed spectrum map."""
    df = pd.read_parquet(path, columns=MZDUCK_COLUMNS)
    spectra: dict[int, SpectrumRecord] = {}
    for row in df.itertuples(index=False):
        scan_number = int(row.scan_number)
        if scan_number in spectra:
            raise ValueError(f"Duplicate scan_number={scan_number} in {path}")
        spectra[scan_number] = _build_spectrum_record(
            scan_number=scan_number,
            retention_time=float(row.rt_seconds),
            mz_array=row.mz_array,
            intensity_array=row.intensity_array,
        )
    return spectra


def _load_text_mgf_spectra(path: Path) -> dict[int, SpectrumRecord]:
    """Load plain-text MGF or gzipped MGF into a ScanNr-indexed spectrum map."""
    spectra: dict[int, SpectrumRecord] = {}
    open_fn = gzip.open if path.name.endswith(".gz") else Path.open
    open_kwargs = {"mode": "rt", "encoding": "utf-8"}
    with open_fn(path, **open_kwargs) as handle:
        with pyteomics_mgf.MGF(handle) as reader:
            for spectrum in reader:
                params = spectrum["params"]
                scan_number = _scan_number_from_mgf_params(params)
                if scan_number in spectra:
                    raise ValueError(f"Duplicate scan_number={scan_number} in {path}")
                spectra[scan_number] = _build_spectrum_record(
                    scan_number=scan_number,
                    retention_time=float(params.get("rtinseconds", 0.0) or 0.0),
                    mz_array=spectrum["m/z array"],
                    intensity_array=spectrum["intensity array"],
                )
    return spectra


def load_spectra(path: Path) -> dict[int, SpectrumRecord]:
    """Load supported spectrum inputs into a ScanNr-indexed spectrum map."""
    normalized_name = _strip_gzip_suffix(path)
    if normalized_name.endswith(".parquet"):
        return _load_mzduck_parquet_spectra(path)
    if normalized_name.endswith(".mgf"):
        return _load_text_mgf_spectra(path)
    raise ValueError(
        f"Unsupported spectrum format for {path}. Expected .mgf.parquet, .mgf, or .mgf.gz."
    )


def read_pin_parquet(path: Path, max_psms: int | None = None) -> pd.DataFrame:
    """Backward-compatible wrapper around read_pin_table()."""
    return read_pin_table(path, max_psms=max_psms)


def load_mzduck_spectra(path: Path) -> dict[int, SpectrumRecord]:
    """Backward-compatible wrapper around load_spectra()."""
    return load_spectra(path)


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
