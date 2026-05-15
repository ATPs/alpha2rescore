"""Spectrum file reading and peak matching."""

from __future__ import annotations

import gzip
import io
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Union
from urllib.parse import unquote

import numpy as np


@dataclass
class ObservedSpectrum:
    """An observed MS2 spectrum."""

    mz: np.ndarray
    intensity: np.ndarray
    identifier: str
    precursor_mz: float
    precursor_charge: int
    retention_time: Optional[float] = None


def _open_text_file(path: Path) -> io.TextIOBase:
    """Open plain or gzipped text files as UTF-8 streams."""
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def normalize_spectrum_id(identifier: object, spectrum_id_pattern: Optional[str] = None) -> str:
    """Normalize a spectrum identifier with an optional regular expression."""
    if identifier is None:
        return ""
    identifier_str = unquote(str(identifier))
    if not spectrum_id_pattern:
        return identifier_str

    import re

    match = re.search(spectrum_id_pattern, identifier_str)
    if not match:
        return identifier_str
    return match.group(1) if match.lastindex else match.group(0)


def _sanitize_mgf_text(path: Path) -> io.StringIO:
    """Return a cleaned in-memory MGF stream that skips malformed peak lines.

    Empty or broken `BEGIN IONS` blocks are dropped. Only simple peak lines with
    numeric m/z and intensity values are retained.
    """
    cleaned_blocks: list[str] = []
    current_block: list[str] = []
    in_block = False
    kept_blocks = 0
    dropped_blocks = 0
    dropped_lines = 0

    with _open_text_file(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            upper = line.upper()
            if upper == "BEGIN IONS":
                in_block = True
                current_block = ["BEGIN IONS\n"]
                continue

            if upper == "END IONS":
                if in_block and any(
                    not item.startswith(("BEGIN IONS", "END IONS")) and "=" not in item
                    for item in current_block
                ):
                    current_block.append("END IONS\n")
                    cleaned_blocks.extend(current_block)
                    kept_blocks += 1
                else:
                    dropped_blocks += 1
                in_block = False
                current_block = []
                continue

            if not in_block:
                continue

            if "=" in line:
                current_block.append(line + "\n")
                continue

            parts = line.split()
            if len(parts) < 2:
                dropped_lines += 1
                continue
            try:
                float(parts[0])
                float(parts[1])
            except ValueError:
                dropped_lines += 1
                continue
            current_block.append(f"{parts[0]} {parts[1]}\n")

    if dropped_lines or dropped_blocks:
        warnings.warn(
            f"Sanitized {path.name}: dropped {dropped_lines} malformed peak lines and "
            f"{dropped_blocks} empty/broken MGF blocks",
            RuntimeWarning,
            stacklevel=2,
        )
        logging.getLogger(__name__).warning(
            "Sanitized %s: dropped %d malformed peak lines and %d empty/broken MGF blocks",
            path,
            dropped_lines,
            dropped_blocks,
        )

    if kept_blocks == 0 and cleaned_blocks:
        # Defensive fallback: emit what we kept even if the block filter missed a case.
        return io.StringIO("".join(cleaned_blocks))

    return io.StringIO("".join(cleaned_blocks))


def read_mgf(spectrum_file: Union[str, Path]) -> Generator[ObservedSpectrum, None, None]:
    """Yield ObservedSpectrum objects from an MGF file using pyteomics."""
    from pyteomics import mgf

    path = Path(spectrum_file)
    if not path.exists():
        raise FileNotFoundError(f"Spectrum file not found: {spectrum_file}")

    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-1:] == [".mgf"] or suffixes[-2:] == [".mgf", ".gz"]:
        for spec_dict in mgf.read(_sanitize_mgf_text(path), use_index=False):
            yield _mgf_dict_to_spectrum(spec_dict)
    else:
        raise ValueError(f"Unsupported spectrum file format: {path.suffix}. Supported: .mgf")


def read_mzml(spectrum_file: Union[str, Path]) -> Generator[ObservedSpectrum, None, None]:
    """Yield ObservedSpectrum objects from an mzML file using pyteomics."""
    from pyteomics import mzml

    path = Path(spectrum_file)
    if not path.exists():
        raise FileNotFoundError(f"Spectrum file not found: {spectrum_file}")

    for spec_dict in mzml.read(str(path)):
        yield _mzml_dict_to_spectrum(spec_dict)


def _mgf_dict_to_spectrum(spec_dict: dict) -> ObservedSpectrum:
    """Convert a pyteomics MGF dict to ObservedSpectrum."""
    mz_array = np.array(spec_dict["m/z array"], dtype=np.float32)
    intensity_array = np.array(spec_dict["intensity array"], dtype=np.float32)

    # Sort by m/z (some files have unsorted peaks)
    sort_order = np.argsort(mz_array)
    mz_array = mz_array[sort_order]
    intensity_array = intensity_array[sort_order]

    params = spec_dict.get("params", {})
    identifier = params.get("title", params.get("TITLE", ""))
    pepmass = params.get("pepmass", params.get("PEPMASS", (0, None)))
    precursor_mz = float(pepmass[0]) if isinstance(pepmass, (list, tuple)) else float(pepmass)
    charge = params.get("charge", params.get("CHARGE", 0))
    if isinstance(charge, (list, tuple)):
        charge = charge[0] if charge else 0
    charge = int(str(charge).rstrip("+").rstrip("-"))

    rt = params.get("rtinseconds", params.get("RTINSECONDS", None))
    if rt is not None:
        rt = float(rt)

    return ObservedSpectrum(
        mz=mz_array,
        intensity=intensity_array,
        identifier=str(identifier),
        precursor_mz=precursor_mz,
        precursor_charge=charge,
        retention_time=rt,
    )


def _mzml_dict_to_spectrum(spec_dict: dict) -> ObservedSpectrum:
    """Convert a pyteomics mzML dict to ObservedSpectrum."""
    mz_array = np.asarray(spec_dict.get("m/z array", []), dtype=np.float32)
    intensity_array = np.asarray(spec_dict.get("intensity array", []), dtype=np.float32)

    sort_order = np.argsort(mz_array)
    mz_array = mz_array[sort_order]
    intensity_array = intensity_array[sort_order]

    identifier = spec_dict.get("id", spec_dict.get("nativeID", ""))
    precursor_mz = 0.0
    precursor_charge = 0
    retention_time = None

    precursor_list = spec_dict.get("precursorList", {})
    precursor_items = precursor_list.get("precursor", []) if isinstance(precursor_list, dict) else []
    if precursor_items:
        selected_ions = precursor_items[0].get("selectedIonList", {}).get("selectedIon", [])
        if selected_ions:
            first_ion = selected_ions[0]
            precursor_mz = float(first_ion.get("selected ion m/z", 0.0) or 0.0)
            precursor_charge = int(first_ion.get("charge state", 0) or 0)

    scan_list = spec_dict.get("scanList", {})
    scan_items = scan_list.get("scan", []) if isinstance(scan_list, dict) else []
    if scan_items:
        rt = scan_items[0].get("scan start time")
        if rt is not None:
            retention_time = float(rt)

    return ObservedSpectrum(
        mz=mz_array,
        intensity=intensity_array,
        identifier=str(identifier),
        precursor_mz=precursor_mz,
        precursor_charge=precursor_charge,
        retention_time=retention_time,
    )


def read_spectrum_file(spectrum_file: Union[str, Path]) -> Generator[ObservedSpectrum, None, None]:
    """Yield ObservedSpectrum objects from MGF or mzML files."""
    path = Path(spectrum_file)
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-1:] == [".mgf"] or suffixes[-2:] == [".mgf", ".gz"]:
        yield from read_mgf(path)
    elif suffixes[-1:] == [".mzml"] or suffixes[-2:] == [".mzml", ".gz"]:
        yield from read_mzml(path)
    else:
        raise ValueError(
            f"Unsupported spectrum file format: {path.suffixes}. Supported: .mgf, .mzML"
        )


def load_spectrum_index(
    spectrum_file: Union[str, Path],
    spectrum_id_pattern: Optional[str] = None,
) -> dict[str, ObservedSpectrum]:
    """Load a spectrum file into a normalized identifier -> spectrum map."""
    spectrum_dict: dict[str, ObservedSpectrum] = {}
    logger = logging.getLogger(__name__)
    for spec in read_spectrum_file(spectrum_file):
        key = normalize_spectrum_id(spec.identifier, spectrum_id_pattern)
        if key in spectrum_dict:
            warnings.warn(
                f"Duplicate normalized spectrum ID {key!r} in {spectrum_file}; keeping the last entry",
                RuntimeWarning,
                stacklevel=2,
            )
            logger.warning(
                "Duplicate normalized spectrum ID %r in %s; keeping the last entry",
                key,
                spectrum_file,
            )
        spectrum_dict[key] = spec
    return spectrum_dict


def match_peaks(
    theoretical_mz: np.ndarray,
    observed_mz: np.ndarray,
    observed_intensity: np.ndarray,
    ms2_tolerance: float = 0.02,
) -> np.ndarray:
    """Extract observed intensities at theoretical m/z positions.

    For each theoretical m/z, find the closest observed peak within ms2_tolerance.
    If no peak is within tolerance, the intensity is 0.0.

    Parameters
    ----------
    theoretical_mz : shape (n_ions,)
        Theoretical m/z values for fragment ions.
    observed_mz : shape (n_peaks,)
        Observed m/z values, sorted ascending.
    observed_intensity : shape (n_peaks,)
        Observed intensities.
    ms2_tolerance : float
        Mass tolerance in Da.

    Returns
    -------
    matched_intensity : shape (n_ions,)
        Matched intensities (log2 space), or 0.0 where no match found.
    """
    observed_mz = np.asarray(observed_mz, dtype=np.float32)
    observed_intensity = np.asarray(observed_intensity, dtype=np.float32)
    theoretical_mz = np.asarray(theoretical_mz, dtype=np.float32)

    matched = np.zeros(len(theoretical_mz), dtype=np.float32)
    valid = theoretical_mz > 0
    if not np.any(valid):
        return matched

    valid_theo = theoretical_mz[valid]
    idx = np.searchsorted(observed_mz, valid_theo)
    left_idx = np.clip(idx - 1, 0, len(observed_mz) - 1)
    right_idx = np.clip(idx, 0, len(observed_mz) - 1)

    left_dist = valid_theo - observed_mz[left_idx]
    right_dist = observed_mz[right_idx] - valid_theo

    use_left = (idx > 0) & (left_dist <= right_dist) & (left_dist < ms2_tolerance)
    use_right = (idx < len(observed_mz)) & (right_dist < left_dist) & (right_dist < ms2_tolerance)
    chosen_idx = np.where(use_left, left_idx, np.where(use_right, right_idx, -1))

    hit_mask = chosen_idx >= 0
    if np.any(hit_mask):
        matched_values = np.log2(observed_intensity[chosen_idx[hit_mask]] + 0.001)
        valid_indices = np.flatnonzero(valid)
        matched[valid_indices[hit_mask]] = matched_values

    return matched
