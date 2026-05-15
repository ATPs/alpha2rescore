"""Core prediction functions: predict_single, predict_batch, correlate."""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from psm_utils import Peptidoform, PSMList
from psm_utils.io import read_file as psm_read_file

from ._utils.peptidoform import peptidoform_to_row
from .constants import (
    APD_FRAG_TYPES_NON_MODLOSS,
    APD_TO_MS2PIP_MAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL,
    MS2PIP_ION_TYPES,
    MS2PIP_MAX_CHARGE,
    MODEL_TO_ALPHAPEPT_MODEL,
)
from .result import ProcessingResult, calculate_correlations
from .spectrum import (
    load_spectrum_index,
    match_peaks,
    normalize_spectrum_id,
)


def _read_psm_input(psms: Union[PSMList, str, Path], psm_filetype: Optional[str]) -> PSMList:
    """Return a PSMList from an existing PSMList or a psm_utils-supported file."""
    if isinstance(psms, (str, Path)):
        filetype = psm_filetype if psm_filetype is not None else "infer"
        return psm_read_file(Path(psms), filetype=filetype)
    if isinstance(psms, PSMList):
        return psms
    raise TypeError(f"psms must be PSMList, str, or Path, got {type(psms)}")


@lru_cache(maxsize=8)
def _get_model_mgr(device: str, model: str):
    """Initialize and return an AlphaPeptDeep ModelManager."""
    from peptdeep.pretrained_models import ModelManager
    from peptdeep.settings import global_settings, update_global_settings

    pepdeep_home = str(Path.home() / "peptdeep" / "pretrained_models")

    if global_settings.get("PEPTDEEP_HOME", "") != pepdeep_home:
        update_global_settings(
            {
                "PEPTDEEP_HOME": pepdeep_home,
                "torch_device": {"device_type": device},
                "thread_num": 1,
                "model_mgr": {"predict": {"multiprocessing": False}},
            }
        )

    model_mgr = ModelManager(device=device)
    apd_model_type = MODEL_TO_ALPHAPEPT_MODEL.get(model, model)
    model_mgr.load_installed_models(apd_model_type)
    return model_mgr


def _match_result_to_spectrum(
    result: ProcessingResult,
    spectrum_dict: dict[str, "ObservedSpectrum"],
    psm_to_spectrum_id: dict[int, Optional[str]],
    ms2_tolerance: float,
) -> bool:
    """Attach observed intensities for one result and report whether it matched."""
    spectrum_id = psm_to_spectrum_id.get(result.psm_index)
    if not spectrum_id or result.theoretical_mz is None:
        return False

    obs_spec = spectrum_dict.get(spectrum_id)
    if obs_spec is None:
        return False

    obs_intensity: dict[str, np.ndarray] = {}
    matched_any = False
    for ion_type in MS2PIP_ION_TYPES:
        theo_mz = result.theoretical_mz.get(ion_type)
        if theo_mz is None:
            continue
        matched = match_peaks(theo_mz.ravel(), obs_spec.mz, obs_spec.intensity, ms2_tolerance)
        matched_any = matched_any or bool(np.any(matched > 0))
        obs_intensity[ion_type] = matched.reshape(theo_mz.shape)

    result.observed_intensity = obs_intensity if obs_intensity else None
    return matched_any


def attach_observed_intensities(
    results: list[ProcessingResult],
    spectrum_dict: dict[str, "ObservedSpectrum"],
    psm_to_spectrum_id: dict[int, Optional[str]],
    ms2_tolerance: float = 0.02,
    processes: int = 1,
) -> int:
    """Attach observed intensities to predicted results.

    This is the CPU-side part that can be run after GPU prediction is finished.
    """
    if not results:
        return 0

    if processes > 1 and len(results) > 1:
        with ThreadPoolExecutor(max_workers=processes) as executor:
            matched_flags = list(
                executor.map(
                    lambda result: _match_result_to_spectrum(
                        result, spectrum_dict, psm_to_spectrum_id, ms2_tolerance
                    ),
                    results,
                )
            )
    else:
        matched_flags = [
            _match_result_to_spectrum(result, spectrum_dict, psm_to_spectrum_id, ms2_tolerance)
            for result in results
        ]

    return int(sum(matched_flags))


def _apd_results_to_processing_result(
    apd_result: dict, psm_indices: Optional[np.ndarray] = None
) -> list[ProcessingResult]:
    """Convert AlphaPeptDeep predict_all output to a list of ProcessingResult.

    Parameters
    ----------
    apd_result : dict
        {"precursor_df": ..., "fragment_mz_df": ..., "fragment_intensity_df": ...}
    psm_indices : np.ndarray or None
        Original PSM indices. If None, sequential indices are used.

    Returns
    -------
    list[ProcessingResult]
    """
    precursor_df = apd_result["precursor_df"].reset_index(drop=False)
    fragment_mz_df = apd_result["fragment_mz_df"]
    fragment_intensity_df = apd_result["fragment_intensity_df"]

    # Columns present in the fragment DataFrames
    available_cols = [
        c for c in APD_FRAG_TYPES_NON_MODLOSS if c in fragment_mz_df.columns
    ]

    results = []
    for df_idx, row in precursor_df.iterrows():
        start = int(row["frag_start_idx"])
        stop = int(row["frag_stop_idx"])
        n_positions = stop - start

        # Build theoretical_mz and predicted_intensity dicts
        theoretical_mz: dict[str, np.ndarray] = {}
        predicted_intensity: dict[str, np.ndarray] = {}

        for ion_type in MS2PIP_ION_TYPES:
            # Build (n_positions, 3) array for this ion type
            mz_arr = np.zeros((n_positions, MS2PIP_MAX_CHARGE), dtype=np.float32)
            int_arr = np.zeros((n_positions, MS2PIP_MAX_CHARGE), dtype=np.float32)

            for apd_col, (mapped_ion, charge_idx) in APD_TO_MS2PIP_MAP.items():
                if mapped_ion != ion_type or apd_col not in available_cols:
                    continue
                mz_arr[:, charge_idx] = fragment_mz_df[apd_col].iloc[start:stop].to_numpy(
                    dtype=np.float32, copy=False
                )
                raw_int = fragment_intensity_df[apd_col].iloc[start:stop].to_numpy(
                    dtype=np.float32, copy=False
                )
                int_arr[:, charge_idx] = np.log2(raw_int + 0.001)

            theoretical_mz[ion_type] = mz_arr
            predicted_intensity[ion_type] = int_arr

        psm_idx = (
            int(row.get("_psm_idx", df_idx))
            if psm_indices is None
            else int(psm_indices[df_idx])
            if df_idx < len(psm_indices)
            else df_idx
        )

        results.append(
            ProcessingResult(
                psm_index=psm_idx,
                theoretical_mz=theoretical_mz,
                predicted_intensity=predicted_intensity,
            )
        )

    return results


def predict_single(
    peptidoform: Union[str, Peptidoform],
    model: str = DEFAULT_MODEL,
    device: str = "cpu",
) -> ProcessingResult:
    """Predict fragmentation spectrum for a single peptidoform.

    Parameters
    ----------
    peptidoform : str or Peptidoform
        ProForma string (e.g. "PGAQANPYSR/3") or Peptidoform object.
    model : str
        MS2PIP-compatible model name. Default: "HCD".
    device : str
        "cpu" or "gpu". Default: "cpu".

    Returns
    -------
    ProcessingResult
    """
    row = peptidoform_to_row(peptidoform)
    df = pd.DataFrame([row])
    model_mgr = _get_model_mgr(device=device, model=model)
    result = model_mgr.predict_all(
        df,
        predict_items=["ms2"],
        frag_types=list(APD_FRAG_TYPES_NON_MODLOSS),
        multiprocessing=False,
    )
    return _apd_results_to_processing_result(result)[0]


def predict_batch(
    psms: Union[PSMList, str, Path],
    model: str = DEFAULT_MODEL,
    device: str = "cpu",
    psm_filetype: Optional[str] = None,
    processes: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[ProcessingResult]:
    """Predict fragmentation spectra for a batch of PSMs.

    Parameters
    ----------
    psms : PSMList, str, or Path
        PSMList or path to a PSM file in a psm_utils-supported format.
    model : str
        Model name. Default: "HCD".
    device : str
        "cpu" or "gpu". Default: "cpu".
    psm_filetype : str or None
        Override PSM file format detection.
    processes : int
        Not used in this version (reserved for future parallelism).
    chunk_size : int
        Precursors per AlphaPeptDeep prediction chunk. Default: 5000.

    Returns
    -------
    list[ProcessingResult]
    """
    psm_list = _read_psm_input(psms, psm_filetype)
    if len(psm_list) == 0:
        return []

    # Build input DataFrame
    rows = []
    for idx, psm in enumerate(psm_list):
        row = peptidoform_to_row(psm.peptidoform)
        row["_psm_idx"] = idx
        rows.append(row)
    full_df = pd.DataFrame(rows)

    # Initialize model once
    model_mgr = _get_model_mgr(device=device, model=model)

    # Predict in chunks
    all_results = []
    for chunk_start in range(0, len(full_df), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(full_df))
        chunk = full_df.iloc[chunk_start:chunk_end].copy()

        apd_result = model_mgr.predict_all(
            chunk,
            predict_items=["ms2"],
            frag_types=list(APD_FRAG_TYPES_NON_MODLOSS),
            multiprocessing=False,
        )
        chunk_results = _apd_results_to_processing_result(apd_result)
        all_results.extend(chunk_results)

    all_results.sort(key=lambda result: result.psm_index)

    # Attach PSMs to results
    for result in all_results:
        if result.psm_index < len(psm_list):
            result.psm = psm_list[result.psm_index]

    return all_results


def correlate(
    psms: Union[PSMList, str, Path],
    spectrum_file: Union[str, Path],
    model: str = DEFAULT_MODEL,
    device: str = "cpu",
    ms2_tolerance: float = 0.02,
    psm_filetype: Optional[str] = None,
    spectrum_id_pattern: Optional[str] = None,
    processes: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[ProcessingResult]:
    """Predict and correlate spectra with observed MS2 peaks.

    Parameters
    ----------
    psms : PSMList, str, or Path
        PSMs with peptidoforms and spectrum IDs.
    spectrum_file : str or Path
        Path to an MGF, mzML, or mzML.gz spectrum file.
    model : str
        Model name. Default: "HCD".
    device : str
        "cpu" or "gpu". Default: "cpu".
    ms2_tolerance : float
        MS2 mass tolerance in Da. Default: 0.02.
    psm_filetype : str or None
        Override PSM file format detection.
    spectrum_id_pattern : str or None
        Regex to extract spectrum ID from spectrum title. Default: "(.*)" (match all).
    processes : int
        CPU worker count for observed-spectrum matching.
    chunk_size : int
        Precursors per prediction chunk.

    Returns
    -------
    list[ProcessingResult]
        Results with both predicted_intensity and observed_intensity populated.
    """
    import logging

    psm_list = _read_psm_input(psms, psm_filetype)
    if len(psm_list) == 0:
        return []

    logger = logging.getLogger(__name__)
    spectrum_dict = load_spectrum_index(spectrum_file, spectrum_id_pattern=spectrum_id_pattern)
    logger.info("Loaded %d spectra from %s", len(spectrum_dict), spectrum_file)

    # Build input DataFrame
    rows = []
    psm_to_spec = {}  # psm_index -> matched spectrum identifier
    for idx, psm in enumerate(psm_list):
        row = peptidoform_to_row(psm.peptidoform)
        row["_psm_idx"] = idx
        rows.append(row)
        spec_id = getattr(psm, "spectrum_id", None)
        if spec_id is not None:
            psm_to_spec[idx] = normalize_spectrum_id(spec_id, spectrum_id_pattern)
        else:
            psm_to_spec[idx] = None
    full_df = pd.DataFrame(rows)

    # Predict
    model_mgr = _get_model_mgr(device=device, model=model)
    all_results = []

    for chunk_start in range(0, len(full_df), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(full_df))
        chunk = full_df.iloc[chunk_start:chunk_end].copy()

        apd_result = model_mgr.predict_all(
            chunk,
            predict_items=["ms2"],
            frag_types=list(APD_FRAG_TYPES_NON_MODLOSS),
            multiprocessing=False,
        )
        chunk_results = _apd_results_to_processing_result(apd_result)
        all_results.extend(chunk_results)

    all_results.sort(key=lambda result: result.psm_index)

    for result in all_results:
        if result.psm_index < len(psm_list):
            result.psm = psm_list[result.psm_index]

    matched_count = attach_observed_intensities(
        all_results,
        spectrum_dict,
        psm_to_spec,
        ms2_tolerance=ms2_tolerance,
        processes=processes,
    )

    calculate_correlations(all_results)

    match_fraction = matched_count / len(all_results) if all_results else 0.0
    logger.info(
        "Matched %d/%d PSMs to spectra (%.1f%%)",
        matched_count,
        len(all_results),
        match_fraction * 100.0,
    )
    if all_results and match_fraction < 0.90:
        warnings.warn(
            f"Low spectrum match fraction: {matched_count}/{len(all_results)} "
            f"({match_fraction:.1%}) for {spectrum_file}",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.warning(
            "Low spectrum match fraction: %d/%d (%.1f%%) for %s",
            matched_count,
            len(all_results),
            match_fraction * 100.0,
            spectrum_file,
        )

    return all_results
