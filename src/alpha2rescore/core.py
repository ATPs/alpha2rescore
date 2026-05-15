"""Core alpha2rescore build orchestration."""

from __future__ import annotations

import copy
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .config import Alpha2RescoreConfig, BuildResult
from .deeplc_module import (
    DEEPLC_BASE_FEATURES,
    DEEPLC_FEATURES,
    build_deeplc_base_features,
    finalize_deeplc_features,
    load_deeplc_calibration,
)
from .features import FEATURE_NAMES, PredictedSpectrum, calculate_feature_dict, empty_feature_dict
from .io import (
    insert_feature_columns,
    load_mzduck_spectra,
    read_cache_parquet,
    read_json,
    read_pin_parquet,
    write_json_atomic,
    write_parquet_atomic,
    write_pin_gz,
)
from .logging_utils import format_duration, log
from .peptides import (
    extract_charge,
    make_precursor_key,
    make_psm_key,
    parse_pin_peptide,
)
from .subprocess_utils import run_python_module


FINAL_FEATURE_COLUMNS = FEATURE_NAMES + DEEPLC_FEATURES
COMPONENT = "alpha2rescore.build"


def _augment_pin_df(pin_df: pd.DataFrame, idn: str) -> pd.DataFrame:
    augmented = pin_df.copy()
    charges = []
    sequences = []
    var_mods = []
    precursor_keys = []
    psm_keys = []
    labels = []
    for row in augmented.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        charge = extract_charge(row_series)
        parsed = parse_pin_peptide(str(row.Peptide))
        label = int(row.Label)
        charges.append(charge)
        sequences.append(parsed.sequence)
        var_mods.append(parsed.var_mod_sites_unimod)
        labels.append(label)
        precursor_key = make_precursor_key(
            label=label,
            sequence=parsed.sequence,
            var_mod_sites_unimod=parsed.var_mod_sites_unimod,
            charge=charge,
        )
        precursor_keys.append(precursor_key)
        psm_keys.append(
            make_psm_key(
                idn=idn,
                spec_id=row.SpecId,
                peptide=row.Peptide,
                charge=charge,
                label=label,
            )
        )

    augmented["charge"] = charges
    augmented["label_int"] = labels
    augmented["sequence"] = sequences
    augmented["var_mod_sites_unimod"] = var_mods
    augmented["precursor_key"] = precursor_keys
    augmented["psm_key"] = psm_keys
    return augmented


def _merge_cache(
    cache_df: pd.DataFrame,
    new_df: pd.DataFrame,
    key_column: str,
) -> pd.DataFrame:
    if cache_df.empty:
        merged = new_df.copy()
    elif new_df.empty:
        merged = cache_df.copy()
    else:
        merged = pd.concat([cache_df, new_df], ignore_index=True)
    if merged.empty:
        return merged
    return merged.drop_duplicates(subset=[key_column], keep="last").reset_index(drop=True)


def _run_postgres_lookup(
    precursor_df: pd.DataFrame,
    config: Alpha2RescoreConfig,
) -> pd.DataFrame:
    started_at = perf_counter()
    log(
        f"dispatching PostgreSQL lookup for {len(precursor_df)} precursor keys",
        COMPONENT,
    )
    with tempfile.TemporaryDirectory(prefix="alpha2rescore.lookup.") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_path = temp_dir / "lookup_input.parquet"
        output_path = temp_dir / "lookup_output.parquet"
        precursor_df.to_parquet(input_path, index=False)
        run_python_module(
            python_executable=config.postgres_python,
            module="alpha2rescore.postgres_helper",
            args=[
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--db-host",
                config.postgres_host,
                "--db-port",
                str(config.postgres_port),
                "--db-name",
                config.postgres_dbname,
                "--db-user",
                config.postgres_user,
                "--db-password-file",
                config.postgres_password_file,
                "--db-schema",
                config.postgres_schema,
            ],
            src_root=config.helper_src_root,
            extra_env=config.extra_env,
        )
        output_df = pd.read_parquet(output_path)
    log(
        f"PostgreSQL lookup returned {len(output_df)} rows in {format_duration(perf_counter() - started_at)}",
        COMPONENT,
    )
    return output_df


def _run_local_prediction(
    missing_df: pd.DataFrame,
    config: Alpha2RescoreConfig,
) -> pd.DataFrame:
    started_at = perf_counter()
    log(
        f"dispatching local AlphaPept prediction for {len(missing_df)} precursor keys",
        COMPONENT,
    )
    with tempfile.TemporaryDirectory(prefix="alpha2rescore.localpred.") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_path = temp_dir / "predict_input.parquet"
        output_path = temp_dir / "predict_output.parquet"
        missing_df.to_parquet(input_path, index=False)
        run_python_module(
            python_executable=config.alphapept_python,
            module="alpha2rescore.alphapept_helper",
            args=[
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--device",
                config.alphapept_device,
                "--processes",
                str(config.alphapept_processes),
            ],
            src_root=config.helper_src_root,
            extra_env=config.extra_env,
        )
        output_df = pd.read_parquet(output_path)
    log(
        f"local AlphaPept prediction returned {len(output_df)} rows in {format_duration(perf_counter() - started_at)}",
        COMPONENT,
    )
    return output_df


def _update_precursor_cache(
    current_rows: pd.DataFrame,
    config: Alpha2RescoreConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    cache_path = config.cache_dir / "precursor_cache.parquet"
    cached = read_cache_parquet(cache_path, key_column="precursor_key")
    current_precursors = current_rows.loc[
        :,
        ["precursor_key", "label_int", "sequence", "var_mod_sites_unimod", "charge"],
    ].drop_duplicates(subset=["precursor_key"])
    current_precursors = current_precursors.rename(
        columns={"label_int": "label", "sequence": "pep_seq"}
    )

    cached_keys = set(cached["precursor_key"].tolist()) if not cached.empty else set()
    missing_precursors = current_precursors[
        ~current_precursors["precursor_key"].isin(cached_keys)
    ].reset_index(drop=True)

    lookup_df = pd.DataFrame()
    local_df = pd.DataFrame()
    if not missing_precursors.empty:
        lookup_df = _run_postgres_lookup(missing_precursors, config)
        local_pending = lookup_df[
            (lookup_df["prediction_source"] == "missing_prediction")
            & lookup_df["variant_id"].notna()
        ].copy()
        if not local_pending.empty:
            local_df = _run_local_prediction(local_pending, config)
            local_df = local_df.set_index("precursor_key")
            lookup_df = lookup_df.set_index("precursor_key")
            for column in [
                "prediction_source",
                "b_mz",
                "b_intensity",
                "y_mz",
                "y_intensity",
            ]:
                lookup_df.loc[local_df.index, column] = local_df[column]
            lookup_df = lookup_df.reset_index()
        cached = _merge_cache(cached, lookup_df, "precursor_key")
        write_parquet_atomic(cached, cache_path)

    stats = {
        "precursor_count": int(len(current_precursors)),
        "postgres_prediction_hits": int(
            (lookup_df["prediction_source"] == "postgres").sum() if not lookup_df.empty else 0
        ),
        "local_prediction_count": int(len(local_df)),
    }
    return cached, stats


def _predicted_from_row(row: pd.Series) -> PredictedSpectrum | None:
    if row.get("b_mz") is None or row.get("b_intensity") is None:
        return None

    def _to_float32_matrix(value) -> np.ndarray:
        array = np.asarray(value, dtype=object)
        if array.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if array.dtype != object:
            return np.asarray(array, dtype=np.float32)
        return np.stack([np.asarray(item, dtype=np.float32) for item in array], axis=0)

    return PredictedSpectrum(
        b_mz=_to_float32_matrix(row["b_mz"]),
        b_intensity=_to_float32_matrix(row["b_intensity"]),
        y_mz=_to_float32_matrix(row["y_mz"]),
        y_intensity=_to_float32_matrix(row["y_intensity"]),
    )


def _build_precursor_prediction_lookup(
    precursor_cache: pd.DataFrame,
) -> dict[str, PredictedSpectrum | None]:
    lookup: dict[str, PredictedSpectrum | None] = {}
    if precursor_cache.empty:
        return lookup
    for row in precursor_cache.itertuples(index=False):
        row_dict = row._asdict()
        lookup[str(row_dict["precursor_key"])] = _predicted_from_row(pd.Series(row_dict))
    return lookup


def _iter_batches(items: list[tuple[str, int, str]], batch_size: int):
    if batch_size < 1:
        raise ValueError("alpha_feature_batch_size must be >= 1")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _score_alpha_feature_batch(
    batch: list[tuple[str, int, str]],
    precursor_prediction_lookup: dict[str, PredictedSpectrum | None],
    spectra_by_scan: dict[int, object],
    ms2_tolerance: float,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for psm_key, scan_number, precursor_key in batch:
        feature_dict = empty_feature_dict()
        observed = spectra_by_scan.get(int(scan_number))
        predicted = precursor_prediction_lookup.get(str(precursor_key))
        if observed is not None and predicted is not None:
            feature_dict = calculate_feature_dict(
                predicted=predicted,
                observed=observed,
                ms2_tolerance=ms2_tolerance,
            )
        feature_dict["psm_key"] = psm_key
        rows.append(feature_dict)
    return rows


def _log_alpha_progress(
    completed_batches: int,
    total_batches: int,
    batch_size: int,
    total_items: int,
) -> None:
    if total_batches <= 1:
        return
    progress_every = max(1, total_batches // 10)
    if (
        completed_batches == 1
        or completed_batches == total_batches
        or completed_batches % progress_every == 0
    ):
        processed = min(completed_batches * batch_size, total_items)
        log(
            f"alpha feature scoring progress: {processed}/{total_items} PSMs "
            f"({completed_batches}/{total_batches} batches)",
            COMPONENT,
        )


def _update_alpha_feature_cache(
    current_rows: pd.DataFrame,
    precursor_cache: pd.DataFrame,
    spectra_by_scan: dict[int, object],
    config: Alpha2RescoreConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    cache_path = config.cache_dir / "alpha_features.parquet"
    cached = read_cache_parquet(cache_path, key_column="psm_key")
    cached_keys = set(cached["psm_key"].tolist()) if not cached.empty else set()
    pending = current_rows[~current_rows["psm_key"].isin(cached_keys)].copy()

    new_rows: list[dict[str, float]] = []
    if not pending.empty:
        precursor_prediction_lookup = _build_precursor_prediction_lookup(precursor_cache)
        pending_items = list(
            pending.loc[:, ["psm_key", "ScanNr", "precursor_key"]].itertuples(
                index=False, name=None
            )
        )
        total_batches = (
            (len(pending_items) + config.alpha_feature_batch_size - 1)
            // config.alpha_feature_batch_size
        )
        if config.alpha_feature_threads <= 1 or len(pending_items) <= config.alpha_feature_batch_size:
            for batch_index, batch in enumerate(
                _iter_batches(pending_items, config.alpha_feature_batch_size),
                start=1,
            ):
                new_rows.extend(
                    _score_alpha_feature_batch(
                        batch=batch,
                        precursor_prediction_lookup=precursor_prediction_lookup,
                        spectra_by_scan=spectra_by_scan,
                        ms2_tolerance=config.ms2_tolerance,
                    )
                )
                _log_alpha_progress(
                    completed_batches=batch_index,
                    total_batches=total_batches,
                    batch_size=config.alpha_feature_batch_size,
                    total_items=len(pending_items),
                )
        else:
            with ThreadPoolExecutor(max_workers=config.alpha_feature_threads) as executor:
                for batch_index, batch_rows in enumerate(
                    executor.map(
                    _score_alpha_feature_batch,
                    list(_iter_batches(pending_items, config.alpha_feature_batch_size)),
                    [precursor_prediction_lookup] * (
                        (len(pending_items) + config.alpha_feature_batch_size - 1)
                        // config.alpha_feature_batch_size
                    ),
                    [spectra_by_scan]
                    * (
                        (len(pending_items) + config.alpha_feature_batch_size - 1)
                        // config.alpha_feature_batch_size
                    ),
                    [config.ms2_tolerance]
                    * (
                        (len(pending_items) + config.alpha_feature_batch_size - 1)
                        // config.alpha_feature_batch_size
                    ),
                    ),
                    start=1,
                ):
                    new_rows.extend(batch_rows)
                    _log_alpha_progress(
                        completed_batches=batch_index,
                        total_batches=total_batches,
                        batch_size=config.alpha_feature_batch_size,
                        total_items=len(pending_items),
                    )

    new_df = pd.DataFrame(new_rows)
    cached = _merge_cache(cached, new_df, "psm_key")
    write_parquet_atomic(cached, cache_path)
    return cached, {
        "alpha_cache_hits": int(len(current_rows) - len(pending)),
        "alpha_cache_misses": int(len(pending)),
    }


def _update_deeplc_cache(
    current_rows: pd.DataFrame,
    config: Alpha2RescoreConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    cache_path = config.cache_dir / "deeplc_base_features.parquet"
    cached = read_cache_parquet(cache_path, key_column="psm_key")
    cached_keys = set(cached["psm_key"].tolist()) if not cached.empty else set()
    pending = current_rows[~current_rows["psm_key"].isin(cached_keys)].copy()

    if not pending.empty:
        new_df = build_deeplc_base_features(
            pin_df=current_rows,
            pending_df=pending,
            config=config,
        )
        cached = _merge_cache(cached, new_df, "psm_key")
        write_parquet_atomic(cached, cache_path)

    return cached, {
        "deeplc_cache_hits": int(len(current_rows) - len(pending)),
        "deeplc_cache_misses": int(len(pending)),
    }


def _build_feature_frame(
    current_rows: pd.DataFrame,
    alpha_cache: pd.DataFrame,
    deeplc_cache: pd.DataFrame,
) -> pd.DataFrame:
    alpha_subset = current_rows.loc[:, ["psm_key"]].merge(alpha_cache, on="psm_key", how="left")
    for column in FEATURE_NAMES:
        alpha_subset[column] = alpha_subset[column].fillna(0.0)

    deeplc_subset = deeplc_cache[
        deeplc_cache["psm_key"].isin(current_rows["psm_key"])
    ].copy()
    deeplc_features = finalize_deeplc_features(current_rows, deeplc_subset)
    merged = alpha_subset.merge(deeplc_features, on="psm_key", how="left")
    for column in DEEPLC_FEATURES:
        merged[column] = merged[column].fillna(0.0)
    return merged.loc[:, ["psm_key", *FINAL_FEATURE_COLUMNS]]


def _write_manifest(
    config: Alpha2RescoreConfig,
    pin_df: pd.DataFrame,
    precursor_stats: dict[str, int],
    alpha_stats: dict[str, int],
    deeplc_stats: dict[str, int],
) -> None:
    manifest = read_json(config.cache_dir / "manifest.json", {})
    manifest.update(
        {
            "idn": config.idn,
            "pin_parquet": str(config.pin_parquet),
            "mgf_parquet": str(config.mgf_parquet),
            "pin_size": config.pin_parquet.stat().st_size,
            "mgf_size": config.mgf_parquet.stat().st_size,
            "pin_mtime_ns": config.pin_parquet.stat().st_mtime_ns,
            "mgf_mtime_ns": config.mgf_parquet.stat().st_mtime_ns,
            "psm_count": int(len(pin_df)),
            "precursor_count": int(precursor_stats["precursor_count"]),
            "alpha_cache_rows": int(alpha_stats["alpha_cache_hits"] + alpha_stats["alpha_cache_misses"]),
            "deeplc_cache_rows": int(deeplc_stats["deeplc_cache_hits"] + deeplc_stats["deeplc_cache_misses"]),
            "deeplc_calibration_present": load_deeplc_calibration(config.idn, config.cache_dir)
            is not None,
        }
    )
    write_json_atomic(manifest, config.cache_dir / "manifest.json")


def _build_features_internal(
    config: Alpha2RescoreConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build feature rows and return per-run counters used by API and CLI."""
    build_started_at = perf_counter()
    runtime_config = copy.copy(config)
    runtime_config.cache_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"starting build for idn={runtime_config.idn} pin={runtime_config.pin_parquet} "
        f"mgf={runtime_config.mgf_parquet}",
        COMPONENT,
    )

    stage_started_at = perf_counter()
    original_pin_df = read_pin_parquet(runtime_config.pin_parquet, runtime_config.max_psms)
    current_rows = _augment_pin_df(original_pin_df, runtime_config.idn)
    log(
        f"loaded {len(original_pin_df)} PSM rows and derived {current_rows['precursor_key'].nunique()} precursor keys "
        f"in {format_duration(perf_counter() - stage_started_at)}",
        COMPONENT,
    )

    stage_started_at = perf_counter()
    spectra_by_scan = load_mzduck_spectra(runtime_config.mgf_parquet)
    current_rows["observed_retention_time"] = [
        float(spectra_by_scan.get(int(scan)).retention_time)
        if int(scan) in spectra_by_scan
        else 0.0
        for scan in current_rows["ScanNr"].tolist()
    ]
    log(
        f"loaded {len(spectra_by_scan)} mzDuck spectra in {format_duration(perf_counter() - stage_started_at)}",
        COMPONENT,
    )

    stage_started_at = perf_counter()
    precursor_cache, precursor_stats = _update_precursor_cache(current_rows, runtime_config)
    log(
        f"precursor cache ready: total={precursor_stats['precursor_count']} "
        f"postgres_hits={precursor_stats['postgres_prediction_hits']} "
        f"local_predictions={precursor_stats['local_prediction_count']} "
        f"in {format_duration(perf_counter() - stage_started_at)}",
        COMPONENT,
    )

    stage_started_at = perf_counter()
    alpha_cache, alpha_stats = _update_alpha_feature_cache(
        current_rows=current_rows,
        precursor_cache=precursor_cache,
        spectra_by_scan=spectra_by_scan,
        config=runtime_config,
    )
    log(
        f"alpha feature cache ready: hits={alpha_stats['alpha_cache_hits']} "
        f"misses={alpha_stats['alpha_cache_misses']} threads={runtime_config.alpha_feature_threads} "
        f"in {format_duration(perf_counter() - stage_started_at)}",
        COMPONENT,
    )

    stage_started_at = perf_counter()
    deeplc_cache, deeplc_stats = _update_deeplc_cache(current_rows, runtime_config)
    log(
        f"DeepLC cache ready: hits={deeplc_stats['deeplc_cache_hits']} "
        f"misses={deeplc_stats['deeplc_cache_misses']} processes={runtime_config.deeplc_processes} "
        f"in {format_duration(perf_counter() - stage_started_at)}",
        COMPONENT,
    )

    stage_started_at = perf_counter()
    feature_df = _build_feature_frame(current_rows, alpha_cache, deeplc_cache)
    combined = insert_feature_columns(original_pin_df, feature_df, FINAL_FEATURE_COLUMNS)
    _write_manifest(runtime_config, original_pin_df, precursor_stats, alpha_stats, deeplc_stats)
    log(
        f"assembled final feature frame in {format_duration(perf_counter() - stage_started_at)}",
        COMPONENT,
    )
    stats = {
        "psm_count": int(len(original_pin_df)),
        "precursor_count": int(precursor_stats["precursor_count"]),
        "alpha_cache_hits": int(alpha_stats["alpha_cache_hits"]),
        "alpha_cache_misses": int(alpha_stats["alpha_cache_misses"]),
        "postgres_prediction_hits": int(precursor_stats["postgres_prediction_hits"]),
        "local_prediction_count": int(precursor_stats["local_prediction_count"]),
        "deeplc_cache_hits": int(deeplc_stats["deeplc_cache_hits"]),
        "deeplc_cache_misses": int(deeplc_stats["deeplc_cache_misses"]),
        "build_seconds": float(perf_counter() - build_started_at),
    }
    log(
        f"finished build preparation for idn={runtime_config.idn} in {format_duration(stats['build_seconds'])}",
        COMPONENT,
    )
    return combined, stats


def build_features(
    pin_parquet,
    mgf_parquet,
    config: Alpha2RescoreConfig,
) -> pd.DataFrame:
    """Build feature-augmented PIN rows without writing the final file."""
    runtime_config = copy.copy(config)
    runtime_config.pin_parquet = Path(pin_parquet)
    runtime_config.mgf_parquet = Path(mgf_parquet)
    combined, _stats = _build_features_internal(runtime_config)
    return combined


def build_pin(config: Alpha2RescoreConfig) -> BuildResult:
    """Build and write the final gzipped alpha2rescore PIN."""
    combined_df, stats = _build_features_internal(config)
    write_started_at = perf_counter()
    output_pin = config.out_dir / f"{config.idn}.comet_alpha2rescore.pin.gz"
    write_pin_gz(combined_df, FINAL_FEATURE_COLUMNS, output_pin)
    log(
        f"wrote final pin {output_pin} in {format_duration(perf_counter() - write_started_at)}",
        COMPONENT,
    )

    return BuildResult(
        idn=config.idn,
        output_pin=output_pin,
        cache_dir=config.cache_dir,
        psm_count=stats["psm_count"],
        precursor_count=stats["precursor_count"],
        alpha_cache_hits=stats["alpha_cache_hits"],
        alpha_cache_misses=stats["alpha_cache_misses"],
        postgres_prediction_hits=stats["postgres_prediction_hits"],
        local_prediction_count=stats["local_prediction_count"],
        deeplc_cache_hits=stats["deeplc_cache_hits"],
        deeplc_cache_misses=stats["deeplc_cache_misses"],
        metadata={
            "feature_columns": FINAL_FEATURE_COLUMNS,
            "build_seconds": stats["build_seconds"],
        },
    )
