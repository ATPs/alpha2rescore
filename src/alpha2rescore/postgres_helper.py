"""PostgreSQL lookup helper for alpha2rescore."""

from __future__ import annotations

import argparse
import math
from time import perf_counter
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from .logging_utils import format_duration, log


COMPONENT = "alpha2rescore.postgres_helper"
FETCH_BATCH_SIZE = 2048


def read_password(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bulk-resolve peptides against ProtInsight PostgreSQL and, when present, "
            "normalize AlphaPept fragment arrays into b/y matrices."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="Input parquet with precursor keys")
    parser.add_argument("--output", required=True, type=Path, help="Output parquet path")
    parser.add_argument("--db-host", default="10.110.120.2")
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-name", default="proteome")
    parser.add_argument("--db-user", default="xlab")
    parser.add_argument(
        "--db-password-file",
        default="/data/users/x/.ssh/20250505xcweb.server2.xlab.postgresql.passwd",
    )
    parser.add_argument("--db-schema", default="protein_hs")
    return parser.parse_args()


def load_peak_code_map(conn, schema: str) -> dict[int, tuple[str, int, int, str]]:
    query = f"""
        SELECT peak_code, ion_type, position, fragment_charge, neutral_loss
        FROM "{schema}"."fragment_peak_codes"
    """
    mapping: dict[int, tuple[str, int, int, str]] = {}
    with conn.cursor() as cursor:
        cursor.execute(query)
        for peak_code, ion_type, position, fragment_charge, neutral_loss in cursor.fetchall():
            mapping[int(peak_code)] = (
                str(ion_type),
                int(position),
                int(fragment_charge),
                str(neutral_loss),
            )
    return mapping


def reconstruct_prediction(
    sequence: str,
    peak_codes,
    mz_values,
    intensity_values,
    peak_code_map: dict[int, tuple[str, int, int, str]],
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]]:
    n_positions = max(len(sequence) - 1, 0)
    b_mz = np.zeros((n_positions, 3), dtype=np.float32)
    b_intensity = np.zeros((n_positions, 3), dtype=np.float32)
    y_mz = np.zeros((n_positions, 3), dtype=np.float32)
    y_intensity = np.zeros((n_positions, 3), dtype=np.float32)

    if n_positions == 0:
        return (
            b_mz.tolist(),
            b_intensity.tolist(),
            y_mz.tolist(),
            y_intensity.tolist(),
        )

    if not peak_codes or not mz_values or not intensity_values:
        return (
            b_mz.tolist(),
            b_intensity.tolist(),
            y_mz.tolist(),
            y_intensity.tolist(),
        )

    for peak_code, mz_value, intensity_value in zip(peak_codes, mz_values, intensity_values):
        peak_meta = peak_code_map.get(int(peak_code))
        if peak_meta is None:
            continue
        ion_type, position, fragment_charge, neutral_loss = peak_meta
        if neutral_loss != "none" or ion_type not in {"b", "y"}:
            continue
        if fragment_charge < 1 or fragment_charge > 3:
            continue
        if position < 1 or position > n_positions:
            continue
        if not math.isfinite(float(mz_value)) or not math.isfinite(float(intensity_value)):
            continue
        if float(mz_value) <= 0 or float(intensity_value) <= 0:
            continue
        column = fragment_charge - 1
        if ion_type == "b":
            row = position - 1
            b_mz[row, column] = float(mz_value)
            b_intensity[row, column] = float(np.log2(float(intensity_value) + 0.001))
        else:
            row = n_positions - position
            y_mz[row, column] = float(mz_value)
            y_intensity[row, column] = float(np.log2(float(intensity_value) + 0.001))
    return b_mz.tolist(), b_intensity.tolist(), y_mz.tolist(), y_intensity.tolist()


def fetch_group(
    conn,
    group_df: pd.DataFrame,
    schema: str,
    pep_table: str,
    variant_table: str,
    alphapep_table: str,
) -> pd.DataFrame:
    if group_df.empty:
        return pd.DataFrame()

    started_at = perf_counter()
    log(
        f"starting lookup for {len(group_df)} precursor keys via {schema}.{pep_table}/{variant_table}",
        COMPONENT,
    )
    with conn.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS alpha2rescore_query_precursors")
        cursor.execute(
            """
            CREATE TEMP TABLE alpha2rescore_query_precursors (
                precursor_key text NOT NULL,
                label integer NOT NULL,
                pep_seq text NOT NULL,
                var_mod_sites_unimod text NOT NULL,
                charge smallint NOT NULL
            ) ON COMMIT DROP
            """
        )
        rows = [
            (
                str(precursor_key),
                int(label),
                str(pep_seq),
                str(var_mod_sites_unimod or ""),
                int(charge),
            )
            for precursor_key, label, pep_seq, var_mod_sites_unimod, charge in group_df.loc[
                :,
                ["precursor_key", "label", "pep_seq", "var_mod_sites_unimod", "charge"],
            ].itertuples(index=False, name=None)
        ]
        execute_values(
            cursor,
            """
            INSERT INTO alpha2rescore_query_precursors
            (precursor_key, label, pep_seq, var_mod_sites_unimod, charge)
            VALUES %s
            """,
            rows,
            page_size=1000,
        )
        cursor.execute("ANALYZE alpha2rescore_query_precursors")
        resolve_started_at = perf_counter()
        cursor.execute("DROP TABLE IF EXISTS alpha2rescore_resolved_precursors")
        cursor.execute(
            f"""
            CREATE TEMP TABLE alpha2rescore_resolved_precursors ON COMMIT DROP AS
            SELECT *
            FROM (
                SELECT
                    q.precursor_key,
                    q.label,
                    q.pep_seq,
                    q.var_mod_sites_unimod,
                    q.charge,
                    v.variant_id,
                    COALESCE(v.fixed_mod_sites_unimod, '') AS fixed_mod_sites_unimod,
                    COALESCE(array_to_string(v.all_unimods, ';'), '') AS all_unimods,
                    CASE
                        WHEN v.variant_id IS NULL THEN NULL
                        ELSE (v.variant_id * 10 + q.charge)
                    END AS precursor_id
                FROM alpha2rescore_query_precursors q
                LEFT JOIN "{schema}"."{pep_table}" p
                    ON p.pep_seq = q.pep_seq
                LEFT JOIN "{schema}"."{variant_table}" v
                    ON v.peptide_id = p.peptide_id
                    AND v.var_mod_sites_unimod = q.var_mod_sites_unimod
            ) resolved
            ORDER BY precursor_id NULLS LAST
            """
        )
        cursor.execute("ANALYZE alpha2rescore_resolved_precursors")
        log(
            f"{variant_table}: resolved {len(group_df)} precursor keys in "
            f"{format_duration(perf_counter() - resolve_started_at)}",
            COMPONENT,
        )

        query = f"""
            SELECT
                r.precursor_key,
                r.label,
                r.pep_seq,
                r.var_mod_sites_unimod,
                r.charge,
                r.variant_id,
                r.fixed_mod_sites_unimod,
                r.all_unimods,
                r.precursor_id,
                a.peak_code,
                a.mz,
                a.intensity
            FROM alpha2rescore_resolved_precursors r
            LEFT JOIN "{schema}"."{alphapep_table}" a
                ON a.precursor_id = r.precursor_id
        """
        cursor_name = f"alpha2rescore_lookup_{uuid4().hex}"

    records = []
    fetched = 0
    with conn.cursor(name=cursor_name) as stream_cursor:
        stream_cursor.itersize = FETCH_BATCH_SIZE
        stream_cursor.execute(query)
        while True:
            batch = stream_cursor.fetchmany(FETCH_BATCH_SIZE)
            if not batch:
                break
            records.extend(batch)
            fetched += len(batch)
            if fetched % (FETCH_BATCH_SIZE * 10) == 0:
                log(
                    f"{variant_table}: fetched {fetched} lookup rows so far",
                    COMPONENT,
                )

    with conn.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS alpha2rescore_resolved_precursors")
        cursor.execute("DROP TABLE IF EXISTS alpha2rescore_query_precursors")

    log(
        f"finished {variant_table}: {fetched} rows in {format_duration(perf_counter() - started_at)}",
        COMPONENT,
    )

    columns = [
        "precursor_key",
        "label",
        "pep_seq",
        "var_mod_sites_unimod",
        "charge",
        "variant_id",
        "fixed_mod_sites_unimod",
        "all_unimods",
        "precursor_id",
        "peak_code",
        "mz",
        "intensity",
    ]
    return pd.DataFrame(records, columns=columns)


def collapse_lookup_rows(
    lookup_df: pd.DataFrame,
    peak_code_map: dict[int, tuple[str, int, int, str]],
) -> pd.DataFrame:
    rows = []
    if lookup_df.empty:
        return pd.DataFrame(
            columns=[
                "precursor_key",
                "label",
                "sequence",
                "var_mod_sites_unimod",
                "fixed_mod_sites_unimod",
                "all_unimods",
                "charge",
                "variant_id",
                "precursor_id",
                "prediction_source",
                "b_mz",
                "b_intensity",
                "y_mz",
                "y_intensity",
            ]
        )

    duplicate_mask = lookup_df["precursor_key"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_variants = (
            lookup_df.loc[duplicate_mask, ["precursor_key", "variant_id"]]
            .groupby("precursor_key", sort=False)["variant_id"]
            .agg(lambda values: sorted({int(value) for value in values.dropna().tolist()}))
        )
        ambiguous = {
            key: values
            for key, values in duplicate_variants.items()
            if len(values) > 1
        }
        if ambiguous:
            sample_key = next(iter(ambiguous))
            raise ValueError(
                f"Ambiguous variant_id values for precursor_key={sample_key}: "
                f"{ambiguous[sample_key]}"
            )
        lookup_df = lookup_df.drop_duplicates(subset=["precursor_key"], keep="first").reset_index(
            drop=True
        )

    for row in lookup_df.itertuples(index=False):
        variant_id = row.variant_id
        precursor_id = row.precursor_id
        peak_codes = row.peak_code
        mz_values = row.mz
        intensity_values = row.intensity

        if pd.isna(variant_id):
            prediction_source = "missing_variant"
            b_mz = None
            b_intensity = None
            y_mz = None
            y_intensity = None
        elif peak_codes is None or mz_values is None or intensity_values is None:
            prediction_source = "missing_prediction"
            b_mz = None
            b_intensity = None
            y_mz = None
            y_intensity = None
        else:
            prediction_source = "postgres"
            b_mz, b_intensity, y_mz, y_intensity = reconstruct_prediction(
                sequence=str(row.pep_seq),
                peak_codes=peak_codes,
                mz_values=mz_values,
                intensity_values=intensity_values,
                peak_code_map=peak_code_map,
            )

        rows.append(
            {
                "precursor_key": row.precursor_key,
                "label": int(row.label),
                "sequence": str(row.pep_seq),
                "var_mod_sites_unimod": str(row.var_mod_sites_unimod or ""),
                "fixed_mod_sites_unimod": str(row.fixed_mod_sites_unimod or ""),
                "all_unimods": str(row.all_unimods or ""),
                "charge": int(row.charge),
                "variant_id": None if pd.isna(variant_id) else int(variant_id),
                "precursor_id": None if pd.isna(precursor_id) else int(precursor_id),
                "prediction_source": prediction_source,
                "b_mz": b_mz,
                "b_intensity": b_intensity,
                "y_mz": y_mz,
                "y_intensity": y_intensity,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    started_at = perf_counter()
    log(f"reading precursor input from {args.input}", COMPONENT)
    input_df = pd.read_parquet(args.input)
    if input_df.empty:
        pd.DataFrame().to_parquet(args.output, index=False)
        log("input was empty; wrote empty output parquet", COMPONENT)
        return
    log(f"loaded {len(input_df)} precursor rows", COMPONENT)

    conn = psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=read_password(args.db_password_file),
        application_name="alpha2rescore_lookup",
    )
    try:
        log(
            f"connected to postgresql host={args.db_host} db={args.db_name} schema={args.db_schema}",
            COMPONENT,
        )
        peak_code_map = load_peak_code_map(conn, args.db_schema)
        log(f"loaded {len(peak_code_map)} fragment peak-code entries", COMPONENT)
        target_df = fetch_group(
            conn=conn,
            group_df=input_df[input_df["label"] == 1].copy(),
            schema=args.db_schema,
            pep_table="protinsight_trypsin_pep",
            variant_table="protinsight_trypsin_variant",
            alphapep_table="protinsight_trypsin_variant_alphapep",
        )
        decoy_df = fetch_group(
            conn=conn,
            group_df=input_df[input_df["label"] != 1].copy(),
            schema=args.db_schema,
            pep_table="protinsight_trypsin_decoy_pep",
            variant_table="protinsight_trypsin_decoy_variant",
            alphapep_table="protinsight_trypsin_decoy_variant_alphapep",
        )
        combined = pd.concat([target_df, decoy_df], ignore_index=True)
        log(f"collapsing {len(combined)} lookup rows into precursor predictions", COMPONENT)
        output_df = collapse_lookup_rows(combined, peak_code_map)
    finally:
        conn.close()
        log("closed postgresql connection", COMPONENT)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(args.output, index=False)
    log(
        f"wrote {len(output_df)} precursor rows to {args.output} in {format_duration(perf_counter() - started_at)}",
        COMPONENT,
    )


if __name__ == "__main__":
    main()
