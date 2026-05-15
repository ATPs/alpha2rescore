"""Local AlphaPept prediction helper for missing precursors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .logging_utils import format_duration, log


SUPPORTED_PREDICTED_FRAG_TYPES = ("b_z1", "b_z2", "y_z1", "y_z2")
COMPONENT = "alpha2rescore.alphapept_helper"


class ModMappingError(ValueError):
    """Raised when a stored UniMod site cannot be converted to AlphaBase format."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict missing AlphaPept fragment arrays locally for the precursor rows "
            "that are absent from PostgreSQL."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="Input parquet of missing precursors")
    parser.add_argument("--output", required=True, type=Path, help="Output parquet path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--model-type", default="generic")
    return parser.parse_args()


@dataclass(frozen=True)
class AlphaBaseModMapper:
    residue_mods: dict[tuple[int, str], list[str]]
    nterm_mods: dict[int, list[str]]
    cterm_mods: dict[int, list[str]]
    nterm_residue_mods: dict[tuple[int, str], list[str]]
    cterm_residue_mods: dict[tuple[int, str], list[str]]

    @classmethod
    def from_alphabase(cls) -> "AlphaBaseModMapper":
        from alphabase.constants.modification import MOD_DF

        residue_mods: dict[tuple[int, str], list[str]] = {}
        nterm_mods: dict[int, list[str]] = {}
        cterm_mods: dict[int, list[str]] = {}
        nterm_residue_mods: dict[tuple[int, str], list[str]] = {}
        cterm_residue_mods: dict[tuple[int, str], list[str]] = {}

        for mod_name_raw, unimod_id_raw in MOD_DF[["mod_name", "unimod_id"]].itertuples(
            index=False, name=None
        ):
            try:
                unimod_id = int(unimod_id_raw)
            except (TypeError, ValueError):
                continue
            if unimod_id < 0:
                continue

            mod_name = str(mod_name_raw)
            if "@" not in mod_name:
                continue
            site_expr = mod_name.rsplit("@", 1)[1]
            if site_expr in {"Any_N-term", "Protein_N-term"}:
                _append_unique(nterm_mods, unimod_id, mod_name)
            elif site_expr in {"Any_C-term", "Protein_C-term"}:
                _append_unique(cterm_mods, unimod_id, mod_name)
            elif "^" in site_expr:
                residue_text, terminal = site_expr.split("^", 1)
                if len(residue_text) != 1:
                    continue
                residue = residue_text.upper()
                if terminal in {"Any_N-term", "Protein_N-term"}:
                    _append_unique(nterm_residue_mods, (unimod_id, residue), mod_name)
                elif terminal in {"Any_C-term", "Protein_C-term"}:
                    _append_unique(cterm_residue_mods, (unimod_id, residue), mod_name)
            elif len(site_expr) == 1 and site_expr.isalpha():
                _append_unique(residue_mods, (unimod_id, site_expr.upper()), mod_name)
        return cls(
            residue_mods=residue_mods,
            nterm_mods=nterm_mods,
            cterm_mods=cterm_mods,
            nterm_residue_mods=nterm_residue_mods,
            cterm_residue_mods=cterm_residue_mods,
        )

    def resolve(
        self,
        sequence: str,
        db_pos: int,
        unimod_id: int,
        variant_id: int,
        source_field: str,
    ) -> tuple[str, int]:
        seq_len = len(sequence)
        if db_pos < 0 and db_pos != -1:
            raise ModMappingError(
                f"Unsupported db_pos={db_pos} for variant_id={variant_id}, {source_field}"
            )
        if db_pos == seq_len:
            return self._pick_one(
                self.nterm_mods.get(unimod_id, []),
                ("@Any_N-term", "@Protein_N-term"),
                variant_id,
                source_field,
            ), 0
        if db_pos == -1:
            return self._pick_one(
                self.cterm_mods.get(unimod_id, []),
                ("@Any_C-term", "@Protein_C-term"),
                variant_id,
                source_field,
            ), -1

        residue = sequence[db_pos].upper()
        if db_pos == 0:
            nterm_specific = self.nterm_residue_mods.get((unimod_id, residue), [])
            if nterm_specific:
                return self._pick_one(
                    nterm_specific,
                    ("^Any_N-term", "^Protein_N-term"),
                    variant_id,
                    source_field,
                ), 0

        ordinary = self.residue_mods.get((unimod_id, residue), [])
        if ordinary:
            return self._pick_one(ordinary, (), variant_id, source_field), db_pos + 1

        if db_pos == seq_len - 1:
            cterm_specific = self.cterm_residue_mods.get((unimod_id, residue), [])
            if cterm_specific:
                return self._pick_one(
                    cterm_specific,
                    ("^Any_C-term", "^Protein_C-term"),
                    variant_id,
                    source_field,
                ), -1

        raise ModMappingError(
            f"No AlphaBase mapping for variant_id={variant_id}, field={source_field}, "
            f"sequence={sequence}, db_pos={db_pos}, unimod_id={unimod_id}"
        )

    @staticmethod
    def _pick_one(
        candidates: list[str],
        preferred_suffixes: tuple[str, ...],
        variant_id: int,
        source_field: str,
    ) -> str:
        unique = list(dict.fromkeys(candidates))
        if not unique:
            raise ModMappingError(
                f"No AlphaBase candidates for variant_id={variant_id}, field={source_field}"
            )
        for suffix in preferred_suffixes:
            preferred = [value for value in unique if value.endswith(suffix)]
            if len(preferred) == 1:
                return preferred[0]
        if len(unique) == 1:
            return unique[0]
        raise ModMappingError(
            f"Ambiguous AlphaBase candidates for variant_id={variant_id}, field={source_field}: "
            f"{', '.join(unique)}"
        )


def _append_unique(mapping: dict, key, value: str) -> None:
    values = mapping.setdefault(key, [])
    if value not in values:
        values.append(value)


def parse_mod_site_pairs(raw_sites: str) -> list[tuple[int, int]]:
    cleaned = str(raw_sites or "").strip()
    if not cleaned:
        return []
    pairs: list[tuple[int, int]] = []
    for token in cleaned.split(";"):
        if not token:
            continue
        pos_text, unimod_text = token.split(":", 1)
        pairs.append((int(pos_text), int(unimod_text)))
    return pairs


def convert_variant_mods(
    mapper: AlphaBaseModMapper,
    variant_id: int,
    sequence: str,
    var_mod_sites_unimod: str,
    fixed_mod_sites_unimod: str,
) -> tuple[str, str]:
    mods: list[str] = []
    sites: list[str] = []
    for field_name, raw_sites in (
        ("var_mod_sites_unimod", var_mod_sites_unimod),
        ("fixed_mod_sites_unimod", fixed_mod_sites_unimod),
    ):
        for db_pos, unimod_id in parse_mod_site_pairs(raw_sites):
            mod_name, mod_site = mapper.resolve(
                sequence=sequence,
                db_pos=db_pos,
                unimod_id=unimod_id,
                variant_id=variant_id,
                source_field=field_name,
            )
            mods.append(mod_name)
            sites.append(str(mod_site))
    return ";".join(mods), ";".join(sites)


def build_model_manager(device: str, model_type: str):
    from peptdeep.pretrained_models import ModelManager
    from peptdeep.settings import update_global_settings

    update_global_settings(
        {
            "PEPTDEEP_HOME": "/data/p/alphabase",
            "torch_device": {"device_type": device},
            "thread_num": 1,
            "model_mgr": {"predict": {"multiprocessing": False}},
        }
    )
    model_mgr = ModelManager(device=device)
    model_mgr.load_installed_models(model_type)
    return model_mgr


def normalize_prediction_row(
    precursor_df: pd.DataFrame,
    fragment_mz_df: pd.DataFrame,
    fragment_intensity_df: pd.DataFrame,
    original_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    precursor_df = precursor_df.reset_index(drop=True)
    fragment_mz_df = fragment_mz_df.loc[:, list(SUPPORTED_PREDICTED_FRAG_TYPES)]
    fragment_intensity_df = fragment_intensity_df.loc[:, list(SUPPORTED_PREDICTED_FRAG_TYPES)]

    for idx, row in precursor_df.iterrows():
        start = int(row["frag_start_idx"])
        stop = int(row["frag_stop_idx"])
        n_positions = stop - start
        b_mz = np.zeros((n_positions, 3), dtype=np.float32)
        b_intensity = np.zeros((n_positions, 3), dtype=np.float32)
        y_mz = np.zeros((n_positions, 3), dtype=np.float32)
        y_intensity = np.zeros((n_positions, 3), dtype=np.float32)

        if n_positions > 0:
            b_mz[:, 0] = fragment_mz_df["b_z1"].iloc[start:stop].to_numpy(dtype=np.float32)
            b_mz[:, 1] = fragment_mz_df["b_z2"].iloc[start:stop].to_numpy(dtype=np.float32)
            y_mz[:, 0] = fragment_mz_df["y_z1"].iloc[start:stop].to_numpy(dtype=np.float32)
            y_mz[:, 1] = fragment_mz_df["y_z2"].iloc[start:stop].to_numpy(dtype=np.float32)

            b_intensity[:, 0] = np.log2(
                fragment_intensity_df["b_z1"].iloc[start:stop].to_numpy(dtype=np.float32) + 0.001
            )
            b_intensity[:, 1] = np.log2(
                fragment_intensity_df["b_z2"].iloc[start:stop].to_numpy(dtype=np.float32) + 0.001
            )
            y_intensity[:, 0] = np.log2(
                fragment_intensity_df["y_z1"].iloc[start:stop].to_numpy(dtype=np.float32) + 0.001
            )
            y_intensity[:, 1] = np.log2(
                fragment_intensity_df["y_z2"].iloc[start:stop].to_numpy(dtype=np.float32) + 0.001
            )

        source_row = original_df.iloc[idx]
        rows.append(
            {
                "precursor_key": source_row["precursor_key"],
                "label": int(source_row["label"]),
                "sequence": str(source_row["sequence"]),
                "var_mod_sites_unimod": str(source_row["var_mod_sites_unimod"] or ""),
                "fixed_mod_sites_unimod": str(source_row["fixed_mod_sites_unimod"] or ""),
                "all_unimods": str(source_row.get("all_unimods", "") or ""),
                "charge": int(source_row["charge"]),
                "variant_id": int(source_row["variant_id"]),
                "precursor_id": int(source_row["precursor_id"]),
                "prediction_source": "local",
                "b_mz": b_mz.tolist(),
                "b_intensity": b_intensity.tolist(),
                "y_mz": y_mz.tolist(),
                "y_intensity": y_intensity.tolist(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    started_at = perf_counter()
    log(f"reading local-prediction input from {args.input}", COMPONENT)
    input_df = pd.read_parquet(args.input)
    if input_df.empty:
        pd.DataFrame().to_parquet(args.output, index=False)
        log("input was empty; wrote empty output parquet", COMPONENT)
        return
    log(f"loaded {len(input_df)} precursor rows for local prediction", COMPONENT)

    stage_started_at = perf_counter()
    mapper = AlphaBaseModMapper.from_alphabase()
    prediction_input = input_df.copy()
    mods = []
    mod_sites = []
    for row in prediction_input.itertuples(index=False):
        row_mods, row_mod_sites = convert_variant_mods(
            mapper=mapper,
            variant_id=int(row.variant_id),
            sequence=str(row.sequence),
            var_mod_sites_unimod=str(row.var_mod_sites_unimod or ""),
            fixed_mod_sites_unimod=str(row.fixed_mod_sites_unimod or ""),
        )
        mods.append(row_mods)
        mod_sites.append(row_mod_sites)
    log(
        f"converted variant modifications for {len(prediction_input)} rows in "
        f"{format_duration(perf_counter() - stage_started_at)}",
        COMPONENT,
    )

    precursor_df = pd.DataFrame(
        {
            "sequence": prediction_input["sequence"].astype(str).tolist(),
            "mods": mods,
            "mod_sites": mod_sites,
            "charge": prediction_input["charge"].astype(int).tolist(),
        }
    )

    stage_started_at = perf_counter()
    model_mgr = build_model_manager(device=args.device, model_type=args.model_type)
    log(
        f"loaded AlphaPept model manager on device={args.device} in "
        f"{format_duration(perf_counter() - stage_started_at)}",
        COMPONENT,
    )

    stage_started_at = perf_counter()
    result = model_mgr.predict_all(
        precursor_df.copy(),
        predict_items=["ms2"],
        frag_types=list(SUPPORTED_PREDICTED_FRAG_TYPES),
        multiprocessing=args.processes > 1,
        process_num=max(args.processes, 1),
    )
    log(
        f"predicted MS2 for {len(precursor_df)} precursor rows in "
        f"{format_duration(perf_counter() - stage_started_at)}",
        COMPONENT,
    )

    stage_started_at = perf_counter()
    output_df = normalize_prediction_row(
        precursor_df=result["precursor_df"],
        fragment_mz_df=result["fragment_mz_df"],
        fragment_intensity_df=result["fragment_intensity_df"],
        original_df=input_df.reset_index(drop=True),
    )
    log(
        f"normalized AlphaPept output into {len(output_df)} cache rows in "
        f"{format_duration(perf_counter() - stage_started_at)}",
        COMPONENT,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(args.output, index=False)
    log(
        f"wrote local prediction output to {args.output} in "
        f"{format_duration(perf_counter() - started_at)}",
        COMPONENT,
    )


if __name__ == "__main__":
    main()
