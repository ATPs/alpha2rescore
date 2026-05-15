"""Convert ProForma peptidoforms to AlphaPeptDeep input DataFrames."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence, Union

import pandas as pd
from psm_utils import Peptidoform, PSM, PSMList


_NTERM_SITE_LABELS = {"Any_N-term", "Protein_N-term"}
_CTERM_SITE_LABELS = {"Any_C-term", "Protein_C-term"}
_TEXT_MOD_NAME_ALIASES = {
    "pyro-glu": {"Q": "Gln->pyro-Glu", "E": "Glu->pyro-Glu"},
}


class ModMappingError(ValueError):
    """Raised when a ProForma modification cannot be mapped to AlphaBase."""


class AlphaBaseModMapper:
    """Resolve UniMod or textual ProForma modifications to AlphaBase names/sites."""

    def __init__(
        self,
        *,
        residue_mods_by_unimod: dict[tuple[int, str], list[str]],
        residue_mods_by_name: dict[tuple[str, str], list[str]],
        nterm_mods_by_unimod: dict[int, list[str]],
        nterm_mods_by_name: dict[str, list[str]],
        cterm_mods_by_unimod: dict[int, list[str]],
        cterm_mods_by_name: dict[str, list[str]],
        nterm_residue_mods_by_unimod: dict[tuple[int, str], list[str]],
        nterm_residue_mods_by_name: dict[tuple[str, str], list[str]],
        cterm_residue_mods_by_unimod: dict[tuple[int, str], list[str]],
        cterm_residue_mods_by_name: dict[tuple[str, str], list[str]],
    ) -> None:
        self.residue_mods_by_unimod = residue_mods_by_unimod
        self.residue_mods_by_name = residue_mods_by_name
        self.nterm_mods_by_unimod = nterm_mods_by_unimod
        self.nterm_mods_by_name = nterm_mods_by_name
        self.cterm_mods_by_unimod = cterm_mods_by_unimod
        self.cterm_mods_by_name = cterm_mods_by_name
        self.nterm_residue_mods_by_unimod = nterm_residue_mods_by_unimod
        self.nterm_residue_mods_by_name = nterm_residue_mods_by_name
        self.cterm_residue_mods_by_unimod = cterm_residue_mods_by_unimod
        self.cterm_residue_mods_by_name = cterm_residue_mods_by_name

    @classmethod
    @lru_cache(maxsize=1)
    def from_alphabase(cls) -> "AlphaBaseModMapper":
        """Build a resolver from the installed AlphaBase modification table."""
        from alphabase.constants.modification import MOD_DF

        residue_mods_by_unimod: dict[tuple[int, str], list[str]] = {}
        residue_mods_by_name: dict[tuple[str, str], list[str]] = {}
        nterm_mods_by_unimod: dict[int, list[str]] = {}
        nterm_mods_by_name: dict[str, list[str]] = {}
        cterm_mods_by_unimod: dict[int, list[str]] = {}
        cterm_mods_by_name: dict[str, list[str]] = {}
        nterm_residue_mods_by_unimod: dict[tuple[int, str], list[str]] = {}
        nterm_residue_mods_by_name: dict[tuple[str, str], list[str]] = {}
        cterm_residue_mods_by_unimod: dict[tuple[int, str], list[str]] = {}
        cterm_residue_mods_by_name: dict[tuple[str, str], list[str]] = {}

        for mod_name_raw, unimod_id_raw in MOD_DF[["mod_name", "unimod_id"]].itertuples(
            index=False, name=None
        ):
            mod_name = str(mod_name_raw)
            if "@" not in mod_name:
                continue

            try:
                unimod_id = int(unimod_id_raw)
            except (TypeError, ValueError):
                unimod_id = None
            if unimod_id is not None and unimod_id < 0:
                unimod_id = None

            base_key = _normalized_name_key(_base_mod_name(mod_name))
            site_expr = _site_expression(mod_name)
            if site_expr in _NTERM_SITE_LABELS:
                if unimod_id is not None:
                    _append_unique(nterm_mods_by_unimod, unimod_id, mod_name)
                _append_unique(nterm_mods_by_name, base_key, mod_name)
                continue
            if site_expr in _CTERM_SITE_LABELS:
                if unimod_id is not None:
                    _append_unique(cterm_mods_by_unimod, unimod_id, mod_name)
                _append_unique(cterm_mods_by_name, base_key, mod_name)
                continue
            if "^" in site_expr:
                residue_text, terminal = site_expr.split("^", 1)
                if len(residue_text) != 1 or not residue_text.isalpha():
                    continue
                residue = residue_text.upper()
                if terminal in _NTERM_SITE_LABELS:
                    if unimod_id is not None:
                        _append_unique(
                            nterm_residue_mods_by_unimod, (unimod_id, residue), mod_name
                        )
                    _append_unique(nterm_residue_mods_by_name, (base_key, residue), mod_name)
                elif terminal in _CTERM_SITE_LABELS:
                    if unimod_id is not None:
                        _append_unique(
                            cterm_residue_mods_by_unimod, (unimod_id, residue), mod_name
                        )
                    _append_unique(cterm_residue_mods_by_name, (base_key, residue), mod_name)
                continue
            if len(site_expr) == 1 and site_expr.isalpha():
                residue = site_expr.upper()
                if unimod_id is not None:
                    _append_unique(residue_mods_by_unimod, (unimod_id, residue), mod_name)
                _append_unique(residue_mods_by_name, (base_key, residue), mod_name)

        return cls(
            residue_mods_by_unimod=residue_mods_by_unimod,
            residue_mods_by_name=residue_mods_by_name,
            nterm_mods_by_unimod=nterm_mods_by_unimod,
            nterm_mods_by_name=nterm_mods_by_name,
            cterm_mods_by_unimod=cterm_mods_by_unimod,
            cterm_mods_by_name=cterm_mods_by_name,
            nterm_residue_mods_by_unimod=nterm_residue_mods_by_unimod,
            nterm_residue_mods_by_name=nterm_residue_mods_by_name,
            cterm_residue_mods_by_unimod=cterm_residue_mods_by_unimod,
            cterm_residue_mods_by_name=cterm_residue_mods_by_name,
        )

    def resolve_nterm(self, token: str, sequence: str) -> tuple[str, str]:
        """Resolve a peptide N-terminal modification to `(mod_name, mod_site)`."""
        if not sequence:
            raise ModMappingError("Cannot resolve an N-terminal modification on an empty sequence")
        residue = sequence[0].upper()
        if "@" in token:
            return token, _site_for_mod_name(token, default_site="0")

        unimod_id = _parse_unimod_id(token)
        if unimod_id is not None:
            candidates = self.nterm_residue_mods_by_unimod.get((unimod_id, residue), [])
            if candidates:
                return self._pick_one(candidates, ("^Any_N-term", "^Protein_N-term")), "0"
            candidates = self.nterm_mods_by_unimod.get(unimod_id, [])
            if candidates:
                return self._pick_one(candidates, ("@Any_N-term", "@Protein_N-term")), "0"
            raise ModMappingError(
                f"No AlphaBase N-terminal mapping found for UniMod {unimod_id} on {sequence}"
            )

        base_key = _name_lookup_key(token, residue)
        candidates = self.nterm_residue_mods_by_name.get((base_key, residue), [])
        if candidates:
            return self._pick_one(candidates, ("^Any_N-term", "^Protein_N-term")), "0"
        candidates = self.nterm_mods_by_name.get(base_key, [])
        if candidates:
            return self._pick_one(candidates, ("@Any_N-term", "@Protein_N-term")), "0"
        return f"{token}@Any_N-term", "0"

    def resolve_cterm(self, token: str, sequence: str) -> tuple[str, str]:
        """Resolve a peptide C-terminal modification to `(mod_name, mod_site)`."""
        if not sequence:
            raise ModMappingError("Cannot resolve a C-terminal modification on an empty sequence")
        residue = sequence[-1].upper()
        if "@" in token:
            return token, _site_for_mod_name(token, default_site="-1")

        unimod_id = _parse_unimod_id(token)
        if unimod_id is not None:
            candidates = self.cterm_residue_mods_by_unimod.get((unimod_id, residue), [])
            if candidates:
                return self._pick_one(candidates, ("^Any_C-term", "^Protein_C-term")), "-1"
            candidates = self.cterm_mods_by_unimod.get(unimod_id, [])
            if candidates:
                return self._pick_one(candidates, ("@Any_C-term", "@Protein_C-term")), "-1"
            raise ModMappingError(
                f"No AlphaBase C-terminal mapping found for UniMod {unimod_id} on {sequence}"
            )

        base_key = _name_lookup_key(token, residue)
        candidates = self.cterm_residue_mods_by_name.get((base_key, residue), [])
        if candidates:
            return self._pick_one(candidates, ("^Any_C-term", "^Protein_C-term")), "-1"
        candidates = self.cterm_mods_by_name.get(base_key, [])
        if candidates:
            return self._pick_one(candidates, ("@Any_C-term", "@Protein_C-term")), "-1"
        return f"{token}@Any_C-term", "-1"

    def resolve_residue(self, token: str, sequence: str, position: int) -> tuple[str, str]:
        """Resolve a residue-attached modification to `(mod_name, mod_site)`."""
        residue = sequence[position - 1].upper()
        if "@" in token:
            return token, _site_for_mod_name(token, default_site=str(position))

        unimod_id = _parse_unimod_id(token)
        if unimod_id is not None:
            return self._resolve_residue_from_unimod(unimod_id, sequence, residue, position)

        base_key = _name_lookup_key(token, residue)
        return self._resolve_residue_from_name(base_key, token, sequence, residue, position)

    def _resolve_residue_from_unimod(
        self, unimod_id: int, sequence: str, residue: str, position: int
    ) -> tuple[str, str]:
        if position == 1:
            candidates = self.nterm_residue_mods_by_unimod.get((unimod_id, residue), [])
            if candidates:
                return self._pick_one(candidates, ("^Any_N-term", "^Protein_N-term")), "0"
        candidates = self.residue_mods_by_unimod.get((unimod_id, residue), [])
        if candidates:
            return self._pick_one(candidates, ()), str(position)
        if position == len(sequence):
            candidates = self.cterm_residue_mods_by_unimod.get((unimod_id, residue), [])
            if candidates:
                return self._pick_one(candidates, ("^Any_C-term", "^Protein_C-term")), "-1"
        raise ModMappingError(
            f"No AlphaBase residue mapping found for UniMod {unimod_id} on "
            f"{sequence} at position {position}"
        )

    def _resolve_residue_from_name(
        self, base_key: str, token: str, sequence: str, residue: str, position: int
    ) -> tuple[str, str]:
        if position == 1:
            candidates = self.nterm_residue_mods_by_name.get((base_key, residue), [])
            if candidates:
                return self._pick_one(candidates, ("^Any_N-term", "^Protein_N-term")), "0"
        candidates = self.residue_mods_by_name.get((base_key, residue), [])
        if candidates:
            return self._pick_one(candidates, ()), str(position)
        if position == len(sequence):
            candidates = self.cterm_residue_mods_by_name.get((base_key, residue), [])
            if candidates:
                return self._pick_one(candidates, ("^Any_C-term", "^Protein_C-term")), "-1"
        return f"{token}@{residue}", str(position)

    @staticmethod
    def _pick_one(candidates: Sequence[str], preferred_suffixes: Sequence[str]) -> str:
        unique_candidates = list(dict.fromkeys(candidates))
        if not unique_candidates:
            raise ModMappingError("Expected at least one AlphaBase modification candidate")
        for suffix in preferred_suffixes:
            preferred = [name for name in unique_candidates if name.endswith(suffix)]
            if len(preferred) == 1:
                return preferred[0]
        if len(unique_candidates) == 1:
            return unique_candidates[0]
        raise ModMappingError(
            f"Ambiguous AlphaBase modification mapping candidates: {', '.join(unique_candidates)}"
        )


def _iter_modifications(modifications) -> Iterable:
    """Yield zero or more modifications from a psm_utils modification field."""
    if modifications is None:
        return ()
    if isinstance(modifications, (list, tuple)):
        return modifications
    return (modifications,)


def _append_unique(mapping: dict, key, value: str) -> None:
    values = mapping.setdefault(key, [])
    if value not in values:
        values.append(value)


def _normalized_name_key(name: str) -> str:
    return name.strip().casefold()


def _base_mod_name(mod_name: str) -> str:
    return mod_name.rsplit("@", 1)[0]


def _site_expression(mod_name: str) -> str:
    return mod_name.rsplit("@", 1)[1]


def _site_for_mod_name(mod_name: str, default_site: str) -> str:
    site_expr = _site_expression(mod_name)
    if site_expr in _NTERM_SITE_LABELS or site_expr.endswith("^Any_N-term") or site_expr.endswith(
        "^Protein_N-term"
    ):
        return "0"
    if site_expr in _CTERM_SITE_LABELS or site_expr.endswith("^Any_C-term") or site_expr.endswith(
        "^Protein_C-term"
    ):
        return "-1"
    return default_site


def _parse_unimod_id(token: str) -> int | None:
    text = token.strip()
    if text.isdigit():
        return int(text)
    upper_text = text.upper()
    if upper_text.startswith("UNIMOD:") and text.split(":", 1)[1].isdigit():
        return int(text.split(":", 1)[1])
    return None


def _name_lookup_key(token: str, residue: str) -> str:
    token_key = _normalized_name_key(token)
    residue_specific_aliases = _TEXT_MOD_NAME_ALIASES.get(token_key)
    if residue_specific_aliases:
        aliased = residue_specific_aliases.get(residue.upper())
        if aliased is not None:
            return _normalized_name_key(aliased)
    return token_key


def _mod_token(modification) -> str:
    """Return a safe string token for a modification without touching slow properties."""
    value = getattr(modification, "value", None)
    return str(value) if value is not None else str(modification)


def peptidoform_to_row(peptidoform: Union[str, Peptidoform]) -> dict:
    """Parse a single ProForma peptidoform into a dict with AlphaPeptDeep columns.

    Returns a dict with keys: sequence, mods, mod_sites, charge.

    Example
    -------
    >>> peptidoform_to_row("RNVIM[Oxidation]DKVAK/2")
    {'sequence': 'RNVIMDKVAK', 'mods': 'Oxidation@M', 'mod_sites': '5', 'charge': 2}
    """
    if isinstance(peptidoform, str):
        peptidoform = Peptidoform(peptidoform)
    elif not isinstance(peptidoform, Peptidoform):
        raise TypeError(
            f"peptidoform must be a ProForma string or Peptidoform, got {type(peptidoform)}"
        )

    if peptidoform.precursor_charge is None:
        raise ValueError(f"Peptidoform must include a precursor charge: {peptidoform}")

    seq_parts = [str(item[0] if isinstance(item, tuple) else item) for item in peptidoform.parsed_sequence]
    sequence = "".join(seq_parts)
    mod_names = []
    mod_sites = []
    mapper = AlphaBaseModMapper.from_alphabase()

    properties = getattr(peptidoform, "properties", {})
    for mod in _iter_modifications(properties.get("n_term")):
        mod_name, mod_site = mapper.resolve_nterm(_mod_token(mod), sequence)
        mod_names.append(mod_name)
        mod_sites.append(mod_site)

    for i, item in enumerate(peptidoform.parsed_sequence):
        if isinstance(item, tuple):
            aa, mods = item
        else:
            aa, mods = item, None
        for mod in _iter_modifications(mods):
            mod_name, mod_site = mapper.resolve_residue(_mod_token(mod), sequence, i + 1)
            mod_names.append(mod_name)
            mod_sites.append(mod_site)

    for mod in _iter_modifications(properties.get("c_term")):
        mod_name, mod_site = mapper.resolve_cterm(_mod_token(mod), sequence)
        mod_names.append(mod_name)
        mod_sites.append(mod_site)

    return {
        "sequence": sequence,
        "mods": ";".join(mod_names),
        "mod_sites": ";".join(mod_sites),
        "charge": peptidoform.precursor_charge,
    }


def psm_list_to_df(psm_list: PSMList) -> pd.DataFrame:
    """Convert a PSMList to an AlphaPeptDeep-compatible DataFrame.

    Returns a DataFrame with columns: sequence, mods, mod_sites, charge.
    """
    rows = []
    for psm in psm_list:
        row = peptidoform_to_row(psm.peptidoform)
        row["psm_index"] = psm.spectrum_id if psm.spectrum_id is not None else id(psm)
        rows.append(row)
    df = pd.DataFrame(rows)
    # Preserve original psm_index for result mapping
    # (predict_all may reorder rows internally)
    return df


def psm_list_to_df_with_index(psm_list: PSMList) -> tuple[pd.DataFrame, list[int]]:
    """Convert PSMList to DataFrame, returning (df, original_indices).

    The original_indices list maps each row in the returned DataFrame back to
    its position in the PSMList. This survives AlphaPeptDeep's internal reordering.
    """
    rows = []
    original_indices = []
    for idx, psm in enumerate(psm_list):
        row = peptidoform_to_row(psm.peptidoform)
        row["_psm_idx"] = idx
        rows.append(row)
        original_indices.append(idx)
    return pd.DataFrame(rows), original_indices
