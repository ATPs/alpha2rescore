"""Fragment type mapping and defaults for alphapeptms2."""

from pathlib import Path

# Default model configurations matching MS2PIP conventions
DEFAULT_MODEL = "HCD"
MODEL_TO_ALPHAPEPT_MODEL = {
    "HCD": "generic",
}

# AlphaPeptDeep fragment column names (from the generic model)
# Confirmed 2026-05-11 from AlphaPeptDeep pretrained_models_v3.zip
APD_FRAG_TYPES_NON_MODLOSS = ["b_z1", "b_z2", "y_z1", "y_z2"]
APD_FRAG_TYPES_MODLOSS = [
    "b_modloss_z1", "b_modloss_z2",
    "y_modloss_z1", "y_modloss_z2",
]

# Mapping from AlphaPeptDeep column -> (ms2pip_ion_type, charge_index)
# e.g. "b_z1" -> ("b", 0), "y_z2" -> ("y", 1)
def _parse_apd_column(col: str):
    """Parse 'b_z1' -> ('b', 1) or 'y_modloss_z2' -> ('y', 2)."""
    parts = col.split("_")
    ion_type = parts[0]  # "b" or "y"
    charge = int(parts[-1].lstrip("z"))  # 1, 2, or 3
    return ion_type, charge

# Build the mapping
APD_TO_MS2PIP_MAP = {}
for col in APD_FRAG_TYPES_NON_MODLOSS:
    ion_type, charge = _parse_apd_column(col)
    APD_TO_MS2PIP_MAP[col] = (ion_type, charge - 1)  # 0-indexed charge

# MS2PIP ion types and their max charge
MS2PIP_ION_TYPES = ["b", "y"]
MS2PIP_MAX_CHARGE = 3  # b1-b3, y1-y3

# Default AlphaPeptDeep settings
PEPTDEEP_HOME = str(Path.home() / "peptdeep" / "pretrained_models")
SUPPORTED_DEVICES = ("cpu", "gpu")

# Default chunk size for predict_batch (precursors per batch)
DEFAULT_CHUNK_SIZE = 5000
