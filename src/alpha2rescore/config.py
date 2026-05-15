"""Configuration dataclasses for alpha2rescore."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MAIN_PYTHON = "/data/p/anaconda3/envs/ms2rescore_3_2_1/bin/python"
DEFAULT_ALPHAPEPT_PYTHON = "/data/p/anaconda3/envs/alphabase/bin/python"
DEFAULT_POSTGRES_PYTHON = "/data/p/anaconda3/bin/python"

DEFAULT_POSTGRES_HOST = "10.110.120.2"
DEFAULT_POSTGRES_PORT = 5432
DEFAULT_POSTGRES_DBNAME = "proteome"
DEFAULT_POSTGRES_USER = "xlab"
DEFAULT_POSTGRES_PASSWORD_FILE = (
    "/data/users/x/.ssh/20250505xcweb.server2.xlab.postgresql.passwd"
)
DEFAULT_POSTGRES_SCHEMA = "protein_hs"
DEFAULT_ALPHA_FEATURE_THREADS = max(1, min(8, os.cpu_count() or 1))
DEFAULT_ALPHA_FEATURE_BATCH_SIZE = 512


@dataclass(slots=True)
class Alpha2RescoreConfig:
    """Runtime configuration for one alpha2rescore build."""

    pin_parquet: Path
    mgf_parquet: Path
    out_dir: Path
    idn: str
    cache_dir: Path | None = None
    max_psms: int | None = None
    ms2_tolerance: float = 0.02
    alpha_feature_threads: int = DEFAULT_ALPHA_FEATURE_THREADS
    alpha_feature_batch_size: int = DEFAULT_ALPHA_FEATURE_BATCH_SIZE
    deeplc_calibration_fraction: float = 0.15
    deeplc_processes: int = 1
    alphapept_device: str = "cpu"
    alphapept_processes: int = 1
    alphapept_python: str = DEFAULT_ALPHAPEPT_PYTHON
    postgres_python: str = DEFAULT_POSTGRES_PYTHON
    main_python: str = DEFAULT_MAIN_PYTHON
    postgres_host: str = DEFAULT_POSTGRES_HOST
    postgres_port: int = DEFAULT_POSTGRES_PORT
    postgres_dbname: str = DEFAULT_POSTGRES_DBNAME
    postgres_user: str = DEFAULT_POSTGRES_USER
    postgres_password_file: str = DEFAULT_POSTGRES_PASSWORD_FILE
    postgres_schema: str = DEFAULT_POSTGRES_SCHEMA
    recalibrate_deeplc: bool = False
    local_prediction_only: bool = True
    keep_all_psms: bool = True
    helper_src_root: Path | None = None
    extra_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pin_parquet = Path(self.pin_parquet)
        self.mgf_parquet = Path(self.mgf_parquet)
        self.out_dir = Path(self.out_dir)
        if self.cache_dir is None:
            self.cache_dir = self.out_dir / "cache" / str(self.idn)
        else:
            self.cache_dir = Path(self.cache_dir)
        if self.helper_src_root is None:
            self.helper_src_root = Path(__file__).resolve().parents[1]
        else:
            self.helper_src_root = Path(self.helper_src_root)


@dataclass(slots=True)
class BuildResult:
    """High-level build result for CLI and API callers."""

    idn: str
    output_pin: Path
    cache_dir: Path
    psm_count: int
    precursor_count: int
    alpha_cache_hits: int
    alpha_cache_misses: int
    postgres_prediction_hits: int
    local_prediction_count: int
    deeplc_cache_hits: int
    deeplc_cache_misses: int
    metadata: dict[str, Any] = field(default_factory=dict)
