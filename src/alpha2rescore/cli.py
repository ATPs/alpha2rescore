"""CLI entrypoint for alpha2rescore."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    Alpha2RescoreConfig,
    DEFAULT_ALPHAPEPT_PYTHON,
    DEFAULT_ALPHA_FEATURE_BATCH_SIZE,
    DEFAULT_ALPHA_FEATURE_THREADS,
    DEFAULT_POSTGRES_DBNAME,
    DEFAULT_POSTGRES_HOST,
    DEFAULT_POSTGRES_PASSWORD_FILE,
    DEFAULT_POSTGRES_PORT,
    DEFAULT_POSTGRES_PYTHON,
    DEFAULT_POSTGRES_SCHEMA,
    DEFAULT_POSTGRES_USER,
)
from .core import build_pin


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """Preserve examples while still showing defaults."""


PIN_SUFFIXES = [
    ".pin.parquet",
    ".pin.tsv.gz",
    ".pin.tsv",
    ".pin.gz",
    ".pin",
    ".tsv.gz",
    ".tsv",
]
SPECTRUM_SUFFIXES = [
    ".mgf.parquet",
    ".mgf.gz",
    ".mgf",
]


def _strip_known_suffix(name: str, suffixes: list[str]) -> str:
    for suffix in sorted(suffixes, key=len, reverse=True):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _resolve_directory_file(directory: str, idn: str, suffixes: list[str], label: str) -> Path:
    base = Path(directory)
    for suffix in suffixes:
        candidate = base / f"{idn}{suffix}"
        if candidate.exists():
            return candidate
    expected = ", ".join(f"{idn}{suffix}" for suffix in suffixes)
    raise SystemExit(f"Could not find {label} for idn={idn} in {directory}. Tried: {expected}")


def resolve_paths(args: argparse.Namespace) -> tuple[str, Path, Path]:
    if args.pin_file or args.spectrum_file:
        if not args.pin_file or not args.spectrum_file:
            raise SystemExit("--pin-file and --spectrum-file must be provided together")
        pin_path = Path(args.pin_file)
        mgf_path = Path(args.spectrum_file)
        idn = args.idn or _strip_known_suffix(pin_path.name, PIN_SUFFIXES)
        return idn, pin_path, mgf_path

    if not args.pin_dir or not args.mgf_dir or not args.idn:
        raise SystemExit(
            "Use either explicit --pin-file/--spectrum-file or "
            "--pin-dir/--mgf-dir/--idn"
        )

    idn = str(args.idn)
    pin_path = _resolve_directory_file(args.pin_dir, idn, PIN_SUFFIXES, "PIN file")
    mgf_path = _resolve_directory_file(args.mgf_dir, idn, SPECTRUM_SUFFIXES, "spectrum file")
    return idn, pin_path, mgf_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpha2rescore",
        description=(
            "Fast incremental AlphaPept + DeepLC feature generation for Percolator PIN files.\n\n"
            "Typical PXD010154 usage:\n"
            "  alpha2rescore build \\\n"
            "    --pin-dir /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/ms2pin.parquet \\\n"
            "    --mgf-dir /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/mzDuck \\\n"
            "    --idn 1554451 \\\n"
            "    --out-dir /XCLabServer002_fastIO/ms2rescore-test/alpha2rescore\n\n"
            "  alpha2rescore build \\\n"
            "    --pin-file /path/to/1554451.pin \\\n"
            "    --spectrum-file /path/to/1554451.mgf \\\n"
            "    --out-dir /XCLabServer002_fastIO/ms2rescore-test/alpha2rescore\n\n"
            "Notes:\n"
            "  - peptide identity comes from the PIN `Peptide` column only.\n"
            "  - supported PIN inputs: .pin.parquet, .pin, .pin.tsv, .tsv, and .gz variants.\n"
            "  - supported spectrum inputs: mzDuck .mgf.parquet, .mgf, and .mgf.gz.\n"
            "  - text MGF spectra are matched by `ScanNr` using `SCANS=` first, then `TITLE` fallback.\n"
            "  - Alpha spectral features support explicit multithreading.\n"
            "  - PostgreSQL predictions are reused when present; missing ones are predicted locally\n"
            "    and cached locally only.\n"
            "  - DeepLC keeps one small calibration pickle per idn under the cache directory."
        ),
        formatter_class=HelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="Build one AlphaPept+DeepLC Percolator PIN for one idn",
        formatter_class=HelpFormatter,
        description=(
            "Build one gzipped PIN with 71 AlphaPept spectral features plus 6 DeepLC RT features.\n\n"
            "Examples:\n"
            "  alpha2rescore build \\\n"
            "    --pin-file /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/ms2pin.parquet/1554451.pin.parquet \\\n"
            "    --spectrum-file /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/mzDuck/1554451.mgf.parquet \\\n"
            "    --out-dir /XCLabServer002_fastIO/ms2rescore-test/alpha2rescore\n\n"
            "  alpha2rescore build \\\n"
            "    --pin-file /path/to/1554451.pin \\\n"
            "    --spectrum-file /path/to/1554451.mgf.gz \\\n"
            "    --out-dir /XCLabServer002_fastIO/ms2rescore-test/alpha2rescore\n\n"
            "  alpha2rescore build \\\n"
            "    --pin-dir /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/ms2pin.parquet \\\n"
            "    --mgf-dir /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/mzDuck \\\n"
            "    --idn 1554451 \\\n"
            "    --out-dir /XCLabServer002_fastIO/ms2rescore-test/alpha2rescore \\\n"
            "    --alpha-feature-threads 8 \\\n"
            "    --deeplc-processes 4\n"
        ),
    )
    build.add_argument(
        "--pin-file",
        "--pin-parquet",
        dest="pin_file",
        help="Explicit PIN input file: .pin.parquet, .pin, .pin.tsv, .tsv, or gzip variant",
    )
    build.add_argument(
        "--spectrum-file",
        "--mgf-parquet",
        dest="spectrum_file",
        help="Explicit spectrum file: mzDuck .mgf.parquet, .mgf, or .mgf.gz",
    )
    build.add_argument(
        "--pin-dir",
        help="Directory containing <idn>.pin.parquet, <idn>.pin, or <idn>.pin.tsv files",
    )
    build.add_argument(
        "--mgf-dir",
        help="Directory containing <idn>.mgf.parquet, <idn>.mgf, or <idn>.mgf.gz files",
    )
    build.add_argument("--idn", help="Run idn such as 1554451")
    build.add_argument("--out-dir", required=True, help="Output directory for final PIN and caches")
    build.add_argument(
        "--max-psms",
        type=int,
        default=None,
        help="Limit to the first N PSMs for smoke tests and profiling",
    )
    build.add_argument("--ms2-tolerance", type=float, default=0.02, help="Fragment m/z tolerance in Da")
    build.add_argument(
        "--alpha-feature-threads",
        type=int,
        default=DEFAULT_ALPHA_FEATURE_THREADS,
        help="Thread count for AlphaPept spectral feature matching and scoring",
    )
    build.add_argument(
        "--alpha-feature-batch-size",
        type=int,
        default=DEFAULT_ALPHA_FEATURE_BATCH_SIZE,
        help="PSM batch size per worker for AlphaPept spectral feature matching",
    )
    build.add_argument(
        "--deeplc-calibration-fraction",
        type=float,
        default=0.15,
        help="Fraction of top target PSMs used for first DeepLC calibration",
    )
    build.add_argument("--deeplc-processes", type=int, default=1, help="DeepLC worker count")
    build.add_argument(
        "--recalibrate-deeplc",
        action="store_true",
        help="Ignore cached DeepLC calibration and recalibrate from the current idn file",
    )
    build.add_argument(
        "--alphapept-python",
        default=DEFAULT_ALPHAPEPT_PYTHON,
        help="Python executable for local missing AlphaPept prediction helper",
    )
    build.add_argument(
        "--alphapept-device",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device for local missing AlphaPept prediction helper",
    )
    build.add_argument(
        "--alphapept-processes",
        type=int,
        default=1,
        help="Worker count for local missing AlphaPept prediction helper",
    )
    build.add_argument(
        "--postgres-python",
        default=DEFAULT_POSTGRES_PYTHON,
        help="Python executable for PostgreSQL lookup helper",
    )
    build.add_argument("--db-host", default=DEFAULT_POSTGRES_HOST, help="PostgreSQL host")
    build.add_argument("--db-port", type=int, default=DEFAULT_POSTGRES_PORT, help="PostgreSQL port")
    build.add_argument("--db-name", default=DEFAULT_POSTGRES_DBNAME, help="PostgreSQL database name")
    build.add_argument("--db-user", default=DEFAULT_POSTGRES_USER, help="PostgreSQL user")
    build.add_argument(
        "--db-password-file",
        default=DEFAULT_POSTGRES_PASSWORD_FILE,
        help="File containing the PostgreSQL password",
    )
    build.add_argument("--db-schema", default=DEFAULT_POSTGRES_SCHEMA, help="PostgreSQL schema")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "build":
        raise SystemExit(f"Unsupported command: {args.command}")

    idn, pin_path, mgf_path = resolve_paths(args)
    config = Alpha2RescoreConfig(
        pin_parquet=pin_path,
        mgf_parquet=mgf_path,
        out_dir=Path(args.out_dir),
        idn=idn,
        max_psms=args.max_psms,
        ms2_tolerance=args.ms2_tolerance,
        alpha_feature_threads=args.alpha_feature_threads,
        alpha_feature_batch_size=args.alpha_feature_batch_size,
        deeplc_calibration_fraction=args.deeplc_calibration_fraction,
        deeplc_processes=args.deeplc_processes,
        recalibrate_deeplc=args.recalibrate_deeplc,
        alphapept_python=args.alphapept_python,
        alphapept_device=args.alphapept_device,
        alphapept_processes=args.alphapept_processes,
        postgres_python=args.postgres_python,
        postgres_host=args.db_host,
        postgres_port=args.db_port,
        postgres_dbname=args.db_name,
        postgres_user=args.db_user,
        postgres_password_file=args.db_password_file,
        postgres_schema=args.db_schema,
    )
    result = build_pin(config)
    print(f"Built {result.output_pin}")
    print(
        f"psms={result.psm_count} precursors={result.precursor_count} "
        f"postgres_hits={result.postgres_prediction_hits} local_predictions={result.local_prediction_count} "
        f"alpha_cache_hits={result.alpha_cache_hits} deeplc_cache_hits={result.deeplc_cache_hits}"
    )
