# alpha2rescore

`alpha2rescore` builds fast, incremental Percolator PIN files from:

- ProtInsight Comet PIN parquet or tab-separated text:
  - `ms2pin.parquet/<idn>.pin.parquet`
  - `<idn>.pin`
  - `<idn>.pin.tsv`
- ProtInsight mzDuck spectra parquet or text MGF:
  - `mzDuck/<idn>.mgf.parquet`
  - `<idn>.mgf`
  - `<idn>.mgf.gz`

It combines:

- AlphaPept MS2 spectral features
- DeepLC retention-time features
- PostgreSQL reuse of previously predicted AlphaPept precursor spectra
- local fallback prediction for missing precursors
- per-idn incremental caches
- streamed stage logs for long runs

The current implementation is optimized first for a ProtInsight workflow using
parquet PIN inputs together with mzDuck spectra. Local benchmark paths and lab
deployment details are recorded in:

- `docs/development/alphapeptms2/20260515_local_environment_notes.md`

## Origins and thanks

This package builds directly on prior work from:

- `alphapeptms2`
- [MS2Rescore](https://github.com/CompOmics/ms2rescore)

The AlphaPeptDeep, alphapeptms2, and MS2Rescore authors established most of the
practical conventions that made this package possible. `alpha2rescore` reuses
their ideas with a narrower focus on fast incremental PIN generation for this
ProtInsight workflow.

The bundled `alphapeptms2` compatibility copy now lives inside this package.
Local lab deployment notes are preserved under:

- `docs/development/alphapeptms2/AGENTS.md`
- `docs/development/alphapeptms2/`

## Main design points

### What this package does

For one `idn`, `alpha2rescore`:

1. reads the input PIN file from parquet or tab-separated text
2. reads the matching spectrum file from mzDuck parquet or text MGF
3. parses the `Peptide` column into database lookup keys
4. queries PostgreSQL for existing AlphaPept precursor predictions
5. locally predicts only missing precursor spectra
6. builds 71 AlphaPept spectral features per PSM
7. builds 6 DeepLC RT features per PSM
8. writes a gzipped Percolator PIN text file

### Important mapping rules

- peptide identity comes from `Peptide`
- `Proteins` is ignored for mapping
- spectra are matched by `ScanNr`
- for mzDuck parquet:
  - `ScanNr -> scan_number`
- for text MGF:
  - `ScanNr -> SCANS`
  - fallback: `ScanNr -> TITLE` when `SCANS` is absent and the title contains a parseable scan id
- variable modification lookup key is derived from `[U:id]` peptide text
- the DB-style `var_mod_sites_unimod` string must be sorted by site position

Examples:

- `LHWLVM[U:35]RK` -> `sequence=LHWLVMRK`, `var_mod_sites_unimod=5:35`
- `[U:1]-M[U:35]LQFLLEVNK` -> `sequence=MLQFLLEVNK`, `var_mod_sites_unimod=0:35;10:1`
- `Q[U:28]AVKLVKANK` -> `sequence=QAVKLVKANK`, `var_mod_sites_unimod=0:28`

### Target and decoy routing

`Label` controls which PostgreSQL tables are used:

- `Label == 1`:
  - target peptide table
  - target variant table
  - target AlphaPept prediction table
- `Label != 1`:
  - decoy peptide table
  - decoy variant table
  - decoy AlphaPept prediction table

### Incremental behavior

Caching is per `idn`.

Once the cache exists, reruns only compute:

- new precursor lookups
- newly missing local precursor predictions
- new Alpha feature rows
- new DeepLC base feature rows

The final PIN is always rewritten in original row order.

## Package layout

```text
alpha2rescore/
  README.md
  AGENTS.md
  pyproject.toml
  docs/
    development/
      alpha2rescore/
        20260513plan.md
      alphapeptms2/
        AGENTS.md
        20260511design.md
        20260511fix1.md
        20260515_local_environment_notes.md
  src/alpha2rescore/
    __init__.py
    __main__.py
    alphapeptms2/
      __init__.py
      __main__.py
      core.py
      result.py
      spectrum.py
      constants.py
      _utils/
    alphapept_helper.py
    cli.py
    config.py
    core.py
    deeplc_module.py
    features.py
    io.py
    peptides.py
    postgres_helper.py
    subprocess_utils.py
  src/alphapeptms2/
    __init__.py
    __main__.py
    core.py
    result.py
    spectrum.py
    constants.py
    _utils/
  tests/
    alphapeptms2/
```

## Environment and dependencies

Use an activated Python environment with the package dependencies installed for
installation, CLI usage, AlphaPept prediction, DeepLC, and PostgreSQL lookup.

The examples below assume you have already activated the environment you want
to use.

The bundled PostgreSQL helper prefers `psycopg` in this environment and keeps a
`psycopg2` fallback for compatibility.

### PostgreSQL

Database connection settings are deployment-specific. Configure them with:

- `--db-host`
- `--db-port`
- `--db-name`
- `--db-user`
- `--db-password-file`
- `--db-schema`

Local lab defaults are documented in:

- `docs/development/alphapeptms2/20260515_local_environment_notes.md`

## Installation

Install editable:

```bash
python -m pip uninstall -y alphapeptms2
python -m pip install -e .
```

Check help:

```bash
alpha2rescore --help
alpha2rescore build --help
alphapeptms2 --help
```

## Bundled alphapeptms2

The bundled `alphapeptms2` layer is a small compatibility wrapper around
AlphaPeptDeep (`peptdeep`). It preserves the MS2PIP-like public surface:

- `predict_single()`
- `predict_batch()`
- `correlate()`
- CLI commands for the same workflows

Canonical imports now live under `alpha2rescore.alphapeptms2`:

```python
from alpha2rescore.alphapeptms2 import predict_single, predict_batch, correlate
```

The legacy compatibility surface is also installed:

```python
from alphapeptms2 import predict_single, predict_batch, correlate
from alphapeptms2 import core
```

For clean compatibility testing, remove any old standalone `alphapeptms2`
editable install from the active environment before importing the top-level
`alphapeptms2` package.

### alphapeptms2 Python API

Predict one charged ProForma peptidoform:

```python
from alphapeptms2 import predict_single

result = predict_single("PGAQANPYSR/3", model="HCD", device="cpu")
print(result.psm_index)
print(result.theoretical_mz["b"].shape)
print(result.predicted_intensity["y"].shape)
```

Predict a `PSMList`:

```python
from psm_utils import PSM, PSMList
from alphapeptms2 import predict_batch

psms = PSMList(
    psm_list=[
        PSM(peptidoform="PGAQANPYSR/3", spectrum_id="scan=1001"),
        PSM(peptidoform="PEPTIDEK/2", spectrum_id="scan=1002"),
    ]
)

results = predict_batch(psms, model="HCD", device="cpu")
```

Predict from a PSM file:

```python
from alphapeptms2 import predict_batch

results = predict_batch(
    "/path/to/psms.tsv",
    psm_filetype=None,
    model="HCD",
    device="cpu",
)
```

Correlate predicted and observed spectra:

```python
from alphapeptms2 import correlate

results = correlate(
    "/path/to/psms.tsv",
    "/path/to/spectra.mzML.gz",
    model="HCD",
    device="cpu",
    ms2_tolerance=0.02,
    spectrum_id_pattern=r"scan=(.*)",
    processes=8,
)
```

Split GPU prediction from CPU-side matching when needed:

```python
from alphapeptms2 import calculate_correlations, predict_batch
from alphapeptms2.core import attach_observed_intensities
from alphapeptms2.spectrum import load_spectrum_index, normalize_spectrum_id
from psm_utils import io as psm_io

psms = psm_io.read_file("/path/to/psms.tsv")
results = predict_batch(psms, device="gpu")
spectrum_index = load_spectrum_index("/path/to/spectra.mzML.gz", r"scan=(.*)")
psm_to_spec = {
    idx: normalize_spectrum_id(psm.spectrum_id, r"scan=(.*)")
    for idx, psm in enumerate(psms)
}
attach_observed_intensities(results, spectrum_index, psm_to_spec, processes=16)
calculate_correlations(results)
```

### alphapeptms2 CLI

After installation:

```bash
alphapeptms2 --help
python -m alphapeptms2 --help
```

Predict one peptidoform:

```bash
alphapeptms2 predict-single "PGAQANPYSR/3" --model HCD --device cpu
```

Predict from a file:

```bash
alphapeptms2 predict-batch /path/to/psms.tsv \
  --model HCD \
  --device cpu \
  --output predictions.json
```

Correlate with observed spectra:

```bash
alphapeptms2 correlate /path/to/psms.tsv /path/to/spectra.mgf \
  --device cpu \
  --ms2-tolerance 0.02 \
  --spectrum-id-pattern 'scan=(.*)' \
  --output correlations.json
```

### alphapeptms2 Result Layout

Each call returns one or more `ProcessingResult` objects:

```python
class ProcessingResult:
    psm_index: int
    psm: Optional[psm_utils.PSM]
    theoretical_mz: dict[str, np.ndarray]
    predicted_intensity: dict[str, np.ndarray]
    observed_intensity: Optional[dict[str, np.ndarray]]
    correlation: Optional[float]
```

The MS2 arrays are keyed by ion type and have shape
`(n_fragment_positions, 3)`. Column 0 is charge 1, column 1 is charge 2, and
column 2 is the zero-filled charge-3 column for the default model.

Predicted and observed matched intensities use:

```python
np.log2(raw_intensity + 0.001)
```

### alphapeptms2 Notes

- Default public model is `HCD`, mapped internally to AlphaPeptDeep `generic`.
- Current exposed fragment mapping is:
  - `b_z1 -> b, charge 1`
  - `b_z2 -> b, charge 2`
  - `y_z1 -> y, charge 1`
  - `y_z2 -> y, charge 2`
- ProForma parsing uses `psm_utils.Peptidoform` and converts modifications into
  AlphaBase-compatible names and sites.
- Residue-attached terminal cases may normalize to terminal sites, for example
  `Q[UNIMOD:28]PEPTIDE/2 -> Gln->pyro-Glu@Q^Any_N-term` with `mod_sites="0"`.
- MGF input is sanitized before parsing; `.mgf`, `.mgf.gz`, `.mzML`, and
  `.mzML.gz` are supported for `correlate()`.
- Correlation excludes zero theoretical m/z values, so the zero-filled default
  charge-3 columns do not affect the score.
- Current limitations:
  - only non-mod-loss `b` and `y` fragments are exposed
  - `processes` affects CPU-side observed-spectrum matching, not prediction
  - there is no package-level multi-process GPU scheduler yet

## CLI usage

### Explicit file paths

```bash
alpha2rescore build \
  --pin-file /path/to/ms2pin.parquet/1554451.pin.parquet \
  --spectrum-file /path/to/mzduck/1554451.mgf.parquet \
  --out-dir /path/to/output
```

Text-format example:

```bash
alpha2rescore build \
  --pin-file /path/to/1554451.pin \
  --spectrum-file /path/to/1554451.mgf.gz \
  --out-dir /path/to/output
```

### Directory plus idn

```bash
alpha2rescore build \
  --pin-dir /path/to/ms2pin.parquet \
  --mgf-dir /path/to/mzduck \
  --idn 1554451 \
  --out-dir /path/to/output
```

### Smoke test on first N PSMs

```bash
alpha2rescore build \
  --pin-parquet /path/to/ms2pin.parquet/1554451.pin.parquet \
  --mgf-parquet /path/to/mzduck/1554451.mgf.parquet \
  --out-dir /path/to/output \
  --max-psms 500
```

### Large file with multithreading

```bash
alpha2rescore build \
  --pin-parquet /path/to/ms2pin.parquet/1554451.pin.parquet \
  --mgf-parquet /path/to/mzduck/1554451.mgf.parquet \
  --out-dir /path/to/output \
  --alpha-feature-threads 8 \
  --alpha-feature-batch-size 512 \
  --deeplc-processes 4
```

## Key CLI options

### Input/output

- `--pin-file`
  - alias:
    `--pin-parquet`
- `--spectrum-file`
  - alias:
    `--mgf-parquet`
- `--pin-dir`
- `--mgf-dir`
- `--idn`
- `--out-dir`

### Runtime control

- `--max-psms`
  - only for smoke runs or profiling
- `--ms2-tolerance`
  - fragment matching tolerance in Da

### Alpha spectral feature parallelism

- `--alpha-feature-threads`
  - number of threads used for AlphaPept spectral feature matching and scoring
- `--alpha-feature-batch-size`
  - number of PSMs processed per worker batch

Practical note:

- this is the main CPU hot path for large files with many uncached PSMs
- increasing this option is the most direct way to speed up fresh feature generation
- current large-sample profiling shows Alpha spectral features are the main fresh-run bottleneck after the PostgreSQL lookup optimization

### DeepLC

- `--deeplc-processes`
- `--deeplc-calibration-fraction`
- `--recalibrate-deeplc`

### Local AlphaPept prediction helper

- `--alphapept-python`
- `--alphapept-device`
- `--alphapept-processes`

### PostgreSQL helper

- `--postgres-python`
- `--db-host`
- `--db-port`
- `--db-name`
- `--db-user`
- `--db-password-file`
- `--db-schema`

## Output files

For `idn=1554451`, the final output is:

- `1554451.comet_alpha2rescore.pin.gz`

The file is a gzipped text PIN suitable as Percolator input.

Feature columns are inserted before:

- `Peptide`
- `Proteins`

## Cache layout

Per-idn cache directory:

- `<out-dir>/cache/<idn>/`

Current cache files:

- `precursor_cache.parquet`
  - precursor lookup results
  - PostgreSQL-reused predictions
  - locally predicted missing precursor arrays
- `alpha_features.parquet`
  - per-PSM 71 Alpha spectral features
- `deeplc_base_features.parquet`
  - per-PSM base RT predictions
- `<idn>.deeplc_calibration.pkl`
  - small reusable DeepLC calibration file
- `manifest.json`
  - basic run metadata

## Python API

```python
from pathlib import Path
from alpha2rescore import Alpha2RescoreConfig, build_pin

config = Alpha2RescoreConfig(
    pin_parquet=Path("/path/to/ms2pin.parquet/1554451.pin.parquet"),
    mgf_parquet=Path("/path/to/mzduck/1554451.mgf.parquet"),
    out_dir=Path("/path/to/output"),
    idn="1554451",
    alpha_feature_threads=8,
    deeplc_processes=4,
)
result = build_pin(config)
print(result.output_pin)
```

## Performance notes

### What is already fast

- PostgreSQL lookup is bulk, not per-PSM
- helper and main-stage logs are streamed live, so long runs are inspectable while they are running
- mzDuck is read directly from parquet
- precursor predictions are cached by precursor key
- Alpha features are cached by stable PSM key
- DeepLC base predictions are cached by stable PSM key
- Alpha feature scoring uses a Numba JIT numeric kernel when `numba` is installed

### PostgreSQL lookup findings

Large-run profiling on `1554451` showed:

- the original slow path was dominated by `DataFileRead` waits on the 28 GB
  target/decoy `*_variant_alphapep` tables
- the main fix was not “add more indexes”
- the useful fixes were:
  - remove `COALESCE(v.var_mod_sites_unimod, '')` from the variant join
  - resolve `variant_id -> precursor_id` first
  - materialize resolved precursor rows ordered by `precursor_id`
  - then fetch AlphaPept arrays from PostgreSQL in precursor-id order

Observed helper-only full-precursor improvement:

- old runtime:
  - about `634s`
- current runtime:
  - about `54.77s`

Observed full fresh `1554451` build on real data:

- `101531` PSM rows
- `82470` precursor keys
- total wall time:
  - `1067.81s`
- dominant stage:
  - Alpha spectral features at `869.74s`
- PostgreSQL helper:
  - `50.15s`
- DeepLC:
  - `68.26s`

That full-run timing was measured before the Numba feature-scoring kernel was added.

Observed full fresh `1554451` build after the Numba kernel:

- total wall time:
  - `935.75s`
- Alpha spectral features:
  - `30.43s`
- PostgreSQL helper:
  - `808.95s`
- DeepLC:
  - `58.26s`

The Alpha stage improved from `869.74s` to `30.43s` on the same full idn. This specific fresh
run was dominated by PostgreSQL `DataFileRead` time while reading the large target/decoy
`*_variant_alphapep` tables.

### Where multithreading helps

Multithreading currently targets:

- Alpha spectral feature calculation

This helps most when:

- there are many uncached PSMs
- PostgreSQL already has precursor predictions
- local MS2 prediction count is low or zero
- but current large-run observation still shows limited effective CPU scaling from Python threads alone, so more threads are not the whole answer

Recommended starting point for larger fresh runs on this host:

- `--alpha-feature-threads 16`
- `--alpha-feature-batch-size 512`
- `--deeplc-processes 8`

Observed 10k staged run with preloaded precursor cache:

- Alpha features:
  - `104.69s` before the Numba kernel
- DeepLC:
  - `19.99s`

Current real-sample inner-loop benchmark on 1000 PSMs from `1554451`:

- Numba kernel:
  - `0.1011s`
  - about `9895.7 PSM/s`
- NumPy fallback:
  - `1.1999s`
  - about `833.4 PSM/s`

This benchmark covers real mzDuck spectra, real precursor-cache predictions, and the same
`calculate_feature_dict` entry point used by the build path. It does not include PIN reading,
cache merging, PostgreSQL lookup, or DeepLC.

The next optimization priority is measuring full fresh-run Alpha feature time after the Numba
kernel, then deciding whether cache merge and row materialization are now the larger cost.

### When cache dominates

Second runs are much faster because:

- precursor cache is reused
- Alpha feature cache is reused
- DeepLC feature cache is reused

## Development notes

### External package boundaries

Do not edit these directories as part of `alpha2rescore` work:

- `DeepLC`
- `IM2Deep`
- `ms2pip`
- `ms2rescore`

Use wrapper logic inside `alpha2rescore` instead.

### Helper module boundaries

- `postgres_helper.py`
  - only PostgreSQL lookup and reconstruction
- `alphapept_helper.py`
  - only local missing precursor prediction
- `core.py`
  - orchestration and cache merge logic

### Current assumptions

- v1 targets ProtInsight parquet inputs, not legacy text MGF/mzML workflows
- missing local predictions are cached locally only
- `Peptide` is the canonical identity source

## Troubleshooting

### Long run appears silent

Current versions print:

- main build stage boundaries
- PostgreSQL helper progress
- local AlphaPept helper progress
- Alpha feature batch progress

For full-file runs, prefer tmux and capture a log file:

```bash
/usr/bin/tmux new-session -d -s alpha2rescore_full \
  "cd /path/to/alpha2rescore && \
   alpha2rescore build ... \
   > /path/to/run.log 2>&1"
```

### `postgres_helper` fails on import

Check that it is run under:

- the same Python interpreter used to install `alpha2rescore`

and that `PYTHONPATH` includes:

- the repository `src/` directory when running directly from source without installation

The helper prefers `psycopg` and can fall back to `psycopg2` if that is the
available PostgreSQL client.

### `alphapept_helper` fails on import

Check that it is run under:

- the same Python interpreter used to install `alpha2rescore`

### DeepLC prints TensorFlow startup logs

This is expected in the current environment. It does not block output generation.

### Large Alpha stage is still slow

The previous hot spots were:

- repeated `corrcoef`
- repeated `quantile`
- Spearman rank work

These are now handled by the Numba kernel in `features.py` when `numba` is available.
If more speed is needed, first profile full fresh runs again; the remaining cost may have moved
to cache materialization, row dictionaries, or thread scheduling rather than raw feature math.

### Numeric `nan` suspicion in output PIN

Be careful:

- peptide text may legitimately contain the substring `NAN`
- grep hits are not enough to prove numeric feature failure

## Validation status

Detailed local implementation notes, reproduced lab commands, and local file
paths are recorded in:

- `docs/development/alphapeptms2/20260515_local_environment_notes.md`
