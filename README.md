# alpha2rescore

`alpha2rescore` builds fast, incremental Percolator PIN files from:

- ProtInsight Comet PIN parquet:
  - `ms2pin.parquet/<idn>.pin.parquet`
- ProtInsight mzDuck spectra parquet:
  - `mzDuck/<idn>.mgf.parquet`

It combines:

- AlphaPept MS2 spectral features
- DeepLC retention-time features
- PostgreSQL reuse of previously predicted AlphaPept precursor spectra
- local fallback prediction for missing precursors
- per-idn incremental caches
- streamed stage logs for long runs

The current implementation is optimized first for:

- `/data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/ms2pin.parquet`
- `/data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/mzDuck`

## Main design points

### What this package does

For one `idn`, `alpha2rescore`:

1. reads the input PIN parquet
2. reads the matching mzDuck spectrum parquet
3. parses the `Peptide` column into database lookup keys
4. queries PostgreSQL for existing AlphaPept precursor predictions
5. locally predicts only missing precursor spectra
6. builds 71 AlphaPept spectral features per PSM
7. builds 6 DeepLC RT features per PSM
8. writes a gzipped Percolator PIN text file

### Important mapping rules

- peptide identity comes from `Peptide`
- `Proteins` is ignored for mapping
- spectra are matched by:
  - `ScanNr -> mzDuck.scan_number`
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
  plan/
    20260513plan.md
  pyproject.toml
  src/alpha2rescore/
    __init__.py
    __main__.py
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
  tests/
```

## Environment and dependencies

### Main runtime

Use the ms2rescore environment for the main package:

- Python:
  - `/data/p/anaconda3/envs/ms2rescore_3_2_1/bin/python`

This environment is used for:

- reading PIN parquet
- reading mzDuck parquet
- Alpha feature generation
- DeepLC
- final PIN writing
- Alpha feature profiling and multithreaded scoring

### Local helper runtimes

The package intentionally uses separate Python interpreters for helper tasks.

#### PostgreSQL lookup helper

- interpreter:
  - `/data/p/anaconda3/bin/python`

Reason:

- this environment already has PostgreSQL client dependencies

#### Local AlphaPept missing-prediction helper

- interpreter:
  - `/data/p/anaconda3/envs/alphabase/bin/python`

Reason:

- AlphaPeptDeep and AlphaBase dependencies live there

### PostgreSQL

Default database settings:

- host:
  - `10.110.120.2`
- port:
  - `5432`
- db:
  - `proteome`
- user:
  - `xlab`
- password file:
  - `/data/users/x/.ssh/20250505xcweb.server2.xlab.postgresql.passwd`
- schema:
  - `protein_hs`

## Installation

Install editable:

```bash
/data/p/anaconda3/envs/ms2rescore_3_2_1/bin/python -m pip install -e /data/p/ms2rescore/alpha2rescore
```

Check help:

```bash
/data/p/anaconda3/envs/ms2rescore_3_2_1/bin/alpha2rescore --help
/data/p/anaconda3/envs/ms2rescore_3_2_1/bin/alpha2rescore build --help
```

## CLI usage

### Explicit file paths

```bash
alpha2rescore build \
  --pin-parquet /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/ms2pin.parquet/1554451.pin.parquet \
  --mgf-parquet /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/mzDuck/1554451.mgf.parquet \
  --out-dir /XCLabServer002_fastIO/ms2rescore-test/alpha2rescore
```

### Directory plus idn

```bash
alpha2rescore build \
  --pin-dir /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/ms2pin.parquet \
  --mgf-dir /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/mzDuck \
  --idn 1554451 \
  --out-dir /XCLabServer002_fastIO/ms2rescore-test/alpha2rescore
```

### Smoke test on first N PSMs

```bash
alpha2rescore build \
  --pin-parquet /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/ms2pin.parquet/1554451.pin.parquet \
  --mgf-parquet /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/mzDuck/1554451.mgf.parquet \
  --out-dir /XCLabServer002_fastIO/ms2rescore-test/alpha2rescore \
  --max-psms 500
```

### Large file with multithreading

```bash
alpha2rescore build \
  --pin-parquet /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/ms2pin.parquet/1554451.pin.parquet \
  --mgf-parquet /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/mzDuck/1554451.mgf.parquet \
  --out-dir /XCLabServer002_fastIO/ms2rescore-test/alpha2rescore-large \
  --alpha-feature-threads 8 \
  --alpha-feature-batch-size 512 \
  --deeplc-processes 4
```

## Key CLI options

### Input/output

- `--pin-parquet`
- `--mgf-parquet`
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
    pin_parquet=Path("/data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/ms2pin.parquet/1554451.pin.parquet"),
    mgf_parquet=Path("/data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/mzDuck/1554451.mgf.parquet"),
    out_dir=Path("/XCLabServer002_fastIO/ms2rescore-test/alpha2rescore"),
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
  - `104.69s` on the current code
- DeepLC:
  - `19.99s`

This means the next optimization priority is Alpha feature math, not DeepLC and not PostgreSQL.
The most promising next technical direction is Numba or deeper NumPy refactoring in `features.py`, not C++.

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
  "cd /data/p/ms2rescore && \
   /data/p/anaconda3/envs/ms2rescore_3_2_1/bin/alpha2rescore build ... \
   > /XCLabServer002_fastIO/ms2rescore-test/alpha2rescore-full/run.log 2>&1"
```

### `postgres_helper` fails on import

Check that it is run under:

- `/data/p/anaconda3/bin/python`

and that `PYTHONPATH` includes:

- `/data/p/ms2rescore/alpha2rescore/src`

### `alphapept_helper` fails on import

Check that it is run under:

- `/data/p/anaconda3/envs/alphabase/bin/python`

### DeepLC prints TensorFlow startup logs

This is expected in the current environment. It does not block output generation.

### Large Alpha stage is still slow

Current evidence points here first:

- repeated `corrcoef`
- repeated `quantile`
- Spearman rank work

Low-risk cleanup already improved serial Alpha scoring by about `16%`.
If more speed is needed, prefer NumPy/Numba work here before considering C++.

### Numeric `nan` suspicion in output PIN

Be careful:

- peptide text may legitimately contain the substring `NAN`
- grep hits are not enough to prove numeric feature failure

## Validation status

Implementation notes and reproduced test commands are recorded in:

- `/data/p/ms2rescore/ms2rescore-test/notes/20260513_alpha2rescore_v1_implementation.md`
