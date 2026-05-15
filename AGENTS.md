# AGENTS.md for alpha2rescore

## Scope

This file applies to:

- `/data/p/ms2rescore/alpha2rescore`

Its purpose is to guide later development of `alpha2rescore`.

## Main objective

`alpha2rescore` should remain a fast, incremental generator of Percolator PIN files for:

- ProtInsight parquet inputs:
  - `ms2pin.parquet/<idn>.pin.parquet`
  - `mzDuck/<idn>.mgf.parquet`
- text-format compatibility inputs:
  - `<idn>.pin`
  - `<idn>.pin.tsv`
  - `<idn>.mgf`
  - `<idn>.mgf.gz`

The priority order is:

1. correctness of peptide-to-precursor mapping
2. speed on repeated incremental runs
3. speed on fresh large-file runs
4. minimal coupling to external package source trees

## Hard constraints

Do not edit these source trees:

- `/data/p/ms2rescore/DeepLC`
- `/data/p/ms2rescore/IM2Deep`
- `/data/p/ms2rescore/ms2pip`
- `/data/p/ms2rescore/ms2rescore`

Wrap them or call them, but do not patch them from this package.

## Canonical data rules

### Identity source

- use `Peptide`
- do not use `Proteins` for precursor mapping

### Spectrum matching

- use `ScanNr -> mzDuck.scan_number`
- for text MGF support:
  - use `SCANS=` first
  - fall back to parsing `TITLE` only when `SCANS=` is absent

### Modification parsing

Parse `[U:id]` peptides into:

- `sequence`
- `var_mod_sites_unimod`

Important invariant:

- `var_mod_sites_unimod` must be sorted by DB site position before lookup

Example:

- `[U:1]-M[U:35]LQFLLEVNK`
- correct:
  - `0:35;10:1`
- wrong:
  - `10:1;0:35`

### Target/decoy

`Label` selects target vs decoy tables.

## Runtime environments

### Main package

- `/data/p/anaconda3/envs/alphabase/bin/python`

### PostgreSQL helper

- `/data/p/anaconda3/envs/alphabase/bin/python`

### Local AlphaPept helper

- `/data/p/anaconda3/envs/alphabase/bin/python`

The helper should prefer `psycopg` and keep `psycopg2` only as a compatibility fallback.

## Performance guidance

### Current hot path

Current large-run behavior is:

- PostgreSQL lookup is now bulk and materially improved by ordered precursor-id fetches
- the main fresh-run CPU hot path is Alpha spectral feature generation
- DeepLC is secondary when `--deeplc-processes` is set high enough
- validated full `1554451` fresh run settings:
  - `--alpha-feature-threads 16`
  - `--alpha-feature-batch-size 512`
  - `--deeplc-processes 8`
- after the Numba feature kernel, full `1554451` Alpha scoring measured `30.43s`; one same-day fresh run was dominated by PostgreSQL lookup at `808.95s` due `DataFileRead` waits

Preferred optimization order:

1. keep PostgreSQL lookup in precursor-id order for alphapept-array fetches
2. avoid repeated precursor-array decoding
3. parallelize per-PSM spectral scoring
4. reduce Alpha feature math overhead before adding new languages
5. avoid rewriting unaffected cache entries

Current proven low-risk Alpha targets:

- the repeated `numpy.quantile`, `numpy.corrcoef`, and Spearman rank hot path now has a Numba kernel in `features.py`
- next measurements should check whether cache materialization, row dictionary creation, or thread scheduling dominate after JIT scoring
- poor effective CPU scaling from Python threads alone was observed before the Numba kernel and should be remeasured

Do not jump to C++ before measuring whether NumPy/SciPy/Numba changes are insufficient.

### Parallelism rules

When improving performance:

- keep PostgreSQL lookup bulk, not per-row
- keep alphapept-array fetches ordered by `precursor_id` when possible
- prefer multithreading for Alpha feature scoring
- keep DeepLC parallelism configurable separately
- keep local AlphaPept helper parallelism configurable separately

Avoid changes that:

- require shared mutable global state across helpers
- force all stages to use the same thread/process count

## Cache invariants

Per-idn cache directory:

- `<out-dir>/cache/<idn>/`

Current files:

- `precursor_cache.parquet`
- `alpha_features.parquet`
- `deeplc_base_features.parquet`
- `<idn>.deeplc_calibration.pkl`
- `manifest.json`

Required behavior:

- reruns must reuse cache whenever the stable key already exists
- final PIN must preserve original row order
- new rows must merge into cache by key, keeping the latest version

Stable keys:

- PSM key:
  - `idn|SpecId|Peptide|charge|Label`
- precursor key:
  - `Label|sequence|var_mod_sites_unimod|charge`

## File ownership

### Keep orchestration in

- `src/alpha2rescore/core.py`

### Keep bundled alphapeptms2 canonical code in

- `src/alpha2rescore/alphapeptms2/`

### Keep top-level alphapeptms2 compatibility wrappers thin in

- `src/alphapeptms2/`

### Keep PostgreSQL lookup logic in

- `src/alpha2rescore/postgres_helper.py`

### Keep local missing prediction logic in

- `src/alpha2rescore/alphapept_helper.py`

### Keep peptide parsing logic in

- `src/alpha2rescore/peptides.py`

### Keep spectral feature math in

- `src/alpha2rescore/features.py`

### Keep DeepLC persistence and RT feature logic in

- `src/alpha2rescore/deeplc_module.py`

Do not smear helper-specific logic across unrelated files.

## Validation expectations

At minimum, after nontrivial changes:

1. run `compileall` on `src`
2. run CLI help
3. run a small smoke build with `--max-psms`
4. rerun the same smoke build and confirm cache reuse

Before claiming performance work is complete:

1. run a larger real file
2. record thread/process settings
3. record whether the run was fresh or cached
4. record output path and row count
5. record stage timings for:
   - precursor lookup
   - Alpha features
   - DeepLC

## Documentation expectations

Keep these files current when behavior changes:

- `/data/p/ms2rescore/alpha2rescore/README.md`
- `/data/p/ms2rescore/alpha2rescore/AGENTS.md`

Also record reproducible test notes under:

- `/data/p/ms2rescore/ms2rescore-test/notes`

## Large-run practice

For long full-file runs:

- prefer tmux if the run may take a while
- write outputs under:
  - `/XCLabServer002_fastIO/ms2rescore-test/`

## Current known limitation

The integrated `missing_prediction -> local prediction -> final feature build` path was validated as a standalone helper invocation, but may not naturally trigger in every real sample if PostgreSQL already covers all precursors after parser fixes.

For large fresh runs, Alpha feature scoring is still the main scaling target even after the PostgreSQL lookup rewrite.
