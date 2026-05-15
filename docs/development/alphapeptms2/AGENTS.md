# alphapeptms2 Development Guide

## Scope

This guide covers the bundled `alphapeptms2` implementation under
`src/alpha2rescore/alphapeptms2/` and the top-level compatibility layer under
`src/alphapeptms2/`, both of which provide a thin wrapper around AlphaPeptDeep
(`peptdeep`) with an MS2PIP-like API.

Keep changes local to those package directories unless the caller explicitly
asks for cross-project integration.

## Required Environment

Use the `alphabase` conda environment:

```bash
/data/p/anaconda3/envs/alphabase/bin/python
```

Install this package in editable mode from the repository root:

```bash
/data/p/anaconda3/envs/alphabase/bin/python -m pip install -e /data/p/ms2rescore/alpha2rescore
```

If dependencies must be installed, install them into this environment only.

## Files To Read First

Before changing behavior, read:

- `20260511design.md`
- `../../../README.md`
- The relevant module under `../../../src/alpha2rescore/alphapeptms2/`
- `../../../src/alphapeptms2/` when compatibility imports or CLI entry points are involved

After user-facing behavior changes, update `../../../README.md`. If
future-development rules change, update this `AGENTS.md` too.

## API Compatibility Rules

Public imports must remain available from `alphapeptms2/__init__.py`:

- `predict_single`
- `predict_batch`
- `correlate`
- `ProcessingResult`

`ProcessingResult` should stay aligned with MS2PIP-style fields:

- `psm_index`
- `psm`
- `theoretical_mz`
- `predicted_intensity`
- `observed_intensity`
- `correlation`

For `theoretical_mz`, `predicted_intensity`, and `observed_intensity`, use dicts
keyed by ion type (`"b"`, `"y"`). Values are NumPy arrays with shape
`(n_fragment_positions, 3)`, where columns are charge 1, charge 2, and charge 3.

AlphaPeptDeep may reorder precursor rows internally. Preserve the original PSM
index through `_psm_idx`, attach the original `PSM`, and return batch results in
input order.

## AlphaPeptDeep Conventions

AlphaPeptDeep input DataFrames must contain:

- `sequence`
- `mods`
- `mod_sites`
- `charge`

Modification names should match AlphaBase names, for example:

- `Oxidation@M`
- `Carbamidomethyl@C`
- `Acetyl@Any_N-term`
- `Amidated@Any_C-term`
- `Gln->pyro-Glu@Q^Any_N-term`

Modification sites use AlphaBase conventions:

- `0` for peptide N-terminus
- `-1` for peptide C-terminus
- `1..n` for residue positions

Important edge case:

- When AlphaBase only defines a residue-specific terminal form, a ProForma
  residue-attached modification on the first or last residue can still resolve
  to site `0` or `-1`. Example: `Q[UNIMOD:28]...` becomes
  `Gln->pyro-Glu@Q^Any_N-term` with `mod_sites=0`.

Predicted intensities are stored in log space:

```python
np.log2(raw_intensity + 0.001)
```

Keep charge-3 arrays zero-filled until AlphaPeptDeep model output includes those
fragments.

## Spectrum And Correlation Rules

MGF is read with `pyteomics.mgf` after a light sanitation pass that removes
malformed peak lines and drops empty/broken blocks. mzML and mzML.gz are read
with `pyteomics.mzml`.

When `spectrum_id_pattern` is provided, apply it to both observed spectrum
identifiers and PSM `spectrum_id` values. This keeps matching symmetric.

Use `alphapeptms2.spectrum.load_spectrum_index()` and
`alphapeptms2.core.attach_observed_intensities()` when a caller needs to split
GPU prediction from CPU-side matching.

Calculate correlations only after observed intensities have been populated.
Exclude zero theoretical m/z entries from correlation so padded fragment columns
do not affect the result.

Warn on unexpectedly low match fractions and on duplicate normalized spectrum
IDs. Do not silently overwrite those conditions without a good reason.

## Testing

Run focused tests before handing work back:

```bash
/data/p/anaconda3/envs/alphabase/bin/python -m pytest -q
```

For changes touching AlphaPeptDeep prediction, also run a live smoke test:

```bash
/data/p/anaconda3/envs/alphabase/bin/python - <<'PY'
from alphapeptms2 import predict_single
r = predict_single("PGAQANPYSR/3", device="cpu")
print(r.theoretical_mz["b"].shape, r.predicted_intensity["y"].shape)
PY
```

Use small synthetic MGF files for correlation tests. Avoid committing large
spectra, pretrained models, or generated prediction outputs.

## Documentation Standard

Complex functions should document:

- Purpose
- Inputs
- Output structure
- Small examples
- Important edge cases

Simple helpers only need a brief docstring.

## Change Discipline

Prefer small compatibility fixes over broad refactors. Do not introduce new
frameworks or global repository tooling for this package unless explicitly
requested.

Do not commit credentials, local model files, or large data files.
