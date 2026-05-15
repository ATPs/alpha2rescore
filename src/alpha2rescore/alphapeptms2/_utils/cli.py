"""CLI definition using Click."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np

from ..constants import DEFAULT_MODEL, SUPPORTED_DEVICES


@click.group()
@click.version_option(version="0.1.0", prog_name="alphapeptms2")
def main():
    """alphapeptms2 - AlphaPeptDeep MS2 prediction with MS2PIP-compatible CLI."""
    pass


@main.command("predict-single")
@click.argument("peptidoform", type=str)
@click.option("--model", default=DEFAULT_MODEL, help="Prediction model (default: HCD)")
@click.option(
    "--device", default="cpu", type=click.Choice(SUPPORTED_DEVICES), help="Inference device"
)
@click.option("--output", "-o", default=None, type=click.Path(), help="Output file (JSON)")
def predict_single_cmd(peptidoform: str, model: str, device: str, output: str | None):
    """Predict MS2 spectrum for a single ProForma peptidoform.

    Example: alphapeptms2 predict-single "PGAQANPYSR/3" --model HCD
    """
    from .. import predict_single

    click.echo(f"Predicting spectrum for: {peptidoform}")
    result = predict_single(peptidoform, model=model, device=device)

    _print_result(result)
    if output:
        _write_result_json(result, output)


@main.command("predict-batch")
@click.argument("psm_file", type=click.Path(exists=True))
@click.option("--model", default=DEFAULT_MODEL, help="Prediction model (default: HCD)")
@click.option(
    "--device", default="cpu", type=click.Choice(SUPPORTED_DEVICES), help="Inference device"
)
@click.option("--psm-filetype", default=None, help="PSM file format (auto-detected if omitted)")
@click.option("--output", "-o", default=None, type=click.Path(), help="Output file (JSON)")
@click.option("--chunk-size", default=5000, type=int, help="Precursors per prediction chunk")
def predict_batch_cmd(
    psm_file: str,
    model: str,
    device: str,
    psm_filetype: str | None,
    output: str | None,
    chunk_size: int,
):
    """Predict MS2 spectra for PSMs from a file.

    Example: alphapeptms2 predict-batch peptides.tsv --model HCD
    """
    from .. import predict_batch

    click.echo(f"Predicting spectra for: {psm_file}")
    results = predict_batch(
        psm_file, model=model, device=device, psm_filetype=psm_filetype, chunk_size=chunk_size
    )

    click.echo(f"Predicted {len(results)} spectra.")
    if output:
        _write_batch_json(results, output)


@main.command("correlate")
@click.argument("psm_file", type=click.Path(exists=True))
@click.argument("spectrum_file", type=click.Path(exists=True))
@click.option("--model", default=DEFAULT_MODEL, help="Prediction model (default: HCD)")
@click.option(
    "--device", default="cpu", type=click.Choice(SUPPORTED_DEVICES), help="Inference device"
)
@click.option(
    "--ms2-tolerance", default=0.02, type=float, help="MS2 mass tolerance in Da (default: 0.02)"
)
@click.option("--psm-filetype", default=None, help="PSM file format (auto-detected if omitted)")
@click.option(
    "--spectrum-id-pattern",
    default=None,
    help="Regex to extract spectrum ID from spectrum title",
)
@click.option("--output", "-o", default=None, type=click.Path(), help="Output file (JSON)")
@click.option("--chunk-size", default=5000, type=int, help="Precursors per prediction chunk")
def correlate_cmd(
    psm_file: str,
    spectrum_file: str,
    model: str,
    device: str,
    ms2_tolerance: float,
    psm_filetype: str | None,
    spectrum_id_pattern: str | None,
    output: str | None,
    chunk_size: int,
):
    """Predict and correlate spectra with observed peaks.

    Example: alphapeptms2 correlate peptides.tsv spectra.mgf --model HCD
    """
    from .. import correlate

    click.echo(f"Correlating PSMs from {psm_file} with spectra from {spectrum_file}")
    results = correlate(
        psm_file,
        spectrum_file,
        model=model,
        device=device,
        ms2_tolerance=ms2_tolerance,
        psm_filetype=psm_filetype,
        spectrum_id_pattern=spectrum_id_pattern,
        chunk_size=chunk_size,
    )

    n_with_corr = sum(1 for r in results if r.correlation is not None)
    correlations = [r.correlation for r in results if r.correlation is not None]
    click.echo(f"Correlated {n_with_corr}/{len(results)} spectra")
    if correlations:
        click.echo(f"Median correlation: {float(np.median(correlations)):.4f}")

    if output:
        _write_batch_json(results, output)


def _print_result(result):
    """Print a single ProcessingResult to stdout."""
    click.echo(f"  psm_index: {result.psm_index}")
    if result.theoretical_mz:
        for ion_type, mz_arr in result.theoretical_mz.items():
            n_peaks = (mz_arr > 0).sum()
            click.echo(f"  theoretical_mz[{ion_type}]: shape={mz_arr.shape}, nonzero_peaks={n_peaks}")
    if result.predicted_intensity:
        for ion_type, int_arr in result.predicted_intensity.items():
            nonzero = (int_arr > np.log2(0.001)).sum()
            click.echo(f"  predicted_intensity[{ion_type}]: shape={int_arr.shape}, nonzero={nonzero}")
    if result.correlation is not None:
        click.echo(f"  correlation: {result.correlation:.4f}")


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)


def _write_result_json(result, path: str):
    """Write a single result as JSON."""
    _write_batch_json([result], path)


def _write_batch_json(results: list, path: str):
    """Write batch results as JSON."""
    data = []
    for r in results:
        entry = {
            "psm_index": r.psm_index,
            "correlation": r.correlation,
        }
        if r.theoretical_mz:
            entry["n_b_ions"] = int((r.theoretical_mz.get("b", np.array([])) > 0).sum())
            entry["n_y_ions"] = int((r.theoretical_mz.get("y", np.array([])) > 0).sum())
        data.append(entry)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
    click.echo(f"Wrote results to {path}")


if __name__ == "__main__":
    main()
