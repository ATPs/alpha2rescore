"""Compatibility tests for the bundled alphapeptms2 package."""

import importlib


def test_canonical_import_exposes_public_api():
    module = importlib.import_module("alpha2rescore.alphapeptms2")

    assert callable(module.predict_single)
    assert callable(module.predict_batch)
    assert callable(module.correlate)
    assert callable(module.calculate_correlations)


def test_top_level_compat_import_exposes_core_module():
    module = importlib.import_module("alphapeptms2")
    core_module = importlib.import_module("alphapeptms2.core")

    assert callable(module.predict_single)
    assert core_module.predict_batch is not None
