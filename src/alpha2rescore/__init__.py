"""alpha2rescore public API."""

from .config import Alpha2RescoreConfig, BuildResult

__all__ = [
    "Alpha2RescoreConfig",
    "BuildResult",
    "build_features",
    "build_pin",
    "load_deeplc_calibration",
    "save_deeplc_calibration",
]


def build_features(*args, **kwargs):
    from .core import build_features as _build_features

    return _build_features(*args, **kwargs)


def build_pin(*args, **kwargs):
    from .core import build_pin as _build_pin

    return _build_pin(*args, **kwargs)


def load_deeplc_calibration(*args, **kwargs):
    from .deeplc_module import load_deeplc_calibration as _load_deeplc_calibration

    return _load_deeplc_calibration(*args, **kwargs)


def save_deeplc_calibration(*args, **kwargs):
    from .deeplc_module import save_deeplc_calibration as _save_deeplc_calibration

    return _save_deeplc_calibration(*args, **kwargs)
