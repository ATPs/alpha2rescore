"""Compatibility re-export for alphapeptms2.spectrum."""

import sys

from alpha2rescore.alphapeptms2 import spectrum as _spectrum

sys.modules[__name__] = _spectrum
