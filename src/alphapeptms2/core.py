"""Compatibility re-export for alphapeptms2.core."""

import sys

from alpha2rescore.alphapeptms2 import core as _core

sys.modules[__name__] = _core
