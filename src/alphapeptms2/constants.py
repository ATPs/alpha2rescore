"""Compatibility re-export for alphapeptms2.constants."""

import sys

from alpha2rescore.alphapeptms2 import constants as _constants

sys.modules[__name__] = _constants
