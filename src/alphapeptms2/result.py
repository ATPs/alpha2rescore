"""Compatibility re-export for alphapeptms2.result."""

import sys

from alpha2rescore.alphapeptms2 import result as _result

sys.modules[__name__] = _result
