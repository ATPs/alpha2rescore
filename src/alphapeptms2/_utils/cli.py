"""Compatibility re-export for alphapeptms2._utils.cli."""

import sys

from alpha2rescore.alphapeptms2._utils import cli as _cli

sys.modules[__name__] = _cli
