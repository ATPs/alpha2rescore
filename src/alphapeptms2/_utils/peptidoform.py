"""Compatibility re-export for alphapeptms2._utils.peptidoform."""

import sys

from alpha2rescore.alphapeptms2._utils import peptidoform as _peptidoform

sys.modules[__name__] = _peptidoform
