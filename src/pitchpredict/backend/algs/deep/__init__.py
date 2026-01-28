# SPDX-License-Identifier: MIT
"""DEPRECATED: Use pitchpredict.backend.algs.xlstm instead.

This module is deprecated and will be removed in a future version.
Import from pitchpredict.backend.algs.xlstm instead.
Training/data utilities now live under tools/deep.
"""

import warnings

warnings.warn(
    "pitchpredict.backend.algs.deep is deprecated. "
    "Use pitchpredict.backend.algs.xlstm instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from xlstm for backwards compatibility
from pitchpredict.backend.algs.xlstm import XlstmAlgorithm

# Alias for backwards compatibility
DeepAlgorithm = XlstmAlgorithm

__all__ = ["DeepAlgorithm", "XlstmAlgorithm"]
