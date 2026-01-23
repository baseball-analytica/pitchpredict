# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from collections.abc import Callable

from pitchpredict.backend.algs.base import PitchPredictAlgorithm


def get_algorithm_by_name(algorithm_name: str) -> PitchPredictAlgorithm:
    """
    Get the algorithm by name.
    """
    raise NotImplementedError("Not implemented")
