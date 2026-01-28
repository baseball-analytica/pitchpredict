# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from typing import Any

from pitchpredict.backend.algs.base import PitchPredictAlgorithm
from pitchpredict.backend.algs.similarity import SimilarityAlgorithm
from pitchpredict.backend.algs.xlstm import XlstmAlgorithm


# Registry of algorithm names to their classes
_ALGORITHM_REGISTRY: dict[str, type[PitchPredictAlgorithm]] = {
    "similarity": SimilarityAlgorithm,
    "xlstm": XlstmAlgorithm,
    "deep": XlstmAlgorithm,  # Alias for backwards compatibility
}

# Cached algorithm instances
_algorithm_instances: dict[str, PitchPredictAlgorithm] = {}


def get_algorithm_by_name(
    algorithm_name: str,
    **kwargs: Any,
) -> PitchPredictAlgorithm:
    """Get an algorithm instance by name.

    Args:
        algorithm_name: Name of the algorithm ('similarity', 'xlstm', or 'deep')
        **kwargs: Additional arguments passed to the algorithm constructor

    Returns:
        PitchPredictAlgorithm instance

    Raises:
        ValueError: If the algorithm name is not recognized
    """
    if algorithm_name not in _ALGORITHM_REGISTRY:
        available = ", ".join(sorted(_ALGORITHM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. Available: {available}"
        )

    # Use cached instance if available and no custom kwargs
    cache_key = algorithm_name
    if not kwargs and cache_key in _algorithm_instances:
        return _algorithm_instances[cache_key]

    # Create new instance
    algorithm_class = _ALGORITHM_REGISTRY[algorithm_name]
    instance = algorithm_class(name=algorithm_name, **kwargs)

    # Cache if no custom kwargs
    if not kwargs:
        _algorithm_instances[cache_key] = instance

    return instance


def get_available_algorithms() -> list[str]:
    """Get list of available algorithm names."""
    return sorted(_ALGORITHM_REGISTRY.keys())


__all__ = [
    "PitchPredictAlgorithm",
    "SimilarityAlgorithm",
    "XlstmAlgorithm",
    "get_algorithm_by_name",
    "get_available_algorithms",
]
