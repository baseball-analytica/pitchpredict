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
}

# Backwards-compat aliases (not listed in available algorithms)
_ALGORITHM_ALIASES: dict[str, str] = {
    "deep": "xlstm",
}

# Cached algorithm instances
_algorithm_instances: dict[str, PitchPredictAlgorithm] = {}


def resolve_algorithm_name(algorithm_name: str) -> str:
    """Resolve an algorithm name to its canonical registry name."""
    return _ALGORITHM_ALIASES.get(algorithm_name, algorithm_name)


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
    resolved_name = resolve_algorithm_name(algorithm_name)
    if resolved_name not in _ALGORITHM_REGISTRY:
        available = ", ".join(sorted(_ALGORITHM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. Available: {available}"
        )

    # Use cached instance if available and no custom kwargs
    cache_key = resolved_name
    if not kwargs and cache_key in _algorithm_instances:
        return _algorithm_instances[cache_key]

    # Create new instance
    algorithm_class = _ALGORITHM_REGISTRY[resolved_name]
    instance = algorithm_class(name=resolved_name, **kwargs)

    # Cache if no custom kwargs
    if not kwargs:
        _algorithm_instances[cache_key] = instance

    return instance


def get_available_algorithms(include_aliases: bool = False) -> list[str]:
    """Get list of available algorithm names."""
    names = sorted(_ALGORITHM_REGISTRY.keys())
    if include_aliases:
        names.extend(sorted(_ALGORITHM_ALIASES.keys()))
    return names


__all__ = [
    "PitchPredictAlgorithm",
    "SimilarityAlgorithm",
    "XlstmAlgorithm",
    "get_algorithm_by_name",
    "get_available_algorithms",
    "resolve_algorithm_name",
]
