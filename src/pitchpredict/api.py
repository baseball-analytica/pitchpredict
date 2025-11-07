# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import os
from typing import Any

from pitchpredict.backend.caching import PitchPredictCache
from pitchpredict.backend.logging import init_logger


class PitchPredict:
    """
    Predict MLB pitcher/batter behavior and outcomes using a given context.
    """

    def __init__(
        self,
        enable_cache: bool = True,
        cache_dir: str = ".pitchpredict_cache",
        enable_logging: bool = True,
        log_dir: str = ".pitchpredict_logs",
        log_level_console: str = "INFO",
        log_level_file: str = "INFO",
        fuzzy_player_lookup: bool = True,
        algorithms: list[str] = ["similarity", "deep"]
    ) -> None:
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        self.enable_logging = enable_logging
        self.log_dir = log_dir
        self.log_level_console = log_level_console
        self.log_level_file = log_level_file
        self.fuzzy_player_lookup = fuzzy_player_lookup
        self.algorithms = algorithms

    def __post_init__(self) -> None:
        """
        Perform post-initialization tasks, including validation.
        """
        # check cache stuff
        if self.enable_cache:
            self.cache = PitchPredictCache(cache_dir=self.cache_dir)
        
        # check logging stuff
        if self.enable_logging:
            init_logger(
                log_dir=self.log_dir,
                log_level_console=self.log_level_console,
                log_level_file=self.log_level_file,
            )
        
        # check algorithms
        VALID_ALGORITHMS = ["similarity", "deep"]
        if self.algorithms == []:
            raise ValueError("at least one algorithm must be specified")
        for algorithm in self.algorithms:
            if algorithm not in VALID_ALGORITHMS:
                raise ValueError(f"unrecognized algorithm: {algorithm}")

    