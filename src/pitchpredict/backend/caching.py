# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import os
import logging


class PitchPredictCache:
    """
    A local cache for PitchPredict data.
    """

    def __init__(
        self,
        cache_dir: str = ".pitchpredict_cache",
    ) -> None:
        self.cache_dir = cache_dir
        self.logger = logging.getLogger("pitchpredict.backend.caching")
    
    def __post_init__(self) -> None:
        """
        Perform post-initialization tasks, including validation.
        """
        self.logger.debug("running post-initialization tasks for cache")

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            self.logger.info(f"created cache directory at {self.cache_dir}")
        else:
            self.logger.info(f"cache directory already exists at {self.cache_dir}")

        self.logger.debug("cache initialized")