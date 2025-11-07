# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import os


class PitchPredictCache:
    """
    A local cache for PitchPredict data.
    """

    def __init__(
        self,
        cache_dir: str = ".pitchpredict_cache",
    ) -> None:
        self.cache_dir = cache_dir
    
    def __post_init__(self) -> None:
        """
        Perform post-initialization tasks, including validation.
        """
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)