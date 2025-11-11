# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from typing import Any

from pitchpredict.backend.algs.base import PitchPredictAlgorithm
from pitchpredict.backend.algs.similarity.base import SimilarityAlgorithm
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
        algorithms: dict[str, PitchPredictAlgorithm] | None = None,
    ) -> None:
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        self.enable_logging = enable_logging
        self.log_dir = log_dir
        self.log_level_console = log_level_console
        self.log_level_file = log_level_file
        self.fuzzy_player_lookup = fuzzy_player_lookup
        if algorithms is None:
            algorithms = {
                "similarity": SimilarityAlgorithm(),
            }
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

    async def predict_pitcher(
        self,
        pitcher_name: str,
        batter_name: str,
        balls: int,
        strikes: int,
        score_bat: int,
        score_fld: int,
        game_date: str,
        algorithm: str,
    ) -> dict[str, Any]:
        """
        Given a context, predict the pitcher's next pitch and its outcome.
        """
        alg = self.algorithms.get(algorithm)
        if alg is None:
            raise ValueError(f"unrecognized algorithm: {algorithm}")
        return await alg.predict_pitcher(
            pitcher_name=pitcher_name,
            batter_name=batter_name,
            balls=balls,
            strikes=strikes,
            score_bat=score_bat,
            score_fld=score_fld,
            game_date=game_date,
        )

    async def predict_batter(
        self,
        batter_name: str,
        pitcher_name: str,
        balls: int,
        strikes: int,
        score_bat: int,
        score_fld: int,
        game_date: str,
        pitch_type: str,
        pitch_speed: float,
        pitch_x: float,
        pitch_y: float,
        algorithm: str,
    ) -> dict[str, Any]:
        """
        Given a context, predict the batter's next outcome.
        """
        alg = self.algorithms.get(algorithm)
        if alg is None:
            raise ValueError(f"unrecognized algorithm: {algorithm}")
        return await alg.predict_batter(
            batter_name=batter_name,
            pitcher_name=pitcher_name,
            balls=balls,
            strikes=strikes,
            score_bat=score_bat,
            score_fld=score_fld,
            game_date=game_date,
            pitch_type=pitch_type,
            pitch_speed=pitch_speed,
            pitch_x=pitch_x,
            pitch_y=pitch_y,
        )