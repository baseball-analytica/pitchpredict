# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import logging
from typing import Any

from fastapi import HTTPException
import pybaseball # type: ignore

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
        if self.enable_cache:
            pybaseball.cache.enable()
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
        self.__post_init__()

    def __post_init__(self) -> None:
        """
        Perform post-initialization tasks, including validation.
        """
        # check logging stuff
        if self.enable_logging:
            init_logger(
                log_dir=self.log_dir,
                log_level_console=self.log_level_console,
                log_level_file=self.log_level_file,
            )
            self.logger = logging.getLogger("pitchpredict")
            self.logger.info("logging initialized")

        # check cache stuff
        if self.enable_cache:
            self.cache = PitchPredictCache(cache_dir=self.cache_dir)
            self.logger.info("cache initialized")
        
        # check algorithms
        VALID_ALGORITHMS = ["similarity", "deep"]
        if self.algorithms == []:
            self.logger.error("algorithms is an empty list")
            raise ValueError("at least one algorithm must be specified")
        for algorithm in self.algorithms:
            if algorithm not in VALID_ALGORITHMS:
                self.logger.error(f"unrecognized algorithm: {algorithm}")
                raise ValueError(f"unrecognized algorithm: {algorithm}")

        self.logger.debug("post-initialization tasks completed")

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
        self.logger.debug("predict_pitcher called")

        alg = self.algorithms.get(algorithm)
        if alg is None:
            self.logger.error(f"unrecognized algorithm: {algorithm}")
            raise HTTPException(status_code=400, detail=f"unrecognized algorithm: {algorithm}")
        self.logger.debug(f"using algorithm: {algorithm}")

        try:
            result = await alg.predict_pitcher(
                pitcher_name=pitcher_name,
                batter_name=batter_name,
                balls=balls,
                strikes=strikes,
                score_bat=score_bat,
                score_fld=score_fld,
                game_date=game_date,
            )
            self.logger.debug("predict_pitcher completed")
            return result
        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))

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
        pitch_z: float,
        algorithm: str,
    ) -> dict[str, Any]:
        """
        Given a context, predict the batter's next outcome.
        """
        self.logger.debug("predict_batter called")

        alg = self.algorithms.get(algorithm)
        if alg is None:
            self.logger.error(f"unrecognized algorithm: {algorithm}")
            raise HTTPException(status_code=400, detail=f"unrecognized algorithm: {algorithm}")
        self.logger.debug(f"using algorithm: {algorithm}")

        try:
            result = await alg.predict_batter(
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
                pitch_z=pitch_z,
            )
            self.logger.debug("predict_batter completed")
            return result
        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def predict_batted_ball(
        self,
        launch_speed: float,
        launch_angle: float,
        algorithm: str,
        spray_angle: float | None = None,
        bb_type: str | None = None,
        outs: int | None = None,
        bases_state: int | None = None,
        batter_id: int | None = None,
        game_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Given batted ball parameters, predict outcome probabilities.
        """
        self.logger.debug("predict_batted_ball called")

        alg = self.algorithms.get(algorithm)
        if alg is None:
            self.logger.error(f"unrecognized algorithm: {algorithm}")
            raise HTTPException(status_code=400, detail=f"unrecognized algorithm: {algorithm}")
        self.logger.debug(f"using algorithm: {algorithm}")

        try:
            result = await alg.predict_batted_ball(
                launch_speed=launch_speed,
                launch_angle=launch_angle,
                spray_angle=spray_angle,
                bb_type=bb_type,
                outs=outs,
                bases_state=bases_state,
                batter_id=batter_id,
                game_date=game_date,
            )
            self.logger.debug("predict_batted_ball completed")
            return result
        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))