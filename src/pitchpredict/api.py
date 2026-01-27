# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import logging
from typing import Any, Literal

from fastapi import HTTPException
import pybaseball  # type: ignore

from pitchpredict.backend.algs.base import PitchPredictAlgorithm
from pitchpredict.backend.algs.similarity.base import SimilarityAlgorithm
from pitchpredict.backend.caching import PitchPredictCache
from pitchpredict.backend.fetching import (
    get_player_id_from_name,
    get_player_record_from_id,
    get_player_records_from_name,
)
from pitchpredict.backend.logging import init_logger
import pitchpredict.types.api as api_types


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
        self.logger = logging.getLogger("pitchpredict")
        if self.enable_logging:
            init_logger(
                log_dir=self.log_dir,
                log_level_console=self.log_level_console,
                log_level_file=self.log_level_file,
            )
            self.logger.info("logging initialized")
        else:
            # Disable logging by setting a high level and adding a NullHandler
            self.logger.setLevel(logging.CRITICAL + 1)
            if not self.logger.handlers:
                self.logger.addHandler(logging.NullHandler())

        # check cache stuff
        if self.enable_cache:
            self.cache = PitchPredictCache(cache_dir=self.cache_dir)
            self.logger.info("cache initialized")
            for alg in self.algorithms.values():
                if hasattr(alg, "set_cache"):
                    alg.set_cache(self.cache)
                elif hasattr(alg, "cache"):
                    alg.cache = self.cache

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

    async def get_player_id_from_name(
        self,
        player_name: str,
        fuzzy_lookup: bool | None = None,
    ) -> int:
        """
        Resolve a player's MLBAM ID from their name.
        """
        self.logger.debug("get_player_id_from_name called")

        if fuzzy_lookup is None:
            fuzzy_lookup = self.fuzzy_player_lookup
        try:
            return await get_player_id_from_name(
                player_name=player_name,
                fuzzy_lookup=fuzzy_lookup,
                cache=getattr(self, "cache", None),
            )
        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_player_records_from_name(
        self,
        player_name: str,
        fuzzy_lookup: bool | None = None,
        limit: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Resolve player records from a name, returning up to `limit` candidates.
        """
        self.logger.debug("get_player_records_from_name called")

        if fuzzy_lookup is None:
            fuzzy_lookup = self.fuzzy_player_lookup
        try:
            return await get_player_records_from_name(
                player_name=player_name,
                fuzzy_lookup=fuzzy_lookup,
                limit=limit,
                cache=getattr(self, "cache", None),
            )
        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_player_record_from_id(self, mlbam_id: int) -> dict[str, Any]:
        """
        Resolve a player record from an MLBAM ID.
        """
        self.logger.debug("get_player_record_from_id called")

        try:
            return await get_player_record_from_id(
                mlbam_id=mlbam_id,
                cache=getattr(self, "cache", None),
            )
        except HTTPException as e:
            self.logger.error(f"encountered HTTPException: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"encountered Exception: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def predict_pitcher(
        self,
        pitcher_id: int,
        batter_id: int,
        prev_pitches: list[api_types.Pitch] | None = None,
        algorithm: str = "similarity",
        sample_size: int = 1,
        pitcher_age: int | None = None,
        pitcher_throws: Literal["L", "R"] | None = None,
        batter_age: int | None = None,
        batter_hits: Literal["L", "R"] | None = None,
        count_balls: int | None = None,
        count_strikes: int | None = None,
        outs: int | None = None,
        bases_state: int | None = None,
        score_bat: int | None = None,
        score_fld: int | None = None,
        inning: int | None = None,
        pitch_number: int | None = None,
        number_through_order: int | None = None,
        game_date: str | None = None,
        fielder_2_id: int | None = None,
        fielder_3_id: int | None = None,
        fielder_4_id: int | None = None,
        fielder_5_id: int | None = None,
        fielder_6_id: int | None = None,
        fielder_7_id: int | None = None,
        fielder_8_id: int | None = None,
        fielder_9_id: int | None = None,
        batter_days_since_prev_game: int | None = None,
        pitcher_days_since_prev_game: int | None = None,
        strike_zone_top: float | None = None,
        strike_zone_bottom: float | None = None,
    ) -> api_types.PredictPitcherResponse:
        """
        Given a context, predict the pitcher's next pitch and its outcome.
        """
        self.logger.debug("predict_pitcher called")

        request = api_types.PredictPitcherRequest(
            pitcher_id=pitcher_id,
            batter_id=batter_id,
            prev_pitches=prev_pitches,
            algorithm=algorithm,
            sample_size=sample_size,
            pitcher_age=pitcher_age,
            pitcher_throws=pitcher_throws,
            batter_age=batter_age,
            batter_hits=batter_hits,
            count_balls=count_balls,
            count_strikes=count_strikes,
            outs=outs,
            bases_state=bases_state,
            score_bat=score_bat,
            score_fld=score_fld,
            inning=inning,
            pitch_number=pitch_number,
            number_through_order=number_through_order,
            game_date=game_date,
            fielder_2_id=fielder_2_id,
            fielder_3_id=fielder_3_id,
            fielder_4_id=fielder_4_id,
            fielder_5_id=fielder_5_id,
            fielder_6_id=fielder_6_id,
            fielder_7_id=fielder_7_id,
            fielder_8_id=fielder_8_id,
            fielder_9_id=fielder_9_id,
            batter_days_since_prev_game=batter_days_since_prev_game,
            pitcher_days_since_prev_game=pitcher_days_since_prev_game,
            strike_zone_top=strike_zone_top,
            strike_zone_bottom=strike_zone_bottom,
        )

        alg = self.algorithms.get(request.algorithm)
        if alg is None:
            self.logger.error(f"unrecognized algorithm: {request.algorithm}")
            raise HTTPException(
                status_code=400, detail=f"unrecognized algorithm: {request.algorithm}"
            )
        self.logger.debug(f"using algorithm: {request.algorithm}")

        try:
            result = await alg.predict_pitcher(request=request)
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
            raise HTTPException(
                status_code=400, detail=f"unrecognized algorithm: {algorithm}"
            )
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
            raise HTTPException(
                status_code=400, detail=f"unrecognized algorithm: {algorithm}"
            )
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
