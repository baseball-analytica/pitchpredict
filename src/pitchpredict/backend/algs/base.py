# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from abc import abstractmethod
from typing import Any

import pitchpredict.types.api as api_types


class PitchPredictAlgorithm:
    """
    Base class for a PitchPredict algorithm.
    """

    def __init__(
        self,
        name: str,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.kwargs = kwargs

    @abstractmethod
    async def predict_pitcher(
        self,
        request: api_types.PredictPitcherRequest,
        **kwargs: Any,
    ) -> api_types.PredictPitcherResponse:
        """
        Predict the pitcher's next pitch and its outcome.
        """
        pass

    @abstractmethod
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
    ) -> dict[str, Any]:
        """
        Predict the batter's next outcome.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """
        Get the metadata for the algorithm, including usage information.
        """
        pass

    @abstractmethod
    def get_pitcher_prediction_metadata(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Get the metadata for the pitcher prediction, including usage information.
        """
        pass

    @abstractmethod
    def get_batter_prediction_metadata(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Get the metadata for the batter prediction, including usage information.
        """
        pass

    @abstractmethod
    async def predict_batted_ball(
        self,
        launch_speed: float,
        launch_angle: float,
        spray_angle: float | None = None,
        bb_type: str | None = None,
        outs: int | None = None,
        bases_state: int | None = None,
        batter_id: int | None = None,
        game_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Predict batted ball outcome probabilities given exit velocity, launch angle, and optional context.
        """
        pass

    @abstractmethod
    def get_batted_ball_prediction_metadata(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Get the metadata for the batted ball prediction, including usage information.
        """
        pass