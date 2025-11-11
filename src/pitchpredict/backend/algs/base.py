# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from abc import abstractmethod
from typing import Any


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
        pitcher_name: str,
        batter_name: str,
        balls: int,
        strikes: int,
        score_bat: int,
        score_fld: int,
        game_date: str,
    ) -> dict[str, Any]:
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
        pitch_y: float,
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