# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from typing import Any
from pydantic import BaseModel


class PredictPitcherRequest(BaseModel):
    pitcher_name: str
    batter_name: str
    balls: int
    strikes: int
    score_bat: int
    score_fld: int
    game_date: str
    algorithm: str


class PredictPitcherResponse(BaseModel):
    basic_pitch_data: dict[str, Any]
    detailed_pitch_data: dict[str, Any]
    basic_outcome_data: dict[str, Any]
    detailed_outcome_data: dict[str, Any]
    prediction_metadata: dict[str, Any]

class PredictBatterRequest(BaseModel):
    batter_name: str
    pitcher_name: str
    balls: int
    strikes: int
    score_bat: int
    score_fld: int
    game_date: str
    pitch_type: str
    pitch_speed: float
    pitch_x: float
    pitch_y: float
    algorithm: str


class PredictBatterResponse(BaseModel):
    basic_outcome_data: dict[str, Any]
    detailed_outcome_data: dict[str, Any]
    prediction_metadata: dict[str, Any]


class PredictBattedBallRequest(BaseModel):
    # Required fields
    launch_speed: float
    launch_angle: float
    algorithm: str

    # Optional fields
    spray_angle: float | None = None
    bb_type: str | None = None
    outs: int | None = None
    bases_state: int | None = None
    batter_id: int | None = None
    game_date: str | None = None


class PredictBattedBallResponse(BaseModel):
    basic_outcome_data: dict[str, Any]
    detailed_outcome_data: dict[str, Any]
    prediction_metadata: dict[str, Any]

