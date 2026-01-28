# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from typing import Any, Literal
from pydantic import BaseModel, Field


class Pitch(BaseModel):
    """
    An individual pitch that can be included in a larger sequence.

    For xLSTM/deep algorithm, pa_id is required to group pitches into plate appearances.
    """

    pitch_type: str
    speed: float
    spin_rate: float
    spin_axis: float
    release_pos_x: float
    release_pos_z: float
    release_extension: float
    vx0: float
    vy0: float
    vz0: float
    ax: float
    ay: float
    az: float
    plate_pos_x: float
    plate_pos_z: float
    result: str
    # Required for xLSTM/deep history grouping
    pa_id: int | None = None
    # Optional: per-pitch context overrides for xLSTM
    batter_id: int | None = None
    batter_age: int | None = None
    batter_hits: Literal["L", "R"] | None = None
    count_balls: int | None = None
    count_strikes: int | None = None
    outs: int | None = None
    bases_state: int | None = None
    score_bat: int | None = None
    score_fld: int | None = None
    inning: int | None = None
    pitch_number: int | None = None
    number_through_order: int | None = None


class PredictPitcherRequest(BaseModel):
    """
    Common type for all pitcher prediction API requests.
    """

    pitcher_id: int
    batter_id: int
    prev_pitches: list[Pitch] | None = None
    algorithm: str = "similarity"
    sample_size: int = 1  # how many pitches should the algorithm generate?
    # optional context fields
    pitcher_age: int | None = None
    pitcher_throws: Literal["L", "R"] | None = None
    batter_age: int | None = None
    batter_hits: Literal["L", "R"] | None = (
        None  # if the batter is a switch hitter, they will still be on just one side of the plate
    )
    count_balls: int | None = None
    count_strikes: int | None = None
    outs: int | None = None
    bases_state: int | None = None
    score_bat: int | None = None
    score_fld: int | None = None
    inning: int | None = None
    pitch_number: int | None = None
    number_through_order: int | None = None
    game_date: str | None = None
    fielder_2_id: int | None = None
    fielder_3_id: int | None = None
    fielder_4_id: int | None = None
    fielder_5_id: int | None = None
    fielder_6_id: int | None = None
    fielder_7_id: int | None = None
    fielder_8_id: int | None = None
    fielder_9_id: int | None = None
    batter_days_since_prev_game: int | None = None
    pitcher_days_since_prev_game: int | None = None
    strike_zone_top: float | None = None
    strike_zone_bottom: float | None = None


class PredictPitcherResponse(BaseModel):
    """
    Common type for all pitcher prediction API responses.
    """

    basic_pitch_data: dict[str, Any]
    detailed_pitch_data: dict[str, Any]
    basic_outcome_data: dict[str, Any]
    detailed_outcome_data: dict[str, Any]
    pitches: list[Pitch]
    prediction_metadata: dict[str, Any]


class PredictBatterRequest(BaseModel):
    """
    Predict batter outcome request.

    Includes the same context fields used by xLSTM/predict_pitcher plus
    the pitch parameters required for batter similarity scoring.
    """

    pitcher_id: int
    batter_id: int
    pitch_type: str
    pitch_speed: float
    pitch_x: float
    pitch_z: float
    prev_pitches: list[Pitch] | None = None
    algorithm: str = "similarity"
    sample_size: int = 1  # how many pitches should the algorithm generate?
    # optional context fields (same as PredictPitcherRequest)
    pitcher_age: int | None = None
    pitcher_throws: Literal["L", "R"] | None = None
    batter_age: int | None = None
    batter_hits: Literal["L", "R"] | None = None
    count_balls: int | None = None
    count_strikes: int | None = None
    outs: int | None = None
    bases_state: int | None = None
    score_bat: int | None = None
    score_fld: int | None = None
    inning: int | None = None
    pitch_number: int | None = None
    number_through_order: int | None = None
    game_date: str | None = None
    fielder_2_id: int | None = None
    fielder_3_id: int | None = None
    fielder_4_id: int | None = None
    fielder_5_id: int | None = None
    fielder_6_id: int | None = None
    fielder_7_id: int | None = None
    fielder_8_id: int | None = None
    fielder_9_id: int | None = None
    batter_days_since_prev_game: int | None = None
    pitcher_days_since_prev_game: int | None = None
    strike_zone_top: float | None = None
    strike_zone_bottom: float | None = None


class PredictBatterResponse(BaseModel):
    algorithm_metadata: dict[str, Any] = Field(default_factory=dict)
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


class PlayerLookupResponse(BaseModel):
    query: str
    fuzzy: bool
    results: list[dict[str, Any]]


class PlayerRecordResponse(BaseModel):
    mlbam_id: int
    record: dict[str, Any]
