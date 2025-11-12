# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from typing import Literal
from pydantic import BaseModel


class PitchToken(BaseModel):
    type: str
    speed: float
    release_pos_x: float
    release_pos_z: float
    plate_pos_x: float
    plate_pos_z: float
    event: str
    end_of_pa: bool = False


class PitchContext(BaseModel):
    pitcher_age: int
    pitcher_throws: Literal["L", "R"]
    batter_age: int
    batter_hits: Literal["L", "R"] # if the batter is a switch hitter, they will still be on just one side of the plate
    count_balls: int
    count_strikes: int
    outs: int
    runner_on_first: bool
    runner_on_second: bool
    runner_on_third: bool
    score_bat: int
    score_fld: int
    inning: int
    pitch_number: int
    number_through_order: int
    game_date: str


