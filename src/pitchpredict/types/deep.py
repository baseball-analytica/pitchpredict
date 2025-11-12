# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from typing import Literal

from pydantic import BaseModel
import torch


class PitchToken(BaseModel):
    type: str
    speed: float
    release_pos_x: float
    release_pos_z: float
    plate_pos_x: float
    plate_pos_z: float
    event: str
    end_of_pa: bool = False

    def to_tensor(self) -> torch.Tensor:
        """
        Convert the pitch token to a tensor.
        """
        is_CH = 1 if self.type == "CH" else -1
        is_CU = 1 if self.type == "CU" else -1
        is_FC = 1 if self.type == "FC" else -1
        is_EP = 1 if self.type == "EP" else -1
        is_FO = 1 if self.type == "FO" else -1
        is_FF = 1 if self.type == "FF" else -1
        is_KN = 1 if self.type == "KN" else -1
        is_KC = 1 if self.type == "KC" else -1
        is_SC = 1 if self.type == "SC" else -1
        is_SI = 1 if self.type == "SI" else -1
        is_SL = 1 if self.type == "SL" else -1
        is_SV = 1 if self.type == "SV" else -1
        is_FS = 1 if self.type == "FS" else -1
        is_ST = 1 if self.type == "ST" else -1

        event_is_none = 1 if self.event == "" else -1

        tensor = torch.tensor([
            is_CH,
            is_CU,
            is_FC,
            is_EP,
            is_FO,
            is_FF,
            is_KN,
            is_KC,
            is_SC,
            is_SI,
            is_SL,
            is_SV,
            is_FS,
            is_ST,
            self.speed,
            self.release_pos_x,
            self.release_pos_z,
            self.plate_pos_x,
            self.plate_pos_z,
            event_is_none,
        ])

        return tensor


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

    def to_tensor(self) -> torch.Tensor:
        """
        Convert the pitch context to a tensor.
        """
        pitcher_is_lefty = 1 if self.pitcher_throws == "L" else -1
        batter_is_lefty = 1 if self.batter_hits == "L" else -1
        game_year = int(self.game_date.split("-")[0])

        tensor = torch.tensor([
            self.pitcher_age,
            pitcher_is_lefty,
            self.batter_age,
            batter_is_lefty,
            self.count_balls,
            self.count_strikes,
            self.outs,
            self.runner_on_first,
            self.runner_on_second,
            self.runner_on_third,
            self.score_bat,
            self.score_fld,
            self.inning,
            self.pitch_number,
            self.number_through_order,
            game_year,
        ])

        return tensor