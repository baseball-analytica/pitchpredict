# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel
import torch



class PitchTokenType(Enum):
    STREAM = 0
    STREAM_END = 1
    PITCH = 2
    PITCH_END = 3
    IS_CH = 4
    IS_CU = 5
    IS_FC = 6
    IS_EP = 7
    IS_FO = 8
    IS_FF = 9
    IS_KN = 10
    IS_KC = 11
    IS_SC = 12
    IS_SI = 13
    IS_SL = 14
    IS_SV = 15
    IS_FS = 16
    IS_ST = 17
    SPEED = 18
    RELEASE_POS_X = 19
    RELEASE_POS_Z = 20
    PLATE_POS_X = 21
    PLATE_POS_Z = 22
    EVENT_IS_NONE = 24

@dataclass
class PitchToken:
    """
    A single token output by the deep model.
    """
    type: PitchTokenType
    value: float

    def to_tensor(self) -> torch.Tensor:
        """
        Convert the pitch token to a tensor.
        """
        token_type_tensor = torch.tensor(
            [
                1.0 if self.type == PitchTokenType.STREAM else -1.0,
                1.0 if self.type == PitchTokenType.STREAM_END else -1.0,
                1.0 if self.type == PitchTokenType.PITCH else -1.0,
                1.0 if self.type == PitchTokenType.PITCH_END else -1.0,
                1.0 if self.type == PitchTokenType.IS_CH else -1.0,
                1.0 if self.type == PitchTokenType.IS_CU else -1.0,
                1.0 if self.type == PitchTokenType.IS_FC else -1.0,
                1.0 if self.type == PitchTokenType.IS_EP else -1.0,
                1.0 if self.type == PitchTokenType.IS_FO else -1.0,
                1.0 if self.type == PitchTokenType.IS_FF else -1.0,
                1.0 if self.type == PitchTokenType.IS_KN else -1.0,
                1.0 if self.type == PitchTokenType.IS_KC else -1.0,
                1.0 if self.type == PitchTokenType.IS_SC else -1.0,
                1.0 if self.type == PitchTokenType.IS_SI else -1.0,
                1.0 if self.type == PitchTokenType.IS_SL else -1.0,
                1.0 if self.type == PitchTokenType.IS_SV else -1.0,
                1.0 if self.type == PitchTokenType.IS_FS else -1.0,
                1.0 if self.type == PitchTokenType.IS_ST else -1.0,
                1.0 if self.type == PitchTokenType.SPEED else -1.0,
                1.0 if self.type == PitchTokenType.RELEASE_POS_X else -1.0,
                1.0 if self.type == PitchTokenType.RELEASE_POS_Z else -1.0,
                1.0 if self.type == PitchTokenType.PLATE_POS_X else -1.0,
                1.0 if self.type == PitchTokenType.PLATE_POS_Z else -1.0,
                1.0 if self.type == PitchTokenType.EVENT_IS_NONE else -1.0,
            ], 
            dtype=torch.float32
        )
        value_tensor = torch.tensor(
            [self.value],
            dtype=torch.float32
        )
        return torch.cat((token_type_tensor, value_tensor), dim=0)

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> "PitchToken":
        """
        Convert a tensor to a pitch token.
        """
        token_type_tensor = tensor[:, 0]
        value_tensor = tensor[:, 1]
        return PitchToken(
            type=PitchTokenType(token_type_tensor.argmax().item()),
            value=value_tensor.item(),
        )


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