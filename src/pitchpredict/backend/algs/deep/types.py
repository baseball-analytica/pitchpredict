# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from dataclasses import dataclass
from enum import auto, Enum
from typing import Literal

from pydantic import BaseModel
import torch



class PitchTokenType(Enum):
    PA_START = auto()
    PA_END = auto()
    PITCH = auto()
    PITCH_END = auto()
    IS_CH = auto()
    IS_CU = auto()
    IS_FC = auto()
    IS_EP = auto()
    IS_FO = auto()
    IS_FF = auto()
    IS_KN = auto()
    IS_KC = auto()
    IS_SC = auto()
    IS_SI = auto()
    IS_SL = auto()
    IS_SV = auto()
    IS_FS = auto()
    IS_ST = auto()
    SPEED_IS_LT65 = auto()
    SPEED_IS_65 = auto()
    SPEED_IS_66 = auto()
    SPEED_IS_67 = auto()
    SPEED_IS_68 = auto()
    SPEED_IS_69 = auto()
    SPEED_IS_70 = auto()
    SPEED_IS_71 = auto()
    SPEED_IS_72 = auto()
    SPEED_IS_73 = auto()
    SPEED_IS_74 = auto()
    SPEED_IS_75 = auto()
    SPEED_IS_76 = auto()
    SPEED_IS_77 = auto()
    SPEED_IS_78 = auto()
    SPEED_IS_79 = auto()
    SPEED_IS_80 = auto()
    SPEED_IS_81 = auto()
    SPEED_IS_82 = auto()
    SPEED_IS_83 = auto()
    SPEED_IS_84 = auto()
    SPEED_IS_85 = auto()
    SPEED_IS_86 = auto()
    SPEED_IS_87 = auto()
    SPEED_IS_88 = auto()
    SPEED_IS_89 = auto()
    SPEED_IS_90 = auto()
    SPEED_IS_91 = auto()
    SPEED_IS_92 = auto()
    SPEED_IS_93 = auto()
    SPEED_IS_94 = auto()
    SPEED_IS_95 = auto()
    SPEED_IS_96 = auto()
    SPEED_IS_97 = auto()
    SPEED_IS_98 = auto()
    SPEED_IS_99 = auto()
    SPEED_IS_100 = auto()
    SPEED_IS_101 = auto()
    SPEED_IS_102 = auto()
    SPEED_IS_103 = auto()
    SPEED_IS_104 = auto()
    SPEED_IS_105 = auto()
    SPEED_IS_GT105 = auto()
    RELEASE_POS_X_IS_LTN4 = auto()
    RELEASE_POS_X_IS_N4_N375 = auto()
    RELEASE_POS_X_IS_N375_N350 = auto()
    RELEASE_POS_X_IS_N350_N325 = auto()
    RELEASE_POS_X_IS_N325_N300 = auto()
    RELEASE_POS_X_IS_N300_N275 = auto()
    RELEASE_POS_X_IS_N275_N250 = auto()
    RELEASE_POS_X_IS_N250_N225 = auto()
    RELEASE_POS_X_IS_N225_N200 = auto()
    RELEASE_POS_X_IS_N200_N175 = auto()
    RELEASE_POS_X_IS_N175_N150 = auto()
    RELEASE_POS_X_IS_N150_N125 = auto()
    RELEASE_POS_X_IS_N125_N100 = auto()
    RELEASE_POS_X_IS_N100_N075 = auto()
    RELEASE_POS_X_IS_N075_N050 = auto()
    RELEASE_POS_X_IS_N050_N025 = auto()
    RELEASE_POS_X_IS_N025_0 = auto()
    RELEASE_POS_X_IS_0_025 = auto()
    RELEASE_POS_X_IS_025_050 = auto()
    RELEASE_POS_X_IS_050_075 = auto()
    RELEASE_POS_X_IS_075_1 = auto()
    RELEASE_POS_X_IS_1_125 = auto()
    RELEASE_POS_X_IS_125_150 = auto()
    RELEASE_POS_X_IS_150_175 = auto()
    RELEASE_POS_X_IS_175_2 = auto()
    RELEASE_POS_X_IS_2_225 = auto()
    RELEASE_POS_X_IS_225_250 = auto()
    RELEASE_POS_X_IS_250_275 = auto()
    RELEASE_POS_X_IS_275_3 = auto()
    RELEASE_POS_X_IS_3_325 = auto()
    RELEASE_POS_X_IS_325_350 = auto()
    RELEASE_POS_X_IS_350_375 = auto()
    RELEASE_POS_X_IS_375_4 = auto()
    RELEASE_POS_X_IS_GT4 = auto()
    RELEASE_POS_Z_IS_LT4 = auto()
    RELEASE_POS_Z_IS_4_425 = auto()
    RELEASE_POS_Z_IS_425_450 = auto()
    RELEASE_POS_Z_IS_450_475 = auto()
    RELEASE_POS_Z_IS_475_5 = auto()
    RELEASE_POS_Z_IS_5_525 = auto()
    RELEASE_POS_Z_IS_525_550 = auto()
    RELEASE_POS_Z_IS_550_575 = auto()
    RELEASE_POS_Z_IS_575_6 = auto()
    RELEASE_POS_Z_IS_6_625 = auto()
    RELEASE_POS_Z_IS_625_650 = auto()
    RELEASE_POS_Z_IS_650_675 = auto()
    RELEASE_POS_Z_IS_675_7 = auto()
    RELEASE_POS_Z_IS_GT7 = auto()
    PLATE_POS_X_IS_LTN2 = auto()
    PLATE_POS_X_IS_N2_N175 = auto()
    PLATE_POS_X_IS_N175_N150 = auto()
    PLATE_POS_X_IS_N150_N125 = auto()
    PLATE_POS_X_IS_N125_N100 = auto()
    PLATE_POS_X_IS_N100_N075 = auto()
    PLATE_POS_X_IS_N075_N050 = auto()
    PLATE_POS_X_IS_N050_N025 = auto()
    PLATE_POS_X_IS_N025_0 = auto()
    PLATE_POS_X_IS_0_025 = auto()
    PLATE_POS_X_IS_025_050 = auto()
    PLATE_POS_X_IS_050_075 = auto()
    PLATE_POS_X_IS_075_1 = auto()
    PLATE_POS_X_IS_1_125 = auto()
    PLATE_POS_X_IS_125_150 = auto()
    PLATE_POS_X_IS_150_175 = auto()
    PLATE_POS_X_IS_175_2 = auto()
    PLATE_POS_X_IS_GT2 = auto()
    PLATE_POS_Z_IS_LTN1 = auto()
    PLATE_POS_Z_IS_N1_N075 = auto()
    PLATE_POS_Z_IS_N075_N050 = auto()
    PLATE_POS_Z_IS_N050_N025 = auto()
    PLATE_POS_Z_IS_N025_0 = auto()
    PLATE_POS_Z_IS_0_025 = auto()
    PLATE_POS_Z_IS_025_050 = auto()
    PLATE_POS_Z_IS_050_075 = auto()
    PLATE_POS_Z_IS_075_1 = auto()
    PLATE_POS_Z_IS_1_125 = auto()
    PLATE_POS_Z_IS_125_150 = auto()
    PLATE_POS_Z_IS_150_175 = auto()
    PLATE_POS_Z_IS_175_2 = auto()
    PLATE_POS_Z_IS_2_225 = auto()
    PLATE_POS_Z_IS_225_250 = auto()
    PLATE_POS_Z_IS_250_275 = auto()
    PLATE_POS_Z_IS_275_3 = auto()
    PLATE_POS_Z_IS_3_325 = auto()
    PLATE_POS_Z_IS_325_350 = auto()
    PLATE_POS_Z_IS_350_375 = auto()
    PLATE_POS_Z_IS_375_4 = auto()
    PLATE_POS_Z_IS_4_425 = auto()
    PLATE_POS_Z_IS_425_450 = auto()
    PLATE_POS_Z_IS_450_475 = auto()
    PLATE_POS_Z_IS_475_5 = auto()
    PLATE_POS_Z_IS_GT5 = auto()
    RESULT_IS_BALL = auto()
    RESULT_IS_BALL_IN_DIRT = auto()
    RESULT_IS_CALLED_STRIKE = auto()
    RESULT_IS_FOUL = auto()
    RESULT_IS_FOUL_BUNT = auto()
    RESULT_IS_FOUL_TIP_BUNT = auto()
    RESULT_IS_FOUL_PITCHOUT = auto()
    RESULT_IS_PITCHOUT = auto()
    RESULT_IS_HIT_BY_PITCH = auto()
    RESULT_IS_INTENTIONAL_BALL = auto()
    RESULT_IS_IN_PLAY = auto()
    RESULT_IS_MISSED_BUNT = auto()
    RESULT_IS_FOUL_TIP = auto()
    RESULT_IS_SWINGING_PITCHOUT = auto()
    RESULT_IS_SWINGING_STRIKE = auto()
    RESULT_IS_SWINGING_STRIKE_BLOCKED = auto()





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