# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from enum import auto, Enum
from typing import Literal

from pydantic import BaseModel
import torch


class PitchToken(Enum):
    SESSION_START = auto()  # when a pitcher enters a game
    SESSION_END = auto()  # when the pitcher leaves the game
    PA_START = auto()  # start of a plate appearance
    PA_END = auto()
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
    IS_FA = auto()
    IS_CS = auto()
    IS_PO = auto()
    IS_UN = auto()
    IS_IN = auto()
    IS_AB = auto()
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
    SPIN_RATE_IS_LT750 = auto()
    SPIN_RATE_IS_750_1000 = auto()
    SPIN_RATE_IS_1000_1250 = auto()
    SPIN_RATE_IS_1250_1500 = auto()
    SPIN_RATE_IS_1500_1750 = auto()
    SPIN_RATE_IS_1750_2000 = auto()
    SPIN_RATE_IS_2000_2250 = auto()
    SPIN_RATE_IS_2250_2500 = auto()
    SPIN_RATE_IS_2500_2750 = auto()
    SPIN_RATE_IS_2750_3000 = auto()
    SPIN_RATE_IS_3000_3250 = auto()
    SPIN_RATE_IS_GT3250 = auto()
    SPIN_AXIS_IS_0_30 = auto()
    SPIN_AXIS_IS_30_60 = auto()
    SPIN_AXIS_IS_60_90 = auto()
    SPIN_AXIS_IS_90_120 = auto()
    SPIN_AXIS_IS_120_150 = auto()
    SPIN_AXIS_IS_150_180 = auto()
    SPIN_AXIS_IS_180_210 = auto()
    SPIN_AXIS_IS_210_240 = auto()
    SPIN_AXIS_IS_240_270 = auto()
    SPIN_AXIS_IS_270_300 = auto()
    SPIN_AXIS_IS_300_330 = auto()
    SPIN_AXIS_IS_330_360 = auto()
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
    RELEASE_EXTENSION_IS_LT5 = auto()
    RELEASE_EXTENSION_IS_5_55 = auto()
    RELEASE_EXTENSION_IS_55_6 = auto()
    RELEASE_EXTENSION_IS_6_65 = auto()
    RELEASE_EXTENSION_IS_65_7 = auto()
    RELEASE_EXTENSION_IS_7_75 = auto()
    RELEASE_EXTENSION_IS_GT75 = auto()
    VX0_IS_LTN15 = auto()
    VX0_IS_N15_N10 = auto()
    VX0_IS_N10_N5 = auto()
    VX0_IS_N5_0 = auto()
    VX0_IS_0_5 = auto()
    VX0_IS_5_10 = auto()
    VX0_IS_10_15 = auto()
    VX0_IS_GT15 = auto()
    VY0_IS_LTN150 = auto()
    VY0_IS_N150_N140 = auto()
    VY0_IS_N140_N130 = auto()
    VY0_IS_N130_N120 = auto()
    VY0_IS_N120_N110 = auto()
    VY0_IS_N110_N100 = auto()
    VY0_IS_GTN100 = auto()
    VZ0_IS_LTN10 = auto()
    VZ0_IS_N10_N5 = auto()
    VZ0_IS_N5_0 = auto()
    VZ0_IS_0_5 = auto()
    VZ0_IS_5_10 = auto()
    VZ0_IS_10_15 = auto()
    VZ0_IS_GT15 = auto()
    AX_IS_LTN25 = auto()
    AX_IS_N25_N20 = auto()
    AX_IS_N20_N15 = auto()
    AX_IS_N15_N10 = auto()
    AX_IS_N10_N5 = auto()
    AX_IS_N5_0 = auto()
    AX_IS_0_5 = auto()
    AX_IS_5_10 = auto()
    AX_IS_10_15 = auto()
    AX_IS_15_20 = auto()
    AX_IS_20_25 = auto()
    AX_IS_GT25 = auto()
    AY_IS_LT15 = auto()
    AY_IS_15_20 = auto()
    AY_IS_20_25 = auto()
    AY_IS_25_30 = auto()
    AY_IS_30_35 = auto()
    AY_IS_35_40 = auto()
    AY_IS_GT40 = auto()
    AZ_IS_LTN45 = auto()
    AZ_IS_N45_N40 = auto()
    AZ_IS_N40_N35 = auto()
    AZ_IS_N35_N30 = auto()
    AZ_IS_N30_N25 = auto()
    AZ_IS_N25_N20 = auto()
    AZ_IS_N20_N15 = auto()
    AZ_IS_GTN15 = auto()
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
    RESULT_IS_BLOCKED_BALL = auto()
    RESULT_IS_AUTOMATIC_BALL = auto()
    RESULT_IS_AUTOMATIC_STRIKE = auto()


class TokenCategory(Enum):
    """
    Categories for grouping PitchToken values by their semantic role.
    """

    SESSION_START = auto()
    SESSION_END = auto()
    PA_START = auto()
    PA_END = auto()
    PITCH_TYPE = auto()
    SPEED = auto()
    SPIN_RATE = auto()
    SPIN_AXIS = auto()
    RELEASE_POS_X = auto()
    RELEASE_POS_Z = auto()
    VX0 = auto()
    VY0 = auto()
    VZ0 = auto()
    AX = auto()
    AY = auto()
    AZ = auto()
    RELEASE_EXTENSION = auto()
    PLATE_POS_X = auto()
    PLATE_POS_Z = auto()
    RESULT = auto()


# Token range boundaries (inclusive) for category classification
_TOKEN_RANGES: list[tuple[PitchToken, PitchToken, TokenCategory]] = [
    (PitchToken.SESSION_START, PitchToken.SESSION_START, TokenCategory.SESSION_START),
    (PitchToken.SESSION_END, PitchToken.SESSION_END, TokenCategory.SESSION_END),
    (PitchToken.PA_START, PitchToken.PA_START, TokenCategory.PA_START),
    (PitchToken.PA_END, PitchToken.PA_END, TokenCategory.PA_END),
    (PitchToken.IS_CH, PitchToken.IS_AB, TokenCategory.PITCH_TYPE),
    (PitchToken.SPEED_IS_LT65, PitchToken.SPEED_IS_GT105, TokenCategory.SPEED),
    (
        PitchToken.SPIN_RATE_IS_LT750,
        PitchToken.SPIN_RATE_IS_GT3250,
        TokenCategory.SPIN_RATE,
    ),
    (
        PitchToken.SPIN_AXIS_IS_0_30,
        PitchToken.SPIN_AXIS_IS_330_360,
        TokenCategory.SPIN_AXIS,
    ),
    (
        PitchToken.RELEASE_POS_X_IS_LTN4,
        PitchToken.RELEASE_POS_X_IS_GT4,
        TokenCategory.RELEASE_POS_X,
    ),
    (
        PitchToken.RELEASE_POS_Z_IS_LT4,
        PitchToken.RELEASE_POS_Z_IS_GT7,
        TokenCategory.RELEASE_POS_Z,
    ),
    (
        PitchToken.RELEASE_EXTENSION_IS_LT5,
        PitchToken.RELEASE_EXTENSION_IS_GT75,
        TokenCategory.RELEASE_EXTENSION,
    ),
    (PitchToken.VX0_IS_LTN15, PitchToken.VX0_IS_GT15, TokenCategory.VX0),
    (PitchToken.VY0_IS_LTN150, PitchToken.VY0_IS_GTN100, TokenCategory.VY0),
    (PitchToken.VZ0_IS_LTN10, PitchToken.VZ0_IS_GT15, TokenCategory.VZ0),
    (PitchToken.AX_IS_LTN25, PitchToken.AX_IS_GT25, TokenCategory.AX),
    (PitchToken.AY_IS_LT15, PitchToken.AY_IS_GT40, TokenCategory.AY),
    (PitchToken.AZ_IS_LTN45, PitchToken.AZ_IS_GTN15, TokenCategory.AZ),
    (
        PitchToken.PLATE_POS_X_IS_LTN2,
        PitchToken.PLATE_POS_X_IS_GT2,
        TokenCategory.PLATE_POS_X,
    ),
    (
        PitchToken.PLATE_POS_Z_IS_LTN1,
        PitchToken.PLATE_POS_Z_IS_GT5,
        TokenCategory.PLATE_POS_Z,
    ),
    (
        PitchToken.RESULT_IS_BALL,
        PitchToken.RESULT_IS_AUTOMATIC_STRIKE,
        TokenCategory.RESULT,
    ),
]


def get_category(token: PitchToken) -> TokenCategory:
    """
    Return the TokenCategory for a given PitchToken.
    """
    val = token.value
    for start, end, category in _TOKEN_RANGES:
        if start.value <= val <= end.value:
            return category
    raise ValueError(f"Unknown token: {token}")


# Pre-computed cache of tokens per category for efficiency
_CATEGORY_TOKENS: dict[TokenCategory, list[PitchToken]] = {}


def _init_category_tokens() -> None:
    """
    Initialize the category tokens cache.
    """
    for start, end, category in _TOKEN_RANGES:
        tokens = [PitchToken(v) for v in range(start.value, end.value + 1)]
        _CATEGORY_TOKENS[category] = tokens


_init_category_tokens()


def get_tokens_in_category(category: TokenCategory) -> list[PitchToken]:
    """
    Return all PitchToken values belonging to a given category.
    """
    return _CATEGORY_TOKENS[category]


# Grammar: maps each category to valid next token categories
_NEXT_CATEGORY: dict[TokenCategory, list[TokenCategory]] = {
    TokenCategory.SESSION_START: [TokenCategory.PA_START, TokenCategory.SESSION_END],
    TokenCategory.PA_START: [TokenCategory.PITCH_TYPE],
    TokenCategory.PITCH_TYPE: [TokenCategory.SPEED],
    TokenCategory.SPEED: [TokenCategory.SPIN_RATE],
    TokenCategory.SPIN_RATE: [TokenCategory.SPIN_AXIS],
    TokenCategory.SPIN_AXIS: [TokenCategory.RELEASE_POS_X],
    TokenCategory.RELEASE_POS_X: [TokenCategory.RELEASE_POS_Z],
    TokenCategory.RELEASE_POS_Z: [TokenCategory.VX0],
    TokenCategory.VX0: [TokenCategory.VY0],
    TokenCategory.VY0: [TokenCategory.VZ0],
    TokenCategory.VZ0: [TokenCategory.AX],
    TokenCategory.AX: [TokenCategory.AY],
    TokenCategory.AY: [TokenCategory.AZ],
    TokenCategory.AZ: [TokenCategory.RELEASE_EXTENSION],
    TokenCategory.RELEASE_EXTENSION: [TokenCategory.PLATE_POS_X],
    TokenCategory.PLATE_POS_X: [TokenCategory.PLATE_POS_Z],
    TokenCategory.PLATE_POS_Z: [TokenCategory.RESULT],
    TokenCategory.RESULT: [TokenCategory.PA_END, TokenCategory.PITCH_TYPE],
    TokenCategory.PA_END: [TokenCategory.PA_START, TokenCategory.SESSION_END],
    TokenCategory.SESSION_END: [TokenCategory.SESSION_START],
}


# Pre-computed cache of valid next tokens for each token
_VALID_NEXT_TOKENS: dict[PitchToken, list[PitchToken]] = {}


def _init_valid_next_tokens() -> None:
    """
    Initialize the valid next tokens cache.
    """
    for token in PitchToken:
        category = get_category(token)
        next_categories = _NEXT_CATEGORY[category]
        next_tokens: list[PitchToken] = []
        for next_cat in next_categories:
            next_tokens.extend(_CATEGORY_TOKENS[next_cat])
        _VALID_NEXT_TOKENS[token] = next_tokens


_init_valid_next_tokens()


def valid_next_tokens(token: PitchToken) -> list[PitchToken]:
    """
    Return all grammatically valid next PitchToken values given a current token.

    This implements the pitch sequence grammar where each pitch consists of
    16 tokens in a fixed order (PITCH_TYPE -> SPEED -> ... -> RESULT),
    bounded by structural tokens (SESSION_START/END, PA_START/END).

    Args:
        token: The current PitchToken in the sequence.

    Returns:
        A list of all valid next PitchToken values.
    """
    return _VALID_NEXT_TOKENS[token]


class PitchContext(BaseModel):
    pitcher_id: int
    batter_id: int
    pitcher_age: int
    pitcher_throws: Literal["L", "R"]
    batter_age: int
    batter_hits: Literal[
        "L", "R"
    ]  # if the batter is a switch hitter, they will still be on just one side of the plate
    count_balls: int
    count_strikes: int
    outs: int
    bases_state: int
    score_bat: int
    score_fld: int
    inning: int
    pitch_number: int
    number_through_order: int
    game_date: str
    fielder_2_id: int
    fielder_3_id: int
    fielder_4_id: int
    fielder_5_id: int
    fielder_6_id: int
    fielder_7_id: int
    fielder_8_id: int
    fielder_9_id: int
    batter_days_since_prev_game: int
    pitcher_days_since_prev_game: int
    strike_zone_top: float
    strike_zone_bottom: float

    def to_tensor(self) -> torch.Tensor:
        """
        Convert the pitch context to a tensor.
        """
        pitcher_is_lefty = 1 if self.pitcher_throws == "L" else -1
        batter_is_lefty = 1 if self.batter_hits == "L" else -1
        game_year = int(self.game_date.split("-")[0])

        tensor = torch.tensor(
            [
                self.pitcher_id,
                self.batter_id,
                self.pitcher_age,
                pitcher_is_lefty,
                self.batter_age,
                batter_is_lefty,
                self.count_balls,
                self.count_strikes,
                self.outs,
                self.bases_state,
                self.score_bat,
                self.score_fld,
                self.inning,
                self.pitch_number,
                self.number_through_order,
                game_year,
                self.fielder_2_id,
                self.fielder_3_id,
                self.fielder_4_id,
                self.fielder_5_id,
                self.fielder_6_id,
                self.fielder_7_id,
                self.fielder_8_id,
                self.fielder_9_id,
                self.batter_days_since_prev_game,
                self.pitcher_days_since_prev_game,
                self.strike_zone_top,
                self.strike_zone_bottom,
            ]
        )

        return tensor
