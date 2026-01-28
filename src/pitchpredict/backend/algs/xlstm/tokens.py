# SPDX-License-Identifier: MIT
"""xLSTM token vocabulary and grammar definitions."""

from enum import auto, Enum


class PitchToken(Enum):
    """Token vocabulary for pitch sequences.

    Each pitch is represented as 16 tokens in a fixed order:
    PITCH_TYPE -> SPEED -> SPIN_RATE -> SPIN_AXIS -> RELEASE_POS_X -> RELEASE_POS_Z ->
    VX0 -> VY0 -> VZ0 -> AX -> AY -> AZ -> RELEASE_EXTENSION -> PLATE_POS_X ->
    PLATE_POS_Z -> RESULT

    Structural tokens mark session and plate appearance boundaries.
    """
    PAD = 0  # padding token for sequences shorter than seq_len
    SESSION_START = auto()  # when a pitcher enters a game
    SESSION_END = auto()  # when the pitcher leaves the game
    PA_START = auto()  # start of a plate appearance
    PA_END = auto()

    # Pitch types (21 types)
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

    # Speed bins (43 bins: <65, 65-105, >105)
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

    # Spin rate bins (12 bins)
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

    # Spin axis bins (12 bins, 30-degree intervals)
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

    # Release position X bins (34 bins)
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

    # Release position Z bins (14 bins)
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

    # Release extension bins (7 bins)
    RELEASE_EXTENSION_IS_LT5 = auto()
    RELEASE_EXTENSION_IS_5_55 = auto()
    RELEASE_EXTENSION_IS_55_6 = auto()
    RELEASE_EXTENSION_IS_6_65 = auto()
    RELEASE_EXTENSION_IS_65_7 = auto()
    RELEASE_EXTENSION_IS_7_75 = auto()
    RELEASE_EXTENSION_IS_GT75 = auto()

    # Velocity X bins (8 bins)
    VX0_IS_LTN15 = auto()
    VX0_IS_N15_N10 = auto()
    VX0_IS_N10_N5 = auto()
    VX0_IS_N5_0 = auto()
    VX0_IS_0_5 = auto()
    VX0_IS_5_10 = auto()
    VX0_IS_10_15 = auto()
    VX0_IS_GT15 = auto()

    # Velocity Y bins (7 bins)
    VY0_IS_LTN150 = auto()
    VY0_IS_N150_N140 = auto()
    VY0_IS_N140_N130 = auto()
    VY0_IS_N130_N120 = auto()
    VY0_IS_N120_N110 = auto()
    VY0_IS_N110_N100 = auto()
    VY0_IS_GTN100 = auto()

    # Velocity Z bins (7 bins)
    VZ0_IS_LTN10 = auto()
    VZ0_IS_N10_N5 = auto()
    VZ0_IS_N5_0 = auto()
    VZ0_IS_0_5 = auto()
    VZ0_IS_5_10 = auto()
    VZ0_IS_10_15 = auto()
    VZ0_IS_GT15 = auto()

    # Acceleration X bins (12 bins)
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

    # Acceleration Y bins (7 bins)
    AY_IS_LT15 = auto()
    AY_IS_15_20 = auto()
    AY_IS_20_25 = auto()
    AY_IS_25_30 = auto()
    AY_IS_30_35 = auto()
    AY_IS_35_40 = auto()
    AY_IS_GT40 = auto()

    # Acceleration Z bins (8 bins)
    AZ_IS_LTN45 = auto()
    AZ_IS_N45_N40 = auto()
    AZ_IS_N40_N35 = auto()
    AZ_IS_N35_N30 = auto()
    AZ_IS_N30_N25 = auto()
    AZ_IS_N25_N20 = auto()
    AZ_IS_N20_N15 = auto()
    AZ_IS_GTN15 = auto()

    # Plate position X bins (18 bins)
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

    # Plate position Z bins (26 bins)
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

    # Result tokens (19 outcomes)
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
    """Categories for grouping PitchToken values by their semantic role."""
    PAD = auto()
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
    (PitchToken.PAD, PitchToken.PAD, TokenCategory.PAD),
    (PitchToken.SESSION_START, PitchToken.SESSION_START, TokenCategory.SESSION_START),
    (PitchToken.SESSION_END, PitchToken.SESSION_END, TokenCategory.SESSION_END),
    (PitchToken.PA_START, PitchToken.PA_START, TokenCategory.PA_START),
    (PitchToken.PA_END, PitchToken.PA_END, TokenCategory.PA_END),
    (PitchToken.IS_CH, PitchToken.IS_AB, TokenCategory.PITCH_TYPE),
    (PitchToken.SPEED_IS_LT65, PitchToken.SPEED_IS_GT105, TokenCategory.SPEED),
    (PitchToken.SPIN_RATE_IS_LT750, PitchToken.SPIN_RATE_IS_GT3250, TokenCategory.SPIN_RATE),
    (PitchToken.SPIN_AXIS_IS_0_30, PitchToken.SPIN_AXIS_IS_330_360, TokenCategory.SPIN_AXIS),
    (PitchToken.RELEASE_POS_X_IS_LTN4, PitchToken.RELEASE_POS_X_IS_GT4, TokenCategory.RELEASE_POS_X),
    (PitchToken.RELEASE_POS_Z_IS_LT4, PitchToken.RELEASE_POS_Z_IS_GT7, TokenCategory.RELEASE_POS_Z),
    (PitchToken.RELEASE_EXTENSION_IS_LT5, PitchToken.RELEASE_EXTENSION_IS_GT75, TokenCategory.RELEASE_EXTENSION),
    (PitchToken.VX0_IS_LTN15, PitchToken.VX0_IS_GT15, TokenCategory.VX0),
    (PitchToken.VY0_IS_LTN150, PitchToken.VY0_IS_GTN100, TokenCategory.VY0),
    (PitchToken.VZ0_IS_LTN10, PitchToken.VZ0_IS_GT15, TokenCategory.VZ0),
    (PitchToken.AX_IS_LTN25, PitchToken.AX_IS_GT25, TokenCategory.AX),
    (PitchToken.AY_IS_LT15, PitchToken.AY_IS_GT40, TokenCategory.AY),
    (PitchToken.AZ_IS_LTN45, PitchToken.AZ_IS_GTN15, TokenCategory.AZ),
    (PitchToken.PLATE_POS_X_IS_LTN2, PitchToken.PLATE_POS_X_IS_GT2, TokenCategory.PLATE_POS_X),
    (PitchToken.PLATE_POS_Z_IS_LTN1, PitchToken.PLATE_POS_Z_IS_GT5, TokenCategory.PLATE_POS_Z),
    (PitchToken.RESULT_IS_BALL, PitchToken.RESULT_IS_AUTOMATIC_STRIKE, TokenCategory.RESULT),
]


def get_category(token: PitchToken) -> TokenCategory:
    """Return the TokenCategory for a given PitchToken."""
    val = token.value
    for start, end, category in _TOKEN_RANGES:
        if start.value <= val <= end.value:
            return category
    raise ValueError(f"Unknown token: {token}")


# Pre-computed cache of tokens per category for efficiency
_CATEGORY_TOKENS: dict[TokenCategory, list[PitchToken]] = {}


def _init_category_tokens() -> None:
    """Initialize the category tokens cache."""
    for start, end, category in _TOKEN_RANGES:
        tokens = [PitchToken(v) for v in range(start.value, end.value + 1)]
        _CATEGORY_TOKENS[category] = tokens


_init_category_tokens()


def get_tokens_in_category(category: TokenCategory) -> list[PitchToken]:
    """Return all PitchToken values belonging to a given category."""
    return _CATEGORY_TOKENS[category]


# Grammar: maps each category to valid next token categories
_NEXT_CATEGORY: dict[TokenCategory, list[TokenCategory]] = {
    TokenCategory.PAD: [],  # PAD is not part of the grammar, no valid next tokens
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
    """Initialize the valid next tokens cache."""
    for token in PitchToken:
        category = get_category(token)
        next_categories = _NEXT_CATEGORY[category]
        next_tokens: list[PitchToken] = []
        for next_cat in next_categories:
            next_tokens.extend(_CATEGORY_TOKENS[next_cat])
        _VALID_NEXT_TOKENS[token] = next_tokens


_init_valid_next_tokens()


def valid_next_tokens(token: PitchToken) -> list[PitchToken]:
    """Return all grammatically valid next PitchToken values given a current token.

    This implements the pitch sequence grammar where each pitch consists of
    16 tokens in a fixed order (PITCH_TYPE -> SPEED -> ... -> RESULT),
    bounded by structural tokens (SESSION_START/END, PA_START/END).

    Args:
        token: The current PitchToken in the sequence.

    Returns:
        A list of all valid next PitchToken values.
    """
    return _VALID_NEXT_TOKENS[token]


def valid_next_token_ids(token_id: int) -> list[int]:
    """Return valid next token IDs given a current token ID.

    Convenience wrapper around valid_next_tokens that works with integer IDs.
    """
    token = PitchToken(token_id)
    return [t.value for t in valid_next_tokens(token)]


# The order of token categories within a pitch (16 tokens total)
PITCH_TOKEN_ORDER: list[TokenCategory] = [
    TokenCategory.PITCH_TYPE,
    TokenCategory.SPEED,
    TokenCategory.SPIN_RATE,
    TokenCategory.SPIN_AXIS,
    TokenCategory.RELEASE_POS_X,
    TokenCategory.RELEASE_POS_Z,
    TokenCategory.VX0,
    TokenCategory.VY0,
    TokenCategory.VZ0,
    TokenCategory.AX,
    TokenCategory.AY,
    TokenCategory.AZ,
    TokenCategory.RELEASE_EXTENSION,
    TokenCategory.PLATE_POS_X,
    TokenCategory.PLATE_POS_Z,
    TokenCategory.RESULT,
]

# Number of tokens per pitch
TOKENS_PER_PITCH = len(PITCH_TOKEN_ORDER)

# Vocabulary size
VOCAB_SIZE = len(PitchToken)
