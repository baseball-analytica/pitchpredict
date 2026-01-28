# SPDX-License-Identifier: MIT
"""Encode API Pitch objects to xLSTM token sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pitchpredict.backend.algs.xlstm.tokens import PitchToken, TOKENS_PER_PITCH


# Pitch type mapping
PITCH_TYPE_TO_TOKEN: dict[str, PitchToken] = {
    "CH": PitchToken.IS_CH,
    "CU": PitchToken.IS_CU,
    "FC": PitchToken.IS_FC,
    "EP": PitchToken.IS_EP,
    "FO": PitchToken.IS_FO,
    "FF": PitchToken.IS_FF,
    "KN": PitchToken.IS_KN,
    "KC": PitchToken.IS_KC,
    "SC": PitchToken.IS_SC,
    "SI": PitchToken.IS_SI,
    "SL": PitchToken.IS_SL,
    "SV": PitchToken.IS_SV,
    "FS": PitchToken.IS_FS,
    "ST": PitchToken.IS_ST,
    "FA": PitchToken.IS_FA,
    "CS": PitchToken.IS_CS,
    "PO": PitchToken.IS_PO,
    "UN": PitchToken.IS_UN,
    "IN": PitchToken.IS_IN,
    "AB": PitchToken.IS_AB,
}

# Result mapping (description string -> token)
RESULT_TO_TOKEN: dict[str, PitchToken] = {
    "ball": PitchToken.RESULT_IS_BALL,
    "ball_in_dirt": PitchToken.RESULT_IS_BALL_IN_DIRT,
    "called_strike": PitchToken.RESULT_IS_CALLED_STRIKE,
    "foul": PitchToken.RESULT_IS_FOUL,
    "foul_bunt": PitchToken.RESULT_IS_FOUL_BUNT,
    "bunt_foul_tip": PitchToken.RESULT_IS_FOUL_TIP_BUNT,
    "foul_pitchout": PitchToken.RESULT_IS_FOUL_PITCHOUT,
    "pitchout": PitchToken.RESULT_IS_PITCHOUT,
    "hit_by_pitch": PitchToken.RESULT_IS_HIT_BY_PITCH,
    "intentional_ball": PitchToken.RESULT_IS_INTENTIONAL_BALL,
    "intent_ball": PitchToken.RESULT_IS_INTENTIONAL_BALL,  # alias
    "hit_into_play": PitchToken.RESULT_IS_IN_PLAY,
    "missed_bunt": PitchToken.RESULT_IS_MISSED_BUNT,
    "foul_tip": PitchToken.RESULT_IS_FOUL_TIP,
    "swinging_pitchout": PitchToken.RESULT_IS_SWINGING_PITCHOUT,
    "swinging_strike": PitchToken.RESULT_IS_SWINGING_STRIKE,
    "swinging_strike_blocked": PitchToken.RESULT_IS_SWINGING_STRIKE_BLOCKED,
    "blocked_ball": PitchToken.RESULT_IS_BLOCKED_BALL,
    "automatic_ball": PitchToken.RESULT_IS_AUTOMATIC_BALL,
    "automatic_strike": PitchToken.RESULT_IS_AUTOMATIC_STRIKE,
}

# Valid result strings for API validation
VALID_RESULTS = frozenset(RESULT_TO_TOKEN.keys())


@dataclass
class BinSpec:
    """Specification for a binned feature."""
    low_token: PitchToken  # Token for values below range
    first_bin_token: PitchToken  # First token in the binned range
    high_token: PitchToken  # Token for values above range
    low_threshold: float  # Values < this get low_token
    high_threshold: float  # Values > this get high_token
    bin_base: float  # Start of first bin
    bin_step: float  # Width of each bin
    max_offset: int  # Maximum bin offset (0-indexed)


# Binning specifications for each feature
BIN_SPECS: dict[str, BinSpec] = {
    "speed": BinSpec(
        low_token=PitchToken.SPEED_IS_LT65,
        first_bin_token=PitchToken.SPEED_IS_65,
        high_token=PitchToken.SPEED_IS_GT105,
        low_threshold=65.0,
        high_threshold=105.0,
        bin_base=65.0,
        bin_step=1.0,
        max_offset=40,
    ),
    "spin_rate": BinSpec(
        low_token=PitchToken.SPIN_RATE_IS_LT750,
        first_bin_token=PitchToken.SPIN_RATE_IS_750_1000,
        high_token=PitchToken.SPIN_RATE_IS_GT3250,
        low_threshold=750.0,
        high_threshold=3250.0,
        bin_base=750.0,
        bin_step=250.0,
        max_offset=9,
    ),
    "spin_axis": BinSpec(
        low_token=PitchToken.SPIN_AXIS_IS_0_30,  # No low token, use first bin
        first_bin_token=PitchToken.SPIN_AXIS_IS_0_30,
        high_token=PitchToken.SPIN_AXIS_IS_330_360,  # No high token, use last bin
        low_threshold=0.0,
        high_threshold=360.0,
        bin_base=0.0,
        bin_step=30.0,
        max_offset=11,
    ),
    "release_pos_x": BinSpec(
        low_token=PitchToken.RELEASE_POS_X_IS_LTN4,
        first_bin_token=PitchToken.RELEASE_POS_X_IS_N4_N375,
        high_token=PitchToken.RELEASE_POS_X_IS_GT4,
        low_threshold=-4.0,
        high_threshold=4.0,
        bin_base=-4.0,
        bin_step=0.25,
        max_offset=31,
    ),
    "release_pos_z": BinSpec(
        low_token=PitchToken.RELEASE_POS_Z_IS_LT4,
        first_bin_token=PitchToken.RELEASE_POS_Z_IS_4_425,
        high_token=PitchToken.RELEASE_POS_Z_IS_GT7,
        low_threshold=4.0,
        high_threshold=7.0,
        bin_base=4.0,
        bin_step=0.25,
        max_offset=11,
    ),
    "release_extension": BinSpec(
        low_token=PitchToken.RELEASE_EXTENSION_IS_LT5,
        first_bin_token=PitchToken.RELEASE_EXTENSION_IS_5_55,
        high_token=PitchToken.RELEASE_EXTENSION_IS_GT75,
        low_threshold=5.0,
        high_threshold=7.5,
        bin_base=5.0,
        bin_step=0.5,
        max_offset=4,
    ),
    "vx0": BinSpec(
        low_token=PitchToken.VX0_IS_LTN15,
        first_bin_token=PitchToken.VX0_IS_N15_N10,
        high_token=PitchToken.VX0_IS_GT15,
        low_threshold=-15.0,
        high_threshold=15.0,
        bin_base=-15.0,
        bin_step=5.0,
        max_offset=5,
    ),
    "vy0": BinSpec(
        low_token=PitchToken.VY0_IS_LTN150,
        first_bin_token=PitchToken.VY0_IS_N150_N140,
        high_token=PitchToken.VY0_IS_GTN100,
        low_threshold=-150.0,
        high_threshold=-100.0,
        bin_base=-150.0,
        bin_step=10.0,
        max_offset=4,
    ),
    "vz0": BinSpec(
        low_token=PitchToken.VZ0_IS_LTN10,
        first_bin_token=PitchToken.VZ0_IS_N10_N5,
        high_token=PitchToken.VZ0_IS_GT15,
        low_threshold=-10.0,
        high_threshold=15.0,
        bin_base=-10.0,
        bin_step=5.0,
        max_offset=4,
    ),
    "ax": BinSpec(
        low_token=PitchToken.AX_IS_LTN25,
        first_bin_token=PitchToken.AX_IS_N25_N20,
        high_token=PitchToken.AX_IS_GT25,
        low_threshold=-25.0,
        high_threshold=25.0,
        bin_base=-25.0,
        bin_step=5.0,
        max_offset=9,
    ),
    "ay": BinSpec(
        low_token=PitchToken.AY_IS_LT15,
        first_bin_token=PitchToken.AY_IS_15_20,
        high_token=PitchToken.AY_IS_GT40,
        low_threshold=15.0,
        high_threshold=40.0,
        bin_base=15.0,
        bin_step=5.0,
        max_offset=4,
    ),
    "az": BinSpec(
        low_token=PitchToken.AZ_IS_LTN45,
        first_bin_token=PitchToken.AZ_IS_N45_N40,
        high_token=PitchToken.AZ_IS_GTN15,
        low_threshold=-45.0,
        high_threshold=-15.0,
        bin_base=-45.0,
        bin_step=5.0,
        max_offset=5,
    ),
    "plate_pos_x": BinSpec(
        low_token=PitchToken.PLATE_POS_X_IS_LTN2,
        first_bin_token=PitchToken.PLATE_POS_X_IS_N2_N175,
        high_token=PitchToken.PLATE_POS_X_IS_GT2,
        low_threshold=-2.0,
        high_threshold=2.0,
        bin_base=-2.0,
        bin_step=0.25,
        max_offset=15,
    ),
    "plate_pos_z": BinSpec(
        low_token=PitchToken.PLATE_POS_Z_IS_LTN1,
        first_bin_token=PitchToken.PLATE_POS_Z_IS_N1_N075,
        high_token=PitchToken.PLATE_POS_Z_IS_GT5,
        low_threshold=-1.0,
        high_threshold=5.0,
        bin_base=-1.0,
        bin_step=0.25,
        max_offset=23,
    ),
}


def _bin_value(value: float, spec: BinSpec) -> PitchToken:
    """Bin a continuous value to a token using the given spec."""
    if value < spec.low_threshold:
        return spec.low_token
    if value > spec.high_threshold:
        return spec.high_token

    # Calculate bin index
    raw_offset = round((value - spec.bin_base) / spec.bin_step)
    offset = max(0, min(spec.max_offset, int(raw_offset)))
    return PitchToken(spec.first_bin_token.value + offset)


def encode_pitch_type(pitch_type: str) -> PitchToken:
    """Encode pitch type string to token."""
    pitch_type_upper = pitch_type.upper()
    if pitch_type_upper not in PITCH_TYPE_TO_TOKEN:
        raise ValueError(f"Unknown pitch type: {pitch_type}. Valid types: {list(PITCH_TYPE_TO_TOKEN.keys())}")
    return PITCH_TYPE_TO_TOKEN[pitch_type_upper]


def encode_result(result: str) -> PitchToken:
    """Encode result string to token."""
    result_lower = result.lower()
    if result_lower not in RESULT_TO_TOKEN:
        raise ValueError(f"Unknown result: {result}. Valid results: {list(RESULT_TO_TOKEN.keys())}")
    return RESULT_TO_TOKEN[result_lower]


def encode_speed(speed: float) -> PitchToken:
    """Encode pitch speed (mph) to token."""
    return _bin_value(round(speed), BIN_SPECS["speed"])


def encode_spin_rate(spin_rate: float) -> PitchToken:
    """Encode spin rate (rpm) to token."""
    return _bin_value(round(spin_rate), BIN_SPECS["spin_rate"])


def encode_spin_axis(spin_axis: float) -> PitchToken:
    """Encode spin axis (degrees, 0-360) to token."""
    # Clamp to valid range
    axis = spin_axis % 360
    offset = min(11, int(axis // 30))
    return PitchToken(PitchToken.SPIN_AXIS_IS_0_30.value + offset)


def encode_release_pos_x(x: float) -> PitchToken:
    """Encode release position X (feet) to token."""
    return _bin_value(round(x, 2), BIN_SPECS["release_pos_x"])


def encode_release_pos_z(z: float) -> PitchToken:
    """Encode release position Z (feet) to token."""
    return _bin_value(round(z, 2), BIN_SPECS["release_pos_z"])


def encode_release_extension(ext: float) -> PitchToken:
    """Encode release extension (feet) to token."""
    return _bin_value(round(ext, 2), BIN_SPECS["release_extension"])


def encode_vx0(vx: float) -> PitchToken:
    """Encode initial X velocity (ft/s) to token."""
    return _bin_value(round(vx, 2), BIN_SPECS["vx0"])


def encode_vy0(vy: float) -> PitchToken:
    """Encode initial Y velocity (ft/s) to token."""
    return _bin_value(round(vy, 2), BIN_SPECS["vy0"])


def encode_vz0(vz: float) -> PitchToken:
    """Encode initial Z velocity (ft/s) to token."""
    return _bin_value(round(vz, 2), BIN_SPECS["vz0"])


def encode_ax(ax: float) -> PitchToken:
    """Encode X acceleration (ft/s^2) to token."""
    return _bin_value(round(ax, 2), BIN_SPECS["ax"])


def encode_ay(ay: float) -> PitchToken:
    """Encode Y acceleration (ft/s^2) to token."""
    return _bin_value(round(ay, 2), BIN_SPECS["ay"])


def encode_az(az: float) -> PitchToken:
    """Encode Z acceleration (ft/s^2) to token."""
    return _bin_value(round(az, 2), BIN_SPECS["az"])


def encode_plate_pos_x(x: float) -> PitchToken:
    """Encode plate position X (feet) to token."""
    return _bin_value(round(x, 2), BIN_SPECS["plate_pos_x"])


def encode_plate_pos_z(z: float) -> PitchToken:
    """Encode plate position Z (feet) to token."""
    return _bin_value(round(z, 2), BIN_SPECS["plate_pos_z"])


def encode_pitch(
    pitch_type: str,
    speed: float,
    spin_rate: float,
    spin_axis: float,
    release_pos_x: float,
    release_pos_z: float,
    vx0: float,
    vy0: float,
    vz0: float,
    ax: float,
    ay: float,
    az: float,
    release_extension: float,
    plate_pos_x: float,
    plate_pos_z: float,
    result: str,
) -> list[int]:
    """Encode a complete pitch to 16 tokens.

    The tokens are ordered according to PITCH_TOKEN_ORDER:
    PITCH_TYPE -> SPEED -> SPIN_RATE -> SPIN_AXIS -> RELEASE_POS_X -> RELEASE_POS_Z ->
    VX0 -> VY0 -> VZ0 -> AX -> AY -> AZ -> RELEASE_EXTENSION -> PLATE_POS_X ->
    PLATE_POS_Z -> RESULT

    Returns:
        List of 16 token IDs (integers)
    """
    tokens = [
        encode_pitch_type(pitch_type).value,
        encode_speed(speed).value,
        encode_spin_rate(spin_rate).value,
        encode_spin_axis(spin_axis).value,
        encode_release_pos_x(release_pos_x).value,
        encode_release_pos_z(release_pos_z).value,
        encode_vx0(vx0).value,
        encode_vy0(vy0).value,
        encode_vz0(vz0).value,
        encode_ax(ax).value,
        encode_ay(ay).value,
        encode_az(az).value,
        encode_release_extension(release_extension).value,
        encode_plate_pos_x(plate_pos_x).value,
        encode_plate_pos_z(plate_pos_z).value,
        encode_result(result).value,
    ]
    assert len(tokens) == TOKENS_PER_PITCH
    return tokens


def encode_pitch_from_dict(pitch: dict[str, Any]) -> list[int]:
    """Encode a pitch from a dictionary with field names matching the API.

    Expected keys: pitch_type, speed, spin_rate, spin_axis, release_pos_x, release_pos_z,
    vx0, vy0, vz0, ax, ay, az, release_extension, plate_pos_x, plate_pos_z, result
    """
    return encode_pitch(
        pitch_type=pitch["pitch_type"],
        speed=pitch["speed"],
        spin_rate=pitch["spin_rate"],
        spin_axis=pitch["spin_axis"],
        release_pos_x=pitch["release_pos_x"],
        release_pos_z=pitch["release_pos_z"],
        vx0=pitch["vx0"],
        vy0=pitch["vy0"],
        vz0=pitch["vz0"],
        ax=pitch["ax"],
        ay=pitch["ay"],
        az=pitch["az"],
        release_extension=pitch["release_extension"],
        plate_pos_x=pitch["plate_pos_x"],
        plate_pos_z=pitch["plate_pos_z"],
        result=pitch["result"],
    )
