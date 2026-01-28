# SPDX-License-Identifier: MIT
"""Decode xLSTM tokens to pitch values and summary statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pitchpredict.backend.algs.xlstm.tokens import (
    PitchToken,
    TokenCategory,
    get_category,
    get_tokens_in_category,
    TOKENS_PER_PITCH,
)
from pitchpredict.backend.algs.xlstm.encoding import BIN_SPECS, BinSpec


# Reverse mapping: token -> pitch type string
TOKEN_TO_PITCH_TYPE: dict[PitchToken, str] = {
    PitchToken.IS_CH: "CH",
    PitchToken.IS_CU: "CU",
    PitchToken.IS_FC: "FC",
    PitchToken.IS_EP: "EP",
    PitchToken.IS_FO: "FO",
    PitchToken.IS_FF: "FF",
    PitchToken.IS_KN: "KN",
    PitchToken.IS_KC: "KC",
    PitchToken.IS_SC: "SC",
    PitchToken.IS_SI: "SI",
    PitchToken.IS_SL: "SL",
    PitchToken.IS_SV: "SV",
    PitchToken.IS_FS: "FS",
    PitchToken.IS_ST: "ST",
    PitchToken.IS_FA: "FA",
    PitchToken.IS_CS: "CS",
    PitchToken.IS_PO: "PO",
    PitchToken.IS_UN: "UN",
    PitchToken.IS_IN: "IN",
    PitchToken.IS_AB: "AB",
}

# Reverse mapping: token -> result string
TOKEN_TO_RESULT: dict[PitchToken, str] = {
    PitchToken.RESULT_IS_BALL: "ball",
    PitchToken.RESULT_IS_BALL_IN_DIRT: "ball_in_dirt",
    PitchToken.RESULT_IS_CALLED_STRIKE: "called_strike",
    PitchToken.RESULT_IS_FOUL: "foul",
    PitchToken.RESULT_IS_FOUL_BUNT: "foul_bunt",
    PitchToken.RESULT_IS_FOUL_TIP_BUNT: "bunt_foul_tip",
    PitchToken.RESULT_IS_FOUL_PITCHOUT: "foul_pitchout",
    PitchToken.RESULT_IS_PITCHOUT: "pitchout",
    PitchToken.RESULT_IS_HIT_BY_PITCH: "hit_by_pitch",
    PitchToken.RESULT_IS_INTENTIONAL_BALL: "intentional_ball",
    PitchToken.RESULT_IS_IN_PLAY: "hit_into_play",
    PitchToken.RESULT_IS_MISSED_BUNT: "missed_bunt",
    PitchToken.RESULT_IS_FOUL_TIP: "foul_tip",
    PitchToken.RESULT_IS_SWINGING_PITCHOUT: "swinging_pitchout",
    PitchToken.RESULT_IS_SWINGING_STRIKE: "swinging_strike",
    PitchToken.RESULT_IS_SWINGING_STRIKE_BLOCKED: "swinging_strike_blocked",
    PitchToken.RESULT_IS_BLOCKED_BALL: "blocked_ball",
    PitchToken.RESULT_IS_AUTOMATIC_BALL: "automatic_ball",
    PitchToken.RESULT_IS_AUTOMATIC_STRIKE: "automatic_strike",
}

# Category to BinSpec mapping for continuous features
CATEGORY_TO_BIN_SPEC: dict[TokenCategory, str] = {
    TokenCategory.SPEED: "speed",
    TokenCategory.SPIN_RATE: "spin_rate",
    TokenCategory.SPIN_AXIS: "spin_axis",
    TokenCategory.RELEASE_POS_X: "release_pos_x",
    TokenCategory.RELEASE_POS_Z: "release_pos_z",
    TokenCategory.RELEASE_EXTENSION: "release_extension",
    TokenCategory.VX0: "vx0",
    TokenCategory.VY0: "vy0",
    TokenCategory.VZ0: "vz0",
    TokenCategory.AX: "ax",
    TokenCategory.AY: "ay",
    TokenCategory.AZ: "az",
    TokenCategory.PLATE_POS_X: "plate_pos_x",
    TokenCategory.PLATE_POS_Z: "plate_pos_z",
}


def _decode_binned_value(token: PitchToken, spec: BinSpec) -> float:
    """Decode a binned token to its bin center value."""
    token_val = token.value

    # Check for low/high boundary tokens
    if token_val == spec.low_token.value:
        # Return a value slightly below the threshold
        return spec.low_threshold - spec.bin_step / 2

    if token_val == spec.high_token.value:
        # Return a value slightly above the threshold
        return spec.high_threshold + spec.bin_step / 2

    # Calculate offset from first bin token
    offset = token_val - spec.first_bin_token.value
    if offset < 0 or offset > spec.max_offset:
        raise ValueError(f"Token {token} not in expected range for spec")

    # Return bin center
    return spec.bin_base + offset * spec.bin_step + spec.bin_step / 2


def decode_pitch_type(token: PitchToken | int) -> str:
    """Decode pitch type token to string."""
    if isinstance(token, int):
        token = PitchToken(token)
    if token not in TOKEN_TO_PITCH_TYPE:
        raise ValueError(f"Token {token} is not a pitch type token")
    return TOKEN_TO_PITCH_TYPE[token]


def decode_result(token: PitchToken | int) -> str:
    """Decode result token to string."""
    if isinstance(token, int):
        token = PitchToken(token)
    if token not in TOKEN_TO_RESULT:
        raise ValueError(f"Token {token} is not a result token")
    return TOKEN_TO_RESULT[token]


def decode_speed(token: PitchToken | int) -> float:
    """Decode speed token to mph value."""
    if isinstance(token, int):
        token = PitchToken(token)
    return _decode_binned_value(token, BIN_SPECS["speed"])


def decode_spin_rate(token: PitchToken | int) -> float:
    """Decode spin rate token to rpm value."""
    if isinstance(token, int):
        token = PitchToken(token)
    return _decode_binned_value(token, BIN_SPECS["spin_rate"])


def decode_spin_axis(token: PitchToken | int) -> float:
    """Decode spin axis token to degrees (0-360)."""
    if isinstance(token, int):
        token = PitchToken(token)
    # Spin axis uses simple 30-degree bins
    offset = token.value - PitchToken.SPIN_AXIS_IS_0_30.value
    return offset * 30 + 15  # Return bin center


def decode_continuous_token(token: PitchToken | int, category: TokenCategory) -> float:
    """Decode a continuous feature token to its value."""
    if isinstance(token, int):
        token = PitchToken(token)

    if category == TokenCategory.SPIN_AXIS:
        return decode_spin_axis(token)

    if category not in CATEGORY_TO_BIN_SPEC:
        raise ValueError(f"Category {category} is not a continuous feature")

    spec_name = CATEGORY_TO_BIN_SPEC[category]
    return _decode_binned_value(token, BIN_SPECS[spec_name])


def decode_pitch(tokens: list[int]) -> dict[str, Any]:
    """Decode 16 pitch tokens to a pitch dictionary.

    Args:
        tokens: List of 16 token IDs in order:
            PITCH_TYPE, SPEED, SPIN_RATE, SPIN_AXIS, RELEASE_POS_X, RELEASE_POS_Z,
            VX0, VY0, VZ0, AX, AY, AZ, RELEASE_EXTENSION, PLATE_POS_X, PLATE_POS_Z, RESULT

    Returns:
        Dictionary with pitch fields
    """
    if len(tokens) != TOKENS_PER_PITCH:
        raise ValueError(f"Expected {TOKENS_PER_PITCH} tokens, got {len(tokens)}")

    return {
        "pitch_type": decode_pitch_type(tokens[0]),
        "speed": decode_speed(tokens[1]),
        "spin_rate": decode_spin_rate(tokens[2]),
        "spin_axis": decode_spin_axis(tokens[3]),
        "release_pos_x": decode_continuous_token(tokens[4], TokenCategory.RELEASE_POS_X),
        "release_pos_z": decode_continuous_token(tokens[5], TokenCategory.RELEASE_POS_Z),
        "vx0": decode_continuous_token(tokens[6], TokenCategory.VX0),
        "vy0": decode_continuous_token(tokens[7], TokenCategory.VY0),
        "vz0": decode_continuous_token(tokens[8], TokenCategory.VZ0),
        "ax": decode_continuous_token(tokens[9], TokenCategory.AX),
        "ay": decode_continuous_token(tokens[10], TokenCategory.AY),
        "az": decode_continuous_token(tokens[11], TokenCategory.AZ),
        "release_extension": decode_continuous_token(tokens[12], TokenCategory.RELEASE_EXTENSION),
        "plate_pos_x": decode_continuous_token(tokens[13], TokenCategory.PLATE_POS_X),
        "plate_pos_z": decode_continuous_token(tokens[14], TokenCategory.PLATE_POS_Z),
        "result": decode_result(tokens[15]),
    }


@dataclass
class PitchTypeProbabilities:
    """Pitch type probability distribution from logits."""
    probs: dict[str, float]  # pitch_type -> probability

    @classmethod
    def from_logits(cls, logits: torch.Tensor, temperature: float = 1.0) -> "PitchTypeProbabilities":
        """Compute pitch type probabilities from step-0 logits.

        Args:
            logits: Logits tensor of shape [vocab_size] or [batch, vocab_size]
            temperature: Softmax temperature (1.0 = no scaling)

        Returns:
            PitchTypeProbabilities with normalized probabilities
        """
        if logits.dim() == 2:
            logits = logits[0]  # Take first batch element

        # Get pitch type tokens
        pitch_type_tokens = get_tokens_in_category(TokenCategory.PITCH_TYPE)
        pitch_type_ids = [t.value for t in pitch_type_tokens]

        # Extract logits for pitch types only
        pitch_logits = logits[pitch_type_ids]

        # Apply temperature and softmax
        if temperature != 1.0:
            pitch_logits = pitch_logits / temperature
        probs = torch.softmax(pitch_logits, dim=-1)

        # Build probability dict
        prob_dict = {}
        for token, prob in zip(pitch_type_tokens, probs.tolist()):
            pitch_type = TOKEN_TO_PITCH_TYPE[token]
            prob_dict[pitch_type] = prob

        return cls(probs=prob_dict)


def aggregate_pitch_stats(pitches: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate statistics from multiple generated pitches.

    Args:
        pitches: List of decoded pitch dictionaries

    Returns:
        Dictionary with aggregated statistics matching the similarity response format
    """
    if not pitches:
        return {}

    # Count pitch types
    pitch_type_counts: dict[str, int] = {}
    for p in pitches:
        pt = p["pitch_type"]
        pitch_type_counts[pt] = pitch_type_counts.get(pt, 0) + 1

    total = len(pitches)
    pitch_type_probs = {pt: count / total for pt, count in pitch_type_counts.items()}

    # Aggregate continuous values
    speeds = [p["speed"] for p in pitches]
    plate_x = [p["plate_pos_x"] for p in pitches]
    plate_z = [p["plate_pos_z"] for p in pitches]

    def _stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0}
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        return {"mean": mean, "std": std}

    speed_stats = _stats(speeds)
    plate_x_stats = _stats(plate_x)
    plate_z_stats = _stats(plate_z)

    # Count results
    result_counts: dict[str, int] = {}
    for p in pitches:
        r = p["result"]
        result_counts[r] = result_counts.get(r, 0) + 1

    return {
        "pitch_type_probs": pitch_type_probs,
        "pitch_speed_mean": speed_stats["mean"],
        "pitch_speed_std": speed_stats["std"],
        "pitch_pos_x_mean": plate_x_stats["mean"],
        "pitch_pos_x_std": plate_x_stats["std"],
        "pitch_pos_z_mean": plate_z_stats["mean"],
        "pitch_pos_z_std": plate_z_stats["std"],
        "result_counts": result_counts,
        "sample_count": total,
    }
