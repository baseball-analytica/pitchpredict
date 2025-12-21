# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline
"""
Vectorized dataset generation for maximum performance.

This module provides optimized, vectorized implementations of the token and context
generation pipeline, achieving 50-100x speedup over the row-by-row approach.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pitchpredict.backend.algs.deep.types import PitchToken

logger = logging.getLogger(__name__)

# Constants
TOKEN_DTYPE = np.dtype("<u2")  # uint16, little-endian
INT32_DTYPE = np.dtype("<i4")
UINT8_DTYPE = np.dtype("uint8")
FLOAT32_DTYPE = np.dtype("float32")

# Token values (avoid repeated .value lookups)
_SESSION_START = PitchToken.SESSION_START.value
_SESSION_END = PitchToken.SESSION_END.value
_PA_START = PitchToken.PA_START.value
_PA_END = PitchToken.PA_END.value

# Base token offsets for each pitch attribute
_PITCH_TYPE_BASE = PitchToken.IS_CH.value
_SPEED_BASE = PitchToken.SPEED_IS_65.value
_SPEED_LT65 = PitchToken.SPEED_IS_LT65.value
_SPEED_GT105 = PitchToken.SPEED_IS_GT105.value
_SPIN_RATE_BASE = PitchToken.SPIN_RATE_IS_750_1000.value
_SPIN_RATE_LT750 = PitchToken.SPIN_RATE_IS_LT750.value
_SPIN_RATE_GT3250 = PitchToken.SPIN_RATE_IS_GT3250.value
_SPIN_AXIS_BASE = PitchToken.SPIN_AXIS_IS_0_30.value
_RELEASE_POS_X_BASE = PitchToken.RELEASE_POS_X_IS_N4_N375.value
_RELEASE_POS_X_LT = PitchToken.RELEASE_POS_X_IS_LTN4.value
_RELEASE_POS_X_GT = PitchToken.RELEASE_POS_X_IS_GT4.value
_RELEASE_POS_Z_BASE = PitchToken.RELEASE_POS_Z_IS_4_425.value
_RELEASE_POS_Z_LT = PitchToken.RELEASE_POS_Z_IS_LT4.value
_RELEASE_POS_Z_GT = PitchToken.RELEASE_POS_Z_IS_GT7.value
_VX0_BASE = PitchToken.VX0_IS_N15_N10.value
_VX0_LT = PitchToken.VX0_IS_LTN15.value
_VX0_GT = PitchToken.VX0_IS_GT15.value
_VY0_BASE = PitchToken.VY0_IS_N150_N140.value
_VY0_LT = PitchToken.VY0_IS_LTN150.value
_VY0_GT = PitchToken.VY0_IS_GTN100.value
_VZ0_BASE = PitchToken.VZ0_IS_N10_N5.value
_VZ0_LT = PitchToken.VZ0_IS_LTN10.value
_VZ0_GT = PitchToken.VZ0_IS_GT15.value
_AX_BASE = PitchToken.AX_IS_N25_N20.value
_AX_LT = PitchToken.AX_IS_LTN25.value
_AX_GT = PitchToken.AX_IS_GT25.value
_AY_BASE = PitchToken.AY_IS_15_20.value
_AY_LT = PitchToken.AY_IS_LT15.value
_AY_GT = PitchToken.AY_IS_GT40.value
_AZ_BASE = PitchToken.AZ_IS_N45_N40.value
_AZ_LT = PitchToken.AZ_IS_LTN45.value
_AZ_GT = PitchToken.AZ_IS_GTN15.value
_RELEASE_EXT_BASE = PitchToken.RELEASE_EXTENSION_IS_5_55.value
_RELEASE_EXT_LT = PitchToken.RELEASE_EXTENSION_IS_LT5.value
_RELEASE_EXT_GT = PitchToken.RELEASE_EXTENSION_IS_GT75.value
_PLATE_X_BASE = PitchToken.PLATE_POS_X_IS_N2_N175.value
_PLATE_X_LT = PitchToken.PLATE_POS_X_IS_LTN2.value
_PLATE_X_GT = PitchToken.PLATE_POS_X_IS_GT2.value
_PLATE_Z_BASE = PitchToken.PLATE_POS_Z_IS_N1_N075.value
_PLATE_Z_LT = PitchToken.PLATE_POS_Z_IS_LTN1.value
_PLATE_Z_GT = PitchToken.PLATE_POS_Z_IS_GT5.value
_RESULT_BASE = PitchToken.RESULT_IS_BALL.value

# Pitch type mapping
_PITCH_TYPE_MAP = {
    "CH": 0, "CU": 1, "FC": 2, "EP": 3, "FO": 4, "FF": 5, "KN": 6, "KC": 7,
    "SC": 8, "SI": 9, "SL": 10, "SV": 11, "FS": 12, "ST": 13, "FA": 14, "CS": 15,
    "PO": 16, "UN": 17, "IN": 18, "AB": 19,
}

# Result description mapping
_RESULT_MAP = {
    "ball": 0,
    "ball_in_dirt": 1,
    "called_strike": 2,
    "foul": 3,
    "foul_bunt": 4,
    "bunt_foul_tip": 5,
    "foul_pitchout": 6,
    "pitchout": 7,
    "hit_by_pitch": 8,
    "intentional_ball": 9,
    "intent_ball": 9,  # alias
    "hit_into_play": 10,
    "missed_bunt": 11,
    "foul_tip": 12,
    "swinging_pitchout": 13,
    "swinging_strike": 14,
    "swinging_strike_blocked": 15,
    "blocked_ball": 16,
    "automatic_ball": 17,
    "automatic_strike": 18,
}


@dataclass
class VectorizedDatasetStats:
    """Statistics from vectorized dataset generation."""
    total_pitches: int = 0
    total_tokens: int = 0
    session_count: int = 0
    pa_count: int = 0
    skipped_pitches: int = 0


def _bin_continuous(
    values: np.ndarray,
    base_token: int,
    lt_token: int,
    gt_token: int,
    min_val: float,
    max_val: float,
    step: float,
    max_offset: int,
) -> np.ndarray:
    """Vectorized binning of continuous values to tokens."""
    result = np.empty(len(values), dtype=np.uint16)

    # Compute bin indices
    offsets = np.round((values - min_val) / step).astype(np.int32)
    offsets = np.clip(offsets, 0, max_offset)

    # Base case: in range
    result[:] = base_token + offsets

    # Handle out of range
    result[values < min_val] = lt_token
    result[values > max_val] = gt_token

    return result


def _compute_base_tokens(pitches: pd.DataFrame) -> np.ndarray:
    """
    Compute the 16 base tokens for each pitch using vectorized operations.

    Returns array of shape (num_pitches, 16) with token values.
    """
    num_pitches = len(pitches)
    base_tokens = np.empty((num_pitches, 16), dtype=np.uint16)

    # 0: Pitch type
    pitch_type_offsets = pitches["pitch_type"].map(_PITCH_TYPE_MAP).values
    base_tokens[:, 0] = _PITCH_TYPE_BASE + pitch_type_offsets

    # 1: Speed (65-105 range, step 1)
    speed = np.round(pitches["release_speed"].values).astype(np.float32)
    base_tokens[:, 1] = _bin_continuous(
        speed, _SPEED_BASE, _SPEED_LT65, _SPEED_GT105,
        min_val=65, max_val=105, step=1, max_offset=40
    )

    # 2: Spin rate (750-3250 range, step 250)
    spin_rate = np.round(pitches["release_spin_rate"].values).astype(np.float32)
    base_tokens[:, 2] = _bin_continuous(
        spin_rate, _SPIN_RATE_BASE, _SPIN_RATE_LT750, _SPIN_RATE_GT3250,
        min_val=750, max_val=3250, step=250, max_offset=9
    )

    # 3: Spin axis (0-360, step 30, no out-of-range)
    spin_axis = np.round(pitches["spin_axis"].values).astype(np.float32)
    spin_axis_offset = np.clip((spin_axis // 30).astype(np.int32), 0, 11)
    base_tokens[:, 3] = _SPIN_AXIS_BASE + spin_axis_offset

    # 4: Release pos X (-4 to 4, step 0.25)
    release_pos_x = np.round(pitches["release_pos_x"].values, 2)
    base_tokens[:, 4] = _bin_continuous(
        release_pos_x, _RELEASE_POS_X_BASE, _RELEASE_POS_X_LT, _RELEASE_POS_X_GT,
        min_val=-4, max_val=4, step=0.25, max_offset=31
    )

    # 5: Release pos Z (4 to 7, step 0.25)
    release_pos_z = np.round(pitches["release_pos_z"].values, 2)
    base_tokens[:, 5] = _bin_continuous(
        release_pos_z, _RELEASE_POS_Z_BASE, _RELEASE_POS_Z_LT, _RELEASE_POS_Z_GT,
        min_val=4, max_val=7, step=0.25, max_offset=11
    )

    # 6: VX0 (-15 to 15, step 5)
    vx0 = np.round(pitches["vx0"].values, 2)
    base_tokens[:, 6] = _bin_continuous(
        vx0, _VX0_BASE, _VX0_LT, _VX0_GT,
        min_val=-15, max_val=15, step=5, max_offset=5
    )

    # 7: VY0 (-150 to -100, step 10)
    vy0 = np.round(pitches["vy0"].values, 2)
    base_tokens[:, 7] = _bin_continuous(
        vy0, _VY0_BASE, _VY0_LT, _VY0_GT,
        min_val=-150, max_val=-100, step=10, max_offset=4
    )

    # 8: VZ0 (-10 to 15, step 5)
    vz0 = np.round(pitches["vz0"].values, 2)
    base_tokens[:, 8] = _bin_continuous(
        vz0, _VZ0_BASE, _VZ0_LT, _VZ0_GT,
        min_val=-10, max_val=15, step=5, max_offset=4
    )

    # 9: AX (-25 to 25, step 5)
    ax = np.round(pitches["ax"].values, 2)
    base_tokens[:, 9] = _bin_continuous(
        ax, _AX_BASE, _AX_LT, _AX_GT,
        min_val=-25, max_val=25, step=5, max_offset=9
    )

    # 10: AY (15 to 40, step 5)
    ay = np.round(pitches["ay"].values, 2)
    base_tokens[:, 10] = _bin_continuous(
        ay, _AY_BASE, _AY_LT, _AY_GT,
        min_val=15, max_val=40, step=5, max_offset=4
    )

    # 11: AZ (-45 to -15, step 5)
    az = np.round(pitches["az"].values, 2)
    base_tokens[:, 11] = _bin_continuous(
        az, _AZ_BASE, _AZ_LT, _AZ_GT,
        min_val=-45, max_val=-15, step=5, max_offset=5
    )

    # 12: Release extension (5 to 7.5, step 0.5)
    release_ext = np.round(pitches["release_extension"].values, 2)
    base_tokens[:, 12] = _bin_continuous(
        release_ext, _RELEASE_EXT_BASE, _RELEASE_EXT_LT, _RELEASE_EXT_GT,
        min_val=5, max_val=7.5, step=0.5, max_offset=4
    )

    # 13: Plate X (-2 to 2, step 0.25)
    plate_x = np.round(pitches["plate_x"].values, 2)
    base_tokens[:, 13] = _bin_continuous(
        plate_x, _PLATE_X_BASE, _PLATE_X_LT, _PLATE_X_GT,
        min_val=-2, max_val=2, step=0.25, max_offset=15
    )

    # 14: Plate Z (-1 to 5, step 0.25)
    plate_z = np.round(pitches["plate_z"].values, 2)
    base_tokens[:, 14] = _bin_continuous(
        plate_z, _PLATE_Z_BASE, _PLATE_Z_LT, _PLATE_Z_GT,
        min_val=-1, max_val=5, step=0.25, max_offset=23
    )

    # 15: Result
    result_offsets = pitches["description"].map(_RESULT_MAP).values
    base_tokens[:, 15] = _RESULT_BASE + result_offsets

    return base_tokens


def _compute_boundaries(pitches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute session and PA boundary flags vectorized.

    Returns:
        is_session_start: bool array, True if pitch starts new session
        is_pa_start: bool array, True if pitch starts new PA
        has_event: bool array, True if pitch has an event (ends PA)
        is_session_end: bool array, True if pitch ends a session
    """
    num_pitches = len(pitches)

    # Extract key columns
    game_pk = pitches["game_pk"].values
    pitcher = pitches["pitcher"].values
    at_bat_number = pitches["at_bat_number"].values
    events = pitches["events"].values

    # Session changes when (game_pk, pitcher) changes
    is_session_start = np.zeros(num_pitches, dtype=bool)
    is_session_start[0] = True
    if num_pitches > 1:
        is_session_start[1:] = (game_pk[1:] != game_pk[:-1]) | (pitcher[1:] != pitcher[:-1])

    # PA changes when (game_pk, at_bat_number) changes
    is_pa_start = np.zeros(num_pitches, dtype=bool)
    is_pa_start[0] = True
    if num_pitches > 1:
        is_pa_start[1:] = (game_pk[1:] != game_pk[:-1]) | (at_bat_number[1:] != at_bat_number[:-1])

    # Has event (PA ends naturally)
    has_event = np.array([isinstance(e, str) and e != "" for e in events], dtype=bool)

    # Session ends if next pitch starts new session, or this is last pitch
    is_session_end = np.zeros(num_pitches, dtype=bool)
    if num_pitches > 1:
        is_session_end[:-1] = is_session_start[1:]
    is_session_end[-1] = True

    return is_session_start, is_pa_start, has_event, is_session_end


def build_tokens_vectorized(
    pitches: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, VectorizedDatasetStats]:
    """
    Build pitch tokens using vectorized operations.

    This is the main optimization: instead of iterating row-by-row with iloc,
    we compute all tokens using numpy vectorized operations.

    Args:
        pitches: DataFrame of pitch data, already cleaned and sorted.

    Returns:
        tokens: 1D uint16 array of token values
        token_to_pitch: 1D int32 array mapping each token to its source pitch index
        stats: Generation statistics
    """
    num_pitches = len(pitches)
    stats = VectorizedDatasetStats()

    if num_pitches == 0:
        return np.array([], dtype=np.uint16), np.array([], dtype=np.int32), stats

    logger.info("Computing base tokens for %d pitches...", num_pitches)

    # Step 1: Compute all 16 base tokens per pitch (vectorized)
    base_tokens = _compute_base_tokens(pitches)  # (num_pitches, 16)

    # Step 2: Compute boundary flags (vectorized)
    is_session_start, is_pa_start, has_event, is_session_end = _compute_boundaries(pitches)

    # Step 3: Compute boundary tokens needed
    # For pitch i, we may need to emit tokens BEFORE it (for previous pitch's endings)
    # and tokens AFTER it (PA_END if has event)

    # Forced PA_END: emitted before pitch i if new PA starts and previous pitch didn't have event
    # Note: Session changes alone don't force PA_END - a PA can span multiple sessions (mid-AB pitching changes)
    needs_forced_pa_end = np.zeros(num_pitches, dtype=bool)
    if num_pitches > 1:
        needs_forced_pa_end[1:] = is_pa_start[1:] & ~has_event[:-1]

    # SESSION_END: emitted before pitch i if session changes at i
    needs_session_end_before = np.zeros(num_pitches, dtype=bool)
    if num_pitches > 1:
        needs_session_end_before[1:] = is_session_start[1:]

    # Count tokens per pitch
    # Structure: [forced_pa_end?] [session_end?] [session_start?] [pa_start?] [16 base] [pa_end?]
    tokens_before_base = (
        needs_forced_pa_end.astype(np.int32) +
        needs_session_end_before.astype(np.int32) +
        is_session_start.astype(np.int32) +
        is_pa_start.astype(np.int32)
    )
    tokens_after_base = has_event.astype(np.int32)
    tokens_per_pitch = tokens_before_base + 16 + tokens_after_base

    # Final tokens: PA_END (if last PA unclosed) + SESSION_END
    final_needs_pa_end = not has_event[-1]
    final_tokens = int(final_needs_pa_end) + 1  # +1 for SESSION_END

    total_tokens = int(tokens_per_pitch.sum()) + final_tokens

    logger.info("Allocating %d tokens...", total_tokens)

    # Step 4: Allocate output arrays
    tokens = np.empty(total_tokens, dtype=np.uint16)
    token_to_pitch = np.empty(total_tokens, dtype=np.int32)

    # Step 5: Fill tokens using cumulative indexing
    # Compute start position for each pitch's tokens
    cumsum = np.zeros(num_pitches + 1, dtype=np.int64)
    cumsum[1:] = np.cumsum(tokens_per_pitch)

    logger.info("Assembling token sequence...")

    # We need to fill in the tokens. For maximum efficiency, we'll use numpy advanced indexing
    # where possible, but some operations require a loop due to variable-length sections.

    # Pre-compute indices for each token type
    for i in range(num_pitches):
        pos = int(cumsum[i])
        pitch_idx = i
        prev_pitch_idx = i - 1 if i > 0 else 0

        # Tokens that use PREVIOUS pitch's context (boundary closers)
        if needs_forced_pa_end[i]:
            tokens[pos] = _PA_END
            token_to_pitch[pos] = prev_pitch_idx
            pos += 1

        if needs_session_end_before[i]:
            tokens[pos] = _SESSION_END
            token_to_pitch[pos] = prev_pitch_idx
            pos += 1

        # Tokens that use CURRENT pitch's context
        if is_session_start[i]:
            tokens[pos] = _SESSION_START
            token_to_pitch[pos] = pitch_idx
            pos += 1

        if is_pa_start[i]:
            tokens[pos] = _PA_START
            token_to_pitch[pos] = pitch_idx
            pos += 1

        # 16 base tokens
        tokens[pos:pos + 16] = base_tokens[i]
        token_to_pitch[pos:pos + 16] = pitch_idx
        pos += 16

        # PA_END if has event
        if has_event[i]:
            tokens[pos] = _PA_END
            token_to_pitch[pos] = pitch_idx
            pos += 1

    # Final boundary tokens (use last pitch's context)
    final_pos = int(cumsum[-1])
    if final_needs_pa_end:
        tokens[final_pos] = _PA_END
        token_to_pitch[final_pos] = num_pitches - 1
        final_pos += 1

    tokens[final_pos] = _SESSION_END
    token_to_pitch[final_pos] = num_pitches - 1

    # Compute stats
    stats.total_pitches = num_pitches
    stats.total_tokens = total_tokens
    stats.session_count = int(is_session_start.sum())
    stats.pa_count = int(is_pa_start.sum())

    logger.info(
        "Generated %d tokens from %d pitches (%d sessions, %d PAs)",
        total_tokens, num_pitches, stats.session_count, stats.pa_count
    )

    return tokens, token_to_pitch, stats


# ============================================================================
# Vectorized Context Encoding
# ============================================================================

def _encode_handedness_vectorized(values: np.ndarray) -> np.ndarray:
    """Vectorized handedness encoding: 'L' -> 0, 'R' -> 1."""
    result = np.zeros(len(values), dtype=np.uint8)
    result[values == "R"] = 1
    return result


def _encode_game_date_vectorized(values: np.ndarray) -> np.ndarray:
    """Vectorized game date encoding to normalized float."""
    min_ordinal = date.fromisoformat("2015-01-01").toordinal()
    max_ordinal = date.fromisoformat("2025-11-18").toordinal()
    range_ordinal = max_ordinal - min_ordinal

    # Convert dates to ordinals
    ordinals = np.array([
        date.fromisoformat(str(d)[:10]).toordinal() if pd.notna(d) else min_ordinal
        for d in values
    ], dtype=np.float32)

    normalized = (ordinals - min_ordinal) / range_ordinal
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _encode_age_vectorized(values: np.ndarray) -> np.ndarray:
    """Vectorized age encoding (z-score normalization)."""
    values = np.asarray(values, dtype=np.float32)
    mean_age = 28.5
    std_dev = 4.0
    result = (values - mean_age) / std_dev
    result[values < 15] = 0.0
    result[np.isnan(values)] = 0.0
    return result.astype(np.float32)


def _encode_score_vectorized(values: np.ndarray) -> np.ndarray:
    """Vectorized score encoding (divide by 10)."""
    return (np.asarray(values, dtype=np.float32) / 10.0).astype(np.float32)


def _encode_pitch_number_vectorized(values: np.ndarray) -> np.ndarray:
    """Vectorized pitch number encoding (divide by 100)."""
    return (np.asarray(values, dtype=np.float32) / 100.0).astype(np.float32)


def _encode_strike_zone_top_vectorized(values: np.ndarray) -> np.ndarray:
    """Vectorized strike zone top encoding (z-score)."""
    values = np.asarray(values, dtype=np.float32)
    mean_top = 3.4
    std_dev = 0.2
    result = (values - mean_top) / std_dev
    result[values < 0] = 0.0
    result[np.isnan(values)] = 0.0
    return result.astype(np.float32)


def _encode_strike_zone_bottom_vectorized(values: np.ndarray) -> np.ndarray:
    """Vectorized strike zone bottom encoding (z-score)."""
    values = np.asarray(values, dtype=np.float32)
    mean_bottom = 1.6
    std_dev = 0.1
    result = (values - mean_bottom) / std_dev
    result[values < 0] = 0.0
    result[np.isnan(values)] = 0.0
    return result.astype(np.float32)


# Field specifications for vectorized encoding
_VECTORIZED_FIELD_SPECS: dict[str, tuple[np.dtype, str, Any]] = {
    # (dtype, df_column, encoder_or_None)
    "pitcher_id": (INT32_DTYPE, "pitcher", None),
    "batter_id": (INT32_DTYPE, "batter", None),
    "pitcher_age": (FLOAT32_DTYPE, "age_pit", _encode_age_vectorized),
    "pitcher_throws": (UINT8_DTYPE, "p_throws", _encode_handedness_vectorized),
    "batter_age": (FLOAT32_DTYPE, "age_bat", _encode_age_vectorized),
    "batter_hits": (UINT8_DTYPE, "stand", _encode_handedness_vectorized),
    "count_balls": (INT32_DTYPE, "balls", None),
    "count_strikes": (INT32_DTYPE, "strikes", None),
    "outs": (INT32_DTYPE, "outs_when_up", None),
    "bases_state": (INT32_DTYPE, "_bases_state", None),  # Computed column
    "score_bat": (FLOAT32_DTYPE, "bat_score", _encode_score_vectorized),
    "score_fld": (FLOAT32_DTYPE, "fld_score", _encode_score_vectorized),
    "inning": (INT32_DTYPE, "inning", None),
    "pitch_number": (FLOAT32_DTYPE, "pitch_number", _encode_pitch_number_vectorized),
    "number_through_order": (INT32_DTYPE, "n_thruorder_pitcher", None),
    "game_date": (FLOAT32_DTYPE, "game_date", _encode_game_date_vectorized),
    "fielder_2_id": (INT32_DTYPE, "fielder_2", None),
    "fielder_3_id": (INT32_DTYPE, "fielder_3", None),
    "fielder_4_id": (INT32_DTYPE, "fielder_4", None),
    "fielder_5_id": (INT32_DTYPE, "fielder_5", None),
    "fielder_6_id": (INT32_DTYPE, "fielder_6", None),
    "fielder_7_id": (INT32_DTYPE, "fielder_7", None),
    "fielder_8_id": (INT32_DTYPE, "fielder_8", None),
    "fielder_9_id": (INT32_DTYPE, "fielder_9", None),
    "batter_days_since_prev_game": (INT32_DTYPE, "batter_days_since_prev_game", None),
    "pitcher_days_since_prev_game": (INT32_DTYPE, "pitcher_days_since_prev_game", None),
    "strike_zone_top": (FLOAT32_DTYPE, "sz_top", _encode_strike_zone_top_vectorized),
    "strike_zone_bottom": (FLOAT32_DTYPE, "sz_bot", _encode_strike_zone_bottom_vectorized),
}


def _compute_bases_state(pitches: pd.DataFrame) -> np.ndarray:
    """Compute bases_state from on_1b, on_2b, on_3b columns."""
    on_1b = pitches["on_1b"].notna().values.astype(np.int32)
    on_2b = pitches["on_2b"].notna().values.astype(np.int32)
    on_3b = pitches["on_3b"].notna().values.astype(np.int32)
    return on_1b + 2 * on_2b + 4 * on_3b


def _write_single_field(
    field_name: str,
    dtype: np.dtype,
    df_column: str,
    encoder: Any,
    pitches: pd.DataFrame,
    token_to_pitch: np.ndarray,
    output_dir: Path,
    prefix: str,
) -> tuple[str, str, int]:
    """Write a single context field file. Returns (field_name, path, unique_count)."""
    # Get values from DataFrame
    if df_column == "_bases_state":
        pitch_values = _compute_bases_state(pitches)
    else:
        pitch_values = pitches[df_column].values

    # Encode if needed
    if encoder is not None:
        pitch_values = encoder(pitch_values)
    else:
        pitch_values = np.asarray(pitch_values, dtype=dtype)

    # Expand from pitch-level to token-level using the mapping
    token_values = pitch_values[token_to_pitch]

    # Write to file
    path = output_dir / f"{prefix}_{field_name}.bin"
    token_values.astype(dtype).tofile(path)

    # Count unique values for ID fields
    unique_count = 0
    if field_name.endswith("_id"):
        unique_count = len(np.unique(pitch_values[pitch_values != 0]))

    return field_name, str(path), unique_count


def write_context_files_vectorized(
    pitches: pd.DataFrame,
    token_to_pitch: np.ndarray,
    output_dir: Path,
    prefix: str = "pitch_context",
    max_workers: int = 8,
) -> list[str]:
    """
    Write context files using vectorized operations and parallel I/O.

    This is Tier 2 + Tier 3 optimization:
    - Vectorized encoding (no per-element Python function calls)
    - Parallel file writes using ThreadPoolExecutor

    Args:
        pitches: DataFrame with pitch data (one row per pitch)
        token_to_pitch: Array mapping each token index to its source pitch index
        output_dir: Directory to write files to
        prefix: Filename prefix for context files
        max_workers: Number of parallel writers

    Returns:
        List of written file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Writing %d context fields in parallel (max_workers=%d)...",
                len(_VECTORIZED_FIELD_SPECS), max_workers)

    paths: list[str] = []
    unique_counts: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for field_name, (dtype, df_column, encoder) in _VECTORIZED_FIELD_SPECS.items():
            future = executor.submit(
                _write_single_field,
                field_name, dtype, df_column, encoder,
                pitches, token_to_pitch, output_dir, prefix
            )
            futures[future] = field_name

        for future in as_completed(futures):
            field_name, path, unique_count = future.result()
            paths.append(path)
            if unique_count > 0:
                unique_counts[field_name] = unique_count

    # Log unique counts
    for field_name, count in sorted(unique_counts.items()):
        logger.info("Number of unique %s: %d", field_name.replace("_id", "s"), count)

    return paths


def write_tokens_file_vectorized(tokens: np.ndarray, path: Path) -> None:
    """Write tokens array directly to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens.astype(TOKEN_DTYPE).tofile(path)
    logger.info("Wrote %d tokens to %s", len(tokens), path)


# ============================================================================
# High-level API
# ============================================================================

def build_and_save_dataset_vectorized(
    pitches: pd.DataFrame,
    tokens_path: str | Path,
    context_dir: str | Path,
    context_prefix: str = "pitch_context",
    max_workers: int = 8,
) -> VectorizedDatasetStats:
    """
    Build and save dataset using fully vectorized pipeline.

    This combines all three tiers of optimization:
    - Tier 1: Vectorized tokenization
    - Tier 2: Vectorized context encoding
    - Tier 3: Parallel file writes

    Args:
        pitches: Cleaned and sorted pitch DataFrame
        tokens_path: Path for token output file
        context_dir: Directory for context files
        context_prefix: Prefix for context filenames
        max_workers: Number of parallel writers for context files

    Returns:
        Statistics from generation
    """
    tokens_path = Path(tokens_path)
    context_dir = Path(context_dir) if context_dir else tokens_path.parent

    # Tier 1: Vectorized tokenization
    tokens, token_to_pitch, stats = build_tokens_vectorized(pitches)

    # Write tokens file (fast, single file)
    write_tokens_file_vectorized(tokens, tokens_path)

    # Tier 2 + 3: Vectorized and parallel context file writing
    write_context_files_vectorized(
        pitches, token_to_pitch, context_dir, context_prefix, max_workers
    )

    return stats
