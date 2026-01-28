# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from operator import attrgetter
from typing import Any, Callable

import numpy as np

from tools.deep.types import PitchContext

TOKEN_DTYPE = np.dtype("<u2")
INT32_DTYPE = np.dtype("<i4")
UINT8_DTYPE = np.dtype("uint8")
FLOAT32_DTYPE = np.dtype("float32")


@dataclass(frozen=True)
class _ContextFieldSpec:
    dtype: np.dtype
    getter: Callable[[PitchContext], Any]
    encode: Callable[[Any], Any]
    decode: Callable[[Any], Any]


def _encode_int(value: Any) -> int:
    return int(value)


def _decode_int(value: Any) -> int:
    return int(value)


def _encode_handedness(value: str) -> int:
    if value == "L":
        return 0
    if value == "R":
        return 1
    raise ValueError(f"expected 'L' or 'R', got {value!r}")


def _decode_handedness(value: Any) -> str:
    val = int(value)
    if val == 0:
        return "L"
    if val == 1:
        return "R"
    raise ValueError(f"invalid handedness value: {val}")


def _encode_game_date(value: str) -> float:
    raw = date.fromisoformat(value).toordinal()
    min_date = date.fromisoformat("2015-01-01").toordinal()
    max_date = date.fromisoformat("2025-11-18").toordinal()
    val = (raw - min_date) / (max_date - min_date)
    return max(0.0, min(1.0, val))


def _decode_game_date(value: Any) -> str:
    min_date = date.fromisoformat("2015-01-01").toordinal()
    max_date = date.fromisoformat("2025-11-18").toordinal()
    val = max(0.0, min(1.0, value))
    return date.fromordinal(int(val * (max_date - min_date) + min_date)).isoformat()


def _encode_age(value: int) -> float:
    if not value or value < 15:
        return 0.0

    mean_age = 28.5
    std_dev = 4.0

    return (value - mean_age) / std_dev  # z-score


def _decode_age(value: Any) -> float:
    if value == 0.0:
        return 0

    mean_age = 28.5
    std_dev = 4.0
    return value * std_dev + mean_age


def _encode_score(value: int) -> float:
    return float(value) / 10.0  # normalize to 0-1


def _decode_score(value: Any) -> float:
    return value * 10.0


def _encode_pitch_number(value: int) -> float:
    return float(value) / 100.0  # normalize to 0-1


def _decode_pitch_number(value: Any) -> float:
    return value * 100.0


def _encode_strike_zone_top(value: float) -> float:
    if not value or value < 0.0:
        return 0.0

    mean_top = 3.4
    std_dev = 0.2
    return (value - mean_top) / std_dev  # z-score


def _decode_strike_zone_top(value: Any) -> float:
    if value == 0.0:
        return 0.0

    mean_top = 3.4
    std_dev = 0.2
    return value * std_dev + mean_top


def _encode_strike_zone_bottom(value: float) -> float:
    if not value or value < 0.0:
        return 0.0

    mean_bottom = 1.6
    std_dev = 0.1
    return (value - mean_bottom) / std_dev  # z-score


def _decode_strike_zone_bottom(value: Any) -> float:
    if value == 0.0:
        return 0.0

    mean_bottom = 1.6
    std_dev = 0.1
    return value * std_dev + mean_bottom


_CONTEXT_FIELD_SPECS: dict[str, _ContextFieldSpec] = {
    "pitcher_id": _ContextFieldSpec(
        INT32_DTYPE, attrgetter("pitcher_id"), _encode_int, _decode_int
    ),
    "batter_id": _ContextFieldSpec(
        INT32_DTYPE, attrgetter("batter_id"), _encode_int, _decode_int
    ),
    "pitcher_age": _ContextFieldSpec(
        FLOAT32_DTYPE, attrgetter("pitcher_age"), _encode_age, _decode_age
    ),
    "pitcher_throws": _ContextFieldSpec(
        UINT8_DTYPE,
        attrgetter("pitcher_throws"),
        _encode_handedness,
        _decode_handedness,
    ),
    "batter_age": _ContextFieldSpec(
        FLOAT32_DTYPE, attrgetter("batter_age"), _encode_age, _decode_age
    ),
    "batter_hits": _ContextFieldSpec(
        UINT8_DTYPE, attrgetter("batter_hits"), _encode_handedness, _decode_handedness
    ),
    "count_balls": _ContextFieldSpec(
        INT32_DTYPE, attrgetter("count_balls"), _encode_int, _decode_int
    ),
    "count_strikes": _ContextFieldSpec(
        INT32_DTYPE, attrgetter("count_strikes"), _encode_int, _decode_int
    ),
    "outs": _ContextFieldSpec(
        INT32_DTYPE, attrgetter("outs"), _encode_int, _decode_int
    ),
    "bases_state": _ContextFieldSpec(
        INT32_DTYPE, attrgetter("bases_state"), _encode_int, _decode_int
    ),
    "score_bat": _ContextFieldSpec(
        FLOAT32_DTYPE, attrgetter("score_bat"), _encode_score, _decode_score
    ),
    "score_fld": _ContextFieldSpec(
        FLOAT32_DTYPE, attrgetter("score_fld"), _encode_score, _decode_score
    ),
    "inning": _ContextFieldSpec(
        INT32_DTYPE, attrgetter("inning"), _encode_int, _decode_int
    ),
    "pitch_number": _ContextFieldSpec(
        FLOAT32_DTYPE,
        attrgetter("pitch_number"),
        _encode_pitch_number,
        _decode_pitch_number,
    ),
    "number_through_order": _ContextFieldSpec(
        INT32_DTYPE, attrgetter("number_through_order"), _encode_int, _decode_int
    ),
    "game_date": _ContextFieldSpec(
        FLOAT32_DTYPE, attrgetter("game_date"), _encode_game_date, _decode_game_date
    ),
    "fielder_2_id": _ContextFieldSpec(
        INT32_DTYPE, attrgetter("fielder_2_id"), _encode_int, _decode_int
    ),
    "fielder_3_id": _ContextFieldSpec(
        INT32_DTYPE, attrgetter("fielder_3_id"), _encode_int, _decode_int
    ),
    "fielder_4_id": _ContextFieldSpec(
        INT32_DTYPE, attrgetter("fielder_4_id"), _encode_int, _decode_int
    ),
    "fielder_5_id": _ContextFieldSpec(
        INT32_DTYPE, attrgetter("fielder_5_id"), _encode_int, _decode_int
    ),
    "fielder_6_id": _ContextFieldSpec(
        INT32_DTYPE, attrgetter("fielder_6_id"), _encode_int, _decode_int
    ),
    "fielder_7_id": _ContextFieldSpec(
        INT32_DTYPE, attrgetter("fielder_7_id"), _encode_int, _decode_int
    ),
    "fielder_8_id": _ContextFieldSpec(
        INT32_DTYPE, attrgetter("fielder_8_id"), _encode_int, _decode_int
    ),
    "fielder_9_id": _ContextFieldSpec(
        INT32_DTYPE, attrgetter("fielder_9_id"), _encode_int, _decode_int
    ),
    "batter_days_since_prev_game": _ContextFieldSpec(
        INT32_DTYPE, attrgetter("batter_days_since_prev_game"), _encode_int, _decode_int
    ),
    "pitcher_days_since_prev_game": _ContextFieldSpec(
        INT32_DTYPE,
        attrgetter("pitcher_days_since_prev_game"),
        _encode_int,
        _decode_int,
    ),
    "strike_zone_top": _ContextFieldSpec(
        FLOAT32_DTYPE,
        attrgetter("strike_zone_top"),
        _encode_strike_zone_top,
        _decode_strike_zone_top,
    ),
    "strike_zone_bottom": _ContextFieldSpec(
        FLOAT32_DTYPE,
        attrgetter("strike_zone_bottom"),
        _encode_strike_zone_bottom,
        _decode_strike_zone_bottom,
    ),
}  # keep parallel with dataset.PackedPitchChunk but without x and y


def _context_field_path(prefix: str, field_name: str) -> str:
    is_directory_hint = prefix.endswith(os.sep)
    normalized = prefix.rstrip(os.sep)
    if is_directory_hint and not normalized:
        normalized = os.sep
    if not normalized:
        normalized = "pitch_context"

    if is_directory_hint:
        directory = normalized
        filename = f"{field_name}.bin"
    else:
        directory, basename = os.path.split(normalized)
        if not basename:
            basename = "pitch_context"
        stem, _ = os.path.splitext(basename)
        stem = stem or basename
        filename = f"{stem}_{field_name}.bin"
    return os.path.join(directory, filename) if directory else filename


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
