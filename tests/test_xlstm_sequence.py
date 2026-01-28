# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from pitchpredict.backend.algs.xlstm.sequence import build_history_sequence, ContextDefaults
from pitchpredict.backend.algs.xlstm.tokens import PitchToken
from pitchpredict.backend.algs.xlstm.encoding import encode_pitch_type


def _make_pitch(
    pa_id: int,
    pitch_type: str = "FF",
    result: str = "called_strike",
    count_balls: int | None = None,
    count_strikes: int | None = None,
) -> dict[str, object]:
    pitch: dict[str, object] = {
        "pa_id": pa_id,
        "pitch_type": pitch_type,
        "speed": 95.0,
        "spin_rate": 2200.0,
        "spin_axis": 120.0,
        "release_pos_x": -2.0,
        "release_pos_z": 6.0,
        "release_extension": 6.5,
        "vx0": -7.0,
        "vy0": -135.0,
        "vz0": 2.0,
        "ax": -12.0,
        "ay": 22.0,
        "az": -33.0,
        "plate_pos_x": 0.5,
        "plate_pos_z": 2.0,
        "result": result,
    }
    if count_balls is not None:
        pitch["count_balls"] = count_balls
    if count_strikes is not None:
        pitch["count_strikes"] = count_strikes
    return pitch


def test_build_history_sequence_preserves_order_and_adds_session_end() -> None:
    defaults = ContextDefaults(pitcher_id=1, batter_id=2)
    first = _make_pitch(pa_id=2, pitch_type="FF", result="called_strike")
    second = _make_pitch(pa_id=1, pitch_type="CH", result="ball")

    result = build_history_sequence([first, second], defaults)
    tokens = result.tokens

    assert tokens[0] == PitchToken.SESSION_START.value
    assert tokens[-1] == PitchToken.SESSION_END.value
    assert tokens.count(PitchToken.PA_START.value) == 2
    assert tokens.count(PitchToken.PA_END.value) == 2

    first_pa_start = tokens.index(PitchToken.PA_START.value)
    second_pa_start = tokens.index(PitchToken.PA_START.value, first_pa_start + 1)
    assert tokens[first_pa_start + 1] == encode_pitch_type(first["pitch_type"]).value
    assert tokens[second_pa_start + 1] == encode_pitch_type(second["pitch_type"]).value


def test_build_history_sequence_uses_count_overrides_for_next_pitch() -> None:
    defaults = ContextDefaults(pitcher_id=1, batter_id=2)
    first = _make_pitch(
        pa_id=1,
        result="called_strike",
        count_balls=2,
        count_strikes=1,
    )
    second = _make_pitch(pa_id=1, result="ball")

    seq = build_history_sequence([first, second], defaults)
    ctx = next(ctx for ctx in seq.contexts if int(ctx.pitch_number) == 2)

    assert ctx.count_balls == 2
    assert ctx.count_strikes == 2


def test_build_history_sequence_rejects_non_contiguous_pa_id() -> None:
    defaults = ContextDefaults(pitcher_id=1, batter_id=2)
    pitches = [
        _make_pitch(pa_id=1, result="called_strike"),
        _make_pitch(pa_id=2, result="ball"),
        _make_pitch(pa_id=1, result="foul"),
    ]

    with pytest.raises(ValueError):
        build_history_sequence(pitches, defaults)
