# SPDX-License-Identifier: MIT

from __future__ import annotations

import pandas as pd
import pytest

from pitchpredict.backend.algs.deep.building import _build_pitch_tokens_and_contexts, _clean_pitch_rows
from pitchpredict.backend.algs.deep.types import PitchContext, PitchToken


def _make_pitch_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "pitch_type": "FF",
        "release_speed": 95.0,
        "release_spin_rate": 1800.0,
        "spin_axis": 45.0,
        "release_pos_x": -3.5,
        "release_pos_z": 5.75,
        "plate_x": 0.5,
        "plate_z": 2.0,
        "vx0": -7.0,
        "vy0": -135.0,
        "vz0": 2.0,
        "ax": -12.0,
        "ay": 22.0,
        "az": -33.0,
        "release_extension": 6.25,
        "on_1b": pd.NA,
        "on_2b": pd.NA,
        "on_3b": pd.NA,
        "events": "",
        "description": "called_strike",
        "game_date": pd.Timestamp("2024-04-01"),
        "pitcher": 111,
        "batter": 222,
        "age_pit": 30,
        "p_throws": "L",
        "age_bat": 28,
        "stand": "R",
        "balls": 1,
        "strikes": 0,
        "outs_when_up": 1,
        "bat_score": 2,
        "fld_score": 3,
        "inning": 1,
        "pitch_number": 1,
        "at_bat_number": 5,
        "n_thruorder_pitcher": 2,
        "game_pk": 1001,
        "fielder_2": 2,
        "fielder_3": 3,
        "fielder_4": 4,
        "fielder_5": 5,
        "fielder_6": 6,
        "fielder_7": 7,
        "fielder_8": 8,
        "fielder_9": 9,
        "batter_days_since_prev_game": 1,
        "pitcher_days_since_prev_game": 3,
        "umpire": 77,
        "sz_top": 3.5,
        "sz_bot": 1.5,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_build_pitch_tokens_includes_new_feature_tokens_and_counts() -> None:
    pitches = pd.DataFrame(
        [
            _make_pitch_row(),
            _make_pitch_row(
                pitch_number=2,
                events="single",
                description="hit_into_play",
            ),
            _make_pitch_row(
                game_pk=2002,
                pitcher=333,
                at_bat_number=7,
                pitch_number=1,
                events="",
            ),
            _make_pitch_row(
                game_pk=2002,
                pitcher=333,
                at_bat_number=7,
                pitch_number=2,
                events="strikeout",
                description="swinging_strike",
            ),
        ]
    )

    pitch_tokens, pitch_contexts, stats = await _build_pitch_tokens_and_contexts(pitches)

    assert stats.sessions == 2
    assert stats.session_starts == stats.session_ends == 2
    assert stats.plate_appearances == 2
    assert stats.plate_appearance_starts == stats.plate_appearance_ends == 2

    token_set = set(pitch_tokens)
    assert PitchToken.SPIN_RATE_IS_1750_2000 in token_set
    assert PitchToken.SPIN_AXIS_IS_30_60 in token_set
    assert PitchToken.RELEASE_EXTENSION_IS_6_65 in token_set
    assert PitchToken.VX0_IS_N5_0 in token_set
    assert PitchToken.VY0_IS_N130_N120 in token_set
    assert PitchToken.VZ0_IS_0_5 in token_set
    assert PitchToken.AX_IS_N10_N5 in token_set
    assert PitchToken.AY_IS_20_25 in token_set
    assert PitchToken.AZ_IS_N35_N30 in token_set

    ctx = pitch_contexts[0]
    assert ctx.game_park_id == 1001
    assert ctx.fielder_9_id == 9
    assert ctx.batter_days_since_prev_game == 1
    assert ctx.pitcher_days_since_prev_game == 3
    assert ctx.umpire_id == 77
    assert ctx.strike_zone_top == 3.5
    assert ctx.strike_zone_bottom == 1.5


@pytest.mark.asyncio
async def test_plate_appearance_end_inferred_when_events_missing() -> None:
    pitches = pd.DataFrame(
        [
            _make_pitch_row(events="", at_bat_number=1, pitch_number=1),
            _make_pitch_row(events="", at_bat_number=1, pitch_number=2),
            _make_pitch_row(events="", at_bat_number=2, pitch_number=1),
        ]
    )

    pitch_tokens, _, stats = await _build_pitch_tokens_and_contexts(pitches)

    pa_end_count = sum(1 for tok in pitch_tokens if tok == PitchToken.PA_END)
    assert stats.plate_appearance_starts == stats.plate_appearance_ends == 2
    assert pa_end_count == 2


def test_pitch_context_to_tensor_includes_extended_fields() -> None:
    context = PitchContext(
        pitcher_id=1,
        batter_id=2,
        pitcher_age=29,
        pitcher_throws="R",
        batter_age=31,
        batter_hits="L",
        count_balls=2,
        count_strikes=1,
        outs=1,
        bases_state=5,
        score_bat=3,
        score_fld=4,
        inning=6,
        pitch_number=12,
        number_through_order=2,
        game_date="2023-05-04",
        game_park_id=3001,
        fielder_2_id=10,
        fielder_3_id=11,
        fielder_4_id=12,
        fielder_5_id=13,
        fielder_6_id=14,
        fielder_7_id=15,
        fielder_8_id=16,
        fielder_9_id=17,
        batter_days_since_prev_game=2,
        pitcher_days_since_prev_game=4,
        umpire_id=88,
        strike_zone_top=3.4,
        strike_zone_bottom=1.2,
    )

    tensor = context.to_tensor().tolist()
    assert tensor[15] == 2023  # game year derived from the date
    assert tensor[16] == 3001  # game_park_id
    assert tensor[24] == 17  # fielder_9_id
    assert tensor[25] == 2  # batter_days_since_prev_game
    assert tensor[26] == 4  # pitcher_days_since_prev_game
    assert tensor[27] == 88  # umpire_id
    assert tensor[28] == pytest.approx(3.4)
    assert tensor[29] == pytest.approx(1.2)


def test_clean_pitch_rows_fills_optional_context_columns() -> None:
    row = _make_pitch_row(
        fielder_2=pd.NA,
        fielder_3=pd.NA,
        fielder_4=pd.NA,
        fielder_5=pd.NA,
        fielder_6=pd.NA,
        fielder_7=pd.NA,
        fielder_8=pd.NA,
        fielder_9=pd.NA,
        batter_days_since_prev_game=pd.NA,
        pitcher_days_since_prev_game=pd.NA,
        umpire=pd.NA,
        sz_top=pd.NA,
        sz_bot=pd.NA,
    )
    cleaned = _clean_pitch_rows(pd.DataFrame([row]))

    assert len(cleaned) == 1
    assert cleaned.loc[0, "fielder_2"] == 0
    assert cleaned.loc[0, "fielder_9"] == 0
    assert cleaned.loc[0, "batter_days_since_prev_game"] == 0
    assert cleaned.loc[0, "pitcher_days_since_prev_game"] == 0
    assert cleaned.loc[0, "umpire"] == 0
    assert cleaned.loc[0, "sz_top"] == pytest.approx(0.0)
    assert cleaned.loc[0, "sz_bot"] == pytest.approx(0.0)
