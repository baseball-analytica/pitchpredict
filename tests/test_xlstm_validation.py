# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
from fastapi import HTTPException

from pitchpredict.backend.algs.xlstm.base import XlstmAlgorithm
import pitchpredict.types.api as api_types


def _make_pitch(pa_id: int | None) -> api_types.Pitch:
    return api_types.Pitch(
        pitch_type="FF",
        speed=95.0,
        spin_rate=2200.0,
        spin_axis=120.0,
        release_pos_x=-2.0,
        release_pos_z=6.0,
        release_extension=6.5,
        vx0=-7.0,
        vy0=-135.0,
        vz0=2.0,
        ax=-12.0,
        ay=22.0,
        az=-33.0,
        plate_pos_x=0.5,
        plate_pos_z=2.0,
        result="called_strike",
        pa_id=pa_id,
    )


def test_xlstm_validation_requires_pa_id() -> None:
    alg = XlstmAlgorithm()
    request = api_types.PredictPitcherRequest(
        pitcher_id=1,
        batter_id=2,
        algorithm="xlstm",
        prev_pitches=[_make_pitch(pa_id=None)],
    )

    with pytest.raises(HTTPException) as exc:
        alg._validate_xlstm_request(request)

    assert exc.value.status_code == 400


def test_xlstm_invalid_game_date_raises_400() -> None:
    alg = XlstmAlgorithm()
    request = api_types.PredictPitcherRequest(
        pitcher_id=1,
        batter_id=2,
        algorithm="xlstm",
        prev_pitches=[],
        game_date="20-01",
    )

    with pytest.raises(HTTPException) as exc:
        alg._build_context_defaults(request)

    assert exc.value.status_code == 400
