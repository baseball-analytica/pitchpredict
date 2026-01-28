# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from pitchpredict import server as server_module


class _FakeAPI:
    def __init__(self, result: dict[str, Any]) -> None:
        self.pitcher_calls: list[dict[str, Any]] = []
        self.batter_calls: list[dict[str, Any]] = []
        self._result = result

    async def predict_pitcher(self, **kwargs: Any) -> dict[str, Any]:
        self.pitcher_calls.append(kwargs)
        return self._result

    async def predict_batter(self, **kwargs: Any) -> dict[str, Any]:
        self.batter_calls.append(kwargs)
        return self._result


def _pitcher_result() -> dict[str, Any]:
    return {
        "basic_pitch_data": {},
        "detailed_pitch_data": {},
        "basic_outcome_data": {},
        "detailed_outcome_data": {},
        "pitches": [],
        "prediction_metadata": {},
    }


def _batter_result() -> dict[str, Any]:
    return {
        "algorithm_metadata": {},
        "basic_outcome_data": {},
        "detailed_outcome_data": {},
        "prediction_metadata": {},
    }


def test_predict_pitcher_endpoint_passes_fields(monkeypatch: Any) -> None:
    api = _FakeAPI(result=_pitcher_result())
    monkeypatch.setattr(server_module, "PitchPredict", lambda: api)

    payload = {
        "pitcher_id": 100001,
        "batter_id": 200002,
        "algorithm": "similarity",
        "sample_size": 2,
        "pitcher_age": 29,
        "pitcher_throws": "R",
        "batter_age": 31,
        "batter_hits": "L",
        "count_balls": 1,
        "count_strikes": 2,
        "outs": 1,
        "bases_state": 5,
        "score_bat": 3,
        "score_fld": 2,
        "inning": 6,
        "pitch_number": 12,
        "number_through_order": 2,
        "game_date": "2024-07-04",
        "fielder_2_id": 111111,
        "fielder_3_id": 222222,
        "batter_days_since_prev_game": 1,
        "pitcher_days_since_prev_game": 4,
        "strike_zone_top": 3.5,
        "strike_zone_bottom": 1.5,
    }

    with TestClient(server_module.app) as client:
        response = client.post("/predict/pitcher", json=payload)

    assert response.status_code == 200
    assert response.json() == _pitcher_result()
    assert len(api.pitcher_calls) == 1
    call = api.pitcher_calls[0]
    assert "request" not in call
    for key, value in payload.items():
        assert call[key] == value
    assert call["prev_pitches"] is None


def test_predict_batter_endpoint_passes_fields(monkeypatch: Any) -> None:
    api = _FakeAPI(result=_batter_result())
    monkeypatch.setattr(server_module, "PitchPredict", lambda: api)

    payload = {
        "pitcher_id": 100001,
        "batter_id": 200002,
        "pitch_type": "FF",
        "pitch_speed": 95.0,
        "pitch_x": 0.15,
        "pitch_z": 2.55,
        "algorithm": "similarity",
        "sample_size": 1,
        "count_balls": 0,
        "count_strikes": 1,
        "outs": 1,
        "bases_state": 3,
        "score_bat": 2,
        "score_fld": 1,
        "inning": 5,
        "game_date": "2024-07-04",
        "pitcher_age": 28,
        "pitcher_throws": "R",
        "batter_age": 27,
        "batter_hits": "R",
    }

    with TestClient(server_module.app) as client:
        response = client.post("/predict/batter", json=payload)

    assert response.status_code == 200
    assert response.json() == _batter_result()
    assert len(api.batter_calls) == 1
    call = api.batter_calls[0]
    assert "request" not in call
    for key, value in payload.items():
        assert call[key] == value
    assert call["prev_pitches"] is None
