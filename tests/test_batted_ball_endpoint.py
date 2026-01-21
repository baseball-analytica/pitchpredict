# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from fastapi.testclient import TestClient

from pitchpredict import server as server_module


class _FakeAPI:
    def __init__(self, result: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._result = result
        self._error = error

    async def predict_batted_ball(
        self,
        launch_speed: float,
        launch_angle: float,
        algorithm: str,
        spray_angle: float | None = None,
        bb_type: str | None = None,
        outs: int | None = None,
        bases_state: int | None = None,
        batter_id: int | None = None,
        game_date: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "launch_speed": launch_speed,
                "launch_angle": launch_angle,
                "algorithm": algorithm,
                "spray_angle": spray_angle,
                "bb_type": bb_type,
                "outs": outs,
                "bases_state": bases_state,
                "batter_id": batter_id,
                "game_date": game_date,
            }
        )
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise AssertionError("fake api result is not configured")
        return self._result


def _make_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "launch_speed": 102.5,
        "launch_angle": 17.2,
        "algorithm": "similarity",
    }
    payload.update(overrides)
    return payload


def test_predict_batted_ball_endpoint_success(monkeypatch: Any) -> None:
    expected = {
        "basic_outcome_data": {"hit_probability": 0.42},
        "detailed_outcome_data": {"xwoba": 0.35},
        "prediction_metadata": {"n_batted_balls_sampled": 150},
    }
    api = _FakeAPI(result=expected)
    monkeypatch.setattr(server_module, "PitchPredict", lambda: api)

    with TestClient(server_module.app) as client:
        payload = _make_payload(
            spray_angle=23.4,
            bb_type="line_drive",
            outs=1,
            bases_state=3,
            batter_id=123456,
            game_date="2024-07-04",
        )
        response = client.post("/predict/batted-ball", json=payload)

    assert response.status_code == 200
    assert response.json() == expected
    assert api.calls == [payload]


def test_predict_batted_ball_endpoint_propagates_http_exception(monkeypatch: Any) -> None:
    api = _FakeAPI(error=HTTPException(status_code=400, detail="bad algorithm"))
    monkeypatch.setattr(server_module, "PitchPredict", lambda: api)

    with TestClient(server_module.app) as client:
        response = client.post("/predict/batted-ball", json=_make_payload())

    assert response.status_code == 400
    assert response.json()["detail"] == "bad algorithm"
