# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from pitchpredict.api import PitchPredict
import pitchpredict.types.api as api_types


class _FakeAlgorithm:
    def __init__(self, result: dict[str, Any]) -> None:
        self.calls: list[api_types.PredictPitcherRequest] = []
        self._result = result

    async def predict_pitcher(
        self, request: api_types.PredictPitcherRequest
    ) -> dict[str, Any]:
        self.calls.append(request)
        return self._result


@pytest.mark.asyncio
async def test_predict_pitcher_builds_request_and_calls_algorithm(
    tmp_path: Any,
) -> None:
    expected = {
        "basic_pitch_data": {"pitch_type_probs": {}},
        "detailed_pitch_data": {},
        "basic_outcome_data": {},
        "detailed_outcome_data": {},
        "pitches": [],
        "prediction_metadata": {},
    }
    fake_algorithm = _FakeAlgorithm(result=expected)
    api = PitchPredict(
        enable_cache=False,
        enable_logging=True,
        log_dir=str(tmp_path),
        algorithms={"similarity": fake_algorithm},  # type: ignore
    )

    result = await api.predict_pitcher(
        pitcher_id=11,
        batter_id=22,
        count_balls=1,
        count_strikes=2,
        outs=1,
        algorithm="similarity",
        sample_size=3,
    )

    assert result == expected
    assert len(fake_algorithm.calls) == 1
    request = fake_algorithm.calls[0]
    assert isinstance(request, api_types.PredictPitcherRequest)
    assert request.pitcher_id == 11
    assert request.batter_id == 22
    assert request.count_balls == 1
    assert request.count_strikes == 2
    assert request.outs == 1
    assert request.algorithm == "similarity"
    assert request.sample_size == 3


@pytest.mark.asyncio
async def test_predict_pitcher_raises_for_unknown_algorithm(tmp_path: Any) -> None:
    fake_algorithm = _FakeAlgorithm(result={})
    api = PitchPredict(
        enable_cache=False,
        enable_logging=True,
        log_dir=str(tmp_path),
        algorithms={"similarity": fake_algorithm},  # type: ignore
    )

    with pytest.raises(HTTPException) as exc:
        await api.predict_pitcher(
            pitcher_id=11,
            batter_id=22,
            algorithm="unknown",
        )

    assert exc.value.status_code == 400
