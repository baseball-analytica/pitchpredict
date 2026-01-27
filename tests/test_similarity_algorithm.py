# SPDX-License-Identifier: MIT

from __future__ import annotations

import pandas as pd
import pytest

from pitchpredict.backend.algs.similarity.base import SimilarityAlgorithm
import pitchpredict.backend.algs.similarity.base as similarity_base
import pitchpredict.types.api as api_types


@pytest.mark.asyncio
async def test_get_similar_pitches_for_pitcher_ranks_best_match() -> None:
    pitches = pd.DataFrame(
        [
            {
                "batter": 1,
                "balls": 2,
                "strikes": 1,
                "on_1b": 100,
                "on_2b": 200,
                "on_3b": pd.NA,
                "game_date": "2024-06-15",
            },
            {
                "batter": 2,
                "balls": 2,
                "strikes": 1,
                "on_1b": pd.NA,
                "on_2b": pd.NA,
                "on_3b": pd.NA,
                "game_date": "2024-06-15",
            },
            {
                "batter": 1,
                "balls": 0,
                "strikes": 1,
                "on_1b": 100,
                "on_2b": 200,
                "on_3b": pd.NA,
                "game_date": "2023-06-15",
            },
        ]
    )
    context = api_types.PredictPitcherRequest(
        pitcher_id=10,
        batter_id=1,
        count_balls=2,
        count_strikes=1,
        bases_state=3,
        game_date="2024-06-15",
    )
    weights = {
        "batter_id": 0.4,
        "count_balls": 0.3,
        "bases_state": 0.2,
        "game_date": 0.1,
    }

    algorithm = SimilarityAlgorithm()
    similar = await algorithm._get_similar_pitches_for_pitcher(
        pitches=pitches,
        context=context,
        weights=weights,
        sample_pctg=1.0,
    )

    assert "similarity_score" in similar.columns
    assert similar.iloc[0]["batter"] == 1
    assert similar.iloc[0]["balls"] == 2
    assert similar.iloc[1]["balls"] == 0


def test_sample_pitches_uses_description_when_events_missing() -> None:
    pitches = pd.DataFrame(
        [
            {
                "pitch_type": "FF",
                "release_speed": 95.0,
                "release_spin_rate": 2100.0,
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
                "plate_x": 0.5,
                "plate_z": 2.0,
                "events": pd.NA,
                "description": "called_strike",
            }
        ]
    )

    algorithm = SimilarityAlgorithm()
    sampled = algorithm._sample_pitches(pitches=pitches, n=1)

    assert len(sampled) == 1
    assert sampled[0].result == "called_strike"


@pytest.mark.asyncio
async def test_get_cached_pitches_for_pitcher_reuses_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    async def fake_get_pitches_from_pitcher(
        pitcher_id: int,
        start_date: str,
        end_date: str | None = None,
        cache: object | None = None,
    ) -> pd.DataFrame:
        calls.append(
            {
                "pitcher_id": pitcher_id,
                "start_date": start_date,
                "end_date": end_date,
            }
        )
        return pd.DataFrame(
            [
                {"pitcher": pitcher_id, "game_date": "2024-06-01"},
                {"pitcher": pitcher_id, "game_date": "2024-06-10"},
            ]
        )

    monkeypatch.setattr(
        similarity_base, "get_pitches_from_pitcher", fake_get_pitches_from_pitcher
    )

    algorithm = SimilarityAlgorithm()
    first = await algorithm._get_cached_pitches_for_pitcher(
        pitcher_id=42,
        end_date="2024-06-10",
    )
    second = await algorithm._get_cached_pitches_for_pitcher(
        pitcher_id=42,
        end_date="2024-06-05",
    )

    assert len(calls) == 1
    assert len(first) == 2
    assert len(second) == 1
    assert second.iloc[0]["game_date"] == "2024-06-01"
