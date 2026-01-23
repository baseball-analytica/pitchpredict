# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pitchpredict.backend.caching import PitchPredictCache
from pitchpredict.backend.fetching import (
    get_all_batted_balls,
    get_pitches_from_pitcher,
    get_pitches_to_batter,
    get_player_record_from_id,
    get_player_records_from_name,
)
import pitchpredict.backend.fetching as fetching


@pytest.mark.asyncio
async def test_get_pitches_from_pitcher_appends_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = PitchPredictCache(cache_dir=str(tmp_path))
    cached = pd.DataFrame(
        [
            {"pitcher": 1, "game_date": "2024-06-01"},
            {"pitcher": 1, "game_date": "2024-06-05"},
        ]
    )
    cache.set_pitcher_pitches(pitcher_id=1, end_date="2024-06-05", pitches=cached)

    calls: list[tuple[str, str, int]] = []

    def fake_statcast_pitcher(start_dt: str, end_dt: str, player_id: int) -> pd.DataFrame:
        calls.append((start_dt, end_dt, player_id))
        return pd.DataFrame(
            [
                {"pitcher": player_id, "game_date": "2024-06-10"},
            ]
        )

    monkeypatch.setattr(fetching.pybaseball, "statcast_pitcher", fake_statcast_pitcher)

    result = await get_pitches_from_pitcher(
        pitcher_id=1,
        start_date="2015-01-01",
        end_date="2024-06-10",
        cache=cache,
    )

    assert calls == [("2024-06-06", "2024-06-10", 1)]
    assert len(result) == 3
    updated = cache.get_pitcher_pitches(pitcher_id=1, end_date="2024-06-10")
    assert updated is not None
    assert len(updated) == 3


@pytest.mark.asyncio
async def test_get_pitches_to_batter_appends_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = PitchPredictCache(cache_dir=str(tmp_path))
    cached = pd.DataFrame(
        [
            {"batter": 2, "game_date": "2024-06-03"},
            {"batter": 2, "game_date": "2024-06-07"},
        ]
    )
    cache.set_batter_pitches(batter_id=2, end_date="2024-06-07", pitches=cached)

    calls: list[tuple[str, str, int]] = []

    def fake_statcast_batter(start_dt: str, end_dt: str, player_id: int) -> pd.DataFrame:
        calls.append((start_dt, end_dt, player_id))
        return pd.DataFrame(
            [
                {"batter": player_id, "game_date": "2024-06-12"},
            ]
        )

    monkeypatch.setattr(fetching.pybaseball, "statcast_batter", fake_statcast_batter)

    result = await get_pitches_to_batter(
        batter_id=2,
        start_date="2015-01-01",
        end_date="2024-06-12",
        cache=cache,
    )

    assert calls == [("2024-06-08", "2024-06-12", 2)]
    assert len(result) == 3
    updated = cache.get_batter_pitches(batter_id=2, end_date="2024-06-12")
    assert updated is not None
    assert len(updated) == 3


@pytest.mark.asyncio
async def test_get_all_batted_balls_prepends_fetch_when_before_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = PitchPredictCache(cache_dir=str(tmp_path))
    cached = pd.DataFrame(
        [
            {"game_date": "2024-06-05", "type": "X", "launch_speed": 100.0, "launch_angle": 21.0},
        ]
    )
    cache.set_batted_balls(start_date="2024-06-05", end_date="2024-06-20", batted_balls=cached)

    calls: list[tuple[str, str]] = []

    def fake_statcast(start_dt: str, end_dt: str) -> pd.DataFrame:
        calls.append((start_dt, end_dt))
        return pd.DataFrame(
            [
                {"game_date": "2024-05-25", "type": "X", "launch_speed": 97.0, "launch_angle": 18.0},
                {"game_date": "2024-06-05", "type": "X", "launch_speed": 100.0, "launch_angle": 21.0},
            ]
        )

    monkeypatch.setattr(fetching.pybaseball, "statcast", fake_statcast)

    result = await get_all_batted_balls(
        start_date="2024-05-20",
        end_date="2024-06-20",
        cache=cache,
    )

    assert calls == [("2024-05-20", "2024-06-20")]
    assert len(result) == 2
    assert set(result["game_date"]) == {"2024-05-25", "2024-06-05"}

    updated = cache.get_batted_balls(start_date="2024-05-20", end_date="2024-06-20")
    assert updated is not None
    assert len(updated) == 2


@pytest.mark.asyncio
async def test_get_all_batted_balls_appends_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = PitchPredictCache(cache_dir=str(tmp_path))
    cached = pd.DataFrame(
        [
            {"game_date": "2024-06-01", "type": "X", "launch_speed": 98.0, "launch_angle": 20.0},
        ]
    )
    cache.set_batted_balls(start_date="2024-01-01", end_date="2024-06-01", batted_balls=cached)

    calls: list[tuple[str, str]] = []

    def fake_statcast(start_dt: str, end_dt: str) -> pd.DataFrame:
        calls.append((start_dt, end_dt))
        return pd.DataFrame(
            [
                {"game_date": "2024-06-05", "type": "X", "launch_speed": 102.0, "launch_angle": 25.0},
                {"game_date": "2024-06-05", "type": "B", "launch_speed": 90.0, "launch_angle": 10.0},
            ]
        )

    monkeypatch.setattr(fetching.pybaseball, "statcast", fake_statcast)

    result = await get_all_batted_balls(
        start_date="2024-02-01",
        end_date="2024-06-10",
        cache=cache,
    )

    assert calls == [("2024-06-02", "2024-06-10")]
    assert len(result) == 2
    assert set(result["game_date"]) == {"2024-06-01", "2024-06-05"}

    updated = cache.get_batted_balls(start_date="2024-02-01", end_date="2024-06-10")
    assert updated is not None
    assert len(updated) == 2


@pytest.mark.asyncio
async def test_get_player_records_from_name_uses_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = PitchPredictCache(cache_dir=str(tmp_path))
    records = [
        {
            "name_first": "Aaron",
            "name_last": "Judge",
            "name_full": "Aaron Judge",
            "key_mlbam": 592450,
        }
    ]
    cache.set_player_records(player_name="Aaron Judge", fuzzy_lookup=True, records=records)

    def fail_lookup(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("playerid_lookup should not be called when cached")

    monkeypatch.setattr(fetching.pybaseball, "playerid_lookup", fail_lookup)

    result = await get_player_records_from_name(
        player_name="Aaron Judge",
        fuzzy_lookup=True,
        limit=1,
        cache=cache,
    )

    assert result == records


@pytest.mark.asyncio
async def test_get_player_record_from_id_uses_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = PitchPredictCache(cache_dir=str(tmp_path))
    record = {
        "name_first": "Aaron",
        "name_last": "Judge",
        "name_full": "Aaron Judge",
        "key_mlbam": 592450,
    }
    cache.set_player_record_by_id(mlbam_id=592450, record=record)

    def fail_reverse_lookup(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("playerid_reverse_lookup should not be called when cached")

    monkeypatch.setattr(fetching.pybaseball, "playerid_reverse_lookup", fail_reverse_lookup)

    result = await get_player_record_from_id(
        mlbam_id=592450,
        cache=cache,
    )

    assert result == record
