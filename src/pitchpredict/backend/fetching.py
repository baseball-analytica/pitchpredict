# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from datetime import datetime
import logging

from fastapi import HTTPException
import pandas as pd
import pybaseball # type: ignore

from pitchpredict.backend.caching import PitchPredictCache

logger = logging.getLogger("pitchpredict.backend.fetching")


def _filter_pitches_by_range(pitches: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Return a view of pitches within the requested date range."""
    if "game_date" not in pitches.columns:
        return pitches.copy(deep=False)
    dates = pd.to_datetime(pitches["game_date"], errors="coerce")
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    mask = (dates >= start_ts) & (dates <= end_ts)
    return pitches.loc[mask].copy(deep=False)


def _coerce_record_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def _records_from_player_df(player_ids: pd.DataFrame) -> list[dict[str, object]]:
    records = player_ids.where(pd.notna(player_ids), None).to_dict(orient="records")
    return [
        {key: _coerce_record_value(value) for key, value in record.items()}
        for record in records
    ]


async def get_pitches_from_pitcher(
    pitcher_id: int,
    start_date: str,
    end_date: str | None = None,
    cache: PitchPredictCache | None = None,
) -> pd.DataFrame:
    """
    Given a pitcher's MLBAM ID, get a list of all their pitches thrown between `start_date` and `end_date`.
    Note that this returns a large list of pitches that will be narrowed down later.

    Args:
        pitcher_id: The MLBAM ID of the pitcher.
        start_date: The start date of the period to get pitches for.
        end_date: The end date of the period to get pitches for.

    Returns:
        A pandas DataFrame containing the pitches thrown by the pitcher.
    """
    logger.debug("get_pitches_from_pitcher called")

    if end_date is None:
        logger.debug("end_date is None, using current date")
        end_date = datetime.now().strftime("%Y-%m-%d")

    if cache is not None:
        cached = cache.get_pitcher_pitches(pitcher_id=pitcher_id, end_date=end_date)
        if cached is not None:
            return _filter_pitches_by_range(cached, start_date, end_date)

        cache_state = cache.get_pitcher_cache_state(pitcher_id=pitcher_id)
        if cache_state is not None:
            cached_data, cached_end_date = cache_state
            try:
                cached_end_ts = pd.Timestamp(cached_end_date)
                requested_end_ts = pd.Timestamp(end_date)
                requested_start_ts = pd.Timestamp(start_date)
            except (TypeError, ValueError):
                cached_end_ts = None
            if cached_end_ts is not None and requested_end_ts > cached_end_ts and requested_start_ts <= cached_end_ts:
                # Fetch only the missing tail of data and append to cache.
                fetch_start = (cached_end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                try:
                    logger.debug("fetching pitches from pitcher for cache append")
                    new_pitches = pybaseball.statcast_pitcher(
                        start_dt=fetch_start,
                        end_dt=end_date,
                        player_id=pitcher_id,
                    )
                except Exception as exc:
                    logger.error(f"encountered Exception: {exc}")
                    raise HTTPException(status_code=500, detail=str(exc))

                if new_pitches.empty:
                    logger.info("no new pitches found for pitcher %s between %s and %s", pitcher_id, fetch_start, end_date)
                    combined = cached_data
                else:
                    combined = pd.concat([cached_data, new_pitches], ignore_index=True)
                cache.set_pitcher_pitches(pitcher_id=pitcher_id, end_date=end_date, pitches=combined)
                return _filter_pitches_by_range(combined, start_date, end_date)

    try:
        logger.debug("fetching pitches from pitcher")
        pitches = pybaseball.statcast_pitcher(
            start_dt=start_date,
            end_dt=end_date,
            player_id=pitcher_id,
        )

        if pitches.empty:
            logger.error(f"no pitches found for pitcher with ID {pitcher_id} between {start_date} and {end_date}")
            raise HTTPException(status_code=404, detail=f"no pitches found for pitcher with ID {pitcher_id} between {start_date} and {end_date}")

        logger.debug("pitches fetched successfully")
        if cache is not None:
            cache.set_pitcher_pitches(pitcher_id=pitcher_id, end_date=end_date, pitches=pitches)
        return pitches
    except HTTPException as e:
        logger.error(f"encountered HTTPException: {e}")
        raise e
    except Exception as e:
        logger.error(f"encountered Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_pitches_to_batter(
    batter_id: int,
    start_date: str,
    end_date: str | None = None,
    cache: PitchPredictCache | None = None,
) -> pd.DataFrame:
    """
    Given a batter's MLBAM ID, get a list of all the pitches thrown to them between `start_date` and `end_date`.
    """
    logger.debug("get_pitches_to_batter called")

    if end_date is None:
        logger.debug("end_date is None, using current date")
        end_date = datetime.now().strftime("%Y-%m-%d")

    if cache is not None:
        cached = cache.get_batter_pitches(batter_id=batter_id, end_date=end_date)
        if cached is not None:
            return _filter_pitches_by_range(cached, start_date, end_date)

        cache_state = cache.get_batter_cache_state(batter_id=batter_id)
        if cache_state is not None:
            cached_data, cached_end_date = cache_state
            try:
                cached_end_ts = pd.Timestamp(cached_end_date)
                requested_end_ts = pd.Timestamp(end_date)
                requested_start_ts = pd.Timestamp(start_date)
            except (TypeError, ValueError):
                cached_end_ts = None
            if cached_end_ts is not None and requested_end_ts > cached_end_ts and requested_start_ts <= cached_end_ts:
                # Fetch only the missing tail of data and append to cache.
                fetch_start = (cached_end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                try:
                    logger.debug("fetching pitches to batter for cache append")
                    new_pitches = pybaseball.statcast_batter(
                        start_dt=fetch_start,
                        end_dt=end_date,
                        player_id=batter_id,
                    )
                except Exception as exc:
                    logger.error(f"encountered Exception: {exc}")
                    raise HTTPException(status_code=500, detail=str(exc))

                if new_pitches.empty:
                    logger.info(f"no new pitches found for batter {batter_id} between {fetch_start} and {end_date}")
                    combined = cached_data
                else:
                    combined = pd.concat([cached_data, new_pitches], ignore_index=True)
                cache.set_batter_pitches(batter_id=batter_id, end_date=end_date, pitches=combined)
                return _filter_pitches_by_range(combined, start_date, end_date)

    try:
        logger.debug("fetching pitches to batter")
        pitches = pybaseball.statcast_batter(
            start_dt=start_date,
            end_dt=end_date,
            player_id=batter_id,
        )

        if pitches.empty:
            logger.error(f"no pitches found for batter with ID {batter_id} between {start_date} and {end_date}")
            raise HTTPException(status_code=404, detail=f"no pitches found for batter with ID {batter_id} between {start_date} and {end_date}")

        logger.debug("pitches fetched successfully")
        if cache is not None:
            cache.set_batter_pitches(batter_id=batter_id, end_date=end_date, pitches=pitches)
        return pitches
    except HTTPException as e:
        logger.error(f"encountered HTTPException: {e}")
        raise e
    except Exception as e:
        logger.error(f"encountered Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_player_id_from_name(
    player_name: str,
    fuzzy_lookup: bool = True,
    cache: PitchPredictCache | None = None,
) -> int:
    """
    Given a player's name, get their MLBAM ID.
    """
    logger.debug("get_player_id_from_name called")

    try:
        last_name, first_name = _parse_player_name(player_name)
        logger.debug(f"parsed player name: {last_name}, {first_name}")
        if cache is not None:
            cached = cache.get_player_id(player_name=player_name, fuzzy_lookup=fuzzy_lookup)
            if cached is not None:
                return cached
        player_ids = pybaseball.playerid_lookup(
            last_name,
            first_name,
            fuzzy=fuzzy_lookup
        )

        if player_ids.empty:
            logger.error(f"no player found with name {player_name}")
            raise HTTPException(status_code=404, detail=f"no player found with name {player_name}")

        player_id = int(player_ids.iloc[0]["key_mlbam"])
        logger.info(f"player ID fetched successfully for {player_name}: {player_id}")
        if cache is not None:
            cache.set_player_id(player_name=player_name, player_id=player_id, fuzzy_lookup=fuzzy_lookup)
        return player_id
    except HTTPException as e:
        logger.error(f"encountered HTTPException: {e}")
        raise e
    except Exception as e:
        logger.error(f"encountered Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_player_records_from_name(
    player_name: str,
    fuzzy_lookup: bool = True,
    limit: int = 1,
    cache: PitchPredictCache | None = None,
) -> list[dict[str, object]]:
    """
    Given a player's name, return matching player records from pybaseball.
    """
    logger.debug("get_player_records_from_name called")

    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")

    try:
        last_name, first_name = _parse_player_name(player_name)
        logger.debug(f"parsed player name: {last_name}, {first_name}")
        if cache is not None:
            cached = cache.get_player_records(player_name=player_name, fuzzy_lookup=fuzzy_lookup)
            if cached is not None:
                return cached[:limit]

        player_ids = pybaseball.playerid_lookup(
            last_name,
            first_name,
            fuzzy=fuzzy_lookup
        )

        if player_ids.empty:
            logger.error(f"no player found with name {player_name}")
            raise HTTPException(status_code=404, detail=f"no player found with name {player_name}")

        records = _records_from_player_df(player_ids)
        if cache is not None:
            cache.set_player_records(player_name=player_name, fuzzy_lookup=fuzzy_lookup, records=records)
        return records[:limit]
    except HTTPException as e:
        logger.error(f"encountered HTTPException: {e}")
        raise e
    except Exception as e:
        logger.error(f"encountered Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_player_record_from_id(
    mlbam_id: int,
    cache: PitchPredictCache | None = None,
) -> dict[str, object]:
    """
    Given an MLBAM ID, return the full pybaseball player record.
    """
    logger.debug("get_player_record_from_id called")

    if cache is not None:
        cached = cache.get_player_record_by_id(mlbam_id=mlbam_id)
        if cached is not None:
            return cached

    try:
        if not hasattr(pybaseball, "playerid_reverse_lookup"):
            raise HTTPException(status_code=500, detail="pybaseball.playerid_reverse_lookup is unavailable")

        player_ids = pybaseball.playerid_reverse_lookup([mlbam_id])
        if player_ids.empty:
            logger.error(f"no player found with id {mlbam_id}")
            raise HTTPException(status_code=404, detail=f"no player found with id {mlbam_id}")

        record = _records_from_player_df(player_ids)[0]
        if cache is not None:
            cache.set_player_record_by_id(mlbam_id=mlbam_id, record=record)
        return record
    except HTTPException as e:
        logger.error(f"encountered HTTPException: {e}")
        raise e
    except Exception as e:
        logger.error(f"encountered Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _parse_player_name(name: str) -> tuple[str, str]:
    """
    Parse the given player's name: "First Last" -> ("Last", "First").
    """
    logger.debug("parse_player_name called")

    name_split = name.split(" ")
    if len(name_split) != 2:
        logger.error(f"player name must be in the format 'First Last': {name}")
        raise HTTPException(status_code=400, detail="player name must be in the format 'First Last'")

    logger.debug(f"parsed player name: {name_split[1]}, {name_split[0]}")
    return name_split[1], name_split[0]


async def get_all_pitches(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Get all pitches thrown between `start_date` and `end_date`.
    """
    logger.debug("get_all_pitches called")

    try:
        logger.debug("fetching all pitches")
        pitches = pybaseball.statcast(
            start_dt=start_date,
            end_dt=end_date,
        )

        if pitches.empty:
            logger.error(f"no pitches found between {start_date} and {end_date}")
            raise HTTPException(status_code=404, detail=f"no pitches found between {start_date} and {end_date}")

        logger.debug("pitches fetched successfully")
        return pitches
    except HTTPException as e:
        logger.error(f"encountered HTTPException: {e}")
        raise e
    except Exception as e:
        logger.error(f"encountered Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_all_batted_balls(
    start_date: str | None = None,
    end_date: str | None = None,
    n_seasons: int = 3,
    cache: PitchPredictCache | None = None,
) -> pd.DataFrame:
    """
    Get all batted ball events from Statcast data.

    Args:
        start_date: The start date of the period to get batted balls for. If None, defaults to n_seasons ago.
        end_date: The end date of the period to get batted balls for. If None, defaults to current date.
        n_seasons: Number of seasons to fetch if start_date is not provided (default: 3).

    Returns:
        A pandas DataFrame containing batted ball events with launch_speed and launch_angle data.
    """
    logger.debug("get_all_batted_balls called")

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if start_date is None:
        # Default to n_seasons ago (approximately)
        current_year = datetime.now().year
        start_year = current_year - n_seasons
        start_date = f"{start_year}-03-01"

    if cache is not None:
        cached = cache.get_batted_balls(start_date=start_date, end_date=end_date)
        if cached is not None:
            return cached

        cache_state = cache.get_batted_balls_cache_state()
        if cache_state is not None:
            cached_data, cached_start, cached_end = cache_state
            try:
                cached_start_ts = pd.Timestamp(cached_start)
                cached_end_ts = pd.Timestamp(cached_end)
                requested_start_ts = pd.Timestamp(start_date)
                requested_end_ts = pd.Timestamp(end_date)
            except (TypeError, ValueError):
                cached_end_ts = None
            if (
                cached_end_ts is not None
                and requested_start_ts >= cached_start_ts
                and requested_end_ts > cached_end_ts
            ):
                # Extend cached range by fetching only missing data.
                fetch_start = (cached_end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                try:
                    logger.debug("fetching batted balls for cache append")
                    pitches = pybaseball.statcast(
                        start_dt=fetch_start,
                        end_dt=end_date,
                    )
                except Exception as exc:
                    logger.error(f"encountered Exception: {exc}")
                    raise HTTPException(status_code=500, detail=str(exc))

                if pitches.empty:
                    logger.info("no new batted balls found between %s and %s", fetch_start, end_date)
                    combined = cached_data
                else:
                    batted_balls = pitches[
                        (pitches["type"] == "X") &
                        (pitches["launch_speed"].notna()) &
                        (pitches["launch_angle"].notna())
                    ].copy()
                    combined = pd.concat([cached_data, batted_balls], ignore_index=True)
                cache.set_batted_balls(
                    start_date=cached_start,
                    end_date=end_date,
                    batted_balls=combined,
                )
                return _filter_pitches_by_range(combined, start_date, end_date)

    try:
        logger.debug(f"fetching batted balls from {start_date} to {end_date}")
        pitches = pybaseball.statcast(
            start_dt=start_date,
            end_dt=end_date,
        )

        if pitches.empty:
            logger.error(f"no data found between {start_date} and {end_date}")
            raise HTTPException(status_code=404, detail=f"no data found between {start_date} and {end_date}")

        # Filter to only batted ball events (contact events with launch_speed and launch_angle)
        batted_balls = pitches[
            (pitches["type"] == "X") &
            (pitches["launch_speed"].notna()) &
            (pitches["launch_angle"].notna())
        ].copy()

        if batted_balls.empty:
            logger.error(f"no batted ball events found between {start_date} and {end_date}")
            raise HTTPException(status_code=404, detail=f"no batted ball events found between {start_date} and {end_date}")

        logger.info(f"fetched {batted_balls.shape[0]} batted ball events successfully")
        if cache is not None:
            cache.set_batted_balls(start_date=start_date, end_date=end_date, batted_balls=batted_balls)
        return batted_balls
    except HTTPException as e:
        logger.error(f"encountered HTTPException: {e}")
        raise e
    except Exception as e:
        logger.error(f"encountered Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))
