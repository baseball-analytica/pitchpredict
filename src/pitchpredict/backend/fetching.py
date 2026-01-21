# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging

from fastapi import HTTPException
import pandas as pd
import pybaseball # type: ignore
from tqdm import tqdm

logger = logging.getLogger("pitchpredict.backend.fetching")


async def get_pitches_from_pitcher(
    pitcher_id: int,
    start_date: str,
    end_date: str | None = None,
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
) -> pd.DataFrame:
    """
    Given a batter's MLBAM ID, get a list of all the pitches thrown to them between `start_date` and `end_date`.
    """
    logger.debug("get_pitches_to_batter called")

    if end_date is None:
        logger.debug("end_date is None, using current date")
        end_date = datetime.now().strftime("%Y-%m-%d")

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
) -> int:
    """
    Given a player's name, get their MLBAM ID.
    """
    logger.debug("get_player_id_from_name called")

    try:
        last_name, first_name = _parse_player_name(player_name)
        logger.debug(f"parsed player name: {last_name}, {first_name}")
        player_ids = pybaseball.playerid_lookup(
            last_name,
            first_name,
            fuzzy=fuzzy_lookup
        )

        if player_ids.empty:
            logger.error(f"no player found with name {player_name}")
            raise HTTPException(status_code=404, detail=f"no player found with name {player_name}")

        logger.info(f"player ID fetched successfully for {player_name}: {player_ids.iloc[0]['key_mlbam']}")
        return player_ids.iloc[0]["key_mlbam"]
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
    parallel: bool = True,
    max_workers: int = 8,
    chunk_months: int = 2,
) -> pd.DataFrame:
    """
    Get all pitches thrown between `start_date` and `end_date`.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        parallel: If True, fetch date chunks in parallel (faster)
        max_workers: Number of parallel fetch threads
        chunk_months: Size of each chunk in months (default 2)

    Returns:
        DataFrame with all pitches
    """
    logger.debug("get_all_pitches called")

    if parallel:
        return await get_all_pitches_parallel(
            start_date, end_date,
            max_workers=max_workers,
            chunk_months=chunk_months
        )

    try:
        logger.debug("fetching all pitches (sequential)")
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
        return batted_balls
    except HTTPException as e:
        logger.error(f"encountered HTTPException: {e}")
        raise e
    except Exception as e:
        logger.error(f"encountered Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _generate_date_chunks(
    start_date: str,
    end_date: str,
    chunk_months: int = 2,
) -> list[tuple[str, str]]:
    """
    Split a date range into chunks of approximately chunk_months months.

    Returns list of (start, end) date string tuples.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    chunks = []
    current = start

    while current < end:
        chunk_end = current + relativedelta(months=chunk_months) - timedelta(days=1)
        if chunk_end > end:
            chunk_end = end

        chunks.append((
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d")
        ))

        current = chunk_end + timedelta(days=1)

    return chunks


def _fetch_chunk(chunk: tuple[str, str]) -> pd.DataFrame:
    """Fetch a single date chunk. Called from thread pool."""
    start, end = chunk
    try:
        # Disable pybaseball's internal parallel (we're doing our own)
        df = pybaseball.statcast(start_dt=start, end_dt=end, parallel=False)
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch chunk {start} to {end}: {e}")
        return pd.DataFrame()


async def get_all_pitches_parallel(
    start_date: str,
    end_date: str,
    max_workers: int = 8,
    chunk_months: int = 2,
) -> pd.DataFrame:
    """
    Fetch all pitches in parallel by splitting into date chunks.

    This is faster than pybaseball's default because we fetch multiple
    month-sized chunks simultaneously, rather than sequential 5-day chunks.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_workers: Number of parallel fetch threads (default 8)
        chunk_months: Size of each chunk in months (default 2)

    Returns:
        DataFrame with all pitches, concatenated and deduplicated
    """
    chunks = _generate_date_chunks(start_date, end_date, chunk_months)
    logger.info(f"Fetching {len(chunks)} chunks in parallel (max_workers={max_workers})")

    results: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_chunk, chunk): chunk for chunk in chunks}

        with tqdm(total=len(chunks), desc="Fetching pitch data") as pbar:
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results.append(df)
                        logger.debug(f"Chunk {chunk[0]} to {chunk[1]}: {len(df)} pitches")
                except Exception as e:
                    logger.error(f"Chunk {chunk} failed: {e}")
                pbar.update(1)

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"no pitches found between {start_date} and {end_date}"
        )

    # Concatenate all results
    logger.info(f"Concatenating {len(results)} chunks...")
    pitches = pd.concat(results, ignore_index=True)

    # Drop duplicates (chunks might have slight overlap at boundaries)
    initial_count = len(pitches)
    pitches = pitches.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
    dedup_count = initial_count - len(pitches)
    if dedup_count > 0:
        logger.info(f"Removed {dedup_count} duplicate rows")

    logger.info(f"Fetched {len(pitches):,} total pitches")
    return pitches
