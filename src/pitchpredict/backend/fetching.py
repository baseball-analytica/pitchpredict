# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from datetime import datetime

from fastapi import HTTPException
import pandas as pd
import pybaseball # type: ignore


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
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        pitches = pybaseball.statcast_pitcher(
            start_dt=start_date,
            end_dt=end_date,
            player_id=pitcher_id,
        )

        if pitches.empty:
            raise HTTPException(status_code=404, detail=f"no pitches found for pitcher with ID {pitcher_id} between {start_date} and {end_date}")

        return pitches
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_pitches_to_batter(
    batter_id: int,
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Given a batter's MLBAM ID, get a list of all the pitches thrown to them between `start_date` and `end_date`.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        pitches = pybaseball.statcast_batter(
            start_dt=start_date,
            end_dt=end_date,
            player_id=batter_id,
        )

        if pitches.empty:
            raise HTTPException(status_code=404, detail=f"no pitches found for batter with ID {batter_id} between {start_date} and {end_date}")

        return pitches
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_player_id_from_name(
    player_name: str,
    fuzzy_lookup: bool = True,
) -> int:
    """
    Given a player's name, get their MLBAM ID.
    """
    try:
        last_name, first_name = _parse_player_name(player_name)
        player_ids = pybaseball.playerid_lookup(
            last_name,
            first_name,
            fuzzy=fuzzy_lookup
        )

        if player_ids.empty:
            raise HTTPException(status_code=404, detail=f"no player found with name {player_name}")

        return player_ids.iloc[0]["key_mlbam"]
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _parse_player_name(name: str) -> tuple[str, str]:
    """
    Parse the given player's name: "First Last" -> ("Last", "First").
    """
    name_split = name.split(" ")
    if len(name_split) != 2:
        raise HTTPException(status_code=400, detail="player name must be in the format 'First Last'")

    return name_split[1], name_split[0]
