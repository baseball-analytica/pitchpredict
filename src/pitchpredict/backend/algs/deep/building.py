# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from typing import Any

import pandas as pd
import torch

from pitchpredict.backend.fetching import get_all_pitches
from pitchpredict.types.deep import PitchToken, PitchContext


async def build_deep_model(
    date_start: str,
    date_end: str,
) -> torch.nn.RNN:
    """
    Build a new deep model from scratch using the given parameters.
    """
    # get pitch data for the given date range
    pitches = await get_all_pitches(date_start, date_end)

    pitch_tokens, pitch_contexts = await _build_pitch_tokens_and_contexts(pitches)

    print(f"built {len(pitch_tokens)} pitch tokens and {len(pitch_contexts)} pitch contexts")
    print(pitch_tokens[0].to_tensor())
    print(pitch_contexts[0].to_tensor())

    raise NotImplementedError("Not implemented")


async def _build_pitch_tokens_and_contexts(
    pitches: pd.DataFrame,
) -> tuple[list[PitchToken], list[PitchContext]]:
    """
    Build the pitch tokens and contexts from the given pitches.
    """
    pitch_tokens = []
    pitch_contexts = []

    for index, row in pitches.iterrows():
        event = str(row["events"])
        end_of_pa = event != "" or False
        
        pitch_token = PitchToken(
            type=row["pitch_type"],
            speed=row["release_speed"],
            release_pos_x=row["release_pos_x"],
            release_pos_z=row["release_pos_z"],
            plate_pos_x=row["plate_x"],
            plate_pos_z=row["plate_z"],
            event=event,
            end_of_pa=end_of_pa,
        )
        pitch_tokens.append(pitch_token)

        runner_on_first = row["on_1b"] is not None or False
        runner_on_second = row["on_2b"] is not None or False
        runner_on_third = row["on_3b"] is not None or False
        game_date = row["game_date"].strftime("%Y-%m-%d")

        pitch_context = PitchContext(
            pitcher_age=row["age_pit"],
            pitcher_throws=row["p_throws"],
            batter_age=row["age_bat"],
            batter_hits=row["stand"],
            count_balls=row["balls"],
            count_strikes=row["strikes"],
            outs=row["outs_when_up"],
            runner_on_first=runner_on_first,
            runner_on_second=runner_on_second,
            runner_on_third=runner_on_third,
            score_bat=row["bat_score"],
            score_fld=row["fld_score"],
            inning=row["inning"],
            pitch_number=row["pitch_number"],
            number_through_order=row["n_thruorder_pitcher"],
            game_date=game_date,
        )
        pitch_contexts.append(pitch_context)
        
    return pitch_tokens, pitch_contexts