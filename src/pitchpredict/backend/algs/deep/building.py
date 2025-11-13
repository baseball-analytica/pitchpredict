# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import os

import pandas as pd
import torch

from pitchpredict.backend.algs.deep.nn import DeepPitcherModel, PitchDataset
from pitchpredict.backend.algs.deep.training import train_model
from pitchpredict.backend.fetching import get_all_pitches
from pitchpredict.backend.algs.deep.types import PitchToken, PitchContext


async def build_deep_model(
    date_start: str,
    date_end: str,
    embed_dim: int,
    hidden_size: int,
    vocab_size: int | None = None,
    num_layers: int = 1,
    bidirectional: bool = False,
    dropout: float = 0.0,
    pad_idx: int = 0,
    num_classes: int | None = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_epochs: int = 10,
    model_path: str = os.getcwd() + "/.pitchpredict_models/deep_pitch.pth",
) -> DeepPitcherModel:
    """
    Build a new deep model from scratch using the given parameters.
    """
    # get pitch data for the given date range
    pitches = await get_all_pitches(date_start, date_end)

    pitch_tokens, pitch_contexts = await _build_pitch_tokens_and_contexts(pitches)

    pitch_dataset = _build_pitch_dataset(pitch_tokens, pitch_contexts)
    train_dataset, val_dataset = torch.utils.data.random_split(pitch_dataset, [0.8, 0.2])

    input_dim = pitch_dataset.feature_dim
    dataset_num_classes = pitch_dataset.num_classes
    effective_num_classes = num_classes or vocab_size or dataset_num_classes
    if effective_num_classes < dataset_num_classes:
        raise ValueError(f"num_classes ({effective_num_classes}) must cover all pitch types ({dataset_num_classes})")

    model = DeepPitcherModel(
        input_dim=input_dim,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
        pad_idx=pad_idx,
        num_classes=effective_num_classes,
    ).to(device)

    train_model(
        model=model,
        train_data=train_dataset,
        val_data=val_dataset,
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        model_path=model_path,
        pad_id=pad_idx,
    )

    return model


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
    
    # reverse the order of the pitch tokens and contexts
    pitch_tokens = pitch_tokens[::-1]
    pitch_contexts = pitch_contexts[::-1]
        
    return pitch_tokens, pitch_contexts


def _build_pitch_dataset(
    pitch_tokens: list[PitchToken],
    pitch_contexts: list[PitchContext],
    seed: int = 0,
    pad_id: int = 0,
) -> PitchDataset:
    """
    Build the pitch dataset from the given pitch tokens and contexts.
    """
    return PitchDataset(pitch_tokens, pitch_contexts, seed, pad_id)
