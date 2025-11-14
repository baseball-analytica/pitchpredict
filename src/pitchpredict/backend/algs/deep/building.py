# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import os

import pandas as pd
import torch

from pitchpredict.backend.algs.deep.nn import DeepPitcherModel, PitchDataset
from pitchpredict.backend.algs.deep.training import train_model
from pitchpredict.backend.fetching import get_all_pitches
from pitchpredict.backend.algs.deep.types import PitchToken, PitchContext, PitchTokenType


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
    model_path: str = "./.pitchpredict_models/deep_pitch.pth",
) -> DeepPitcherModel:
    """
    Build a new deep model from scratch using the given parameters.
    """
    # get pitch data for the given date range
    pitches = await get_all_pitches(date_start, date_end)
    pitches = _clean_pitch_rows(pitches)

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

    pitches = pitches.sort_values(by=["game_pk", "at_bat_number", "pitch_number"])

    for index, row in pitches.iterrows():
        if not row["pitch_type"]:
            continue

        raw_event = row["events"]
        event = raw_event if isinstance(raw_event, str) else ""
        end_of_pa = bool(event)
        
        token_pitch = PitchToken(
            type=PitchTokenType.PITCH,
            value=0.0,
        )
        pitch_tokens.append(token_pitch)

        token_is_CH = PitchToken(
            type=PitchTokenType.IS_CH,
            value=1.0 if row["pitch_type"] == "CH" else -1.0,
        )
        pitch_tokens.append(token_is_CH)

        token_is_CU = PitchToken(
            type=PitchTokenType.IS_CU,
            value=1.0 if row["pitch_type"] == "CU" else -1.0,
        )
        pitch_tokens.append(token_is_CU)

        token_is_FC = PitchToken(
            type=PitchTokenType.IS_FC,
            value=1.0 if row["pitch_type"] == "FC" else -1.0,
        )
        pitch_tokens.append(token_is_FC)

        token_is_EP = PitchToken(
            type=PitchTokenType.IS_EP,
            value=1.0 if row["pitch_type"] == "EP" else -1.0,
        )
        pitch_tokens.append(token_is_EP)

        token_is_FO = PitchToken(
            type=PitchTokenType.IS_FO,
            value=1.0 if row["pitch_type"] == "FO" else -1.0,
        )
        pitch_tokens.append(token_is_FO)

        token_is_FF = PitchToken(
            type=PitchTokenType.IS_FF,
            value=1.0 if row["pitch_type"] == "FF" else -1.0,
        )
        pitch_tokens.append(token_is_FF)

        token_is_KN = PitchToken(
            type=PitchTokenType.IS_KN,
            value=1.0 if row["pitch_type"] == "KN" else -1.0,
        )
        pitch_tokens.append(token_is_KN)

        token_is_KC = PitchToken(
            type=PitchTokenType.IS_KC,
            value=1.0 if row["pitch_type"] == "KC" else -1.0,
        )
        pitch_tokens.append(token_is_KC)

        token_is_SC = PitchToken(
            type=PitchTokenType.IS_SC,
            value=1.0 if row["pitch_type"] == "SC" else -1.0,
        )
        pitch_tokens.append(token_is_SC)

        token_is_SI = PitchToken(
            type=PitchTokenType.IS_SI,
            value=1.0 if row["pitch_type"] == "SI" else -1.0,
        )
        pitch_tokens.append(token_is_SI)

        token_is_SL = PitchToken(
            type=PitchTokenType.IS_SL,
            value=1.0 if row["pitch_type"] == "SL" else -1.0,
        )
        pitch_tokens.append(token_is_SL)

        token_is_SV = PitchToken(
            type=PitchTokenType.IS_SV,
            value=1.0 if row["pitch_type"] == "SV" else -1.0,
        )
        pitch_tokens.append(token_is_SV)

        token_is_FS = PitchToken(
            type=PitchTokenType.IS_FS,
            value=1.0 if row["pitch_type"] == "FS" else -1.0,
        )
        pitch_tokens.append(token_is_FS)

        token_is_ST = PitchToken(
            type=PitchTokenType.IS_ST,
            value=1.0 if row["pitch_type"] == "ST" else -1.0,
        )
        pitch_tokens.append(token_is_ST)

        token_speed = PitchToken(
            type=PitchTokenType.SPEED,
            value=row["release_speed"],
        )
        pitch_tokens.append(token_speed)

        token_release_pos_x = PitchToken(
            type=PitchTokenType.RELEASE_POS_X,
            value=row["release_pos_x"],
        )
        pitch_tokens.append(token_release_pos_x)

        token_release_pos_z = PitchToken(
            type=PitchTokenType.RELEASE_POS_Z,
            value=row["release_pos_z"],
        )
        pitch_tokens.append(token_release_pos_z)

        token_plate_pos_x = PitchToken(
            type=PitchTokenType.PLATE_POS_X,
            value=row["plate_x"],
        )
        pitch_tokens.append(token_plate_pos_x)

        token_plate_pos_z = PitchToken(
            type=PitchTokenType.PLATE_POS_Z,
            value=row["plate_z"],
        )
        pitch_tokens.append(token_plate_pos_z)

        token_event_is_none = PitchToken(
            type=PitchTokenType.EVENT_IS_NONE,
            value=1.0 if event is None else -1.0,
        )
        pitch_tokens.append(token_event_is_none)

        token_pitch_end = PitchToken(
            type=PitchTokenType.PITCH_END,
            value=0.0,
        )
        pitch_tokens.append(token_pitch_end)

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


def _clean_pitch_rows(pitches: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that are missing fields required by the deep model.
    """
    required_columns = [
        "pitch_type",
        "release_speed",
        "release_pos_x",
        "release_pos_z",
        "plate_x",
        "plate_z",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "p_throws",
        "stand",
        "balls",
        "strikes",
        "outs_when_up",
        "bat_score",
        "fld_score",
        "inning",
        "pitch_number",
        "n_thruorder_pitcher",
        "game_date",
        "age_pit",
        "age_bat",
    ]
    cleaned = pitches.dropna(subset=required_columns)
    return cleaned.reset_index(drop=True)
