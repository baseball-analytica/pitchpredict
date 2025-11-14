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

        if row["pitch_number"] == 1:
            token_pa_start = PitchToken.PA_START
            pitch_tokens.append(token_pa_start)

        pitch_tokens.append(PitchToken.PITCH)

        match row["pitch_type"]:
            case "CH":
                pitch_tokens.append(PitchToken.IS_CH)
            case "CU":
                pitch_tokens.append(PitchToken.IS_CU)
            case "FC":
                pitch_tokens.append(PitchToken.IS_FC)
            case "EP":
                pitch_tokens.append(PitchToken.IS_EP)
            case "FO":
                pitch_tokens.append(PitchToken.IS_FO)
            case "FF":
                pitch_tokens.append(PitchToken.IS_FF)
            case "KN":
                pitch_tokens.append(PitchToken.IS_KN)
            case "KC":
                pitch_tokens.append(PitchToken.IS_KC)
            case "SC":
                pitch_tokens.append(PitchToken.IS_SC)
            case "SI":
                pitch_tokens.append(PitchToken.IS_SI)
            case "SL":
                pitch_tokens.append(PitchToken.IS_SL)
            case "SV":
                pitch_tokens.append(PitchToken.IS_SV)
            case "FS":
                pitch_tokens.append(PitchToken.IS_FS)
            case "ST":
                pitch_tokens.append(PitchToken.IS_ST)
            case _:
                raise ValueError(f"unknown pitch type: {row['pitch_type']}")

        speed = row["release_speed"].round(0)
        if speed < 65:
            pitch_tokens.append(PitchToken.SPEED_IS_LT65)
        elif speed > 105:
            pitch_tokens.append(PitchToken.SPEED_IS_GT105)
        else:
            token = PitchToken.SPEED_IS_65 + (speed - 65)
            pitch_tokens.append(token)

        release_pos_x = row["release_pos_x"].round(2)
        if release_pos_x < -4:
            pitch_tokens.append(PitchToken.RELEASE_POS_X_IS_LTN4)
        elif release_pos_x > 4:
            pitch_tokens.append(PitchToken.RELEASE_POS_X_IS_GT4)
        else:
            token = PitchToken.RELEASE_POS_X_IS_N4_N375 + (release_pos_x + 4)
            pitch_tokens.append(token)

        release_pos_z = row["release_pos_z"].round(2)
        if release_pos_z < -4:
            pitch_tokens.append(PitchToken.RELEASE_POS_Z_IS_LT4)
        elif release_pos_z > 4:
            pitch_tokens.append(PitchToken.RELEASE_POS_Z_IS_GT7)
        else:
            token = PitchToken.RELEASE_POS_Z_IS_4_425 + (release_pos_z - 4)
            pitch_tokens.append(token)

        plate_pos_x = row["plate_x"].round(2)
        if plate_pos_x < -2:
            pitch_tokens.append(PitchToken.PLATE_POS_X_IS_LTN2)
        elif plate_pos_x > 2:
            pitch_tokens.append(PitchToken.PLATE_POS_X_IS_GT2)
        else:
            token = PitchToken.PLATE_POS_X_IS_N2_N175 + (plate_pos_x + 2)
            pitch_tokens.append(token)

        plate_pos_z = row["plate_z"].round(2)
        if plate_pos_z < -1:
            pitch_tokens.append(PitchToken.PLATE_POS_Z_IS_LTN1)
        elif plate_pos_z > 1:
            pitch_tokens.append(PitchToken.PLATE_POS_Z_IS_GT5)
        else:
            token = PitchToken.PLATE_POS_Z_IS_N1_N075 + (plate_pos_z - 1)
            pitch_tokens.append(token)
        
        match row["description"]:
            case "ball":
                pitch_tokens.append(PitchToken.RESULT_IS_BALL)
            case "ball in dirt":
                pitch_tokens.append(PitchToken.RESULT_IS_BALL_IN_DIRT)
            case "called strike":
                pitch_tokens.append(PitchToken.RESULT_IS_CALLED_STRIKE)
            case "foul":
                pitch_tokens.append(PitchToken.RESULT_IS_FOUL)
            case "foul bunt":
                pitch_tokens.append(PitchToken.RESULT_IS_FOUL_BUNT)
            case "foul tip":
                pitch_tokens.append(PitchToken.RESULT_IS_FOUL_TIP)
            case "foul tip bunt":
                pitch_tokens.append(PitchToken.RESULT_IS_FOUL_TIP_BUNT)
            case "foul tip pitchout":
                pitch_tokens.append(PitchToken.RESULT_IS_FOUL_PITCHOUT)
            case "hit by pitch":
                pitch_tokens.append(PitchToken.RESULT_IS_HIT_BY_PITCH)
            case "intentional ball":
                pitch_tokens.append(PitchToken.RESULT_IS_INTENTIONAL_BALL)
            case "in play":
                pitch_tokens.append(PitchToken.RESULT_IS_IN_PLAY)
            case "missed bunt":
                pitch_tokens.append(PitchToken.RESULT_IS_MISSED_BUNT)
            case "pitchout":
                pitch_tokens.append(PitchToken.RESULT_IS_PITCHOUT)
            case "swinging strike":
                pitch_tokens.append(PitchToken.RESULT_IS_SWINGING_STRIKE)
            case "swinging strike blocked":
                pitch_tokens.append(PitchToken.RESULT_IS_SWINGING_STRIKE_BLOCKED)
            case _:
                raise ValueError(f"unknown result: {row['description']}")

        pitch_tokens.append(PitchToken.PITCH_END)
        
        if end_of_pa:
            pitch_tokens.append(PitchToken.PA_END)


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
