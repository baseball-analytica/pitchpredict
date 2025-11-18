# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import logging

import pandas as pd
import torch
from tqdm import tqdm

from pitchpredict.backend.algs.deep.nn import DeepPitcherModel, PitchDataset
from pitchpredict.backend.algs.deep.training import train_model
from pitchpredict.backend.fetching import get_all_pitches
from pitchpredict.backend.algs.deep.types import PitchToken, PitchContext

logger = logging.getLogger(__name__)

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
    dataset_path: str = "./.pitchpredict_data/pitch_data.bin",
    dataset_log_interval: int = 1000
) -> DeepPitcherModel:
    """
    Build a new deep model from scratch using the given parameters.
    """
    logger.debug("build_deep_model called")

    # get pitch data for the given date range
    pitches = await get_all_pitches(date_start, date_end)
    pitches = _clean_pitch_rows(pitches)

    pitch_tokens, pitch_contexts = await _build_pitch_tokens_and_contexts(pitches, dataset_log_interval)

    pitch_dataset = _build_pitch_dataset(
        pitch_tokens=pitch_tokens,
        pitch_contexts=pitch_contexts,
        seed=0,
        pad_id=pad_idx,
        dataset_path=dataset_path,
        dataset_log_interval=dataset_log_interval,
    )
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

    logger.info("build_deep_model completed successfully")

    return model


async def build_deep_model_from_dataset(
    tokens_path: str,
    contexts_path: str,
    embed_dim: int,
    hidden_size: int,
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
    Build a new deep model from a pre-existing dataset.
    """
    logger.debug("build_deep_model_from_dataset called")

    pitch_dataset = PitchDataset.load(tokens_path, contexts_path, seed=0, pad_id=0, dataset_log_interval=10000)
    train_dataset, val_dataset = torch.utils.data.random_split(pitch_dataset, [0.8, 0.2])

    input_dim = pitch_dataset.feature_dim
    num_classes = pitch_dataset.num_classes

    model = DeepPitcherModel(
        input_dim=input_dim,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
        pad_idx=pad_idx,
        num_classes=num_classes,
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

    logger.info("build_deep_model_from_dataset completed successfully")
    return model


async def _build_pitch_tokens_and_contexts(
    pitches: pd.DataFrame,
    dataset_log_interval: int = 1000,
) -> tuple[list[PitchToken], list[PitchContext]]:
    """
    Build the pitch tokens and contexts from the given pitches.
    """
    logger.debug("_build_pitch_tokens_and_contexts called")

    def _offset_token(base: PitchToken, offset: int, max_offset: int) -> PitchToken:
        clamped = max(0, min(max_offset, offset))
        return PitchToken(base.value + clamped)

    def _bin_index(value: float, base: float, step: float, max_offset: int) -> int:
        raw = round((value - base) / step)
        return max(0, min(max_offset, int(raw)))

    pitch_tokens = []
    pitch_contexts = []

    pitches = pitches.sort_values(by=["game_pk", "at_bat_number", "pitch_number"])

    for i in tqdm(range(pitches.shape[0]), total=pitches.shape[0], desc="tokenizing pitches"):
        row = pitches.iloc[i]

        if not row["pitch_type"]:
            continue

        tokens_this_pitch: list[PitchToken] = []

        raw_event = row["events"]
        event = raw_event if isinstance(raw_event, str) else ""
        end_of_pa = bool(event)

        if row["pitch_number"] == 1:
            token_pa_start = PitchToken.PA_START
            tokens_this_pitch.append(token_pa_start)

        tokens_this_pitch.append(PitchToken.PITCH)

        match row["pitch_type"]:
            case "CH":
                tokens_this_pitch.append(PitchToken.IS_CH)
            case "CU":
                tokens_this_pitch.append(PitchToken.IS_CU)
            case "FC":
                tokens_this_pitch.append(PitchToken.IS_FC)
            case "EP":
                tokens_this_pitch.append(PitchToken.IS_EP)
            case "FO":
                tokens_this_pitch.append(PitchToken.IS_FO)
            case "FF":
                tokens_this_pitch.append(PitchToken.IS_FF)
            case "KN":
                tokens_this_pitch.append(PitchToken.IS_KN)
            case "KC":
                tokens_this_pitch.append(PitchToken.IS_KC)
            case "SC":
                tokens_this_pitch.append(PitchToken.IS_SC)
            case "SI":
                tokens_this_pitch.append(PitchToken.IS_SI)
            case "SL":
                tokens_this_pitch.append(PitchToken.IS_SL)
            case "SV":
                tokens_this_pitch.append(PitchToken.IS_SV)
            case "FS":
                tokens_this_pitch.append(PitchToken.IS_FS)
            case "ST":
                tokens_this_pitch.append(PitchToken.IS_ST)
            case "FA":
                tokens_this_pitch.append(PitchToken.IS_FA)
            case "CS":
                tokens_this_pitch.append(PitchToken.IS_CS)
            case "PO":
                tokens_this_pitch.append(PitchToken.IS_PO)
            case "UN":
                tokens_this_pitch.append(PitchToken.IS_UN)
            case _:
                raise ValueError(f"unknown pitch type: {row['pitch_type']}")

        speed = round(row["release_speed"])
        if speed < 65:
            tokens_this_pitch.append(PitchToken.SPEED_IS_LT65)
        elif speed > 105:
            tokens_this_pitch.append(PitchToken.SPEED_IS_GT105)
        else:
            offset = int(speed - 65)
            token = _offset_token(PitchToken.SPEED_IS_65, offset, max_offset=40)
            tokens_this_pitch.append(token)

        release_pos_x = round(row["release_pos_x"], 2)
        if release_pos_x < -4:
            tokens_this_pitch.append(PitchToken.RELEASE_POS_X_IS_LTN4)
        elif release_pos_x > 4:
            tokens_this_pitch.append(PitchToken.RELEASE_POS_X_IS_GT4)
        else:
            offset = _bin_index(release_pos_x, base=-4.0, step=0.25, max_offset=31)
            token = _offset_token(PitchToken.RELEASE_POS_X_IS_N4_N375, offset, max_offset=31)
            tokens_this_pitch.append(token)

        release_pos_z = round(row["release_pos_z"], 2)
        if release_pos_z < 4:
            tokens_this_pitch.append(PitchToken.RELEASE_POS_Z_IS_LT4)
        elif release_pos_z > 7:
            tokens_this_pitch.append(PitchToken.RELEASE_POS_Z_IS_GT7)
        else:
            offset = _bin_index(release_pos_z, base=4.0, step=0.25, max_offset=11)
            token = _offset_token(PitchToken.RELEASE_POS_Z_IS_4_425, offset, max_offset=11)
            tokens_this_pitch.append(token)

        plate_pos_x = round(row["plate_x"], 2)
        if plate_pos_x < -2:
            tokens_this_pitch.append(PitchToken.PLATE_POS_X_IS_LTN2)
        elif plate_pos_x > 2:
            tokens_this_pitch.append(PitchToken.PLATE_POS_X_IS_GT2)
        else:
            offset = _bin_index(plate_pos_x, base=-2.0, step=0.25, max_offset=15)
            token = _offset_token(PitchToken.PLATE_POS_X_IS_N2_N175, offset, max_offset=15)
            tokens_this_pitch.append(token)

        plate_pos_z = round(row["plate_z"], 2)
        if plate_pos_z < -1:
            tokens_this_pitch.append(PitchToken.PLATE_POS_Z_IS_LTN1)
        elif plate_pos_z > 5:
            tokens_this_pitch.append(PitchToken.PLATE_POS_Z_IS_GT5)
        else:
            offset = _bin_index(plate_pos_z, base=-1.0, step=0.25, max_offset=23)
            token = _offset_token(PitchToken.PLATE_POS_Z_IS_N1_N075, offset, max_offset=23)
            tokens_this_pitch.append(token)
        
        match row["description"]:
            case "ball":
                tokens_this_pitch.append(PitchToken.RESULT_IS_BALL)
            case "ball_in_dirt":
                tokens_this_pitch.append(PitchToken.RESULT_IS_BALL_IN_DIRT)
            case "called_strike":
                tokens_this_pitch.append(PitchToken.RESULT_IS_CALLED_STRIKE)
            case "foul":
                tokens_this_pitch.append(PitchToken.RESULT_IS_FOUL)
            case "foul_bunt":
                tokens_this_pitch.append(PitchToken.RESULT_IS_FOUL_BUNT)
            case "foul_tip":
                tokens_this_pitch.append(PitchToken.RESULT_IS_FOUL_TIP)
            case "bunt_foul_tip":
                tokens_this_pitch.append(PitchToken.RESULT_IS_FOUL_TIP_BUNT)
            case "foul_pitchout":
                tokens_this_pitch.append(PitchToken.RESULT_IS_FOUL_PITCHOUT)
            case "hit_by_pitch":
                tokens_this_pitch.append(PitchToken.RESULT_IS_HIT_BY_PITCH)
            case "intentional_ball":
                tokens_this_pitch.append(PitchToken.RESULT_IS_INTENTIONAL_BALL)
            case "hit_into_play":
                tokens_this_pitch.append(PitchToken.RESULT_IS_IN_PLAY)
            case "missed_bunt":
                tokens_this_pitch.append(PitchToken.RESULT_IS_MISSED_BUNT)
            case "pitchout":
                tokens_this_pitch.append(PitchToken.RESULT_IS_PITCHOUT)
            case "swinging_strike":
                tokens_this_pitch.append(PitchToken.RESULT_IS_SWINGING_STRIKE)
            case "swinging_strike_blocked":
                tokens_this_pitch.append(PitchToken.RESULT_IS_SWINGING_STRIKE_BLOCKED)
            case "blocked_ball":
                tokens_this_pitch.append(PitchToken.RESULT_IS_BLOCKED_BALL)
            case "automatic_ball":
                tokens_this_pitch.append(PitchToken.RESULT_IS_AUTOMATIC_BALL)
            case "automatic_strike":
                tokens_this_pitch.append(PitchToken.RESULT_IS_AUTOMATIC_STRIKE)
            case _:
                raise ValueError(f"unknown result: {row['description']}")
        
        if end_of_pa:
            tokens_this_pitch.append(PitchToken.PA_END)


        runner_on_first = row["on_1b"] is not None or False
        runner_on_second = row["on_2b"] is not None or False
        runner_on_third = row["on_3b"] is not None or False
        match (runner_on_first, runner_on_second, runner_on_third):
            case (False, False, False):
                bases_state = 0
            case (True, False, False):
                bases_state = 1
            case (False, True, False):
                bases_state = 2
            case (True, True, False):
                bases_state = 3
            case (False, False, True):
                bases_state = 4
            case (True, False, True):
                bases_state = 5
            case (False, True, True):
                bases_state = 6
            case (True, True, True):
                bases_state = 7
            case _:
                raise ValueError(f"unknown bases state: {runner_on_first}, {runner_on_second}, {runner_on_third}")

        game_date = row["game_date"].strftime("%Y-%m-%d")

        pitch_context = PitchContext(
            pitcher_id=row["pitcher"],
            batter_id=row["batter"],
            pitcher_age=row["age_pit"],
            pitcher_throws=row["p_throws"],
            batter_age=row["age_bat"],
            batter_hits=row["stand"],
            count_balls=row["balls"],
            count_strikes=row["strikes"],
            outs=row["outs_when_up"],
            bases_state=bases_state,
            score_bat=row["bat_score"],
            score_fld=row["fld_score"],
            inning=row["inning"],
            pitch_number=row["pitch_number"],
            number_through_order=row["n_thruorder_pitcher"],
            game_date=game_date,
        )
        pitch_tokens.extend(tokens_this_pitch)
        pitch_contexts.extend([pitch_context] * len(tokens_this_pitch))
    
    # reverse the order of the pitch tokens and contexts
    pitch_tokens = pitch_tokens[::-1]
    pitch_contexts = pitch_contexts[::-1]

    logger.info("_build_pitch_tokens_and_contexts completed successfully")
    return pitch_tokens, pitch_contexts


def _build_pitch_dataset(
    pitch_tokens: list[PitchToken],
    pitch_contexts: list[PitchContext],
    seed: int = 0,
    pad_id: int = 0,
    dataset_path: str = "./.pitchpredict_data/pitch_data.bin",
    dataset_log_interval: int = 10000,
) -> PitchDataset:
    """
    Build the pitch dataset from the given pitch tokens and contexts.
    """
    logger.debug("_build_pitch_dataset called")

    dataset = PitchDataset(
        pitch_tokens=pitch_tokens,
        pitch_contexts=pitch_contexts,
        seed=seed,
        pad_id=pad_id,
        dataset_log_interval=dataset_log_interval,
    )
    dataset.save(dataset_path)

    logger.info("_build_pitch_dataset completed successfully")
    return dataset


def _clean_pitch_rows(pitches: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that are missing fields required by the deep model.
    """
    logger.debug("_clean_pitch_rows called")

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

    logger.debug("_clean_pitch_rows completed successfully")
    return cleaned.reset_index(drop=True)
