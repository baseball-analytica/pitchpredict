# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import logging
from dataclasses import dataclass

import pandas as pd
from pandas._libs.missing import NAType
import torch
from tqdm import tqdm

from pitchpredict.backend.algs.deep.nn import DeepPitcherModel, PitchDataset
from pitchpredict.backend.algs.deep.training import train_model
from pitchpredict.backend.fetching import get_all_pitches
from pitchpredict.backend.algs.deep.types import PitchToken, PitchContext

logger = logging.getLogger(__name__)


@dataclass
class PitchDatasetStats:
    total_pitches: int = 0
    session_starts: int = 0
    session_ends: int = 0
    plate_appearance_starts: int = 0
    plate_appearance_ends: int = 0

    @property
    def sessions(self) -> int:
        return self.session_starts

    @property
    def plate_appearances(self) -> int:
        return self.plate_appearance_starts

    def validate(self) -> None:
        if self.session_starts != self.session_ends:
            raise ValueError(
                f"mismatched session counts: {self.session_starts} starts vs {self.session_ends} ends"
            )
        if self.plate_appearance_starts != self.plate_appearance_ends:
            raise ValueError(
                f"mismatched plate appearance counts: {self.plate_appearance_starts} starts vs {self.plate_appearance_ends} ends"
            )

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
    tokens_path: str = "./.pitchpredict_data/pitch_data.bin",
    contexts_path: str = "./.pitchpredict_data/pitch_contexts.json",
    dataset_log_interval: int = 1000
) -> DeepPitcherModel:
    """
    Build a new deep model from scratch using the given parameters.
    """
    logger.debug("build_deep_model called")

    # get pitch data for the given date range
    pitches = await get_all_pitches(date_start, date_end)
    pitches = _clean_pitch_rows(pitches)
    pitches = _sort_pitches_by_session(pitches)

    pitch_tokens, pitch_contexts, dataset_stats = await _build_pitch_tokens_and_contexts(pitches, dataset_log_interval)
    logger.info(
        "dataset includes %d sessions and %d plate appearances from %d pitch rows",
        dataset_stats.sessions,
        dataset_stats.plate_appearances,
        dataset_stats.total_pitches,
    )

    pitch_dataset = _build_pitch_dataset(
        pitch_tokens=pitch_tokens,
        pitch_contexts=pitch_contexts,
        seed=0,
        pad_id=pad_idx,
        tokens_path=tokens_path,
        contexts_path=contexts_path,
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
) -> tuple[list[PitchToken], list[PitchContext], PitchDatasetStats]:
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
    stats = PitchDatasetStats()
    current_session: tuple[int, int] | None = None
    last_pa_key: tuple[int, int] | None = None
    last_pa_closed = True
    last_context: PitchContext | None = None

    pitches = pitches.sort_values(by=["game_pk", "at_bat_number", "pitch_number"])

    for i in tqdm(range(pitches.shape[0]), total=pitches.shape[0], desc="tokenizing pitches"):
        row = pitches.iloc[i]

        if not row["pitch_type"]:
            continue

        stats.total_pitches += 1

        session_key = (int(row["game_pk"]), int(row["pitcher"]))
        if session_key != current_session:
            if current_session is not None:
                stats.session_ends += 1
            stats.session_starts += 1
            current_session = session_key

        tokens_this_pitch: list[PitchToken] = []

        pa_key = (int(row["game_pk"]), int(row["at_bat_number"]))
        if pa_key != last_pa_key:
            if last_pa_key is not None and not last_pa_closed and last_context is not None:
                pitch_tokens.append(PitchToken.PA_END)
                pitch_contexts.append(last_context)
                stats.plate_appearance_ends += 1
                last_pa_closed = True
            stats.plate_appearance_starts += 1
            tokens_this_pitch.append(PitchToken.PA_START)
            last_pa_closed = False

        raw_event = row["events"]
        event = raw_event if isinstance(raw_event, str) else ""
        end_of_pa = bool(event)

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
            case "IN":
                tokens_this_pitch.append(PitchToken.IS_IN)
            case "AB":
                tokens_this_pitch.append(PitchToken.IS_AB)
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

        spin_rate = round(row["release_spin_rate"])
        if spin_rate < 750:
            tokens_this_pitch.append(PitchToken.SPIN_RATE_IS_LT750)
        elif spin_rate > 3250:
            tokens_this_pitch.append(PitchToken.SPIN_RATE_IS_GT3250)
        else:
            offset = _bin_index(spin_rate, base=750.0, step=250.0, max_offset=9)
            token = _offset_token(PitchToken.SPIN_RATE_IS_750_1000, offset, max_offset=9)
            tokens_this_pitch.append(token)

        spin_axis = round(row["spin_axis"])
        if spin_axis < 0:
            raise ValueError(f"spin axis out of range: {spin_axis}")
        elif spin_axis > 360:
            raise ValueError(f"spin axis out of range: {spin_axis}")
        else:
            offset = min(11, int(spin_axis // 30))
            token = _offset_token(PitchToken.SPIN_AXIS_IS_0_30, offset, max_offset=11)
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

        vx0 = round(row["vx0"], 2)
        if vx0 < -15.0:
            tokens_this_pitch.append(PitchToken.VX0_IS_LTN15)
        elif vx0 > 15.0:
            tokens_this_pitch.append(PitchToken.VX0_IS_GT15)
        else:
            offset = _bin_index(vx0, base=-15.0, step=5.0, max_offset=5)
            token = _offset_token(PitchToken.VX0_IS_N15_N10, offset, max_offset=5)
            tokens_this_pitch.append(token)

        vy0 = round(row["vy0"], 2)
        if vy0 < -150.0:
            tokens_this_pitch.append(PitchToken.VY0_IS_LTN150)
        elif vy0 > -100.0:
            tokens_this_pitch.append(PitchToken.VY0_IS_GTN100)
        else:
            offset = _bin_index(vy0, base=-150.0, step=10.0, max_offset=4)
            token = _offset_token(PitchToken.VY0_IS_N150_N140, offset, max_offset=4)
            tokens_this_pitch.append(token)

        vz0 = round(row["vz0"], 2)
        if vz0 < -10.0:
            tokens_this_pitch.append(PitchToken.VZ0_IS_LTN10)
        elif vz0 > 15.0:
            tokens_this_pitch.append(PitchToken.VZ0_IS_GT15)
        else:
            offset = _bin_index(vz0, base=-10.0, step=5.0, max_offset=4)
            token = _offset_token(PitchToken.VZ0_IS_N10_N5, offset, max_offset=4)
            tokens_this_pitch.append(token)

        ax = round(row["ax"], 2)
        if ax < -25.0:
            tokens_this_pitch.append(PitchToken.AX_IS_LTN25)
        elif ax > 25.0:
            tokens_this_pitch.append(PitchToken.AX_IS_GT25)
        else:
            offset = _bin_index(ax, base=-25.0, step=5.0, max_offset=9)
            token = _offset_token(PitchToken.AX_IS_N25_N20, offset, max_offset=9)
            tokens_this_pitch.append(token)

        ay = round(row["ay"], 2)
        if ay < 15.0:
            tokens_this_pitch.append(PitchToken.AY_IS_LT15)
        elif ay > 40.0:
            tokens_this_pitch.append(PitchToken.AY_IS_GT40)
        else:
            offset = _bin_index(ay, base=15.0, step=5.0, max_offset=4)
            token = _offset_token(PitchToken.AY_IS_15_20, offset, max_offset=4)
            tokens_this_pitch.append(token)

        az = round(row["az"], 2)
        if az < -45.0:
            tokens_this_pitch.append(PitchToken.AZ_IS_LTN45)
        elif az > -15.0:
            tokens_this_pitch.append(PitchToken.AZ_IS_GTN15)
        else:
            offset = _bin_index(az, base=-45.0, step=5.0, max_offset=5)
            token = _offset_token(PitchToken.AZ_IS_N45_N40, offset, max_offset=5)
            tokens_this_pitch.append(token)

        release_extension = round(row["release_extension"], 2)
        if release_extension < 5.0:
            tokens_this_pitch.append(PitchToken.RELEASE_EXTENSION_IS_LT5)
        elif release_extension > 7.5:
            tokens_this_pitch.append(PitchToken.RELEASE_EXTENSION_IS_GT75)
        else:
            offset = _bin_index(release_extension, base=5.0, step=0.5, max_offset=4)
            token = _offset_token(PitchToken.RELEASE_EXTENSION_IS_5_55, offset, max_offset=4)
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
            case "intentional_ball" | "intent_ball":
                tokens_this_pitch.append(PitchToken.RESULT_IS_INTENTIONAL_BALL)
            case "hit_into_play":
                tokens_this_pitch.append(PitchToken.RESULT_IS_IN_PLAY)
            case "missed_bunt":
                tokens_this_pitch.append(PitchToken.RESULT_IS_MISSED_BUNT)
            case "pitchout":
                tokens_this_pitch.append(PitchToken.RESULT_IS_PITCHOUT)
            case "swinging_pitchout":
                tokens_this_pitch.append(PitchToken.RESULT_IS_SWINGING_PITCHOUT)
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
            stats.plate_appearance_ends += 1
            last_pa_closed = True


        runner_on_first = not isinstance(row["on_1b"], NAType) or False
        runner_on_second = not isinstance(row["on_2b"], NAType) or False
        runner_on_third = not isinstance(row["on_3b"], NAType) or False
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
            game_park_id=row["game_pk"],
            fielder_2_id=row["fielder_2"],
            fielder_3_id=row["fielder_3"],
            fielder_4_id=row["fielder_4"],
            fielder_5_id=row["fielder_5"],
            fielder_6_id=row["fielder_6"],
            fielder_7_id=row["fielder_7"],
            fielder_8_id=row["fielder_8"],
            fielder_9_id=row["fielder_9"],
            batter_days_since_prev_game=row["batter_days_since_prev_game"],
            pitcher_days_since_prev_game=row["pitcher_days_since_prev_game"],
            umpire_id=row["umpire"],
            strike_zone_top=row["sz_top"],
            strike_zone_bottom=row["sz_bot"],
        )
        pitch_tokens.extend(tokens_this_pitch)
        pitch_contexts.extend([pitch_context] * len(tokens_this_pitch))
        last_context = pitch_context
        last_pa_key = pa_key

    if current_session is not None:
        stats.session_ends += 1

    if not last_pa_closed and last_context is not None:
        pitch_tokens.append(PitchToken.PA_END)
        pitch_contexts.append(last_context)
        stats.plate_appearance_ends += 1

    stats.validate()
    logger.info(
        "tokenized %d pitches across %d sessions and %d plate appearances",
        stats.total_pitches,
        stats.sessions,
        stats.plate_appearances,
    )
    logger.info("_build_pitch_tokens_and_contexts completed successfully")
    return pitch_tokens, pitch_contexts, stats


def _build_pitch_dataset(
    pitch_tokens: list[PitchToken],
    pitch_contexts: list[PitchContext],
    seed: int = 0,
    pad_id: int = 0,
    tokens_path: str = "./.pitchpredict_data/pitch_data.bin",
    contexts_path: str = "./.pitchpredict_data/pitch_contexts.json",
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
    dataset.save(tokens_path, contexts_path)

    logger.info("_build_pitch_dataset completed successfully")
    return dataset


def _sort_pitches_by_session(pitches: pd.DataFrame) -> pd.DataFrame:
    """
    Group pitches by (game, pitcher) sessions and order pitches chronologically inside each session.
    """
    logger.debug("_sort_pitches_by_session called")

    if pitches.empty:
        return pitches

    required_columns = ["game_pk", "pitcher", "at_bat_number", "pitch_number", "game_date"]
    missing = [col for col in required_columns if col not in pitches.columns]
    if missing:
        raise ValueError(f"missing required columns for session sorting: {missing}")

    session_start = pitches.groupby(["game_pk", "pitcher"])["at_bat_number"].transform("min")

    sorted_pitches = (
        pitches.assign(_session_start=session_start)
        .sort_values(
            by=["game_date", "game_pk", "_session_start", "pitcher", "at_bat_number", "pitch_number"],
            kind="mergesort",
        )
        .drop(columns="_session_start")
        .reset_index(drop=True)
    )

    logger.debug("_sort_pitches_by_session completed successfully")
    return sorted_pitches


def _clean_pitch_rows(pitches: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that are missing fields required by the deep model.
    """
    logger.debug("_clean_pitch_rows called")

    pitches = pitches.copy()

    required_columns = [
        "pitch_type",
        "release_speed",
        "release_pos_x",
        "release_pos_z",
        "release_spin_rate",
        "spin_axis",
        "plate_x",
        "plate_z",
        "vx0",
        "vy0",
        "vz0",
        "ax",
        "ay",
        "az",
        "release_extension",
        "game_pk",
        "fielder_2",
        "fielder_3",
        "fielder_4",
        "fielder_5",
        "fielder_6",
        "fielder_7",
        "fielder_8",
        "fielder_9",
        "umpire",
        "batter_days_since_prev_game",
        "pitcher_days_since_prev_game",
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
        "sz_top",
        "sz_bot",
    ]
    missing = [col for col in required_columns if col not in pitches.columns]
    if missing:
        raise ValueError(f"missing required columns for cleaning: {missing}")

    fill_zero_int_columns = [
        "fielder_2",
        "fielder_3",
        "fielder_4",
        "fielder_5",
        "fielder_6",
        "fielder_7",
        "fielder_8",
        "fielder_9",
        "umpire",
        "batter_days_since_prev_game",
        "pitcher_days_since_prev_game",
    ]
    fill_zero_float_columns = ["sz_top", "sz_bot"]

    essential_columns = [
        col for col in required_columns if col not in fill_zero_int_columns + fill_zero_float_columns
    ]
    cleaned = pitches.dropna(subset=essential_columns).copy()

    for col in fill_zero_int_columns:
        cleaned[col] = cleaned[col].fillna(0).astype(int)

    for col in fill_zero_float_columns:
        cleaned[col] = cleaned[col].fillna(0.0)

    logger.debug("_clean_pitch_rows completed successfully")
    return cleaned.reset_index(drop=True)
