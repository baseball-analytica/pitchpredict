# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import logging
from pathlib import Path

import pandas as pd

from pitchpredict.backend.fetching import get_all_pitches
from tools.deep.vectorized import (
    VectorizedDatasetStats,
    build_tokens_vectorized,
    write_context_files_vectorized,
    write_tokens_file_vectorized,
)

logger = logging.getLogger(__name__)


async def build_dataset_vectorized(
    date_start: str | None = None,
    date_end: str | None = None,
    tokens_path: str = "./.pitchpredict_data/pitch_seq.bin",
    contexts_path: str = "./.pitchpredict_data/pitch_context",
    split_val_ratio: float = 0.01,
    split_test_ratio: float = 0.01,
    max_workers: int = 8,
    pitches_parquet: str | None = None,
    pitches_df: pd.DataFrame | None = None,
) -> VectorizedDatasetStats:
    """
    Build dataset using the optimized vectorized pipeline.

    This is 50-100x faster than the original row-by-row approach:
    - Tier 1: Vectorized tokenization using numpy
    - Tier 2: Vectorized context encoding
    - Tier 3: Parallel file I/O

    Args:
        date_start: Start date for pitch data (YYYY-MM-DD). Ignored if pitches provided.
        date_end: End date for pitch data (YYYY-MM-DD). Ignored if pitches provided.
        tokens_path: Path for output token file
        contexts_path: Directory/prefix for context files
        split_val_ratio: Fraction of data for validation (0.0 to skip)
        split_test_ratio: Fraction of data for test (0.0 to skip)
        max_workers: Number of parallel threads for file I/O
        pitches_parquet: Path to pre-fetched pitches parquet file (faster than API)
        pitches_df: Pre-loaded DataFrame of pitches (fastest, skips I/O)

    Returns:
        Statistics from dataset generation
    """
    import time
    from tools.deep.split import split_saved_dataset

    logger.info("build_dataset_vectorized called")
    start_time = time.time()

    # Step 1: Get pitch data (from DataFrame, parquet, or API)
    fetch_start = time.time()
    if pitches_df is not None:
        logger.info("Using provided DataFrame with %d pitches", len(pitches_df))
        pitches = pitches_df
    elif pitches_parquet is not None:
        logger.info("Loading pitches from parquet: %s", pitches_parquet)
        pitches = pd.read_parquet(pitches_parquet)
        logger.info("Loaded %d pitches in %.1fs", len(pitches), time.time() - fetch_start)
    elif date_start is not None and date_end is not None:
        logger.info("Fetching pitch data from %s to %s...", date_start, date_end)
        pitches = await get_all_pitches(date_start, date_end)
        logger.info("Fetched %d pitches in %.1fs", len(pitches), time.time() - fetch_start)
    else:
        raise ValueError("Must provide either (date_start, date_end), pitches_parquet, or pitches_df")

    # Step 2: Clean and sort
    logger.info("Cleaning pitch data...")
    clean_start = time.time()
    pitches = _clean_pitch_rows(pitches)
    logger.info("Cleaned to %d pitches in %.1fs", len(pitches), time.time() - clean_start)

    logger.info("Sorting pitches by session...")
    sort_start = time.time()
    pitches = _sort_pitches_by_session(pitches)
    logger.info("Sorted in %.1fs", time.time() - sort_start)

    # Step 3: Vectorized tokenization
    logger.info("Building tokens (vectorized)...")
    token_start = time.time()
    tokens, token_to_pitch, stats = build_tokens_vectorized(pitches)
    logger.info("Built %d tokens in %.1fs", len(tokens), time.time() - token_start)

    # Step 4: Write tokens file
    tokens_path_obj = Path(tokens_path)
    tokens_path_obj.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Writing tokens file...")
    write_tokens_file_vectorized(tokens, tokens_path_obj)

    # Step 5: Write context files (vectorized + parallel)
    contexts_path_obj = Path(contexts_path)
    context_dir = contexts_path_obj.parent
    context_prefix = contexts_path_obj.name

    logger.info("Writing context files (parallel)...")
    ctx_start = time.time()
    write_context_files_vectorized(
        pitches, token_to_pitch, context_dir, context_prefix, max_workers
    )
    logger.info("Wrote context files in %.1fs", time.time() - ctx_start)

    # Step 6: Split if requested
    if split_val_ratio > 0 or split_test_ratio > 0:
        logger.info(
            "Splitting dataset (val=%.1f%%, test=%.1f%%)...",
            split_val_ratio * 100,
            split_test_ratio * 100,
        )
        split_start = time.time()
        split_saved_dataset(
            tokens_path=tokens_path_obj,
            context_prefix=contexts_path_obj,
            val_ratio=split_val_ratio,
            test_ratio=split_test_ratio,
            overwrite=True,
        )
        logger.info("Split completed in %.1fs", time.time() - split_start)

    total_time = time.time() - start_time
    logger.info(
        "build_dataset_vectorized completed in %.1fs: %d tokens from %d pitches (%d sessions, %d PAs)",
        total_time,
        stats.total_tokens,
        stats.total_pitches,
        stats.session_count,
        stats.pa_count,
    )

    return stats


def _sort_pitches_by_session(pitches: pd.DataFrame) -> pd.DataFrame:
    """
    Group pitches by (game, pitcher) sessions and order pitches chronologically inside each session.
    """
    logger.debug("_sort_pitches_by_session called")

    if pitches.empty:
        return pitches

    required_columns = [
        "game_pk",
        "pitcher",
        "at_bat_number",
        "pitch_number",
        "game_date",
    ]
    missing = [col for col in required_columns if col not in pitches.columns]
    if missing:
        raise ValueError(f"missing required columns for session sorting: {missing}")

    session_start = pitches.groupby(["game_pk", "pitcher"])["at_bat_number"].transform(
        "min"
    )

    sorted_pitches = (
        pitches.assign(_session_start=session_start)
        .sort_values(
            by=[
                "game_date",
                "game_pk",
                "_session_start",
                "pitcher",
                "at_bat_number",
                "pitch_number",
            ],
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
        "batter_days_since_prev_game",
        "pitcher_days_since_prev_game",
    ]
    fill_zero_float_columns = ["sz_top", "sz_bot"]

    essential_columns = [
        col
        for col in required_columns
        if col not in fill_zero_int_columns + fill_zero_float_columns
    ]
    cleaned = pitches.dropna(subset=essential_columns).copy()

    for col in fill_zero_int_columns:
        cleaned[col] = cleaned[col].astype("Int64").fillna(0).astype(int)

    for col in fill_zero_float_columns:
        cleaned[col] = cleaned[col].astype("Float64").fillna(0.0).astype(float)

    logger.debug("_clean_pitch_rows completed successfully")
    return cleaned.reset_index(drop=True)
