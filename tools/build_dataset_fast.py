#!/usr/bin/env python3
"""
Build dataset using the optimized vectorized pipeline.

This is 50-100x faster than the original row-by-row approach.

For fastest results, run fetch_pitches.py first to cache the raw data,
then set USE_PARQUET_CACHE = True below.
"""

import asyncio
import warnings
from pathlib import Path

# Suppress pybaseball's pandas FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="pybaseball")

from tools.deep.building import build_dataset_vectorized
from pitchpredict.backend.logging import init_logger
from pybaseball import cache  # type: ignore

# Configuration
DATE_START = "2016-01-01"
DATE_END = "2025-11-18"
TOKENS_PATH = "/raid/kline/pitchpredict/.pitchpredict_session_data/pitch_seq.bin"
CONTEXTS_PATH = "/raid/kline/pitchpredict/.pitchpredict_session_data/pitch_context"
SPLIT_VAL_RATIO = 0.01
SPLIT_TEST_RATIO = 0.01
MAX_WORKERS = 28  # Parallel threads for context file I/O

# Set to parquet path to skip API fetching (much faster!)
# Run fetch_pitches.py first to create this file
PITCHES_PARQUET = "/raid/kline/pitchpredict/.pitchpredict_session_data/raw_pitches.parquet"


async def main():
    init_logger(
        log_level_console="INFO",
        log_level_file="DEBUG",
    )
    cache.enable()

    # Check if parquet cache exists
    parquet_path = Path(PITCHES_PARQUET) if PITCHES_PARQUET else None
    use_parquet = parquet_path and parquet_path.exists()

    if use_parquet:
        print(f"Loading pitches from parquet cache: {parquet_path}")
        print(f"(Run fetch_pitches.py to update the cache)")
    else:
        print(f"Fetching pitches from API: {DATE_START} to {DATE_END}")
        print(f"(Run fetch_pitches.py first to cache for faster subsequent runs)")

    print(f"Output: {TOKENS_PATH}")
    print(f"Using {MAX_WORKERS} parallel workers")
    print()

    stats = await build_dataset_vectorized(
        date_start=None if use_parquet else DATE_START,
        date_end=None if use_parquet else DATE_END,
        tokens_path=TOKENS_PATH,
        contexts_path=CONTEXTS_PATH,
        split_val_ratio=SPLIT_VAL_RATIO,
        split_test_ratio=SPLIT_TEST_RATIO,
        max_workers=MAX_WORKERS,
        pitches_parquet=str(parquet_path) if use_parquet else None,
    )

    print()
    print("=" * 60)
    print("Dataset Generation Complete")
    print("=" * 60)
    print(f"Total pitches: {stats.total_pitches:,}")
    print(f"Total tokens:  {stats.total_tokens:,}")
    print(f"Sessions:      {stats.session_count:,}")
    print(f"Plate appearances: {stats.pa_count:,}")
    print(f"Avg tokens/pitch: {stats.total_tokens / max(1, stats.total_pitches):.1f}")


if __name__ == "__main__":
    asyncio.run(main())
