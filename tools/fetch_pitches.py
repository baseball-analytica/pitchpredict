#!/usr/bin/env python3
"""
Fetch all pitch data and cache to parquet for fast reuse.

Run this ONCE, then use the parquet file for dataset generation.
Loading from parquet: ~5 seconds
Fetching from API: ~5-15 minutes with parallel fetching
"""

import asyncio
import time
import warnings
from pathlib import Path

# Suppress pybaseball's pandas FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="pybaseball")

from pitchpredict.backend.fetching import get_all_pitches
from pitchpredict.backend.logging import init_logger
from pybaseball import cache  # type: ignore

# Configuration
DATE_START = "2016-01-01"
DATE_END = "2025-11-18"
CACHE_PATH = "/raid/kline/pitchpredict/.pitchpredict_session_data/raw_pitches.parquet"

# Parallel fetching settings
MAX_WORKERS = 4  # Parallel HTTP connections (don't go too high or you'll get rate limited)
CHUNK_MONTHS = 2  # Fetch 2 months at a time


async def main():
    init_logger(log_level_console="INFO", log_level_file="DEBUG")
    cache.enable()

    cache_file = Path(CACHE_PATH)

    if cache_file.exists():
        print(f"Cache file already exists: {cache_file}")
        print(f"Size: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")

        # Quick load test
        import pandas as pd
        start = time.time()
        df = pd.read_parquet(cache_file)
        load_time = time.time() - start
        print(f"Loaded {len(df):,} pitches in {load_time:.1f}s")
        print()
        print("To re-fetch, delete the cache file and run again.")
        return

    print(f"Fetching all pitches from {DATE_START} to {DATE_END}")
    print(f"Using {MAX_WORKERS} parallel workers, {CHUNK_MONTHS}-month chunks")
    print()

    start = time.time()
    pitches = await get_all_pitches(
        DATE_START,
        DATE_END,
        parallel=True,
        max_workers=MAX_WORKERS,
        chunk_months=CHUNK_MONTHS,
    )
    fetch_time = time.time() - start

    print()
    print(f"Fetched {len(pitches):,} pitches in {fetch_time:.1f}s ({fetch_time/60:.1f} min)")

    # Save to parquet
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {cache_file}...")
    save_start = time.time()
    pitches.to_parquet(cache_file, compression="zstd")
    save_time = time.time() - save_start

    file_size = cache_file.stat().st_size / 1024 / 1024
    print(f"Saved in {save_time:.1f}s ({file_size:.1f} MB)")
    print()
    print("Done! Use this file with build_dataset_fast.py:")
    print(f'  PITCHES_PARQUET = "{CACHE_PATH}"')


if __name__ == "__main__":
    asyncio.run(main())
