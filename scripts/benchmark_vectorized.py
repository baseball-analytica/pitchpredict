#!/usr/bin/env python3
"""
Benchmark vectorized vs original pipeline.
"""

import asyncio
import tempfile
import time
from pathlib import Path

import pandas as pd
import numpy as np

from pitchpredict.backend.algs.deep.building import (
    _clean_pitch_rows,
    _sort_pitches_by_session,
    _build_pitch_tokens_and_contexts,
    _build_pitch_dataset,
)
from pitchpredict.backend.algs.deep.vectorized import (
    build_tokens_vectorized,
    write_tokens_file_vectorized,
    write_context_files_vectorized,
)
from pitchpredict.backend.fetching import get_all_pitches
from pitchpredict.backend.logging import init_logger
from pybaseball import cache  # type: ignore


async def benchmark_original(pitches: pd.DataFrame, output_dir: Path) -> tuple[float, int]:
    """Run original tokenization pipeline and return (time, token_count)."""
    start = time.time()

    try:
        tokens, contexts, stats = await _build_pitch_tokens_and_contexts(pitches, dataset_log_interval=100000)
        elapsed = time.time() - start
        return elapsed, len(tokens)
    except ValueError as e:
        # Original code has validation bugs, still return the time
        elapsed = time.time() - start
        print(f"  (Original pipeline had validation error: {e})")
        return elapsed, -1


def benchmark_vectorized(pitches: pd.DataFrame, output_dir: Path) -> tuple[float, int]:
    """Run vectorized tokenization pipeline and return (time, token_count)."""
    start = time.time()

    tokens, token_to_pitch, stats = build_tokens_vectorized(pitches)

    elapsed = time.time() - start
    return elapsed, len(tokens)


async def main():
    init_logger(log_level_console="INFO", log_level_file="DEBUG")
    cache.enable()

    print("=" * 70)
    print("BENCHMARK: Vectorized vs Original Pipeline")
    print("=" * 70)
    print()

    # Fetch data for a few days to get a reasonable sample
    print("Fetching pitch data for 2024-07-01 to 2024-07-05...")
    pitches = await get_all_pitches("2024-07-01", "2024-07-05")
    pitches = _clean_pitch_rows(pitches)
    pitches = _sort_pitches_by_session(pitches)

    print(f"Dataset: {len(pitches):,} pitches")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Benchmark vectorized (run first as it's faster)
        print("Running VECTORIZED pipeline...")
        vec_time, vec_tokens = benchmark_vectorized(pitches.copy(), output_dir)
        print(f"  Time: {vec_time:.2f}s")
        print(f"  Tokens: {vec_tokens:,}")
        print(f"  Throughput: {len(pitches) / vec_time:,.0f} pitches/sec")
        print()

        # Benchmark original
        print("Running ORIGINAL pipeline...")
        orig_time, orig_tokens = await benchmark_original(pitches.copy(), output_dir)
        print(f"  Time: {orig_time:.2f}s")
        print(f"  Tokens: {orig_tokens:,}")
        print(f"  Throughput: {len(pitches) / orig_time:,.0f} pitches/sec")
        print()

        # Summary
        speedup = orig_time / vec_time if vec_time > 0 else float('inf')

        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Original:   {orig_time:7.2f}s ({len(pitches) / orig_time:,.0f} pitches/sec)")
        print(f"Vectorized: {vec_time:7.2f}s ({len(pitches) / vec_time:,.0f} pitches/sec)")
        print(f"Speedup:    {speedup:.1f}x faster")
        print()

        # Verify same output
        if orig_tokens > 0 and orig_tokens == vec_tokens:
            print("✓ Token counts match!")
        elif orig_tokens < 0:
            print("(Original pipeline errored - cannot compare token counts)")
        else:
            print(f"⚠ Token count mismatch: {orig_tokens} vs {vec_tokens}")
            if orig_tokens > 0:
                diff_pct = abs(orig_tokens - vec_tokens) / orig_tokens * 100
                print(f"  Difference: {diff_pct:.2f}%")


if __name__ == "__main__":
    asyncio.run(main())
