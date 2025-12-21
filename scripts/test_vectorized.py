#!/usr/bin/env python3
"""
Test the vectorized dataset pipeline against a small dataset.
"""

import asyncio
import tempfile
import os
from pathlib import Path

import numpy as np

from pitchpredict.backend.algs.deep.building import build_dataset_vectorized
from pitchpredict.backend.logging import init_logger
from pitchpredict.backend.algs.deep.nn import TOKEN_DTYPE
from pitchpredict.backend.algs.deep.types import PitchToken
from pybaseball import cache  # type: ignore


async def main():
    init_logger(log_level_console="INFO", log_level_file="DEBUG")
    cache.enable()

    # Use a temp directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        tokens_path = Path(tmpdir) / "pitch_seq.bin"
        contexts_path = Path(tmpdir) / "pitch_context"

        print("=" * 60)
        print("Testing vectorized pipeline with one day of data")
        print("=" * 60)
        print()

        # Build small dataset
        stats = await build_dataset_vectorized(
            date_start="2024-07-04",
            date_end="2024-07-04",
            tokens_path=str(tokens_path),
            contexts_path=str(contexts_path),
            split_val_ratio=0.0,  # No split for testing
            split_test_ratio=0.0,
            max_workers=4,
        )

        print()
        print("=" * 60)
        print("Validation")
        print("=" * 60)

        # Load and validate tokens
        tokens = np.fromfile(tokens_path, dtype=TOKEN_DTYPE)
        print(f"Loaded {len(tokens):,} tokens")
        assert len(tokens) == stats.total_tokens, f"Token count mismatch: {len(tokens)} vs {stats.total_tokens}"

        # Check token values are valid
        min_val, max_val = tokens.min(), tokens.max()
        print(f"Token range: {min_val} - {max_val}")
        assert min_val >= 0, f"Invalid min token: {min_val}"
        max_token = max(t.value for t in PitchToken)
        assert max_val <= max_token, f"Invalid max token: {max_val} (max allowed: {max_token})"

        # Check session structure
        session_starts = np.sum(tokens == PitchToken.SESSION_START.value)
        session_ends = np.sum(tokens == PitchToken.SESSION_END.value)
        pa_starts = np.sum(tokens == PitchToken.PA_START.value)
        pa_ends = np.sum(tokens == PitchToken.PA_END.value)

        print(f"SESSION_START count: {session_starts}")
        print(f"SESSION_END count: {session_ends}")
        print(f"PA_START count: {pa_starts}")
        print(f"PA_END count: {pa_ends}")

        assert session_starts == session_ends, f"Mismatched sessions: {session_starts} starts vs {session_ends} ends"
        assert pa_starts == pa_ends, f"Mismatched PAs: {pa_starts} starts vs {pa_ends} ends"
        assert session_starts == stats.session_count, f"Session count mismatch"
        assert pa_starts == stats.pa_count, f"PA count mismatch"

        # Check context files exist
        context_fields = [
            "pitcher_id", "batter_id", "pitcher_age", "pitcher_throws",
            "batter_age", "batter_hits", "count_balls", "count_strikes",
            "outs", "bases_state", "score_bat", "score_fld", "inning",
            "pitch_number", "number_through_order", "game_date",
            "fielder_2_id", "fielder_3_id", "fielder_4_id", "fielder_5_id",
            "fielder_6_id", "fielder_7_id", "fielder_8_id", "fielder_9_id",
            "batter_days_since_prev_game", "pitcher_days_since_prev_game",
            "strike_zone_top", "strike_zone_bottom",
        ]

        for field in context_fields:
            field_path = Path(tmpdir) / f"pitch_context_{field}.bin"
            assert field_path.exists(), f"Missing context file: {field_path}"
            file_size = field_path.stat().st_size
            # Verify file size matches token count (with appropriate dtype)
            print(f"  {field}: {file_size:,} bytes")

        print()
        print("=" * 60)
        print("Token Sequence Sample (first 100)")
        print("=" * 60)

        token_names = {t.value: t.name for t in PitchToken}
        for i, tok in enumerate(tokens[:100]):
            name = token_names.get(tok, f"UNKNOWN_{tok}")
            if name in ("SESSION_START", "SESSION_END", "PA_START", "PA_END"):
                print(f"  [{i:4d}] ** {name} **")
            elif name.startswith("IS_"):
                print(f"  [{i:4d}] Pitch type: {name}")

        print()
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
