#!/usr/bin/env python3
"""
Test that vectorized output is compatible with PackedPitchDataset.
"""

import asyncio
import tempfile
from pathlib import Path

import torch
import numpy as np

from pitchpredict.backend.algs.deep.building import build_dataset_vectorized
from pitchpredict.backend.algs.deep.dataset import PackedPitchDataset
from pitchpredict.backend.logging import init_logger
from pybaseball import cache  # type: ignore


async def main():
    init_logger(log_level_console="INFO", log_level_file="DEBUG")
    cache.enable()

    with tempfile.TemporaryDirectory() as tmpdir:
        tokens_path = Path(tmpdir) / "pitch_seq.bin"
        contexts_path = Path(tmpdir) / "pitch_context"

        print("=" * 60)
        print("Building dataset with vectorized pipeline")
        print("=" * 60)

        stats = await build_dataset_vectorized(
            date_start="2024-07-04",
            date_end="2024-07-04",
            tokens_path=str(tokens_path),
            contexts_path=str(contexts_path),
            split_val_ratio=0.0,
            split_test_ratio=0.0,
            max_workers=4,
        )

        print()
        print("=" * 60)
        print("Loading with PackedPitchDataset")
        print("=" * 60)

        # Test loading with different sequence lengths
        for seq_len in [64, 128, 256]:
            print(f"\nTesting seq_len={seq_len}:")
            dataset = PackedPitchDataset(tmpdir, seq_len=seq_len)
            print(f"  Total chunks: {len(dataset)}")
            print(f"  Chunk boundaries (first 5): {dataset.chunks[:5]}")

            # Load a few chunks and verify
            for i in range(min(3, len(dataset))):
                chunk = dataset[i]
                print(f"  Chunk {i}:")
                print(f"    x shape: {chunk.x.shape}, y shape: {chunk.y.shape}")
                print(f"    pitcher_id shape: {chunk.pitcher_id.shape}")
                print(f"    x dtype: {chunk.x.dtype}, pitcher_id dtype: {chunk.pitcher_id.dtype}")

                # Verify shapes match
                assert chunk.x.shape == (seq_len,), f"x shape mismatch: {chunk.x.shape}"
                assert chunk.y.shape == (seq_len,), f"y shape mismatch: {chunk.y.shape}"
                assert chunk.pitcher_id.shape == (seq_len,), f"pitcher_id shape mismatch"

        print()
        print("=" * 60)
        print("Testing DataLoader integration")
        print("=" * 60)

        dataset = PackedPitchDataset(tmpdir, seq_len=128)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=4, shuffle=True, num_workers=0
        )

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 2:
                break
            print(f"\nBatch {batch_idx}:")
            print(f"  x shape: {batch.x.shape}")
            print(f"  y shape: {batch.y.shape}")
            print(f"  pitcher_id shape: {batch.pitcher_id.shape}")

        print()
        print("=" * 60)
        print("ALL COMPATIBILITY TESTS PASSED!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
