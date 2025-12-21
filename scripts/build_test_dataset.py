#!/usr/bin/env python3
"""Build a small test dataset from a single day of games."""

import asyncio

import torch
from pitchpredict.backend.algs.deep.building import build_deep_model
from pitchpredict.backend.logging import init_logger
from pybaseball import cache  # type: ignore

async def main():
    init_logger(
        log_level_console="DEBUG",
        log_level_file="DEBUG",
    )
    cache.enable()

    # Just one day of games for testing
    await build_deep_model(
        date_start="2024-07-04",
        date_end="2024-07-04",
        embed_dim=128,
        hidden_size=64,
        num_layers=2,
        bidirectional=False,
        dropout=0.1,
        device=torch.device("cpu"),
        model_path=".pitchpredict_test/model.pth",
        tokens_path=".pitchpredict_test/pitch_seq.bin",
        contexts_path=".pitchpredict_test/pitch_context",
        data_only=True,
    )

if __name__ == "__main__":
    asyncio.run(main())
