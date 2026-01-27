import asyncio

import torch
from pitchpredict.backend.algs.deep.building import build_deep_model
from pitchpredict.backend.logging import init_logger
from pybaseball import cache  # type: ignore

DATA_ONLY = True

# SAMPLE NUM PITCHERS = 537
# SAMPLE NUM BATTERS = 470

# FULL NUM PITCHERS = 2962
# FULL NUM BATTERS = 3701
# FULL NUM TOKENS = 51_145_514
# FULL VOCAB SIZE = 192 (really 174 but padding to multiple of 64)


async def main():
    init_logger(
        log_level_console="DEBUG",
        log_level_file="DEBUG",
    )
    cache.enable()

    model = await build_deep_model(
        date_start="2016-01-01",
        date_end="2025-11-18",
        embed_dim=128,
        hidden_size=64,
        num_layers=2,
        bidirectional=False,
        dropout=0.1,
        device=torch.device("cuda:5"),
        model_path="/raid/kline/pitchpredict/.pitchpredict_models/deep_pitch.pth",
        tokens_path="/raid/kline/pitchpredict/.pitchpredict_data/pitch_seq.bin",
        contexts_path="/raid/kline/pitchpredict/.pitchpredict_data/pitch_context",
        data_only=DATA_ONLY,
    )
    print(model)


if __name__ == "__main__":
    asyncio.run(main())
