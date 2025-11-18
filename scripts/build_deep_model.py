import asyncio

import torch
from pitchpredict.backend.algs.deep.building import build_deep_model, build_deep_model_from_dataset
from pitchpredict.backend.logging import init_logger
from pybaseball import cache # type: ignore

async def main():
    init_logger(
        log_level_console="DEBUG",
        log_level_file="DEBUG",
    )
    cache.enable()

    model = await build_deep_model(
        date_start="2025-04-01",
        date_end="2025-05-01",
        embed_dim=128,
        hidden_size=64,
        num_layers=2,
        bidirectional=False,
        dropout=0.1,
    )
    print(model)

if __name__ == "__main__":
    asyncio.run(main())