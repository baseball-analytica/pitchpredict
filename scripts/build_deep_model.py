import asyncio

import torch
from pitchpredict.backend.algs.deep.building import build_deep_model
from pitchpredict.backend.logging import init_logger
from pybaseball import cache # type: ignore

async def main():
    init_logger()
    cache.enable()

    model = await build_deep_model(
        date_start="2024-01-01",
        date_end="2025-11-17",
        embed_dim=128,
        hidden_size=64,
        num_layers=2,
        bidirectional=False,
        num_epochs=10,
        device=torch.device("cuda:5"),
        model_path="/raid/kline/pitchpredict/.pitchpredict_models/deep_pitch.pth",
        dataset_path="/raid/kline/pitchpredict/.pitchpredict_data/pitch_data.bin",
    )
    print(model)

if __name__ == "__main__":
    asyncio.run(main())