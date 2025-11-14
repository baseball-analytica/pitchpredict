import asyncio

import torch
from pitchpredict.backend.algs.deep.building import build_deep_model
from pybaseball import cache # type: ignore

async def main():
    cache.enable()

    model = await build_deep_model(
        date_start="2023-01-01",
        date_end="2024-12-31",
        embed_dim=128,
        hidden_size=64,
        num_layers=2,
        bidirectional=False,
        num_epochs=100,
        device=torch.device("cuda:5"),
        model_path="/raid/kline/pitchpredict/.pitchpredict_models/deep_pitch.pth",
    )
    print(model)

if __name__ == "__main__":
    asyncio.run(main())