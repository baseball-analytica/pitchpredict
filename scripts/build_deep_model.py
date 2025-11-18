import asyncio

import torch
from pitchpredict.backend.algs.deep.building import build_deep_model_from_dataset
from pitchpredict.backend.logging import init_logger
from pybaseball import cache # type: ignore

async def main():
    init_logger(
        log_level_console="DEBUG",
        log_level_file="DEBUG",
    )
    cache.enable()

    model = await build_deep_model_from_dataset(
        dataset_path="/raid/kline/pitchpredict/.pitchpredict_data/pitch_data.bin",
        embed_dim=128,
        hidden_size=64,
        num_layers=2,
        bidirectional=True,
        dropout=0.3,
        pad_idx=0,
        num_classes=None,
        device=torch.device("cuda:5"),
        batch_size=32,
        learning_rate=0.001,
        num_epochs=10,
        model_path="/raid/kline/pitchpredict/.pitchpredict_models/deep_pitch_001.pth",
    )
    print(model)

if __name__ == "__main__":
    asyncio.run(main())