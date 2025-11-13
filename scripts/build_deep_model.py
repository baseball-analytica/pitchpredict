import asyncio
from pitchpredict.backend.algs.deep.building import build_deep_model

async def main():
    model = await build_deep_model(
        date_start="2023-01-01",
        date_end="2024-12-31",
        embed_dim=128,
        hidden_size=64,
        num_layers=2,
        bidirectional=False,
        num_epochs=100,
        model_path="/raid/kline/pitchpredict/.pitchpredict_models/deep_pitch.pth",
    )
    print(model)

if __name__ == "__main__":
    asyncio.run(main())