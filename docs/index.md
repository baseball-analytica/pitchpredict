# PitchPredict Documentation

PitchPredict is cutting-edge MLB pitch prediction software that predicts pitcher and batter behavior using Statcast data. Open-source and free to use.

**Version:** 0.4.0
**License:** MIT
**Author:** Addison Kline (akline@baseball-analytica.com)

**Technical Writeup:** [Predicting MLB Pitch Sequences with xLSTM](https://panoramic-letter-5ff.notion.site/PitchPredict-Predicting-MLB-Pitch-Sequences-with-xLSTM-6eed91163e064299b726c3bd32eedb95)

## Quick Links

- [Getting Started](getting-started.md) - Get up and running in minutes
- [Installation](installation.md) - Detailed installation instructions
- [Python API Reference](python-api.md) - `PitchPredict` class documentation
- [REST API Reference](rest-api.md) - FastAPI server endpoints
- [CLI Reference](cli.md) - Command-line interface
- [Algorithms](algorithms.md) - Similarity and deep learning algorithms
- [Caching](caching.md) - Cache behavior and storage layout

## Features

- **Two prediction algorithms**: Similarity-based (nearest neighbor) and deep learning (xLSTM)
- **Multiple interfaces**: Python API, REST API server, and CLI
- **Rich data output**: Pitch type probabilities, speed/location distributions, outcome predictions
- **Disk-backed caching**: Parquet cache with incremental Statcast updates
- **Statcast integration**: Uses MLB's Statcast data via [pybaseball](https://github.com/jldbc/pybaseball)

## Basic Usage

### Installation

```bash
uv pip install pitchpredict
```

### Python API

```python
import asyncio
from pitchpredict import PitchPredict

async def main():
    client = PitchPredict()
    result = await client.predict_pitcher(
        pitcher_id=await client.get_player_id_from_name("Clayton Kershaw"),
        batter_id=await client.get_player_id_from_name("Aaron Judge"),
        count_balls=0,
        count_strikes=0,
        score_bat=0,
        score_fld=0,
        game_date="2024-06-15",
        algorithm="similarity"
    )
    print(result.basic_pitch_data)

asyncio.run(main())
```

### CLI

```bash
pitchpredict player lookup "Aaron Judge"
pitchpredict predict pitcher "Zack Wheeler" "Juan Soto" --balls 1 --strikes 2
pitchpredict cache status
```

### REST API Server

```bash
pitchpredict serve --port 8056
```

Then make requests to `http://localhost:8056/predict/pitcher`.

## Data Sources

PitchPredict uses data from:

- [Statcast](https://baseballsavant.mlb.com/statcast_search) - MLB's pitch tracking system
- [Baseball-Reference](https://www.baseball-reference.com/)
- [FanGraphs](https://www.fangraphs.com/)

All data is fetched via the [pybaseball](https://github.com/jldbc/pybaseball) library.

## Links

- [GitHub Repository](https://github.com/baseball-analytica/pitchpredict)
- [baseball-analytica.com](https://baseball-analytica.com)
