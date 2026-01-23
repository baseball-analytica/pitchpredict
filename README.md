# PitchPredict

Cutting-edge MLB pitch-predicting software utilizing the latest Statcast data. Open-source and free to use. Brought to you by [baseball-analytica.com].

[baseball-analytica.com]: https://baseball-analytica.com

**Read our technical writeup:** [Predicting MLB Pitch Sequences with xLSTM][writeup]

[writeup]: https://panoramic-letter-5ff.notion.site/PitchPredict-Predicting-MLB-Pitch-Sequences-with-xLSTM-6eed91163e064299b726c3bd32eedb95

## Features

- **Two prediction algorithms**: Similarity-based (nearest neighbor) and deep learning (xLSTM)
- **Multiple interfaces**: Python API, REST API server, and CLI
- **Rich predictions**: Pitch type probabilities, speed/location distributions, outcome analysis
- **Batted ball predictions**: Outcome probabilities from exit velocity and launch angle with context-aware filtering
- **Disk-backed caching**: Parquet cache with incremental Statcast updates
- **Statcast powered**: Uses MLB's comprehensive pitch tracking data via [pybaseball]

## Installation

### Package Installation

```bash
uv pip install pitchpredict
```

Or with pip:

```bash
pip install pitchpredict
```

Requires Python 3.12 or higher. We recommend using [uv](https://github.com/astral-sh/uv) for faster, more reliable package management.

### Development Installation

```bash
git clone https://github.com/baseball-analytica/pitchpredict.git
cd pitchpredict
uv sync
```

## Quick Start

### Python API

```python
import asyncio
from pybaseball import playerid_lookup
from pitchpredict import PitchPredict

async def main():
    client = PitchPredict()

    # Resolve MLBAM IDs (via pybaseball) for pitcher/batter
    pitcher_id = int(playerid_lookup("Kershaw", "Clayton").iloc[0]["key_mlbam"])
    batter_id = int(playerid_lookup("Judge", "Aaron").iloc[0]["key_mlbam"])

    # Predict pitcher's next pitch
    result = await client.predict_pitcher(
        pitcher_id=pitcher_id,
        batter_id=batter_id,
        count_balls=0,
        count_strikes=0,
        score_bat=0,
        score_fld=0,
        game_date="2024-06-15",
        algorithm="similarity"
    )

    print(result.basic_pitch_data["pitch_type_probs"])
    # {'FF': 0.45, 'SL': 0.30, 'CU': 0.15, 'CH': 0.10}

asyncio.run(main())
```

Pitcher and batter IDs are MLBAM IDs; use `pybaseball.playerid_lookup` as shown above to resolve names.
Pitcher predictions return a `PredictPitcherResponse` model; use attribute access or `model_dump()` for a dict.

Caching is enabled by default and stores data in `.pitchpredict_cache`. Delete the folder to refresh cached data.

### REST API Server

Start the server:

```bash
pitchpredict serve
```

Make a prediction:

```bash
curl -X POST http://localhost:8056/predict/pitcher \
  -H "Content-Type: application/json" \
  -d '{
    "pitcher_id": 477132,
    "batter_id": 592450,
    "count_balls": 0,
    "count_strikes": 0,
    "score_bat": 0,
    "score_fld": 0,
    "game_date": "2024-06-15",
    "algorithm": "similarity"
  }'
```

`pitcher_id` and `batter_id` are MLBAM IDs; use `pybaseball.playerid_lookup` to resolve names.

Predict batted ball outcomes:

```bash
curl -X POST http://localhost:8056/predict/batted-ball \
  -H "Content-Type: application/json" \
  -d '{
    "launch_speed": 95.0,
    "launch_angle": 18.0,
    "algorithm": "similarity"
  }'
```

## Documentation

Full documentation is available in the [docs/](docs/) folder:

- [Getting Started](docs/getting-started.md) - Quick start guide
- [Installation](docs/installation.md) - Detailed installation instructions
- [Python API Reference](docs/python-api.md) - `PitchPredict` class documentation
- [REST API Reference](docs/rest-api.md) - Server endpoints
- [CLI Reference](docs/cli.md) - Command-line interface
- [Algorithms](docs/algorithms.md) - Similarity and deep learning algorithms
- [Caching](docs/caching.md) - Cache behavior and storage layout

## Methodology

PitchPredict offers two algorithms:

### Similarity Algorithm

Finds historical pitches most similar to the current game context using weighted nearest-neighbor analysis:

1. Fetch all pitches thrown by the pitcher from Statcast
2. Score each pitch based on similarity to the current context (batter, count, score, date)
3. Sample the top 5% most similar pitches
4. Aggregate statistics to produce predictions

**Similarity weights for pitcher predictions:**
- Batter: 35%
- Ball count: 20%
- Strike count: 20%
- Batting team score: 10%
- Fielding team score: 10%
- Game date: 5%

**Similarity weights for batted ball predictions:**
- Exit velocity: 45% (continuous, 15 mph tolerance)
- Launch angle: 40% (continuous, 20° tolerance)
- Spray angle: 5%
- Bases state: 5%
- Outs: 3%
- Date recency: 2%

### Deep Learning Algorithm

Uses an xLSTM neural network trained on pitch sequences with a 270-token vocabulary encoding pitch type, speed, spin, location, and result. Considers 28 contextual features including game state, player identities, and matchup history.

## Acknowledgements

PitchPredict would not be possible without [pybaseball], the open-source and MIT-licensed baseball data scraping library. The baseball data itself largely comes from [Statcast], but [Baseball-Reference] and [FanGraphs] are sources as well.

[pybaseball]: https://github.com/jldbc/pybaseball
[Statcast]: https://baseballsavant.mlb.com/statcast_search
[Baseball-Reference]: https://www.baseball-reference.com/
[FanGraphs]: https://www.fangraphs.com/

## License

MIT License - see [LICENSE](LICENSE) for details.
