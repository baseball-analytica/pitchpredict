# PitchPredict

Cutting-edge MLB pitch-predicting software utilizing the latest Statcast data. Open-source and free to use. Brought to you by [baseball-analytica.com].

[baseball-analytica.com]: https://baseball-analytica.com

**Read our technical writeup:** [Predicting MLB Pitch Sequences with xLSTM][writeup]

[writeup]: https://panoramic-letter-5ff.notion.site/PitchPredict-Predicting-MLB-Pitch-Sequences-with-xLSTM-6eed91163e064299b726c3bd32eedb95

## Features

- **Two prediction algorithms**: Similarity-based (nearest neighbor) and xLSTM sequence model
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
from pitchpredict import PitchPredict

async def main():
    client = PitchPredict()

    # Resolve MLBAM IDs (cached) for pitcher/batter
    pitcher_id = await client.get_player_id_from_name("Clayton Kershaw")
    batter_id = await client.get_player_id_from_name("Aaron Judge")

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

Pitcher and batter IDs are MLBAM IDs; use `PitchPredict.get_player_id_from_name` (or the REST `/players/lookup` endpoint) to resolve names.
Pitcher predictions return a `PredictPitcherResponse` model; use attribute access or `model_dump()` for a dict.

Caching is enabled by default and stores data in `.pitchpredict_cache`. Delete the folder to refresh cached data.

### xLSTM Quick Start

For xLSTM predictions, you must pass `prev_pitches` (empty list allowed for cold-start):

```python
result = await client.predict_pitcher(
    pitcher_id=pitcher_id,
    batter_id=batter_id,
    prev_pitches=[],  # required for xLSTM, empty list is cold-start
    game_date="2024-06-15",
    algorithm="xlstm",
)
```

xLSTM loads weights lazily. Weights will download automatically on first use. Alternatively, set `PITCHPREDICT_XLSTM_PATH` to a local checkpoint directory containing `model.safetensors` and `config.json`.

When providing history, each pitch in `prev_pitches` must include a `pa_id` (plate-appearance id).

### CLI

Run predictions and look up players directly from the command line (no server required):

```bash
# Lookup player IDs
pitchpredict player lookup "Aaron Judge"

# Predict next pitch (names or MLBAM IDs)
pitchpredict predict pitcher "Zack Wheeler" "Juan Soto" --balls 1 --strikes 2

# Predict batter outcome given a pitch
pitchpredict predict batter "Aaron Judge" "Gerrit Cole" FF 96.5 0.15 2.85

# Predict batted-ball outcome (use --format json for machine-readable output)
pitchpredict predict batted-ball 102.3 24 --format json
```

Use `--verbose` for detailed tables, and `pitchpredict cache status` to inspect the local cache.

### REST API Server

Start the server:

```bash
pitchpredict serve
```

Make a prediction:

```bash
curl "http://localhost:8056/players/lookup?name=Clayton%20Kershaw&fuzzy=true"
curl "http://localhost:8056/players/lookup?name=Aaron%20Judge&fuzzy=true"
```

Use the returned `key_mlbam` values in the prediction request:

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

`pitcher_id` and `batter_id` are MLBAM IDs; use `/players/lookup` to resolve names.

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

Lookup player IDs:

```bash
curl "http://localhost:8056/players/lookup?name=Aaron%20Judge&fuzzy=true"
```

Lookup player metadata by MLBAM ID:

```bash
curl http://localhost:8056/players/592450
```

## Documentation

Full documentation is available in the [docs/](docs/) folder:

- [Getting Started](docs/getting-started.md) - Quick start guide
- [Installation](docs/installation.md) - Detailed installation instructions
- [Python API Reference](docs/python-api.md) - `PitchPredict` class documentation
- [REST API Reference](docs/rest-api.md) - Server endpoints
- [CLI Reference](docs/cli.md) - Command-line interface
- [Algorithms](docs/algorithms.md) - Similarity and xLSTM algorithms
- [Caching](docs/caching.md) - Cache behavior and storage layout

## Methodology

PitchPredict offers two algorithms (details in [Algorithms](docs/algorithms.md)):

### Similarity Algorithm

Finds historical pitches most similar to the current game context using weighted nearest-neighbor analysis:

1. Fetch all pitches thrown by the pitcher from Statcast (2015-01-01 through the requested `game_date`).
2. Compute similarity scores across contextual features (batter ID, counts, bases, score, inning, date, fielders, rest days, strike zone) using softmaxed weights from `SimilarityWeights`.
3. Sample the top `sample_pctg` (default 0.05) most similar pitches.
4. Aggregate statistics and sample concrete pitches to produce predictions.

**Batted ball predictions** use continuous similarity scoring on exit velocity and launch angle, plus optional spray angle, bases state, outs, and date recency, then sample the top similar events for outcome probabilities and expected stats.

### xLSTM Algorithm

Uses an xLSTM sequence model trained on pitch sequences with a ~260-token vocabulary encoding pitch type, speed, spin, location, and result. The model consumes contextual features (player IDs, count, bases, score, inning, and more) to predict the next pitch token sequence, which is decoded back into pitch attributes and outcomes.

## Acknowledgements

PitchPredict would not be possible without [pybaseball], the open-source and MIT-licensed baseball data scraping library. The baseball data itself largely comes from [Statcast], but [Baseball-Reference] and [FanGraphs] are sources as well.

[pybaseball]: https://github.com/jldbc/pybaseball
[Statcast]: https://baseballsavant.mlb.com/statcast_search
[Baseball-Reference]: https://www.baseball-reference.com/
[FanGraphs]: https://www.fangraphs.com/

## License

MIT License - see [LICENSE](LICENSE) for details.
