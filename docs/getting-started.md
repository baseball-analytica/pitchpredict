# Getting Started

This guide will help you get up and running with PitchPredict in minutes.

## Prerequisites

- Python 3.12 or higher

## Installation

```bash
uv pip install pitchpredict
```

Or with pip: `pip install pitchpredict`

For development installation, see the [Installation Guide](installation.md).

## Quick Start: Python API

```python
import asyncio
from pitchpredict import PitchPredict

async def main():
    # Initialize the client
    client = PitchPredict()

    # Predict a pitcher's next pitch
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

    # View pitch type probabilities
    print("Pitch type probabilities:")
    print(result.basic_pitch_data["pitch_type_probs"])

    # View outcome probabilities
    print("\nOutcome probabilities:")
    print(result.basic_outcome_data["outcome_probs"])

asyncio.run(main())
```

Pitcher and batter IDs are MLBAM IDs; use `PitchPredict.get_player_id_from_name` as shown above to resolve names.
Pitcher predictions return a `PredictPitcherResponse` model; use attribute access or `model_dump()` for a dict.

### xLSTM Quick Start

xLSTM requires `prev_pitches` (empty list allowed for cold-start):

```python
result = await client.predict_pitcher(
    pitcher_id=await client.get_player_id_from_name("Clayton Kershaw"),
    batter_id=await client.get_player_id_from_name("Aaron Judge"),
    prev_pitches=[],  # required for xLSTM
    game_date="2024-06-15",
    algorithm="xlstm",
)
```

When providing history, each pitch in `prev_pitches` must include a `pa_id` (plate-appearance id).

## Quick Start: CLI

Run predictions and player lookups without starting the server:

```bash
# Lookup player IDs
pitchpredict player lookup "Aaron Judge"

# Predict next pitch
pitchpredict predict pitcher "Zack Wheeler" "Juan Soto" --balls 1 --strikes 2

# Predict batter outcome given a pitch
pitchpredict predict batter "Aaron Judge" "Gerrit Cole" FF 96.5 0.15 2.85

# Predict batted-ball outcome
pitchpredict predict batted-ball 102.3 24 --format json
```

Use `--verbose` for detailed tables and `--format json` for machine-readable output.

## Quick Start: REST API Server

Start the server:

```bash
pitchpredict serve
```

The server runs on `http://localhost:8056` by default.

Make a prediction request:

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

## Understanding the Response

### Pitcher Prediction Response

A pitcher prediction returns four data sections:

| Section | Description |
|---------|-------------|
| `basic_pitch_data` | Pitch type probabilities, speed/location means and standard deviations |
| `detailed_pitch_data` | Fastball vs offspeed breakdown, percentile distributions |
| `basic_outcome_data` | Outcome probabilities (strike, ball, contact), swing probability |
| `detailed_outcome_data` | Swing metrics, contact quality, batting average on contact |

Example `basic_pitch_data`:

```json
{
  "pitch_type_probs": {
    "FF": 0.45,
    "SL": 0.30,
    "CU": 0.15,
    "CH": 0.10
  },
  "pitch_speed_mean": 92.5,
  "pitch_speed_std": 4.2,
  "pitch_x_mean": 0.12,
  "pitch_x_std": 0.85,
  "pitch_z_mean": 2.45,
  "pitch_z_std": 0.72
}
```

### Batter Prediction Response

A batter prediction returns outcome-focused data:

| Section | Description |
|---------|-------------|
| `basic_outcome_data` | Outcome probabilities, swing/contact rates |
| `detailed_outcome_data` | Contact quality metrics, expected batting stats |

## Pitch Type Codes

Common pitch type codes in the response:

| Code | Pitch Type |
|------|------------|
| FF | Four-seam fastball |
| SI | Sinker |
| FC | Cutter |
| SL | Slider |
| CU | Curveball |
| CH | Changeup |
| FS | Splitter |
| SV | Sweeper |

## Next Steps

- [Python API Reference](python-api.md) - Full API documentation
- [REST API Reference](rest-api.md) - Server endpoint details
- [Algorithms](algorithms.md) - Learn about similarity vs xLSTM
