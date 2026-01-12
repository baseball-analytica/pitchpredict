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
        pitcher_name="Clayton Kershaw",
        batter_name="Aaron Judge",
        balls=0,
        strikes=0,
        score_bat=0,
        score_fld=0,
        game_date="2024-06-15",
        algorithm="similarity"
    )

    # View pitch type probabilities
    print("Pitch type probabilities:")
    print(result["basic_pitch_data"]["pitch_type_probs"])

    # View outcome probabilities
    print("\nOutcome probabilities:")
    print(result["basic_outcome_data"]["outcome_probs"])

asyncio.run(main())
```

## Quick Start: REST API Server

Start the server:

```bash
pitchpredict serve
```

The server runs on `http://localhost:8056` by default.

Make a prediction request:

```bash
curl -X POST http://localhost:8056/predict/pitcher \
  -H "Content-Type: application/json" \
  -d '{
    "pitcher_name": "Clayton Kershaw",
    "batter_name": "Aaron Judge",
    "balls": 0,
    "strikes": 0,
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
- [Algorithms](algorithms.md) - Learn about similarity vs deep learning
