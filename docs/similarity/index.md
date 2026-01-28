# Similarity Algorithm Deep Dive

This section provides comprehensive documentation for PitchPredict's similarity algorithm. For a high-level overview and comparison with the xLSTM algorithm, see the main [Algorithms](../algorithms.md) documentation.

## Overview

The similarity algorithm uses weighted nearest-neighbor analysis to predict pitcher behavior, batter outcomes, and batted ball results. It works by:

1. Fetching historical Statcast data for the relevant player
2. Scoring each historical pitch/event based on similarity to the current game context
3. Sampling the most similar records
4. Aggregating statistics from the sample to produce predictions

## Prediction Modes

The similarity algorithm supports three prediction modes, each with its own scoring logic and data requirements:

| Mode | Purpose | Documentation |
|------|---------|---------------|
| **Pitcher Prediction** | Predict what pitch a pitcher will throw | [pitcher-prediction.md](pitcher-prediction.md) |
| **Batter Prediction** | Predict batter outcome given an incoming pitch | [batter-prediction.md](batter-prediction.md) |
| **Batted Ball Prediction** | Predict outcome of a ball in play | [batted-ball.md](batted-ball.md) |

## Additional Resources

| Resource | Description |
|----------|-------------|
| [Data Columns Reference](data-columns.md) | Complete list of Statcast columns used by each mode |
| [Configuration Guide](configuration.md) | Customizing weights, sample sizes, and algorithm instances |
| [Code Examples](examples.md) | Practical Python and REST API examples |

## Quick Start

```python
from pitchpredict import PitchPredict

async def predict():
    client = PitchPredict()

    # Pitcher prediction
    result = await client.predict_pitcher(
        pitcher_id=543037,  # Clayton Kershaw
        batter_id=592450,   # Aaron Judge
        count_balls=1,
        count_strikes=2,
        algorithm="similarity"
    )
    print(result.basic_pitch_data["pitch_type_probs"])
```

## Key Concepts

### Similarity Scoring

Each prediction mode calculates a similarity score for every historical record. The score is a weighted sum of per-field scores, where each field contributes based on how closely it matches the current context.

**Score types:**
- **Exact match**: `1.0` if values match, `0.0` otherwise
- **Numeric (tolerance)**: `max(0, 1 - abs(diff) / tolerance)`
- **Bases state**: `1 - (mismatches / 3)` across three base positions

### Sample Selection

After scoring, the algorithm selects the top N% most similar records:
- Default sample percentage: **5%**
- Pitcher and batted ball modes enforce a **minimum of 100 samples**
- Batter mode uses raw top-N% with no minimum floor

### Weight Normalization

Pitcher prediction uses **softmax normalization** on raw weights, ensuring they sum to 1. Batter and batted ball modes use fixed hard-coded weights.

## Source Code

The similarity algorithm is implemented in:
- `src/pitchpredict/backend/algs/similarity/base.py` - Core `SimilarityAlgorithm` class
- `src/pitchpredict/backend/algs/similarity/types.py` - `SimilarityWeights` model
