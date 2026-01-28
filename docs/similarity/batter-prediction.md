# Batter Prediction Mode

The batter prediction mode answers: **"What will the batter do when facing this specific pitch?"**

Unlike pitcher prediction, batter prediction takes a known incoming pitch and predicts the batter's response and outcome.

## Key Differences from Pitcher Prediction

| Aspect | Pitcher Prediction | Batter Prediction |
|--------|-------------------|-------------------|
| **Question** | What pitch will be thrown? | What happens when this pitch is thrown? |
| **Data source** | Pitcher's historical pitches | Batter's historical at-bats |
| **Weight normalization** | Softmax (sum to 1) | Hard-coded fixed weights |
| **Scoring type** | Mix of exact match and tolerance | Exact match only |
| **Minimum samples** | 100 | None (raw top-N%) |
| **Pitch characteristics** | Not used | Used for scoring |

## Required Parameters

Batter prediction requires pitch information that pitcher prediction doesn't need:

```python
result = await client.predict_batter(
    pitcher_id=543037,      # Required
    batter_id=592450,       # Required
    pitch_type="FF",        # Required: pitch type code
    pitch_speed=95.2,       # Required: velocity in mph
    pitch_x=-0.5,           # Required: horizontal location (feet from center)
    pitch_z=2.8,            # Required: vertical location (feet from ground)
    count_balls=1,          # Optional context
    count_strikes=2,        # Optional context
    game_date="2024-06-15"  # Optional context
)
```

## Scoring Weights

Batter prediction uses fixed, hard-coded weights (no softmax):

| Factor | Weight | Statcast Column | Description |
|--------|--------|-----------------|-------------|
| `pitcher_id` | 25% | `pitcher` | Facing the same pitcher |
| `count_balls` | 15% | `balls` | Same ball count |
| `count_strikes` | 15% | `strikes` | Same strike count |
| `game_date` | 10% | `game_date` | Same game date |
| `score_bat` | 5% | `bat_score` | Same batting team score |
| `score_fld` | 5% | `fld_score` | Same fielding team score |
| `pitch_type` | 5% | `pitch_type` | Same pitch type |
| `pitch_speed` | 5% | `release_speed` | Same velocity |
| `pitch_x` | 5% | `plate_x` | Same horizontal location |
| `pitch_z` | 5% | `plate_z` | Same vertical location |

**Total: 100%** (weights sum to 1.0 by design)

## Exact Match Scoring

All batter prediction scores use exact equality—there are no tolerances:

```python
def score_equal(column, target):
    if target is None or column not in pitches.columns:
        return 0.0
    return 1.0 if pitch[column] == target else 0.0
```

This means:
- `pitch_speed=95.2` only matches pitches at exactly 95.2 mph
- `pitch_x=-0.5` only matches pitches at exactly -0.5 feet

### Practical Implications

Exact matching on continuous values like `pitch_speed` and `pitch_x` is very strict:

1. **Most pitches score 0** on these fields (exact matches are rare)
2. **Pitcher and count dominate** since they're more likely to match
3. **Consider this a "same pitcher, same count" filter** more than a pitch-matching algorithm

This is a known limitation. For more nuanced pitch-based predictions, consider:
- Using batted ball prediction for contact outcomes
- Implementing custom scoring with tolerance (see [Configuration](configuration.md))

## Sample Selection

Unlike pitcher and batted ball modes, batter prediction has **no minimum sample floor**:

```python
n_samples = int(total_pitches * sample_pctg)  # No max(100, ...)

if n_samples <= 0:
    return empty_dataframe
```

With a 5% sample rate and 1000 historical pitches, you get 50 samples. With only 100 pitches, you get 5 samples.

## Data Columns

Batter prediction uses a smaller set of columns than pitcher prediction:

```python
BATTER_PITCH_COLUMNS = (
    "game_date",
    "game_date_dt",
    "pitcher",
    "balls",
    "strikes",
    "bat_score",
    "fld_score",
    "pitch_type",
    "release_speed",
    "plate_x",
    "plate_z",
    "type",
    "bat_speed",
    "swing_length",
    "events",
    "bb_type",
    "launch_speed",
    "launch_angle",
    "estimated_ba_using_speedangle",
    "estimated_slg_using_speedangle",
    "estimated_woba_using_speedangle",
)
```

See [Data Columns Reference](data-columns.md) for complete column definitions.

## Output Format

Batter prediction returns outcome data (no pitch data, since the pitch is already known):

```python
{
    "algorithm_metadata": {
        "algorithm_name": "similarity",
        "instance_name": "similarity"
    },
    "basic_outcome_data": {
        "outcome_probs": {"strike": 0.3, "ball": 0.2, "contact": 0.5},
        "swing_probability": 0.6,
        "swing_event_probs": {"swinging_strike": 0.2, "contact": 0.8},
        "contact_probability": 0.5,
        "contact_event_probs": {"single": 0.25, "field_out": 0.6, ...}
    },
    "detailed_outcome_data": {
        "swing_data": {
            "bat_speed_mean": 72.5,
            "bat_speed_std": 3.2,
            # ... percentiles
        },
        "contact_data": {
            "bb_type_probs": {"fly_ball": 0.3, "ground_ball": 0.4, ...},
            "BA": 0.280,
            "exit_velocity_mean": 89.5,
            # ... percentiles and expected stats
        }
    },
    "prediction_metadata": {
        "n_pitches_total": 1500,
        "n_pitches_sampled": 75,
        "sample_pctg": 0.05
    }
}
```

## Use Cases

### Pitch Scouting

Find how a batter performs against specific pitch types from specific pitchers:

```python
# How does Judge handle Kershaw's curveball?
result = await client.predict_batter(
    pitcher_id=477132,   # Kershaw
    batter_id=592450,    # Judge
    pitch_type="CU",
    pitch_speed=75.0,
    pitch_x=0.0,
    pitch_z=2.0
)
print(f"Contact rate: {result['basic_outcome_data']['contact_probability']:.1%}")
```

### Count-Based Analysis

Compare batter performance across counts:

```python
counts = [(0, 0), (0, 2), (3, 0), (3, 2)]
for balls, strikes in counts:
    result = await client.predict_batter(
        pitcher_id=543037,
        batter_id=592450,
        pitch_type="FF",
        pitch_speed=95.0,
        pitch_x=0.0,
        pitch_z=2.5,
        count_balls=balls,
        count_strikes=strikes
    )
    print(f"{balls}-{strikes}: swing rate = {result['basic_outcome_data']['swing_probability']:.1%}")
```

## Limitations

1. **Exact matching is strict**: Continuous fields rarely match exactly
2. **No minimum samples**: Small datasets may produce unreliable results
3. **Pitcher-centric**: Heavy weight on `pitcher_id` assumes same-pitcher data exists
4. **No pitch sequencing**: Each pitch is evaluated independently

For outcome prediction after contact, use [Batted Ball Prediction](batted-ball.md) instead.
