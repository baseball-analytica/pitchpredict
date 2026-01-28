# Batted Ball Prediction Mode

The batted ball prediction mode answers: **"What happens when a ball is hit with this exit velocity and launch angle?"**

This mode is designed for predicting outcomes after contact, using Statcast's batted ball data.

## Key Features

- **Continuous similarity scoring** with defined tolerances (not exact match)
- **Context-aware outcome filtering** removes impossible plays
- **Batted ball type inference** from launch angle
- **Expected statistics** (xBA, xSLG, xwOBA) from similar batted balls

## Required vs Optional Parameters

```python
result = await client.predict_batted_ball(
    # Required
    launch_speed=98.5,      # Exit velocity in mph
    launch_angle=22.0,      # Launch angle in degrees

    # Optional context
    spray_angle=15.0,       # Spray angle (-45 to +45, pull to oppo)
    bb_type="fly_ball",     # Known batted ball type
    outs=1,                 # Current outs (for context filtering)
    bases_state=5,          # Runner situation (for context filtering)
    batter_id=592450,       # For batter-specific patterns
    game_date="2024-06-15"  # For recency weighting
)
```

## Similarity Scoring

Batted ball prediction uses **continuous scoring with tolerances**, unlike batter prediction's exact matching.

### Scoring Formulas

| Factor | Weight | Formula | Tolerance |
|--------|--------|---------|-----------|
| Exit velocity | 45% | `max(0, 1 - abs(diff) / 15.0)` | 15 mph |
| Launch angle | 40% | `max(0, 1 - abs(diff) / 20.0)` | 20° |
| Spray angle | 5% | `max(0, 1 - abs(diff) / 30.0)` | 30° |
| Bases state | 5% | `1 - (mismatches / 3)` | N/A |
| Outs | 3% | `1.0` if equal, `0.0` otherwise | Exact |
| Date | 2% | `max(0, 1 - days_diff / 365.0)` | 365 days |

### Exit Velocity Scoring

A batted ball with EV = 95 mph compared to query EV = 100 mph:
```python
diff = abs(95 - 100)  # = 5
score = max(0, 1 - 5 / 15)  # = 0.67
```

The 15 mph tolerance means:
- Within 7.5 mph → score > 0.5
- Within 15 mph → score > 0
- Beyond 15 mph → score = 0

### Launch Angle Scoring

A batted ball with LA = 30° compared to query LA = 22°:
```python
diff = abs(30 - 22)  # = 8
score = max(0, 1 - 8 / 20)  # = 0.6
```

The 20° tolerance means:
- Within 10° → score > 0.5
- Within 20° → score > 0
- Beyond 20° → score = 0

### Spray Angle Calculation

Spray angle is calculated from Statcast hit coordinates:

```python
# hc_x: 0 = left field line, 125 = center, 250 = right field line
# Convert to spray angle: -45 (pull) to +45 (opposite field)
spray_angle = ((hc_x - 125) / 125) * 45
```

For a right-handed batter:
- Negative spray angle = pulled to left field
- Positive spray angle = hit to right field

## Context-Aware Outcome Filtering

When `outs` and `bases_state` are provided, the algorithm filters out impossible outcomes:

| Outcome | Condition to Include |
|---------|---------------------|
| `sac_fly` | `outs < 2` AND runner on 3B |
| `double_play` | At least one runner on base (`bases_state > 0`) |
| `force_out` | Runner on 1B (`bases_state & 1 != 0`) |

### Example

With `outs=2` and `bases_state=0` (bases empty):
- `sac_fly` removed (need < 2 outs and runner on 3rd)
- `double_play` removed (need runners on base)
- `force_out` removed (need runner on 1st)

This prevents the algorithm from predicting outcomes that can't happen in the current game state.

## Outcome Event Mapping

Raw Statcast events are mapped to outcome categories:

| Statcast Event | Outcome Category |
|----------------|------------------|
| `single` | `single` |
| `double` | `double` |
| `triple` | `triple` |
| `home_run` | `home_run` |
| `field_out` | Redistributed by BB type |
| `grounded_into_double_play` | `double_play` |
| `double_play` | `double_play` |
| `force_out` | `force_out` |
| `fielders_choice` | `force_out` |
| `fielders_choice_out` | `force_out` |
| `sac_fly` | `sac_fly` |
| `sac_fly_double_play` | `sac_fly` |
| `field_error` | `field_error` |
| `sac_bunt` | `groundout` |
| `sac_bunt_double_play` | `double_play` |

### Flyout Redistribution

Generic `field_out` events are redistributed based on inferred batted ball type:

| Inferred BB Type | Launch Angle Range | Flyout Becomes |
|------------------|-------------------|----------------|
| `ground_ball` | LA < 10° | `groundout` |
| `line_drive` | 10° ≤ LA < 25° | `lineout` |
| `fly_ball` | 25° ≤ LA < 50° | `flyout` |
| `popup` | LA ≥ 50° | `popout` |

## Output Format

```python
{
    "algorithm_metadata": {
        "algorithm_name": "similarity",
        "instance_name": "similarity"
    },
    "basic_outcome_data": {
        "outcome_probs": {
            "single": 0.18,
            "double": 0.08,
            "triple": 0.01,
            "home_run": 0.05,
            "flyout": 0.45,
            "groundout": 0.15,
            "lineout": 0.05,
            "double_play": 0.03
        },
        "hit_probability": 0.32,
        "xba": 0.285,
        "bb_type_inferred": "fly_ball"
    },
    "detailed_outcome_data": {
        "sample_launch_speed_mean": 97.2,
        "sample_launch_angle_mean": 23.5,
        "expected_stats": {
            "xBA": 0.285,
            "xSLG": 0.520,
            "xwOBA": 0.345
        }
    },
    "prediction_metadata": {
        "n_batted_balls_sampled": 1500,
        "sample_pctg": 0.05,
        "similarity_weights": {
            "launch_speed": 0.45,
            "launch_angle": 0.40,
            "spray_angle": 0.05,
            "bases_state": 0.05,
            "outs": 0.03,
            "date": 0.02
        }
    }
}
```

## Use Cases

### Evaluating Contact Quality

```python
# Hard-hit line drive
result = await client.predict_batted_ball(
    launch_speed=105.0,
    launch_angle=18.0
)
print(f"Hit probability: {result['basic_outcome_data']['hit_probability']:.1%}")
print(f"xBA: {result['basic_outcome_data']['xba']:.3f}")
```

### Context-Specific Outcomes

```python
# Fly ball with runner on 3rd, 1 out
result = await client.predict_batted_ball(
    launch_speed=95.0,
    launch_angle=35.0,
    outs=1,
    bases_state=4  # Runner on 3rd only
)
# sac_fly will be included in outcome_probs
print(f"Sac fly prob: {result['basic_outcome_data']['outcome_probs'].get('sac_fly', 0):.1%}")
```

### Spray Angle Impact

```python
# Same contact, different spray angles
for spray in [-30, 0, 30]:  # Pull, center, oppo
    result = await client.predict_batted_ball(
        launch_speed=98.0,
        launch_angle=15.0,
        spray_angle=spray
    )
    print(f"Spray {spray:+d}°: hit prob = {result['basic_outcome_data']['hit_probability']:.1%}")
```

## Data Source

Batted ball predictions use league-wide Statcast data (not pitcher or batter specific by default). The algorithm fetches all batted balls and filters by similarity.

To include batter-specific patterns, provide `batter_id`:

```python
result = await client.predict_batted_ball(
    launch_speed=100.0,
    launch_angle=20.0,
    batter_id=592450  # Will boost scores for Judge's batted balls
)
```

Note: `batter_id` currently has 0 weight in the similarity formula, so it doesn't affect scoring. This may change in future versions.
