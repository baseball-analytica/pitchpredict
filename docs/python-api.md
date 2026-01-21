# Python API Reference

The `PitchPredict` class is the main interface for making predictions in Python.

## PitchPredict Class

```python
from pitchpredict import PitchPredict
```

### Constructor

```python
PitchPredict(
    enable_cache: bool = True,
    cache_dir: str = ".pitchpredict_cache",
    enable_logging: bool = True,
    log_dir: str = ".pitchpredict_logs",
    log_level_console: str = "INFO",
    log_level_file: str = "INFO",
    fuzzy_player_lookup: bool = True,
    algorithms: dict[str, PitchPredictAlgorithm] | None = None,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_cache` | `bool` | `True` | Enable caching of Statcast data |
| `cache_dir` | `str` | `".pitchpredict_cache"` | Directory for cache files |
| `enable_logging` | `bool` | `True` | Enable logging to console and file |
| `log_dir` | `str` | `".pitchpredict_logs"` | Directory for log files |
| `log_level_console` | `str` | `"INFO"` | Log level for console output |
| `log_level_file` | `str` | `"INFO"` | Log level for file output |
| `fuzzy_player_lookup` | `bool` | `True` | Enable fuzzy matching for player names |
| `algorithms` | `dict` | `None` | Custom algorithm implementations |

#### Example

```python
# Default configuration
client = PitchPredict()

# Custom configuration
client = PitchPredict(
    enable_cache=True,
    enable_logging=True,
    log_level_console="DEBUG",
    fuzzy_player_lookup=True
)
```

---

## Methods

### predict_pitcher

Predict the pitcher's next pitch given the game context.

```python
async def predict_pitcher(
    pitcher_name: str,
    batter_name: str,
    balls: int,
    strikes: int,
    score_bat: int,
    score_fld: int,
    game_date: str,
    algorithm: str,
) -> dict[str, Any]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `pitcher_name` | `str` | Pitcher's name (e.g., "Clayton Kershaw") |
| `batter_name` | `str` | Batter's name (e.g., "Aaron Judge") |
| `balls` | `int` | Current ball count (0-3) |
| `strikes` | `int` | Current strike count (0-2) |
| `score_bat` | `int` | Batting team's score |
| `score_fld` | `int` | Fielding team's score |
| `game_date` | `str` | Game date in "YYYY-MM-DD" format |
| `algorithm` | `str` | Algorithm to use: `"similarity"` or `"deep"` |

#### Returns

A dictionary containing:

| Key | Type | Description |
|-----|------|-------------|
| `basic_pitch_data` | `dict` | Pitch type probabilities, speed/location means |
| `detailed_pitch_data` | `dict` | Fastball vs offspeed breakdown, percentiles |
| `basic_outcome_data` | `dict` | Outcome probabilities, swing rates |
| `detailed_outcome_data` | `dict` | Contact quality metrics |
| `prediction_metadata` | `dict` | Timing and sample information |

#### Example

```python
import asyncio
from pitchpredict import PitchPredict

async def main():
    client = PitchPredict()

    result = await client.predict_pitcher(
        pitcher_name="Clayton Kershaw",
        batter_name="Aaron Judge",
        balls=1,
        strikes=2,
        score_bat=2,
        score_fld=1,
        game_date="2024-06-15",
        algorithm="similarity"
    )

    # Pitch type probabilities
    print(result["basic_pitch_data"]["pitch_type_probs"])
    # {'FF': 0.42, 'SL': 0.31, 'CU': 0.18, 'CH': 0.09}

    # Average pitch speed
    print(result["basic_pitch_data"]["pitch_speed_mean"])
    # 92.5

    # Outcome probabilities
    print(result["basic_outcome_data"]["outcome_probs"])
    # {'strike': 0.45, 'ball': 0.35, 'contact': 0.20}

asyncio.run(main())
```

---

### predict_batter

Predict the batter's outcome given a specific pitch.

```python
async def predict_batter(
    batter_name: str,
    pitcher_name: str,
    balls: int,
    strikes: int,
    score_bat: int,
    score_fld: int,
    game_date: str,
    pitch_type: str,
    pitch_speed: float,
    pitch_x: float,
    pitch_y: float,
    algorithm: str,
) -> dict[str, Any]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `batter_name` | `str` | Batter's name |
| `pitcher_name` | `str` | Pitcher's name |
| `balls` | `int` | Current ball count (0-3) |
| `strikes` | `int` | Current strike count (0-2) |
| `score_bat` | `int` | Batting team's score |
| `score_fld` | `int` | Fielding team's score |
| `game_date` | `str` | Game date in "YYYY-MM-DD" format |
| `pitch_type` | `str` | Pitch type code (e.g., "FF", "SL") |
| `pitch_speed` | `float` | Pitch speed in mph |
| `pitch_x` | `float` | Horizontal location at plate (feet from center) |
| `pitch_y` | `float` | Vertical location at plate (feet from ground) |
| `algorithm` | `str` | Algorithm to use: `"similarity"` or `"deep"` |

#### Returns

A dictionary containing:

| Key | Type | Description |
|-----|------|-------------|
| `basic_outcome_data` | `dict` | Outcome probabilities, swing rates |
| `detailed_outcome_data` | `dict` | Contact quality, expected stats |
| `prediction_metadata` | `dict` | Timing and sample information |

#### Example

```python
import asyncio
from pitchpredict import PitchPredict

async def main():
    client = PitchPredict()

    result = await client.predict_batter(
        batter_name="Aaron Judge",
        pitcher_name="Clayton Kershaw",
        balls=1,
        strikes=2,
        score_bat=2,
        score_fld=1,
        game_date="2024-06-15",
        pitch_type="FF",
        pitch_speed=95.0,
        pitch_x=0.5,
        pitch_y=2.5,
        algorithm="similarity"
    )

    # Swing probability
    print(result["basic_outcome_data"]["swing_probability"])

    # Contact event probabilities
    print(result["basic_outcome_data"]["contact_event_probs"])

asyncio.run(main())
```

---

### predict_batted_ball

Predict batted ball outcome probabilities given exit velocity, launch angle, and optional game context.

```python
async def predict_batted_ball(
    launch_speed: float,
    launch_angle: float,
    algorithm: str,
    spray_angle: float | None = None,
    bb_type: str | None = None,
    outs: int | None = None,
    bases_state: int | None = None,
    batter_id: int | None = None,
    game_date: str | None = None,
) -> dict[str, Any]
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `launch_speed` | `float` | Yes | Exit velocity in mph |
| `launch_angle` | `float` | Yes | Launch angle in degrees (-90 to 90) |
| `algorithm` | `str` | Yes | Algorithm to use: `"similarity"` |
| `spray_angle` | `float` | No | Horizontal direction (-45 to 45, 0 = center field) |
| `bb_type` | `str` | No | Batted ball type: `"ground_ball"`, `"line_drive"`, `"fly_ball"`, `"popup"` |
| `outs` | `int` | No | Current outs (0-2) |
| `bases_state` | `int` | No | Bases occupied bitmask (1=1B, 2=2B, 4=3B) |
| `batter_id` | `int` | No | MLBAM batter ID |
| `game_date` | `str` | No | Game date in "YYYY-MM-DD" format |

#### Returns

A dictionary containing:

| Key | Type | Description |
|-----|------|-------------|
| `basic_outcome_data` | `dict` | Outcome probabilities, hit probability, xBA, inferred batted ball type |
| `detailed_outcome_data` | `dict` | Sample statistics, expected stats (xBA, xSLG, xwOBA) |
| `prediction_metadata` | `dict` | Sample size and similarity weights |

#### Example

```python
import asyncio
from pitchpredict import PitchPredict

async def main():
    client = PitchPredict()

    # Basic prediction with just exit velocity and launch angle
    result = await client.predict_batted_ball(
        launch_speed=95.0,
        launch_angle=18.0,
        algorithm="similarity"
    )

    # Outcome probabilities
    print(result["basic_outcome_data"]["outcome_probs"])
    # {'single': 0.18, 'double': 0.08, 'triple': 0.01, 'home_run': 0.05, ...}

    # Hit probability
    print(result["basic_outcome_data"]["hit_probability"])
    # 0.32

    # Expected batting average
    print(result["basic_outcome_data"]["xba"])
    # 0.285

    # With game context for context-aware filtering
    result = await client.predict_batted_ball(
        launch_speed=102.0,
        launch_angle=25.0,
        algorithm="similarity",
        outs=1,
        bases_state=5  # runners on 1B and 3B
    )

    # sac_fly will be included (outs < 2 and runner on 3B)
    # double_play will be included (runners on base)
    print(result["basic_outcome_data"]["outcome_probs"])

asyncio.run(main())
```

---

## Response Data Structures

### basic_pitch_data

```python
{
    "pitch_type_probs": {
        "FF": 0.45,  # Four-seam fastball
        "SL": 0.30,  # Slider
        "CU": 0.15,  # Curveball
        "CH": 0.10   # Changeup
    },
    "pitch_speed_mean": 92.5,      # Average speed (mph)
    "pitch_speed_std": 4.2,        # Speed standard deviation
    "pitch_x_mean": 0.12,          # Average horizontal location (feet)
    "pitch_x_std": 0.85,           # Horizontal location std dev
    "pitch_z_mean": 2.45,          # Average vertical location (feet)
    "pitch_z_std": 0.72            # Vertical location std dev
}
```

### detailed_pitch_data

```python
{
    "pitch_prob_fastball": 0.55,   # Probability of fastball
    "pitch_prob_offspeed": 0.45,   # Probability of offspeed
    "pitch_data_fastballs": {
        "pitch_type_probs": {"FF": 0.70, "SI": 0.20, "FC": 0.10},
        "pitch_speed_mean": 95.2,
        "pitch_speed_p05": 92.0,   # 5th percentile
        "pitch_speed_p25": 94.0,   # 25th percentile
        "pitch_speed_p50": 95.5,   # Median
        "pitch_speed_p75": 96.5,   # 75th percentile
        "pitch_speed_p95": 98.0,   # 95th percentile
        # ... location percentiles
    },
    "pitch_data_offspeed": {
        # Similar structure for offspeed pitches
    },
    "pitch_data_overall": {
        # Combined statistics
    }
}
```

### basic_outcome_data

```python
{
    "outcome_probs": {
        "strike": 0.45,
        "ball": 0.35,
        "contact": 0.20
    },
    "swing_probability": 0.55,
    "swing_event_probs": {
        "swinging_strike": 0.30,
        "contact": 0.70
    },
    "contact_probability": 0.20,
    "contact_event_probs": {
        "single": 0.18,
        "double": 0.05,
        "home_run": 0.03,
        "field_out": 0.60,
        # ...
    }
}
```

### detailed_outcome_data

```python
{
    "swing_data": {
        "bat_speed_mean": 72.5,    # mph
        "bat_speed_std": 3.2,
        "swing_length_mean": 7.2,  # feet
        "swing_length_std": 0.8,
        # ... percentiles
    },
    "contact_data": {
        "bb_type_probs": {         # Batted ball types
            "fly_ball": 0.35,
            "ground_ball": 0.40,
            "line_drive": 0.25
        },
        "BA": 0.285,               # Batting average on contact
        "SLG": 0.520,              # Slugging on contact
        "wOBA": 0.380,             # wOBA on contact
        "exit_velocity_mean": 92.0,
        "launch_angle_mean": 12.5,
        "xBA": 0.290,              # Expected BA
        "xSLG": 0.510,             # Expected SLG
        "xwOBA": 0.375             # Expected wOBA
    }
}
```

### prediction_metadata

```python
{
    "start_time": "2024-06-15T10:30:00",
    "end_time": "2024-06-15T10:30:02",
    "duration": 2.5,               # seconds
    "n_pitches_total": 5000,       # Total pitches analyzed
    "n_pitches_sampled": 250,      # Pitches in sample
    "sample_pctg": 0.05            # Sample percentage
}
```

### batted_ball_basic_outcome_data

```python
{
    "outcome_probs": {
        "single": 0.18,
        "double": 0.08,
        "triple": 0.01,
        "home_run": 0.05,
        "groundout": 0.25,
        "flyout": 0.20,
        "lineout": 0.08,
        "popout": 0.02,
        "sac_fly": 0.03,       # Only if outs < 2 AND runner on 3B
        "double_play": 0.04,   # Only if runners on base
        "field_error": 0.02,
        "force_out": 0.04      # Only if force play possible
    },
    "hit_probability": 0.32,       # Probability of a hit
    "xba": 0.285,                  # Expected batting average
    "bb_type_inferred": "line_drive"  # Inferred from launch angle
}
```

### batted_ball_detailed_outcome_data

```python
{
    "sample_launch_speed_mean": 98.5,   # Mean EV of similar batted balls
    "sample_launch_angle_mean": 15.2,   # Mean LA of similar batted balls
    "expected_stats": {
        "xBA": 0.285,              # Expected batting average
        "xSLG": 0.520,             # Expected slugging
        "xwOBA": 0.380             # Expected wOBA
    }
}
```

### batted_ball_prediction_metadata

```python
{
    "n_batted_balls_sampled": 750,   # Number of similar batted balls
    "sample_pctg": 0.05,             # Sample percentage
    "similarity_weights": {
        "launch_speed": 0.45,        # Exit velocity weight
        "launch_angle": 0.40,        # Launch angle weight
        "spray_angle": 0.05,         # Spray angle weight
        "bases_state": 0.05,         # Bases state weight
        "outs": 0.03,                # Outs weight
        "date": 0.02                 # Date recency weight
    }
}
```

---

## Error Handling

PitchPredict raises `HTTPException` for errors:

```python
from fastapi import HTTPException

try:
    result = await client.predict_pitcher(...)
except HTTPException as e:
    if e.status_code == 400:
        print(f"Bad request: {e.detail}")
    elif e.status_code == 500:
        print(f"Server error: {e.detail}")
```

Common errors:

| Status | Description |
|--------|-------------|
| 400 | Invalid algorithm name |
| 500 | Player not found, data fetch error |
