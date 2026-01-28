# Code Examples

Practical examples for using the similarity algorithm.

## Python API Examples

### Basic Pitcher Prediction

```python
import asyncio
from pitchpredict import PitchPredict

async def basic_prediction():
    client = PitchPredict()

    result = await client.predict_pitcher(
        pitcher_id=543037,   # Clayton Kershaw
        batter_id=592450,    # Aaron Judge
        algorithm="similarity"
    )

    # Most likely pitch type
    probs = result.basic_pitch_data["pitch_type_probs"]
    most_likely = max(probs, key=probs.get)
    print(f"Most likely pitch: {most_likely} ({probs[most_likely]:.1%})")

    # Expected velocity
    print(f"Expected velocity: {result.basic_pitch_data['pitch_speed_mean']:.1f} mph")

asyncio.run(basic_prediction())
```

### Full Game Context

```python
async def full_context_prediction():
    client = PitchPredict()

    result = await client.predict_pitcher(
        # Required
        pitcher_id=543037,
        batter_id=592450,

        # Game situation
        count_balls=3,
        count_strikes=2,
        outs=2,
        inning=9,
        bases_state=5,        # Runners on 1st and 3rd (1 + 4)

        # Score
        score_bat=4,          # Batting team winning
        score_fld=3,

        # Player info
        pitcher_throws="L",
        batter_hits="R",
        pitcher_age=36,
        batter_age=32,

        # Timing
        game_date="2024-09-15",
        number_through_order=3,
        pitch_number=8,

        # Optional: fielders affect similarity
        fielder_2_id=543877,  # Catcher

        algorithm="similarity"
    )

    print("Full count, 2 outs, 9th inning prediction:")
    print(f"Pitch type probs: {result.basic_pitch_data['pitch_type_probs']}")
    print(f"Contact probability: {result.basic_outcome_data['contact_probability']:.1%}")

asyncio.run(full_context_prediction())
```

### Batter Pitch Scenario Analysis

```python
async def batter_scenario():
    client = PitchPredict()

    # How does Judge handle high fastballs from Kershaw?
    result = await client.predict_batter(
        pitcher_id=477132,   # Kershaw
        batter_id=592450,    # Judge
        pitch_type="FF",
        pitch_speed=93.0,
        pitch_x=0.0,         # Middle
        pitch_z=3.5,         # High in zone
        count_balls=0,
        count_strikes=0
    )

    print("Judge vs Kershaw high fastball:")
    print(f"Swing probability: {result['basic_outcome_data']['swing_probability']:.1%}")
    print(f"Contact probability: {result['basic_outcome_data']['contact_probability']:.1%}")

    if result['basic_outcome_data']['contact_probability'] > 0:
        contact = result['detailed_outcome_data']['contact_data']
        print(f"Average exit velocity: {contact['exit_velocity_mean']:.1f} mph")
        print(f"xBA: {contact['xBA']:.3f}")

asyncio.run(batter_scenario())
```

### Batted Ball Outcome Prediction

```python
async def batted_ball_prediction():
    client = PitchPredict()

    # Hard-hit line drive
    result = await client.predict_batted_ball(
        launch_speed=105.0,
        launch_angle=18.0,
        spray_angle=-15.0,   # Pulled slightly
        outs=1,
        bases_state=1        # Runner on first
    )

    print("Hard-hit line drive outcomes:")
    probs = result['basic_outcome_data']['outcome_probs']
    for outcome, prob in sorted(probs.items(), key=lambda x: -x[1]):
        if prob > 0.01:
            print(f"  {outcome}: {prob:.1%}")

    print(f"\nHit probability: {result['basic_outcome_data']['hit_probability']:.1%}")
    print(f"xBA: {result['basic_outcome_data']['xba']:.3f}")

asyncio.run(batted_ball_prediction())
```

### Comparing Pitchers Against Same Batter

```python
async def compare_pitchers():
    client = PitchPredict()

    batter_id = 592450  # Judge
    pitchers = [
        (543037, "Kershaw"),
        (477132, "Verlander"),
        (605483, "Cole"),
    ]

    print(f"Pitch type predictions vs Aaron Judge:\n")

    for pitcher_id, name in pitchers:
        result = await client.predict_pitcher(
            pitcher_id=pitcher_id,
            batter_id=batter_id,
            count_balls=0,
            count_strikes=0,
            algorithm="similarity"
        )

        probs = result.basic_pitch_data["pitch_type_probs"]
        top_3 = sorted(probs.items(), key=lambda x: -x[1])[:3]

        print(f"{name}:")
        for pitch, prob in top_3:
            print(f"  {pitch}: {prob:.1%}")
        print()

asyncio.run(compare_pitchers())
```

### Analyzing Sample Quality

```python
async def analyze_sample():
    client = PitchPredict()

    result = await client.predict_pitcher(
        pitcher_id=543037,
        batter_id=592450,
        count_balls=1,
        count_strikes=2,
        algorithm="similarity"
    )

    meta = result.prediction_metadata

    print("Sample Analysis:")
    print(f"  Total pitches available: {meta['n_pitches_total']}")
    print(f"  Pitches in sample: {meta['n_pitches_sampled']}")
    print(f"  Sample percentage: {meta['sample_pctg']:.1%}")
    print(f"  Actual percentage: {meta['n_pitches_sampled'] / meta['n_pitches_total']:.1%}")

    # Check if we hit the minimum
    if meta['n_pitches_sampled'] == 100 and meta['n_pitches_total'] > 2000:
        effective_pctg = 100 / meta['n_pitches_total']
        print(f"  Note: Using minimum 100 samples ({effective_pctg:.1%} effective)")

asyncio.run(analyze_sample())
```

### Count Progression Analysis

```python
async def count_progression():
    client = PitchPredict()

    pitcher_id = 543037
    batter_id = 592450

    counts = [
        (0, 0, "First pitch"),
        (1, 0, "Ahead 1-0"),
        (0, 1, "Behind 0-1"),
        (2, 0, "Ahead 2-0"),
        (0, 2, "Behind 0-2"),
        (3, 2, "Full count"),
    ]

    print("Pitch type by count:\n")

    for balls, strikes, label in counts:
        result = await client.predict_pitcher(
            pitcher_id=pitcher_id,
            batter_id=batter_id,
            count_balls=balls,
            count_strikes=strikes,
            algorithm="similarity"
        )

        probs = result.basic_pitch_data["pitch_type_probs"]

        # Calculate fastball percentage
        fb_pct = sum(probs.get(p, 0) for p in ["FF", "SI", "FC"])

        print(f"{label} ({balls}-{strikes}):")
        print(f"  Fastball: {fb_pct:.1%}")
        print(f"  Offspeed: {1 - fb_pct:.1%}")

asyncio.run(count_progression())
```

## REST API Examples

### Basic Pitcher Prediction

```bash
curl -X POST http://localhost:8056/predict/pitcher \
  -H "Content-Type: application/json" \
  -d '{
    "pitcher_id": 543037,
    "batter_id": 592450,
    "algorithm": "similarity"
  }'
```

### Full Context Request

```bash
curl -X POST http://localhost:8056/predict/pitcher \
  -H "Content-Type: application/json" \
  -d '{
    "pitcher_id": 543037,
    "batter_id": 592450,
    "count_balls": 3,
    "count_strikes": 2,
    "outs": 2,
    "inning": 9,
    "bases_state": 5,
    "score_bat": 4,
    "score_fld": 3,
    "pitcher_throws": "L",
    "batter_hits": "R",
    "game_date": "2024-09-15",
    "algorithm": "similarity"
  }'
```

### Batter Prediction

```bash
curl -X POST http://localhost:8056/predict/batter \
  -H "Content-Type: application/json" \
  -d '{
    "pitcher_id": 477132,
    "batter_id": 592450,
    "pitch_type": "FF",
    "pitch_speed": 95.0,
    "pitch_x": 0.0,
    "pitch_z": 2.5,
    "count_balls": 1,
    "count_strikes": 2,
    "algorithm": "similarity"
  }'
```

### Batted Ball Prediction

```bash
curl -X POST http://localhost:8056/predict/batted_ball \
  -H "Content-Type: application/json" \
  -d '{
    "launch_speed": 105.0,
    "launch_angle": 18.0,
    "spray_angle": -15.0,
    "outs": 1,
    "bases_state": 1,
    "algorithm": "similarity"
  }'
```

### Player Lookup (for IDs)

```bash
# Look up player by name
curl "http://localhost:8056/player/lookup?query=Aaron%20Judge"

# Fuzzy search
curl "http://localhost:8056/player/lookup?query=jdge&fuzzy=true"
```

### Parsing JSON Response

```bash
# Get most likely pitch type
curl -s -X POST http://localhost:8056/predict/pitcher \
  -H "Content-Type: application/json" \
  -d '{"pitcher_id": 543037, "batter_id": 592450, "algorithm": "similarity"}' \
  | jq '.basic_pitch_data.pitch_type_probs | to_entries | max_by(.value)'

# Get hit probability for batted ball
curl -s -X POST http://localhost:8056/predict/batted_ball \
  -H "Content-Type: application/json" \
  -d '{"launch_speed": 100, "launch_angle": 20, "algorithm": "similarity"}' \
  | jq '.basic_outcome_data.hit_probability'
```

## CLI Examples

### Predict Pitcher

```bash
# Basic prediction
pitchpredict predict pitcher "Clayton Kershaw" "Aaron Judge"

# With count
pitchpredict predict pitcher "Clayton Kershaw" "Aaron Judge" \
  --balls 3 --strikes 2

# With full context
pitchpredict predict pitcher "Clayton Kershaw" "Aaron Judge" \
  --balls 1 --strikes 2 \
  --outs 2 \
  --inning 7 \
  --bases 5 \
  --algorithm similarity
```

### Predict Batter

```bash
pitchpredict predict batter "Aaron Judge" "Clayton Kershaw" \
  --pitch-type FF \
  --speed 93.0 \
  --x 0.0 \
  --z 2.5
```

### Player Lookup

```bash
# Find player ID
pitchpredict player lookup "Aaron Judge"

# Fuzzy search
pitchpredict player lookup "jdge" --fuzzy
```
