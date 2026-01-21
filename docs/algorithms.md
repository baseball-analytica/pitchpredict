# Algorithms

PitchPredict supports two prediction algorithms: **similarity** and **deep learning**. This guide explains how each works and when to use them.

## Algorithm Comparison

| Feature | Similarity | Deep Learning |
|---------|-----------|---------------|
| **Approach** | Nearest-neighbor | Neural network (xLSTM) |
| **Speed** | Faster | Slower |
| **Data required** | Historical pitches | Pre-trained model |
| **Availability** | Ready to use | Requires trained model |
| **Sequence awareness** | No | Yes |

## Specifying an Algorithm

```python
# Python API
result = await client.predict_pitcher(
    ...,
    algorithm="similarity"  # or "deep"
)
```

```json
// REST API
{
  "algorithm": "similarity"
}
```

---

## Similarity Algorithm

The similarity algorithm uses weighted nearest-neighbor analysis to find historical pitches most similar to the current game context.

### How It Works

1. **Fetch historical data**: Retrieve all pitches thrown by the pitcher (or to the batter) from Statcast
2. **Calculate similarity scores**: Score each pitch based on how closely it matches the current context
3. **Sample top pitches**: Select the top 5% most similar pitches
4. **Aggregate statistics**: Compute probabilities and distributions from the sample

### Pitcher Prediction Weights

For `predict_pitcher`, similarity is calculated using:

| Factor | Weight | Description |
|--------|--------|-------------|
| Batter | 35% | Same batter = 1.0, different = 0.0 |
| Balls | 20% | Same ball count = 1.0, different = 0.0 |
| Strikes | 20% | Same strike count = 1.0, different = 0.0 |
| Batting team score | 10% | Same score = 1.0, different = 0.0 |
| Fielding team score | 10% | Same score = 1.0, different = 0.0 |
| Game date | 5% | Same date = 1.0, different = 0.0 |

**Similarity formula:**

```
similarity = 0.35 * batter_match +
             0.20 * balls_match +
             0.20 * strikes_match +
             0.10 * score_bat_match +
             0.10 * score_fld_match +
             0.05 * date_match
```

### Batter Prediction Weights

For `predict_batter`, additional pitch characteristics are considered:

| Factor | Weight |
|--------|--------|
| Pitcher | 25% |
| Balls | 15% |
| Strikes | 15% |
| Game date | 10% |
| Batting team score | 5% |
| Fielding team score | 5% |
| Pitch type | 5% |
| Pitch speed | 5% |
| Pitch horizontal location | 5% |
| Pitch vertical location | 5% |

### Batted Ball Prediction Weights

For `predict_batted_ball`, the algorithm uses **continuous similarity scores** (unlike the binary matching above):

| Factor | Weight | Scoring |
|--------|--------|---------|
| Exit velocity | 45% | `max(0, 1 - abs(diff) / 15.0)` — 15 mph tolerance |
| Launch angle | 40% | `max(0, 1 - abs(diff) / 20.0)` — 20° tolerance |
| Spray angle | 5% | `max(0, 1 - abs(diff) / 30.0)` — if provided |
| Bases state | 5% | 1.0 if exact match, 0.5 if both have runners, else 0.0 |
| Outs | 3% | Same outs = 1.0, different = 0.0 |
| Date | 2% | `max(0, 1 - abs(days_diff) / 365.0)` — recency bonus |

**Similarity formula:**

```
ev_score = max(0, 1 - abs(ev_diff) / 15.0)
la_score = max(0, 1 - abs(la_diff) / 20.0)

similarity = 0.45 * ev_score +
             0.40 * la_score +
             0.05 * spray_score +
             0.05 * bases_score +
             0.03 * outs_score +
             0.02 * date_score
```

**Context-aware outcome filtering:**

When game context is provided, impossible outcomes are filtered:

| Outcome | Condition to include |
|---------|---------------------|
| `sac_fly` | `outs < 2` AND runner on 3B |
| `double_play` | At least one runner on base |
| `force_out` | Runner on 1B (force play possible) |

### Sample Percentage

By default, the algorithm samples the top 5% most similar pitches (minimum 100 for batted balls). This can be configured:

```python
from pitchpredict.backend.algs.similarity.base import SimilarityAlgorithm

custom_alg = SimilarityAlgorithm(sample_pctg=0.10)  # Top 10%

client = PitchPredict(
    algorithms={"similarity": custom_alg}
)
```

### Pros and Cons

**Pros:**
- Fast predictions
- No training required
- Transparent methodology
- Works immediately with any pitcher/batter

**Cons:**
- Limited context awareness
- Treats each pitch independently
- Requires exact matches for full weight

---

## Deep Learning Algorithm

The deep learning algorithm uses an xLSTM (Extended Long Short-Term Memory) neural network trained on pitch sequences.

**Technical Writeup:** [Predicting MLB Pitch Sequences with xLSTM](https://panoramic-letter-5ff.notion.site/PitchPredict-Predicting-MLB-Pitch-Sequences-with-xLSTM-6eed91163e064299b726c3bd32eedb95)

### How It Works

1. **Tokenize pitch sequences**: Convert pitch attributes into a 270-token vocabulary
2. **Encode context**: Process 28 contextual features (game state, players, etc.)
3. **Generate predictions**: Use the trained model to predict the next token in the sequence
4. **Decode output**: Convert predicted tokens back to pitch probabilities

### Token Vocabulary

Pitches are encoded using 270 discrete tokens:

| Category | Tokens | Description |
|----------|--------|-------------|
| Session | 2 | SESSION_START, SESSION_END |
| Plate appearance | 2 | PA_START, PA_END |
| Pitch type | 21 | CH, CU, FC, FF, SI, SL, etc. |
| Speed | 43 | LT65, 65-105 (each mph), GT105 |
| Spin rate | 12 | 250 RPM bins from <750 to >3250 |
| Spin axis | 12 | 30-degree bins from 0-360 |
| Release position X | 35 | Horizontal release point |
| Release position Z | 14 | Vertical release point |
| Velocity vectors | 22 | VX0, VY0, VZ0 components |
| Acceleration | 27 | AX, AY, AZ components |
| Release extension | 7 | Extension from rubber |
| Plate position X | 18 | Horizontal at plate |
| Plate position Z | 26 | Vertical at plate |
| Result | 19 | ball, strike, foul, in_play, etc. |

### Token Grammar

Tokens follow a strict grammar for valid sequences:

```
SESSION_START → PA_START → PITCH_TYPE → SPEED → SPIN_RATE →
SPIN_AXIS → RELEASE_POS_X → RELEASE_POS_Z → VX0 → VY0 → VZ0 →
AX → AY → AZ → RELEASE_EXTENSION → PLATE_POS_X → PLATE_POS_Z →
RESULT → (PA_END or next PITCH_TYPE) → ... → SESSION_END
```

### Context Features

The model considers 28 contextual features:

| Feature | Type | Description |
|---------|------|-------------|
| `pitcher_id` | int | MLBAM pitcher ID |
| `batter_id` | int | MLBAM batter ID |
| `pitcher_age` | int | Pitcher's age |
| `pitcher_throws` | L/R | Throwing hand |
| `batter_age` | int | Batter's age |
| `batter_hits` | L/R | Batting side |
| `count_balls` | int | Ball count (0-3) |
| `count_strikes` | int | Strike count (0-2) |
| `outs` | int | Outs in inning (0-2) |
| `bases_state` | int | Runners on base (bitmask) |
| `score_bat` | int | Batting team score |
| `score_fld` | int | Fielding team score |
| `inning` | int | Current inning |
| `pitch_number` | int | Pitch count in PA |
| `number_through_order` | int | Times through order |
| `game_date` | str | Game date |
| `fielder_2_id` - `fielder_9_id` | int | Fielder IDs |
| `batter_days_since_prev_game` | int | Rest days |
| `pitcher_days_since_prev_game` | int | Rest days |
| `strike_zone_top` | float | Top of zone |
| `strike_zone_bottom` | float | Bottom of zone |

### Model Architecture

- **Model**: xLSTM (Extended LSTM with exponential gating)
- **Embedding dimension**: 256
- **Attention heads**: 8
- **Transformer blocks**: 6
- **Context adapter**: Embeds game state features

### Training

Training the deep learning model requires:

1. Building a token dataset from Statcast data
2. Training the xLSTM model
3. Saving checkpoints to `.pitchpredict_models/`

Scripts in the `scripts/` directory handle this:

```bash
python scripts/build_deep_model.py
python scripts/xlstm.py  # For distributed training
```

### Pros and Cons

**Pros:**
- Understands pitch sequences
- Considers full game context
- Can capture complex patterns
- Learns pitcher/batter tendencies

**Cons:**
- Requires trained model
- Slower inference
- More computational resources
- Training requires significant data

---

## Choosing an Algorithm

| Use Case | Recommended |
|----------|-------------|
| Quick predictions | similarity |
| Production API with low latency | similarity |
| Analyzing pitch sequences | deep |
| Research / analysis | deep |
| No trained model available | similarity |

The similarity algorithm is the default and works out of the box. The deep learning algorithm requires a trained model file.
