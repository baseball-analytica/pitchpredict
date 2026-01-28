# Pitcher Prediction Mode

The pitcher prediction mode answers: **"What pitch will this pitcher throw in this situation?"**

## Workflow

```
┌─────────────────────┐
│  1. Receive Request │
│    (pitcher_id,     │
│     game context)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 2. Fetch Historical │
│    Pitches from     │
│    Statcast Cache   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 3. Calculate        │
│    Similarity Score │
│    for Each Pitch   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 4. Select Top N%    │
│    Most Similar     │
│    (min 100)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 5. Aggregate Stats  │
│    & Sample Pitches │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 6. Return Response  │
│    (probabilities,  │
│     distributions)  │
└─────────────────────┘
```

## Field-to-Column Mapping

The similarity algorithm maps API request fields to Statcast columns:

### Exact Match Fields

| API Field | Statcast Column | Description |
|-----------|-----------------|-------------|
| `batter_id` | `batter` | MLBAM batter ID |
| `pitcher_throws` | `p_throws` | Throwing hand (L/R) |
| `batter_hits` | `stand` | Batting side (L/R) |
| `count_balls` | `balls` | Ball count (0-3) |
| `count_strikes` | `strikes` | Strike count (0-2) |
| `outs` | `outs_when_up` | Current outs (0-2) |
| `inning` | `inning` | Inning number |
| `number_through_order` | `n_thruorder_pitcher` | Times through lineup |
| `fielder_2_id` | `fielder_2` | Catcher MLBAM ID |
| `fielder_3_id` | `fielder_3` | First baseman MLBAM ID |
| `fielder_4_id` | `fielder_4` | Second baseman MLBAM ID |
| `fielder_5_id` | `fielder_5` | Third baseman MLBAM ID |
| `fielder_6_id` | `fielder_6` | Shortstop MLBAM ID |
| `fielder_7_id` | `fielder_7` | Left fielder MLBAM ID |
| `fielder_8_id` | `fielder_8` | Center fielder MLBAM ID |
| `fielder_9_id` | `fielder_9` | Right fielder MLBAM ID |

### Numeric Fields (with Tolerance)

| API Field | Statcast Column | Tolerance | Description |
|-----------|-----------------|-----------|-------------|
| `pitcher_age` | `age_pit` | Column std-dev | Pitcher's age |
| `batter_age` | `age_bat` | Column std-dev | Batter's age |
| `score_bat` | `bat_score` | Column std-dev | Batting team score |
| `score_fld` | `fld_score` | Column std-dev | Fielding team score |
| `pitch_number` | `pitch_number` | Column std-dev | Pitch count in PA |
| `batter_days_since_prev_game` | `batter_days_since_prev_game` | Column std-dev | Batter rest days |
| `pitcher_days_since_prev_game` | `pitcher_days_since_prev_game` | Column std-dev | Pitcher rest days |
| `strike_zone_top` | `sz_top` | 0.2 ft | Top of strike zone |
| `strike_zone_bottom` | `sz_bot` | 0.1 ft | Bottom of strike zone |

### Special Fields

| API Field | Statcast Columns | Scoring |
|-----------|------------------|---------|
| `bases_state` | `on_1b`, `on_2b`, `on_3b` | Bitmask comparison |
| `game_date` | `game_date` / `game_date_dt` | Recency scoring |

## Scoring Details

### Exact Match Scoring

For categorical fields, scoring is binary:

```python
score = 1.0 if pitch_value == request_value else 0.0
```

### Numeric Tolerance Scoring

For numeric fields, the score decreases linearly as the difference increases:

```python
tolerance = column_std_dev  # or fixed value for strike zone
diff = abs(pitch_value - request_value)
score = max(0, 1 - diff / tolerance)
```

**Example**: If `pitcher_age` in the request is 30 and the column std-dev is 4:
- A pitch from age 30 → score = 1.0
- A pitch from age 28 → score = 1 - 2/4 = 0.5
- A pitch from age 26 → score = 1 - 4/4 = 0.0
- A pitch from age 24 → score = max(0, 1 - 6/4) = 0.0

### Bases State Scoring

The `bases_state` field uses a bitmask:
- Bit 0 (value 1): Runner on 1st base
- Bit 1 (value 2): Runner on 2nd base
- Bit 2 (value 4): Runner on 3rd base

**Examples:**
- `bases_state = 0`: Bases empty
- `bases_state = 1`: Runner on 1st
- `bases_state = 5`: Runners on 1st and 3rd (1 + 4)
- `bases_state = 7`: Bases loaded (1 + 2 + 4)

The similarity score counts mismatches:

```python
mismatches = (
    (runner_on_1b != target_on_1b) +
    (runner_on_2b != target_on_2b) +
    (runner_on_3b != target_on_3b)
)
score = 1 - (mismatches / 3)
```

**Example**: Request has `bases_state = 5` (1st and 3rd), historical pitch has runner on 1st only:
- 1B: match (1 mismatch = 0)
- 2B: match (both empty, 0)
- 3B: mismatch (1)
- Score = 1 - 1/3 ≈ 0.67

### Game Date Scoring

More recent games receive higher scores:

```python
days_diff = abs(request_date - pitch_date).days
score = max(0, 1 - days_diff / 365)
```

A pitch from exactly one year ago scores 0.0. Pitches from the same day score 1.0.

## Weight Normalization

Raw weights from `SimilarityWeights` are transformed using softmax:

```python
import numpy as np

raw_weights = {"batter_id": 1.0, "count_balls": 0.5, ...}
exp_weights = {k: np.exp(v) for k, v in raw_weights.items()}
normalized = {k: v / sum(exp_weights.values()) for k, v in exp_weights.items()}
```

This ensures:
1. All weights are positive
2. Weights sum to 1.0
3. Higher raw weights have proportionally more influence

## Missing Fields Handling

When a field is missing from the request or not present in the pitch data:

1. The field's contribution is **skipped entirely**
2. Weights are **not renormalized** after skipping
3. The remaining fields determine similarity

**Implication**: Providing more context fields generally improves prediction quality, but the algorithm gracefully handles partial information.

## Sample Selection

After scoring all pitches:

```python
n_samples = max(100, int(total_pitches * sample_pctg))

if n_samples >= total_pitches:
    # Return all pitches sorted by score
    similar = pitches.sort_values("similarity_score", ascending=False)
else:
    # Return top N most similar
    similar = pitches.nlargest(n_samples, "similarity_score")
```

The minimum of 100 samples ensures statistical reliability even with small sample percentages.

## Output Aggregation

From the similar pitches, the algorithm calculates:

**Basic pitch data:**
- `pitch_type_probs`: Probability of each pitch type
- `pitch_speed_mean`, `pitch_speed_std`: Velocity statistics
- `pitch_x_mean`, `pitch_x_std`: Horizontal location statistics
- `pitch_z_mean`, `pitch_z_std`: Vertical location statistics

**Detailed pitch data:**
- Separate statistics for fastballs (FF, FC, SI) vs offspeed
- Percentile distributions (5th, 25th, 50th, 75th, 95th)

**Outcome data:**
- Strike/ball/contact probabilities
- Swing and contact rates
- Exit velocity and launch angle distributions

## Edge Cases

### Sparse Data

When a pitcher has limited historical data:
- The 100-pitch minimum may return all available pitches
- Scores may be dominated by a few high-similarity matches
- Consider using xLSTM for pitchers with < 500 historical pitches

### Identical Scores

When multiple pitches have the same similarity score:
- `nlargest` returns them in arbitrary order
- Sampling from ties is effectively random
- This is intentional—it adds variety to predictions

### New Batters

When `batter_id` has no historical matchups with the pitcher:
- The batter_id contribution is 0 for all pitches
- Other context factors (count, outs, bases) drive similarity
- Predictions reflect how the pitcher behaves in similar situations

## Debugging Tips

### Examine Individual Scores

After prediction, the similar pitches DataFrame contains score columns:

```python
# Access internal state (for debugging)
alg = client._algorithms["similarity"]
# Scores are stored as score_{field} columns in the DataFrame
```

### Check Sample Quality

The response includes metadata:

```python
result.prediction_metadata["n_pitches_total"]    # Total pitches considered
result.prediction_metadata["n_pitches_sampled"]  # Pitches in final sample
result.prediction_metadata["sample_pctg"]        # Sample percentage used
```

If `n_pitches_sampled` equals `n_pitches_total`, the pitcher has limited data.

### Verify Context Impact

Compare predictions with and without specific context fields to understand their influence:

```python
# With full context
full = await client.predict_pitcher(pitcher_id=543037, batter_id=592450,
                                     count_balls=3, count_strikes=2, outs=2)

# Without count
partial = await client.predict_pitcher(pitcher_id=543037, batter_id=592450,
                                        outs=2)

# Compare pitch_type_probs to see count's influence
```
