# Configuration Guide

This guide covers how to customize the similarity algorithm's behavior.

## Creating Custom Algorithm Instances

The `SimilarityAlgorithm` class can be instantiated with custom settings:

```python
from pitchpredict import PitchPredict
from pitchpredict.backend.algs.similarity.base import SimilarityAlgorithm

# Create a custom instance
custom_alg = SimilarityAlgorithm(
    name="my-similarity",  # Custom name for logging/metadata
)

# Use it with PitchPredict
client = PitchPredict(
    algorithms={"similarity": custom_alg}
)
```

## Sample Percentage Tuning

The sample percentage controls how many similar pitches are used for aggregation.

### Default Behavior

- **Default**: 5% (`sample_pctg=0.05`)
- **Pitcher/Batted Ball minimum**: 100 samples
- **Batter**: No minimum (raw percentage)

### Per-Request Override

For pitcher predictions, you can override via the request:

```python
result = await client.predict_pitcher(
    pitcher_id=543037,
    batter_id=592450,
    sample_pctg=0.10  # Use top 10% instead of 5%
)
```

### Guidelines by Dataset Size

| Dataset Size | Recommended Sample % | Rationale |
|--------------|---------------------|-----------|
| < 500 pitches | 20-50% | Small sample, need more data |
| 500-2000 | 10-20% | Moderate sample |
| 2000-5000 | 5-10% | Default range |
| > 5000 | 3-5% | Large dataset, can be selective |

### Trade-offs

| Higher Sample % | Lower Sample % |
|-----------------|----------------|
| More diverse results | More focused on best matches |
| Better for sparse data | Better for dense data |
| May include less-similar pitches | May miss relevant variations |
| More stable probabilities | More context-sensitive |

## SimilarityWeights Configuration

The `SimilarityWeights` class defines raw weights for pitcher prediction:

```python
from pitchpredict.backend.algs.similarity.types import SimilarityWeights

# Default weights
weights = SimilarityWeights()

# View defaults
print(weights.model_dump())
# {
#     'batter_id': 1.0,
#     'pitcher_age': 0.6,
#     'pitcher_throws': 0.4,
#     ...
# }

# Get softmax-normalized weights
normalized = weights.softmax()
print(sum(normalized.values()))  # 1.0
```

### Default Weight Values

| Field | Raw Weight | Softmax (approx) |
|-------|------------|------------------|
| `batter_id` | 1.0 | 11.4% |
| `pitcher_age` | 0.6 | 7.6% |
| `pitcher_throws` | 0.4 | 6.3% |
| `batter_age` | 0.4 | 6.3% |
| `batter_hits` | 0.4 | 6.3% |
| `count_balls` | 0.5 | 6.9% |
| `count_strikes` | 0.5 | 6.9% |
| `outs` | 0.2 | 5.1% |
| `bases_state` | 0.3 | 5.7% |
| `score_bat` | 0.1 | 4.6% |
| `score_fld` | 0.1 | 4.6% |
| `inning` | 0.1 | 4.6% |
| `pitch_number` | 0.1 | 4.6% |
| `number_through_order` | 0.2 | 5.1% |
| `game_date` | 0.05 | 4.4% |
| `fielder_2_id` | 0.3 | 5.7% |
| `fielder_3_id` - `fielder_9_id` | 0.05 each | 4.4% each |
| `batter_days_since_prev_game` | 0.05 | 4.4% |
| `pitcher_days_since_prev_game` | 0.05 | 4.4% |
| `strike_zone_top` | 0.1 | 4.6% |
| `strike_zone_bottom` | 0.1 | 4.6% |

### Softmax Normalization

Raw weights are transformed using softmax:

```python
import numpy as np

def softmax(weights: dict[str, float]) -> dict[str, float]:
    exp_weights = {k: np.exp(v) for k, v in weights.items()}
    total = sum(exp_weights.values())
    return {k: v / total for k, v in exp_weights.items()}
```

This ensures:
1. All weights are positive
2. Weights sum to 1.0
3. Relative differences are preserved but compressed

### Custom Weights

Currently, custom weights require modifying the source or creating a custom algorithm subclass:

```python
from pitchpredict.backend.algs.similarity.base import SimilarityAlgorithm
from pitchpredict.backend.algs.similarity.types import SimilarityWeights

class CustomSimilarityAlgorithm(SimilarityAlgorithm):
    async def predict_pitcher(self, request, **kwargs):
        # Override default weights
        custom_weights = SimilarityWeights(
            batter_id=2.0,      # Double emphasis on batter
            count_balls=1.0,    # Double emphasis on count
            count_strikes=1.0,
            game_date=0.5,      # More weight on recency
        )
        weights = custom_weights.softmax()

        # Continue with modified weights
        pitches = await self._get_cached_pitches_for_pitcher(...)
        similar = await self._get_similar_pitches_for_pitcher(
            pitches=pitches,
            context=request,
            weights=weights,  # Use custom weights
            sample_pctg=kwargs.get("sample_pctg", 0.05)
        )
        # ... rest of implementation
```

## Performance Tuning

### Caching

The algorithm uses in-memory caching for pitcher data:

```python
# Cache is automatically managed
# Pitches are cached by pitcher_id with end_date

# To clear cache (e.g., after data update)
alg = client._algorithms["similarity"]
alg._pitcher_cache.clear()
```

### External Cache

For persistent caching across sessions, configure `PitchPredictCache`:

```python
from pitchpredict import PitchPredict
from pitchpredict.backend.caching import PitchPredictCache

cache = PitchPredictCache(cache_dir="~/.pitchpredict/cache")
client = PitchPredict(cache=cache)
```

See [Caching Documentation](../caching.md) for details.

### Batch Predictions

For multiple predictions, reuse the client to benefit from caching:

```python
client = PitchPredict()

# First call fetches data
result1 = await client.predict_pitcher(pitcher_id=543037, ...)

# Second call uses cached pitcher data
result2 = await client.predict_pitcher(pitcher_id=543037, ...)  # Faster
```

### Memory Considerations

Each cached pitcher stores:
- End date timestamp
- DataFrame with ~57 columns per pitch
- Typical size: 1-5 MB per active pitcher

For high-volume usage, consider:
- Limiting concurrent pitchers in cache
- Periodic cache clearing
- Using external cache with disk persistence

## Logging

The algorithm logs to `pitchpredict.backend.algs.similarity`:

```python
import logging

# Enable debug logging
logging.getLogger("pitchpredict.backend.algs.similarity").setLevel(logging.DEBUG)
```

Debug output includes:
- Pitch counts at each stage
- Timing information
- Warning for sparse data
