# Caching

PitchPredict caches Statcast data on disk to reduce repeated network calls. Caching is enabled by default via `enable_cache=True`.

## What Gets Cached

- Pitcher pitch histories keyed by MLBAM ID and coverage end date.
- Batter pitch histories keyed by MLBAM ID and coverage end date.
- Batted ball datasets keyed by coverage range.
- Player name lookups (normalized name + fuzzy flag) to MLBAM IDs.

## Cache Layout

Cache files live in `cache_dir` (default `.pitchpredict_cache`) using Parquet for DataFrames and JSON for metadata:

- `pitcher/{pitcher_id}.parquet` + `pitcher/{pitcher_id}.meta.json`
- `batter/{batter_id}.parquet` + `batter/{batter_id}.meta.json`
- `batted_ball/batted_balls.parquet` + `batted_ball/batted_balls.meta.json`
- `players/name_to_id.json`
- `players/name_to_records.json`
- `players/id_to_record.json`

## Incremental Updates

When a request asks for an `end_date` later than cached coverage, PitchPredict fetches only the missing tail of data and appends it to the cache. If the requested range starts earlier than the cached range, the full range is fetched.

## Configuration

```python
from pitchpredict import PitchPredict

client = PitchPredict(
    enable_cache=True,
    cache_dir=".pitchpredict_cache",
)
```

To refresh cached data, delete the cache directory and rerun your requests.

## CLI Cache Commands

The CLI exposes cache inspection and management commands:

```bash
# Show cache size and per-category stats
pitchpredict cache status

# Clear specific categories or the full cache
pitchpredict cache clear --category pitcher
pitchpredict cache clear --confirm

# Warm caches for a player or league-wide batted balls
pitchpredict cache warm "Aaron Judge"        # pitcher + batter
pitchpredict cache warm 592450 --type batter
pitchpredict cache warm --type batted-ball --seasons 4
```

All cache commands accept `--cache-dir` to target a non-default location.
