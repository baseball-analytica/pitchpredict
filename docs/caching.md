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
