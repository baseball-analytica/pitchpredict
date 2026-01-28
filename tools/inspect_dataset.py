#!/usr/bin/env python3

"""
Dataset inspection script for PitchPredict.

Loads train, val, and test splits and reports comprehensive statistics including:
- Token counts and distributions
- Context variable ranges and unique ID counts
- File sizes
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Ensure the project sources are importable when the script runs directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from tools.deep.types import (
    PitchToken,
    TokenCategory,
    get_category,
    get_tokens_in_category,
)
from tools.deep.nn import (
    _CONTEXT_FIELD_SPECS,
    _context_field_path,
    TOKEN_DTYPE,
)


# ID fields that should show unique counts
ID_FIELDS = {
    "pitcher_id",
    "batter_id",
    "fielder_2_id",
    "fielder_3_id",
    "fielder_4_id",
    "fielder_5_id",
    "fielder_6_id",
    "fielder_7_id",
    "fielder_8_id",
    "fielder_9_id",
}

# Categorical fields (handedness)
CATEGORICAL_FIELDS = {"pitcher_throws", "batter_hits"}

# Fields that need special decoding display
DECODED_FIELDS = {
    "pitcher_age",
    "batter_age",
    "score_bat",
    "score_fld",
    "pitch_number",
    "game_date",
    "strike_zone_top",
    "strike_zone_bottom",
}


@dataclass
class SplitStats:
    """Statistics for a single dataset split."""

    name: str
    token_count: int = 0
    total_size_bytes: int = 0
    token_counts: Counter = field(default_factory=Counter)
    category_counts: dict[TokenCategory, Counter] = field(default_factory=dict)
    context_stats: dict[str, dict[str, Any]] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect PitchPredict dataset and report statistics."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="/raid/kline/pitchpredict/.pitchpredict_session_data",
        help="Base data directory (default: %(default)s).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated list of splits to load (default: %(default)s).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top tokens to show per category (default: %(default)s).",
    )
    return parser.parse_args()


def get_split_dir(base_dir: Path, split_name: str) -> Path:
    """Get the directory for a given split."""
    if split_name == "train":
        return base_dir
    return base_dir / split_name


def load_tokens(data_dir: Path) -> np.memmap | None:
    """Load tokens from a split directory."""
    tokens_path = data_dir / "pitch_seq.bin"
    if not tokens_path.exists():
        return None
    return np.memmap(tokens_path, dtype=TOKEN_DTYPE, mode="r")


def compute_token_stats(
    tokens: np.ndarray,
) -> tuple[Counter, dict[TokenCategory, Counter]]:
    """Compute token counts and category-wise counts."""
    token_counts: Counter = Counter()
    category_counts: dict[TokenCategory, Counter] = {
        cat: Counter() for cat in TokenCategory
    }

    # Count all tokens
    unique, counts = np.unique(tokens, return_counts=True)
    for token_val, count in zip(unique, counts):
        try:
            token = PitchToken(int(token_val))
            token_counts[token] = int(count)
            category = get_category(token)
            category_counts[category][token] = int(count)
        except ValueError:
            token_counts[f"UNKNOWN_{token_val}"] = int(count)

    return token_counts, category_counts


def compute_context_stats(
    data_dir: Path, token_count: int
) -> dict[str, dict[str, Any]]:
    """Compute statistics for all context fields."""
    context_stats: dict[str, dict[str, Any]] = {}
    context_prefix = data_dir / "pitch_context"

    for field_name, spec in _CONTEXT_FIELD_SPECS.items():
        path = _context_field_path(str(context_prefix), field_name)
        if not os.path.exists(path):
            continue

        data = np.memmap(path, dtype=spec.dtype, mode="r")
        stats: dict[str, Any] = {"file_size": os.path.getsize(path)}

        if field_name in ID_FIELDS:
            # For ID fields, count unique values
            unique_vals = np.unique(data)
            stats["unique_count"] = len(unique_vals)
            stats["min_id"] = int(unique_vals.min()) if len(unique_vals) > 0 else None
            stats["max_id"] = int(unique_vals.max()) if len(unique_vals) > 0 else None
        elif field_name in CATEGORICAL_FIELDS:
            # For categorical fields, show distribution
            unique, counts = np.unique(data, return_counts=True)
            total = counts.sum()
            distribution = {}
            for val, cnt in zip(unique, counts):
                decoded = spec.decode(val)
                distribution[decoded] = {
                    "count": int(cnt),
                    "pct": float(cnt) / total * 100,
                }
            stats["distribution"] = distribution
        else:
            # For numeric fields, compute min/max/mean/std
            stats["raw_min"] = float(data.min())
            stats["raw_max"] = float(data.max())
            stats["raw_mean"] = float(data.mean())
            stats["raw_std"] = float(data.std())

            # Decode values if applicable
            if field_name in DECODED_FIELDS:
                stats["decoded_min"] = spec.decode(stats["raw_min"])
                stats["decoded_max"] = spec.decode(stats["raw_max"])
                stats["decoded_mean"] = spec.decode(stats["raw_mean"])

        context_stats[field_name] = stats

    return context_stats


def compute_split_stats(data_dir: Path, split_name: str) -> SplitStats | None:
    """Compute all statistics for a single split."""
    tokens = load_tokens(data_dir)
    if tokens is None:
        return None

    stats = SplitStats(name=split_name)
    stats.token_count = len(tokens)

    # Token file size
    tokens_path = data_dir / "pitch_seq.bin"
    stats.total_size_bytes = os.path.getsize(tokens_path)

    # Token distribution
    stats.token_counts, stats.category_counts = compute_token_stats(tokens)

    # Context stats
    stats.context_stats = compute_context_stats(data_dir, stats.token_count)

    # Add context file sizes to total
    for field_stats in stats.context_stats.values():
        stats.total_size_bytes += field_stats.get("file_size", 0)

    return stats


def format_number(n: int | float) -> str:
    """Format a number with commas."""
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{n:,}"


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def print_split_summary(stats: SplitStats) -> None:
    """Print summary statistics for a single split."""
    # Count pitches (pitch type tokens), plate appearances (PA_END), sessions (SESSION_END)
    pitch_count = sum(stats.category_counts.get(TokenCategory.PITCH_TYPE, {}).values())
    pa_count = stats.token_counts.get(PitchToken.PA_END, 0)
    session_count = stats.token_counts.get(PitchToken.SESSION_END, 0)

    print(
        f"  {stats.name.upper():8} | Tokens: {format_number(stats.token_count):>15} | "
        f"Pitches: {format_number(pitch_count):>12} | "
        f"PAs: {format_number(pa_count):>10} | "
        f"Size: {format_size(stats.total_size_bytes):>10}"
    )


def print_combined_stats(all_stats: list[SplitStats], top_n: int, args: argparse.Namespace) -> None:
    """Print combined statistics across all splits."""
    print(f"\n{'=' * 60}")
    print(f"  COMBINED STATISTICS")
    print(f"{'=' * 60}")

    # Aggregate totals
    total_tokens = sum(s.token_count for s in all_stats)
    total_size = sum(s.total_size_bytes for s in all_stats)

    # Aggregate token counts
    combined_token_counts: Counter = Counter()
    combined_category_counts: dict[TokenCategory, Counter] = {
        cat: Counter() for cat in TokenCategory
    }
    for stats in all_stats:
        combined_token_counts.update(stats.token_counts)
        for cat, counts in stats.category_counts.items():
            combined_category_counts[cat].update(counts)

    print(f"\n--- Summary ---")
    print(f"Total Tokens: {format_number(total_tokens)}")
    print(f"Total Size: {format_size(total_size)}")

    pitch_count = sum(
        combined_category_counts.get(TokenCategory.PITCH_TYPE, {}).values()
    )
    pa_count = combined_token_counts.get(PitchToken.PA_END, 0)
    session_count = combined_token_counts.get(PitchToken.SESSION_END, 0)

    print(f"Estimated Pitches: {format_number(pitch_count)}")
    print(f"Plate Appearances: {format_number(pa_count)}")
    if session_count > 0:
        print(f"Sessions: {format_number(session_count)}")

    # Compute session length statistics
    all_session_lengths: list[int] = []
    for stats in all_stats:
        split_dir = get_split_dir(Path(args.data_dir), stats.name)
        tokens = load_tokens(split_dir)
        if tokens is not None:
            # Find SESSION_START positions
            session_starts = np.where(tokens == PitchToken.SESSION_START.value)[0]
            if len(session_starts) > 1:
                # Session length = distance between consecutive SESSION_STARTs
                lengths = np.diff(session_starts)
                all_session_lengths.extend(lengths.tolist())
            # Add length of last session (from last SESSION_START to end or SESSION_END)
            if len(session_starts) > 0:
                last_start = session_starts[-1]
                # Find next SESSION_END after last SESSION_START
                session_ends_after = np.where(tokens[last_start:] == PitchToken.SESSION_END.value)[0]
                if len(session_ends_after) > 0:
                    last_len = session_ends_after[0] + 1  # +1 to include SESSION_END
                    all_session_lengths.append(last_len)

    if all_session_lengths:
        lengths_arr = np.array(all_session_lengths)
        print(f"\n--- Session Length Statistics ---")
        print(f"  Min: {format_number(int(lengths_arr.min()))} tokens")
        print(f"  Max: {format_number(int(lengths_arr.max()))} tokens")
        print(f"  Mean: {lengths_arr.mean():.1f} tokens")
        print(f"  Median: {format_number(int(np.median(lengths_arr)))} tokens")
        print(f"  Std Dev: {lengths_arr.std():.1f} tokens")

    # Token distribution
    print(f"\n--- Token Distribution by Category ---")
    for category in TokenCategory:
        cat_counts = combined_category_counts.get(category, {})
        if not cat_counts:
            continue
        total_in_cat = sum(cat_counts.values())
        pct_of_total = (total_in_cat / total_tokens * 100) if total_tokens > 0 else 0
        print(
            f"\n{category.name}: {format_number(total_in_cat)} ({pct_of_total:.1f}% of all tokens)"
        )

        top_tokens = cat_counts.most_common(top_n)
        for token, count in top_tokens:
            token_pct = (count / total_in_cat * 100) if total_in_cat > 0 else 0
            print(f"  {token.name}: {format_number(count)} ({token_pct:.1f}%)")

    # Context variables - aggregate from all splits
    # For numeric fields, we compute weighted averages and combined min/max
    print(f"\n--- Context Variables ---")

    # ID fields - sum unique counts (note: may overcount if same IDs in multiple splits)
    print(f"\nID Fields (unique counts per split summed):")
    for field_name in sorted(ID_FIELDS):
        total_unique = sum(
            s.context_stats.get(field_name, {}).get("unique_count", 0)
            for s in all_stats
        )
        print(f"  {field_name}: {format_number(total_unique)} unique")

    # Compute total unique fielder IDs across all fielder positions (fast method)
    fielder_fields = [f"fielder_{i}_id" for i in range(2, 10)]
    unique_arrays: list[np.ndarray] = []
    for stats in all_stats:
        split_dir = get_split_dir(Path(args.data_dir), stats.name)
        context_prefix = split_dir / "pitch_context"
        for field_name in fielder_fields:
            if field_name in stats.context_stats:
                path = _context_field_path(str(context_prefix), field_name)
                if os.path.exists(path):
                    spec = _CONTEXT_FIELD_SPECS[field_name]
                    data = np.memmap(path, dtype=spec.dtype, mode="r")
                    # Use np.unique which is much faster than .tolist() + set
                    unique_arrays.append(np.unique(data))
    # Combine all unique arrays and get final unique count
    if unique_arrays:
        all_unique = np.unique(np.concatenate(unique_arrays))
        # Exclude 0 (padding/missing value)
        all_unique = all_unique[all_unique != 0]
        print(f"\n  ** Total unique fielders (all positions): {format_number(len(all_unique))} **")
    else:
        print(f"\n  ** Total unique fielders (all positions): 0 **")

    # Categorical fields - aggregate distributions
    print(f"\nCategorical Fields:")
    for field_name in sorted(CATEGORICAL_FIELDS):
        # Aggregate counts across splits
        combined_dist: dict[str, int] = {}
        for stats in all_stats:
            if field_name in stats.context_stats:
                dist = stats.context_stats[field_name].get("distribution", {})
                for key, val in dist.items():
                    combined_dist[key] = combined_dist.get(key, 0) + val["count"]
        total = sum(combined_dist.values())
        if total > 0:
            dist_str = ", ".join(
                f"{k}={v / total * 100:.1f}%" for k, v in sorted(combined_dist.items())
            )
            print(f"  {field_name}: {dist_str}")

    # Numeric fields - compute combined min/max and weighted mean
    print(f"\nNumeric Fields (raw encoded + decoded where applicable):")

    # Get all numeric field names from first split
    numeric_fields = []
    if all_stats:
        for field_name in all_stats[0].context_stats:
            if field_name not in ID_FIELDS and field_name not in CATEGORICAL_FIELDS:
                if "raw_min" in all_stats[0].context_stats[field_name]:
                    numeric_fields.append(field_name)

    for field_name in numeric_fields:
        # Aggregate across splits
        raw_mins = []
        raw_maxs = []
        weighted_sum = 0.0
        total_weight = 0

        for stats in all_stats:
            if field_name in stats.context_stats:
                fs = stats.context_stats[field_name]
                raw_mins.append(fs["raw_min"])
                raw_maxs.append(fs["raw_max"])
                weighted_sum += fs["raw_mean"] * stats.token_count
                total_weight += stats.token_count

        if not raw_mins:
            continue

        raw_min = min(raw_mins)
        raw_max = max(raw_maxs)
        raw_mean = weighted_sum / total_weight if total_weight > 0 else 0

        # Get decode function from first split that has this field
        spec = _CONTEXT_FIELD_SPECS.get(field_name)

        if field_name in DECODED_FIELDS and spec:
            dec_min = spec.decode(raw_min)
            dec_max = spec.decode(raw_max)
            dec_mean = spec.decode(raw_mean)
            print(f"  {field_name}:")
            print(f"    raw: [{raw_min:.3f}, {raw_max:.3f}] (mean: {raw_mean:.3f})")
            print(f"    decoded: [{dec_min}, {dec_max}] (mean: {dec_mean})")
        else:
            print(
                f"  {field_name}: [{raw_min:.3f}, {raw_max:.3f}] (mean: {raw_mean:.3f})"
            )


def main() -> None:
    args = parse_args()
    base_dir = args.data_dir.resolve()
    splits = [s.strip() for s in args.splits.split(",")]

    print("=" * 60)
    print("  PitchPredict Dataset Statistics")
    print("=" * 60)
    print(f"\nBase directory: {base_dir}")
    print(f"Splits to inspect: {', '.join(splits)}")

    all_stats: list[SplitStats] = []

    for split_name in splits:
        split_dir = get_split_dir(base_dir, split_name)
        if not split_dir.exists():
            print(f"\nWarning: Split directory not found: {split_dir}")
            continue

        print(f"\nLoading {split_name} split...", end="", flush=True)
        stats = compute_split_stats(split_dir, split_name)
        if stats is None:
            print(f" FAILED (no pitch_seq.bin found)")
            continue
        print(" done")

        all_stats.append(stats)

    if not all_stats:
        print("\nNo valid splits found!")
        return

    # Print per-split summaries
    print(f"\n--- Per-Split Summary ---")
    for stats in all_stats:
        print_split_summary(stats)

    # Print combined/detailed stats
    print_combined_stats(all_stats, args.top_n, args)

    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
