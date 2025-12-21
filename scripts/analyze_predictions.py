#!/usr/bin/env python3
"""
Analyze saved predictions from detailed_eval_xlstm.py.

Computes per-session and per-pitcher accuracy metrics,
identifies best/worst sessions and pitchers.

Usage:
    uv run python scripts/analyze_predictions.py \
        --predictions predictions.npz \
        --data_dir /path/to/test \
        --output results.json
"""

import argparse
import json
import os

import numpy as np

from pitchpredict.utils.player_lookup import PlayerNameCache, load_session_info


def compute_session_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    top_k_indices: np.ndarray,
    session_ids: np.ndarray,
    session_info: list[dict],
) -> list[dict]:
    """Compute accuracy metrics for each session."""
    unique_sessions = np.unique(session_ids)
    session_metrics = []

    for sid in unique_sessions:
        mask = session_ids == sid
        session_targets = targets[mask]
        session_preds = predictions[mask]

        # Top-1 accuracy
        correct = (session_targets == session_preds).sum()
        n_tokens = len(session_targets)
        accuracy = float(correct / n_tokens)

        # Top-5 accuracy
        session_top_k = top_k_indices[mask]
        top5_correct = (session_top_k == session_targets[:, None]).any(axis=1).sum()
        top5_accuracy = float(top5_correct / n_tokens)

        # Get session info
        info = session_info[sid]

        session_metrics.append({
            "session_id": int(sid),
            "pitcher_id": info["pitcher_id"],
            "batter_id": info["batter_id"],
            "game_date": info["game_date"],
            "n_tokens": int(n_tokens),
            "correct": int(correct),
            "accuracy": accuracy,
            "top5_accuracy": top5_accuracy,
        })

    return session_metrics


def compute_pitcher_metrics(
    session_metrics: list[dict],
    min_tokens: int,
) -> list[dict]:
    """Aggregate session metrics by pitcher."""
    pitcher_stats: dict[int, dict] = {}

    for sm in session_metrics:
        pid = sm["pitcher_id"]
        if pid not in pitcher_stats:
            pitcher_stats[pid] = {
                "pitcher_id": pid,
                "n_sessions": 0,
                "n_tokens": 0,
                "correct": 0,
            }
        pitcher_stats[pid]["n_sessions"] += 1
        pitcher_stats[pid]["n_tokens"] += sm["n_tokens"]
        pitcher_stats[pid]["correct"] += sm["correct"]

    # Compute accuracy and filter by min_tokens
    pitcher_metrics = []
    for ps in pitcher_stats.values():
        if ps["n_tokens"] >= min_tokens:
            ps["accuracy"] = ps["correct"] / ps["n_tokens"]
            pitcher_metrics.append(ps)

    return pitcher_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze predictions from detailed_eval_xlstm.py"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        nargs="+",
        default=["val_preds.npz", "test_preds.npz"],
        help="Path(s) to predictions.npz file(s). Multiple files will be aggregated.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/raid/kline/pitchpredict/.pitchpredict_data_fixed",
        help="Path to base data directory (contains val/, test/ subdirs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analyze_val_test_results.json",
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of best/worst to show (default: 10)",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=100,
        help="Minimum tokens for a pitcher to be included (default: 100)",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="player_names.json",
        help="Path to player name cache file (default: player_names.json)",
    )
    return parser.parse_args()


def infer_split_from_filename(filepath: str) -> str:
    """Infer split name (val/test) from filename."""
    basename = os.path.basename(filepath).lower()
    if "val" in basename:
        return "val"
    elif "test" in basename:
        return "test"
    else:
        # Default to test if can't infer
        print(f"Warning: Could not infer split from '{basename}', assuming 'test'")
        return "test"


def main() -> None:
    args = parse_args()

    # Load and aggregate predictions from all files
    all_targets = []
    all_predictions = []
    all_top_k_indices = []
    all_session_ids = []
    all_session_info = []
    session_id_offset = 0

    print(f"Loading predictions from {len(args.predictions)} file(s)...")

    for pred_path in args.predictions:
        print(f"\n  Loading: {pred_path}")
        preds = np.load(pred_path)

        targets = preds["targets"]
        predictions = preds["predictions"]
        top_k_indices = preds["top_k_indices"]
        session_ids = preds["session_ids"]

        # Infer split and load session info
        split = infer_split_from_filename(pred_path)
        split_dir = os.path.join(args.data_dir, split)
        print(f"    Split: {split}, loading session info from: {split_dir}")
        session_info = load_session_info(split_dir)
        print(f"    Tokens: {len(targets)}, Sessions: {len(session_info)}")

        # Offset session_ids to make them unique across files
        session_ids_offset = session_ids + session_id_offset

        all_targets.append(targets)
        all_predictions.append(predictions)
        all_top_k_indices.append(top_k_indices)
        all_session_ids.append(session_ids_offset)
        all_session_info.extend(session_info)

        session_id_offset += len(session_info)

    # Concatenate all arrays
    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    top_k_indices = np.concatenate(all_top_k_indices)
    session_ids = np.concatenate(all_session_ids)
    session_info = all_session_info

    print(f"\nTotal tokens: {len(targets)}")
    print(f"Total sessions: {len(session_info)}")

    # Overall accuracy
    total_correct = (targets == predictions).sum()
    overall_accuracy = float(total_correct / len(targets))

    top5_correct = (top_k_indices == targets[:, None]).any(axis=1).sum()
    overall_top5_accuracy = float(top5_correct / len(targets))

    print(f"\n{'='*50}")
    print("OVERALL METRICS")
    print(f"{'='*50}")
    print(f"Total tokens: {len(targets)}")
    print(f"Top-1 accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"Top-5 accuracy: {overall_top5_accuracy:.4f} ({overall_top5_accuracy*100:.2f}%)")

    # Per-session metrics
    print("\nComputing per-session metrics...")
    session_metrics = compute_session_metrics(
        targets, predictions, top_k_indices, session_ids, session_info
    )

    # Sort for best/worst
    sessions_by_accuracy = sorted(session_metrics, key=lambda x: x["accuracy"])
    best_sessions = sessions_by_accuracy[-args.top_n:][::-1]
    worst_sessions = sessions_by_accuracy[:args.top_n]

    # Collect unique player IDs for name lookup
    pitcher_ids = set(sm["pitcher_id"] for sm in session_metrics)
    batter_ids = set(sm["batter_id"] for sm in session_metrics)
    all_player_ids = list(pitcher_ids | batter_ids)

    # Look up player names
    print(f"\nLooking up player names (cache: {args.cache_path})...")
    cache = PlayerNameCache(args.cache_path)
    player_names = cache.batch_lookup(all_player_ids)

    # Add names to session metrics
    for sm in session_metrics:
        sm["pitcher_name"] = player_names.get(sm["pitcher_id"], f"Unknown ({sm['pitcher_id']})")
        sm["batter_name"] = player_names.get(sm["batter_id"], f"Unknown ({sm['batter_id']})")

    print(f"\n{'='*90}")
    print(f"TOP {args.top_n} BEST SESSIONS (highest accuracy)")
    print(f"{'='*90}")
    print(f"{'Session':>8}  {'Pitcher':<20} {'vs Batter':<20} {'Date':<12} {'Tokens':>6} {'Acc':>8}")
    print("-" * 90)
    for sm in best_sessions:
        print(f"{sm['session_id']:>8}  {sm['pitcher_name']:<20} {sm['batter_name']:<20} "
              f"{sm['game_date']:<12} {sm['n_tokens']:>6} {sm['accuracy']:>8.4f}")

    print(f"\n{'='*90}")
    print(f"TOP {args.top_n} WORST SESSIONS (lowest accuracy)")
    print(f"{'='*90}")
    print(f"{'Session':>8}  {'Pitcher':<20} {'vs Batter':<20} {'Date':<12} {'Tokens':>6} {'Acc':>8}")
    print("-" * 90)
    for sm in worst_sessions:
        print(f"{sm['session_id']:>8}  {sm['pitcher_name']:<20} {sm['batter_name']:<20} "
              f"{sm['game_date']:<12} {sm['n_tokens']:>6} {sm['accuracy']:>8.4f}")

    # Per-pitcher metrics
    print(f"\nComputing per-pitcher metrics (min_tokens={args.min_tokens})...")
    pitcher_metrics = compute_pitcher_metrics(session_metrics, args.min_tokens)
    print(f"Pitchers with >= {args.min_tokens} tokens: {len(pitcher_metrics)}")

    # Add names to pitcher metrics
    for pm in pitcher_metrics:
        pm["pitcher_name"] = player_names.get(pm["pitcher_id"], f"Unknown ({pm['pitcher_id']})")

    pitchers_by_accuracy = sorted(pitcher_metrics, key=lambda x: x["accuracy"])
    best_pitchers = pitchers_by_accuracy[-args.top_n:][::-1]
    worst_pitchers = pitchers_by_accuracy[:args.top_n]

    print(f"\n{'='*70}")
    print(f"TOP {args.top_n} BEST PREDICTED PITCHERS")
    print(f"{'='*70}")
    print(f"{'Pitcher':<25} {'Sessions':>10} {'Tokens':>10} {'Accuracy':>10}")
    print("-" * 70)
    for pm in best_pitchers:
        print(f"{pm['pitcher_name']:<25} {pm['n_sessions']:>10} {pm['n_tokens']:>10} "
              f"{pm['accuracy']:>10.4f}")

    print(f"\n{'='*70}")
    print(f"TOP {args.top_n} WORST PREDICTED PITCHERS")
    print(f"{'='*70}")
    print(f"{'Pitcher':<25} {'Sessions':>10} {'Tokens':>10} {'Accuracy':>10}")
    print("-" * 70)
    for pm in worst_pitchers:
        print(f"{pm['pitcher_name']:<25} {pm['n_sessions']:>10} {pm['n_tokens']:>10} "
              f"{pm['accuracy']:>10.4f}")

    # Save results
    results = {
        "overall": {
            "total_tokens": int(len(targets)),
            "accuracy": overall_accuracy,
            "top5_accuracy": overall_top5_accuracy,
        },
        "best_sessions": best_sessions,
        "worst_sessions": worst_sessions,
        "best_pitchers": best_pitchers,
        "worst_pitchers": worst_pitchers,
    }

    print(f"\nSaving results to: {args.output}")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
