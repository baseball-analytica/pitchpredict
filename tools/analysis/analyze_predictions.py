#!/usr/bin/env python3
"""
Analyze saved predictions from detailed_eval_xlstm.py.

Computes per-session and per-pitcher accuracy metrics,
identifies best/worst sessions and pitchers.

Usage:
    uv run python -m tools.analysis.analyze_predictions \
        --predictions predictions.npz \
        --data_dir /path/to/test \
        --output results.json
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from tools.utils.player_lookup import PlayerNameCache, load_session_info
from tools.deep.types import PitchToken


# Pitch type token range
PITCH_TYPE_START = PitchToken.IS_CH.value  # 6
PITCH_TYPE_END = PitchToken.IS_AB.value    # 26

# Human-readable pitch type names
PITCH_TYPE_NAMES = {
    PitchToken.IS_CH.value: "CH",   # Changeup
    PitchToken.IS_CU.value: "CU",   # Curveball
    PitchToken.IS_FC.value: "FC",   # Cutter
    PitchToken.IS_EP.value: "EP",   # Eephus
    PitchToken.IS_FO.value: "FO",   # Forkball
    PitchToken.IS_FF.value: "FF",   # Four-seam
    PitchToken.IS_KN.value: "KN",   # Knuckleball
    PitchToken.IS_KC.value: "KC",   # Knuckle-curve
    PitchToken.IS_SC.value: "SC",   # Screwball
    PitchToken.IS_SI.value: "SI",   # Sinker
    PitchToken.IS_SL.value: "SL",   # Slider
    PitchToken.IS_SV.value: "SV",   # Slurve
    PitchToken.IS_FS.value: "FS",   # Splitter
    PitchToken.IS_ST.value: "ST",   # Sweeper
    PitchToken.IS_FA.value: "FA",   # Fastball
    PitchToken.IS_CS.value: "CS",   # Slow Curve
    PitchToken.IS_PO.value: "PO",   # Pitchout
    PitchToken.IS_UN.value: "UN",   # Unknown
    PitchToken.IS_IN.value: "IN",   # Intentional
    PitchToken.IS_AB.value: "AB",   # Automatic Ball
}


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
            "batter_ids": info.get("batter_ids", [info["batter_id"]]),  # List of all batters
            "batter_id": info["batter_id"],  # First batter for backwards compat
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


def compute_pitch_type_confusion(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray, list[str], dict]:
    """
    Compute confusion matrix for pitch type predictions only.

    Args:
        targets: Ground truth token IDs
        predictions: Predicted token IDs

    Returns:
        Tuple of (confusion_matrix, labels, stats_dict)
    """
    # Filter to only pitch type tokens
    mask = (targets >= PITCH_TYPE_START) & (targets <= PITCH_TYPE_END)
    pitch_targets = targets[mask]
    pitch_preds = predictions[mask]

    # Get unique pitch types that appear in targets
    unique_types = sorted(set(pitch_targets))
    labels = [PITCH_TYPE_NAMES.get(t, f"T{t}") for t in unique_types]
    n_classes = len(unique_types)

    # Build token_id -> index mapping
    token_to_idx = {t: i for i, t in enumerate(unique_types)}

    # Compute confusion matrix manually
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_tok, pred_tok in zip(pitch_targets, pitch_preds):
        true_idx = token_to_idx.get(true_tok)
        pred_idx = token_to_idx.get(pred_tok)
        if true_idx is not None and pred_idx is not None:
            cm[true_idx, pred_idx] += 1

    # Compute per-class stats
    stats = {}
    for i, token_id in enumerate(unique_types):
        label = labels[i]
        true_count = cm[i, :].sum()
        pred_count = cm[:, i].sum()
        correct = cm[i, i]
        precision = correct / pred_count if pred_count > 0 else 0.0
        recall = correct / true_count if true_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        stats[label] = {
            "token_id": int(token_id),
            "true_count": int(true_count),
            "pred_count": int(pred_count),
            "correct": int(correct),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    return cm, labels, stats


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    output_path: str,
    title: str = "Pitch Type Confusion Matrix",
) -> None:
    """
    Plot and save confusion matrix heatmap.

    Args:
        cm: Confusion matrix array
        labels: Class labels
        output_path: Path to save the plot
        title: Plot title
    """
    # Normalize by row (true labels) for better visualization
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    ax1 = axes[0]
    im1 = ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks(range(len(labels)))
    ax1.set_yticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title(f"{title} (Counts)")
    plt.colorbar(im1, ax=ax1)

    # Add text annotations for counts
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            if val > 0:
                color = "white" if val > cm.max() / 2 else "black"
                ax1.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=7)

    # Normalized (recall)
    ax2 = axes[1]
    im2 = ax2.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks(range(len(labels)))
    ax2.set_yticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title(f"{title} (Row-Normalized / Recall)")
    plt.colorbar(im2, ax=ax2)

    # Add text annotations for percentages
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_normalized[i, j]
            if val > 0.005:  # Only show if > 0.5%
                color = "white" if val > 0.5 else "black"
                ax2.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix plot saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze predictions from detailed_eval_xlstm.py"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        nargs="+",
        default=["new_val_preds.npz", "new_test_preds.npz"],
        help="Path(s) to predictions.npz file(s). Multiple files will be aggregated.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/raid/kline/pitchpredict/.pitchpredict_session_data",
        help="Path to base data directory (contains val/, test/ subdirs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="new_analyze_val_test_results.json",
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
    parser.add_argument(
        "--confusion_matrix",
        type=str,
        default="pitch_type_confusion.png",
        help="Path to save confusion matrix plot (default: pitch_type_confusion.png)",
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

    # Pitch type confusion matrix
    print("\nComputing pitch type confusion matrix...")
    cm, cm_labels, pitch_type_stats = compute_pitch_type_confusion(targets, predictions)

    # Print pitch type stats
    print(f"\n{'='*80}")
    print("PITCH TYPE METRICS")
    print(f"{'='*80}")
    print(f"{'Type':<6} {'True':>10} {'Pred':>10} {'Correct':>10} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 80)
    for label in cm_labels:
        s = pitch_type_stats[label]
        print(f"{label:<6} {s['true_count']:>10} {s['pred_count']:>10} {s['correct']:>10} "
              f"{s['precision']:>8.4f} {s['recall']:>8.4f} {s['f1']:>8.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(cm, cm_labels, args.confusion_matrix)

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
    batter_ids = set()
    for sm in session_metrics:
        batter_ids.update(sm["batter_ids"])
    all_player_ids = list(pitcher_ids | batter_ids)

    # Look up player names
    print(f"\nLooking up player names (cache: {args.cache_path})...")
    cache = PlayerNameCache(args.cache_path)
    player_names = cache.batch_lookup(all_player_ids)

    # Add names to session metrics
    for sm in session_metrics:
        sm["pitcher_name"] = player_names.get(sm["pitcher_id"], f"Unknown ({sm['pitcher_id']})")
        sm["batter_names"] = [
            player_names.get(bid, f"Unknown ({bid})") for bid in sm["batter_ids"]
        ]
        sm["batter_name"] = sm["batter_names"][0] if sm["batter_names"] else "Unknown"

    def format_batters(names: list[str], max_len: int = 40) -> str:
        """Format list of batter names, truncating if needed."""
        result = ", ".join(names)
        if len(result) > max_len:
            result = result[:max_len - 3] + "..."
        return result

    print(f"\n{'='*110}")
    print(f"TOP {args.top_n} BEST SESSIONS (highest accuracy)")
    print(f"{'='*110}")
    print(f"{'Session':>8}  {'Pitcher':<20} {'Batters':<40} {'Date':<12} {'Tokens':>6} {'Acc':>8}")
    print("-" * 110)
    for sm in best_sessions:
        batters_str = format_batters(sm['batter_names'])
        print(f"{sm['session_id']:>8}  {sm['pitcher_name']:<20} {batters_str:<40} "
              f"{sm['game_date']:<12} {sm['n_tokens']:>6} {sm['accuracy']:>8.4f}")

    print(f"\n{'='*110}")
    print(f"TOP {args.top_n} WORST SESSIONS (lowest accuracy)")
    print(f"{'='*110}")
    print(f"{'Session':>8}  {'Pitcher':<20} {'Batters':<40} {'Date':<12} {'Tokens':>6} {'Acc':>8}")
    print("-" * 110)
    for sm in worst_sessions:
        batters_str = format_batters(sm['batter_names'])
        print(f"{sm['session_id']:>8}  {sm['pitcher_name']:<20} {batters_str:<40} "
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
        "pitch_type_stats": pitch_type_stats,
        "pitch_type_confusion_matrix": {
            "labels": cm_labels,
            "matrix": cm.tolist(),
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
