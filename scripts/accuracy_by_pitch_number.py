#!/usr/bin/env python3
"""
Analyze accuracy by pitch number within a session.

Instead of binning by token position (which creates a sawtooth due to the 16-token
pitch structure), this script groups by actual pitch number (1st pitch, 2nd pitch, etc.)
to see how accuracy changes as the session progresses.

Also breaks down accuracy by token category within each pitch to see which
parts of a pitch get easier/harder with more context.

Usage:
    uv run python scripts/accuracy_by_pitch_number.py \
        --predictions new_val_preds.npz new_test_preds.npz \
        --data_dir /raid/kline/pitchpredict/.pitchpredict_session_data \
        --output_dir new_category_analysis
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from pitchpredict.backend.algs.deep.types import (
    PitchToken,
    TokenCategory,
)
from pitchpredict.backend.algs.deep.nn import TOKEN_DTYPE, _CONTEXT_FIELD_SPECS
from pitchpredict.backend.algs.deep.dataset import SESSION_START_TOKEN


# =============================================================================
# Token Category Info
# =============================================================================

_TOKEN_RANGES = [
    (PitchToken.PAD, PitchToken.PAD, TokenCategory.PAD),
    (PitchToken.SESSION_START, PitchToken.SESSION_START, TokenCategory.SESSION_START),
    (PitchToken.SESSION_END, PitchToken.SESSION_END, TokenCategory.SESSION_END),
    (PitchToken.PA_START, PitchToken.PA_START, TokenCategory.PA_START),
    (PitchToken.PA_END, PitchToken.PA_END, TokenCategory.PA_END),
    (PitchToken.IS_CH, PitchToken.IS_AB, TokenCategory.PITCH_TYPE),
    (PitchToken.SPEED_IS_LT65, PitchToken.SPEED_IS_GT105, TokenCategory.SPEED),
    (PitchToken.SPIN_RATE_IS_LT750, PitchToken.SPIN_RATE_IS_GT3250, TokenCategory.SPIN_RATE),
    (PitchToken.SPIN_AXIS_IS_0_30, PitchToken.SPIN_AXIS_IS_330_360, TokenCategory.SPIN_AXIS),
    (PitchToken.RELEASE_POS_X_IS_LTN4, PitchToken.RELEASE_POS_X_IS_GT4, TokenCategory.RELEASE_POS_X),
    (PitchToken.RELEASE_POS_Z_IS_LT4, PitchToken.RELEASE_POS_Z_IS_GT7, TokenCategory.RELEASE_POS_Z),
    (PitchToken.RELEASE_EXTENSION_IS_LT5, PitchToken.RELEASE_EXTENSION_IS_GT75, TokenCategory.RELEASE_EXTENSION),
    (PitchToken.VX0_IS_LTN15, PitchToken.VX0_IS_GT15, TokenCategory.VX0),
    (PitchToken.VY0_IS_LTN150, PitchToken.VY0_IS_GTN100, TokenCategory.VY0),
    (PitchToken.VZ0_IS_LTN10, PitchToken.VZ0_IS_GT15, TokenCategory.VZ0),
    (PitchToken.AX_IS_LTN25, PitchToken.AX_IS_GT25, TokenCategory.AX),
    (PitchToken.AY_IS_LT15, PitchToken.AY_IS_GT40, TokenCategory.AY),
    (PitchToken.AZ_IS_LTN45, PitchToken.AZ_IS_GTN15, TokenCategory.AZ),
    (PitchToken.PLATE_POS_X_IS_LTN2, PitchToken.PLATE_POS_X_IS_GT2, TokenCategory.PLATE_POS_X),
    (PitchToken.PLATE_POS_Z_IS_LTN1, PitchToken.PLATE_POS_Z_IS_GT5, TokenCategory.PLATE_POS_Z),
    (PitchToken.RESULT_IS_BALL, PitchToken.RESULT_IS_AUTOMATIC_STRIKE, TokenCategory.RESULT),
]

CATEGORY_RANGES: dict[TokenCategory, tuple[int, int]] = {}
for start_tok, end_tok, cat in _TOKEN_RANGES:
    CATEGORY_RANGES[cat] = (start_tok.value, end_tok.value)

# The 16 pitch attribute categories in order
PITCH_CATEGORIES = [
    TokenCategory.PITCH_TYPE,
    TokenCategory.SPEED,
    TokenCategory.SPIN_RATE,
    TokenCategory.SPIN_AXIS,
    TokenCategory.RELEASE_POS_X,
    TokenCategory.RELEASE_POS_Z,
    TokenCategory.RELEASE_EXTENSION,
    TokenCategory.VX0,
    TokenCategory.VY0,
    TokenCategory.VZ0,
    TokenCategory.AX,
    TokenCategory.AY,
    TokenCategory.AZ,
    TokenCategory.PLATE_POS_X,
    TokenCategory.PLATE_POS_Z,
    TokenCategory.RESULT,
]

CATEGORY_DISPLAY_NAMES = {
    TokenCategory.PITCH_TYPE: "Pitch Type",
    TokenCategory.SPEED: "Speed",
    TokenCategory.SPIN_RATE: "Spin Rate",
    TokenCategory.SPIN_AXIS: "Spin Axis",
    TokenCategory.RELEASE_POS_X: "Release X",
    TokenCategory.RELEASE_POS_Z: "Release Z",
    TokenCategory.RELEASE_EXTENSION: "Extension",
    TokenCategory.VX0: "Vel X",
    TokenCategory.VY0: "Vel Y",
    TokenCategory.VZ0: "Vel Z",
    TokenCategory.AX: "Accel X",
    TokenCategory.AY: "Accel Y",
    TokenCategory.AZ: "Accel Z",
    TokenCategory.PLATE_POS_X: "Plate X",
    TokenCategory.PLATE_POS_Z: "Plate Z",
    TokenCategory.RESULT: "Result",
}


def get_target_category(target: int) -> TokenCategory | None:
    """Get the category for a target token."""
    for cat, (start, end) in CATEGORY_RANGES.items():
        if start <= target <= end:
            return cat
    return None


# =============================================================================
# Data Loading
# =============================================================================

def load_tokens(data_dir: str, split: str) -> np.ndarray:
    """Load tokens for a split."""
    if split == "train":
        path = os.path.join(data_dir, "pitch_seq.bin")
    else:
        path = os.path.join(data_dir, split, "pitch_seq.bin")
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def get_session_boundaries(tokens: np.ndarray) -> np.ndarray:
    """Get session start indices."""
    return np.where(tokens == SESSION_START_TOKEN)[0]


def infer_split_from_filename(filepath: str) -> str:
    """Infer split name from filename."""
    basename = os.path.basename(filepath).lower()
    if "val" in basename:
        return "val"
    return "test"


def load_and_align_predictions(
    predictions_file: str,
    data_dir: str,
    split: str,
) -> dict[str, np.ndarray]:
    """
    Load predictions and compute pitch number for each prediction.

    Returns dict with targets, predictions, pitch_number_in_session, and token_category.
    """
    # Load predictions
    preds = np.load(predictions_file)
    targets = preds["targets"]
    predictions = preds["predictions"]
    session_ids = preds["session_ids"]

    # Load tokens to find session structure
    tokens = load_tokens(data_dir, split)
    session_starts = get_session_boundaries(tokens)

    n_preds = len(targets)

    # For each prediction, compute:
    # 1. Which pitch number in the session (0-indexed)
    # 2. Which token category it belongs to
    pitch_number_in_session = np.zeros(n_preds, dtype=np.int32)
    token_category_idx = np.zeros(n_preds, dtype=np.int32)  # Index into PITCH_CATEGORIES

    # Count predictions per session
    unique_sessions, session_counts = np.unique(session_ids, return_counts=True)
    session_pred_counts = dict(zip(unique_sessions, session_counts))

    # Process each session
    idx = 0
    for sess_id in range(len(session_starts)):
        if sess_id not in session_pred_counts:
            continue

        n_sess_preds = session_pred_counts[sess_id]
        sess_start = session_starts[sess_id]

        # Get the tokens for this session's predictions
        # We need to count pitches (PITCH_TYPE tokens) to determine pitch number
        pitch_count = 0
        token_in_pitch = 0  # Which token within the current pitch (0-15)

        for local_pos in range(n_sess_preds):
            abs_pos = sess_start + local_pos
            target_token = targets[idx]
            target_cat = get_target_category(int(target_token))

            # Check if this is a pitch attribute token
            if target_cat in PITCH_CATEGORIES:
                cat_idx = PITCH_CATEGORIES.index(target_cat)

                # If we hit PITCH_TYPE, we're starting a new pitch
                if target_cat == TokenCategory.PITCH_TYPE:
                    pitch_count += 1
                    token_in_pitch = 0

                pitch_number_in_session[idx] = pitch_count
                token_category_idx[idx] = cat_idx
                token_in_pitch += 1
            else:
                # PA_START, PA_END, etc. - assign current pitch number
                pitch_number_in_session[idx] = pitch_count
                token_category_idx[idx] = -1  # Not a pitch attribute

            idx += 1

    return {
        "targets": targets,
        "predictions": predictions,
        "session_ids": session_ids,
        "pitch_number": pitch_number_in_session,
        "token_category_idx": token_category_idx,
    }


# =============================================================================
# Analysis
# =============================================================================

def compute_accuracy_by_pitch_number(
    targets: np.ndarray,
    predictions: np.ndarray,
    pitch_numbers: np.ndarray,
    token_category_idx: np.ndarray,
    max_pitch: int = 100,
    min_samples: int = 500,
) -> dict:
    """
    Compute accuracy by pitch number in session.

    Returns dict with overall accuracy per pitch and per-category accuracy.
    """
    # Overall accuracy by pitch number
    pitch_correct = defaultdict(int)
    pitch_total = defaultdict(int)

    # Per-category accuracy by pitch number
    # cat_pitch_correct[cat_idx][pitch_num] = count
    cat_pitch_correct = defaultdict(lambda: defaultdict(int))
    cat_pitch_total = defaultdict(lambda: defaultdict(int))

    for target, pred, pitch_num, cat_idx in zip(
        targets, predictions, pitch_numbers, token_category_idx
    ):
        if pitch_num > max_pitch:
            continue
        if cat_idx < 0:  # Skip non-pitch-attribute tokens
            continue

        pitch_total[pitch_num] += 1
        cat_pitch_total[cat_idx][pitch_num] += 1

        if target == pred:
            pitch_correct[pitch_num] += 1
            cat_pitch_correct[cat_idx][pitch_num] += 1

    # Compute accuracies
    overall_acc = {}
    for pitch_num in sorted(pitch_total.keys()):
        if pitch_total[pitch_num] >= min_samples:
            overall_acc[pitch_num] = {
                "accuracy": pitch_correct[pitch_num] / pitch_total[pitch_num],
                "total": pitch_total[pitch_num],
            }

    per_category_acc = {}
    for cat_idx in range(len(PITCH_CATEGORIES)):
        cat = PITCH_CATEGORIES[cat_idx]
        cat_acc = {}
        for pitch_num in sorted(cat_pitch_total[cat_idx].keys()):
            if cat_pitch_total[cat_idx][pitch_num] >= min_samples // 16:  # Less samples per category
                cat_acc[pitch_num] = {
                    "accuracy": cat_pitch_correct[cat_idx][pitch_num] / cat_pitch_total[cat_idx][pitch_num],
                    "total": cat_pitch_total[cat_idx][pitch_num],
                }
        per_category_acc[cat.name] = cat_acc

    return {
        "overall": overall_acc,
        "per_category": per_category_acc,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_accuracy_by_pitch_number(
    results: dict,
    output_path: str,
) -> None:
    """Plot overall accuracy by pitch number."""
    overall = results["overall"]

    pitch_nums = sorted(overall.keys())
    accuracies = [overall[p]["accuracy"] for p in pitch_nums]
    totals = [overall[p]["total"] for p in pitch_nums]

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot accuracy
    color1 = "steelblue"
    ax1.plot(pitch_nums, accuracies, 'o-', markersize=4, linewidth=2, color=color1)
    ax1.set_xlabel("Pitch Number in Session", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_title("Prediction Accuracy by Pitch Number in Session\n(Does seeing more pitches help?)", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Dynamic y-axis
    min_acc = min(accuracies)
    max_acc = max(accuracies)
    padding = (max_acc - min_acc) * 0.3
    ax1.set_ylim(max(0, min_acc - padding), min(1, max_acc + padding))

    # Secondary axis for sample counts
    ax2 = ax1.twinx()
    color2 = "gray"
    ax2.bar(pitch_nums, totals, alpha=0.3, color=color2, width=0.8)
    ax2.set_ylabel("Sample Count", fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_category_accuracy_by_pitch(
    results: dict,
    output_path: str,
    categories_to_plot: list[TokenCategory] | None = None,
) -> None:
    """Plot per-category accuracy by pitch number."""
    if categories_to_plot is None:
        # Plot a subset of interesting categories
        categories_to_plot = [
            TokenCategory.PITCH_TYPE,
            TokenCategory.SPEED,
            TokenCategory.VY0,
            TokenCategory.PLATE_POS_X,
            TokenCategory.RESULT,
        ]

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(categories_to_plot)))

    for cat, color in zip(categories_to_plot, colors):
        cat_data = results["per_category"].get(cat.name, {})
        if not cat_data:
            continue

        pitch_nums = sorted(cat_data.keys())
        accuracies = [cat_data[p]["accuracy"] for p in pitch_nums]

        label = CATEGORY_DISPLAY_NAMES.get(cat, cat.name)
        ax.plot(pitch_nums, accuracies, 'o-', markersize=3, linewidth=1.5,
                label=label, color=color, alpha=0.8)

    ax.set_xlabel("Pitch Number in Session", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Per-Category Accuracy by Pitch Number", fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_category_heatmap(
    results: dict,
    output_path: str,
    max_pitch: int = 50,
) -> None:
    """Plot heatmap of accuracy by pitch number and category."""
    # Build matrix
    n_cats = len(PITCH_CATEGORIES)

    # Find common pitch numbers across categories
    all_pitch_nums = set()
    for cat in PITCH_CATEGORIES:
        cat_data = results["per_category"].get(cat.name, {})
        all_pitch_nums.update(p for p in cat_data.keys() if p <= max_pitch)

    pitch_nums = sorted(all_pitch_nums)
    if not pitch_nums:
        return

    # Build accuracy matrix
    acc_matrix = np.full((n_cats, len(pitch_nums)), np.nan)

    for cat_idx, cat in enumerate(PITCH_CATEGORIES):
        cat_data = results["per_category"].get(cat.name, {})
        for pitch_idx, pitch_num in enumerate(pitch_nums):
            if pitch_num in cat_data:
                acc_matrix[cat_idx, pitch_idx] = cat_data[pitch_num]["accuracy"]

    fig, ax = plt.subplots(figsize=(16, 8))

    im = ax.imshow(acc_matrix, aspect='auto', cmap='RdYlGn', vmin=0.2, vmax=1.0)

    # Labels
    ax.set_yticks(range(n_cats))
    ax.set_yticklabels([CATEGORY_DISPLAY_NAMES.get(c, c.name) for c in PITCH_CATEGORIES])

    # Show every 5th pitch number on x-axis
    x_ticks = list(range(0, len(pitch_nums), 5))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([pitch_nums[i] for i in x_ticks])

    ax.set_xlabel("Pitch Number in Session", fontsize=12)
    ax.set_ylabel("Token Category", fontsize=12)
    ax.set_title("Accuracy Heatmap: Category × Pitch Number", fontsize=14)

    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze accuracy by pitch number")
    parser.add_argument(
        "--predictions",
        type=str,
        nargs="+",
        default=["new_val_preds.npz", "new_test_preds.npz"],
        help="Path(s) to predictions.npz file(s)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/raid/kline/pitchpredict/.pitchpredict_session_data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="new_category_analysis",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--max_pitch",
        type=int,
        default=100,
        help="Maximum pitch number to analyze",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("ACCURACY BY PITCH NUMBER IN SESSION")
    print("="*80)

    # Load and process predictions
    print("\nLoading predictions and computing pitch numbers...")

    all_targets = []
    all_predictions = []
    all_pitch_numbers = []
    all_token_categories = []

    for pred_path in args.predictions:
        print(f"  Processing: {pred_path}")
        split = infer_split_from_filename(pred_path)
        data = load_and_align_predictions(pred_path, args.data_dir, split)

        all_targets.append(data["targets"])
        all_predictions.append(data["predictions"])
        all_pitch_numbers.append(data["pitch_number"])
        all_token_categories.append(data["token_category_idx"])

    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    pitch_numbers = np.concatenate(all_pitch_numbers)
    token_categories = np.concatenate(all_token_categories)

    print(f"\nTotal predictions: {len(targets):,}")
    print(f"Max pitch number: {pitch_numbers.max()}")

    # Compute accuracy
    print("\nComputing accuracy by pitch number...")
    results = compute_accuracy_by_pitch_number(
        targets, predictions, pitch_numbers, token_categories,
        max_pitch=args.max_pitch,
    )

    # Print summary
    print("\n" + "="*80)
    print("OVERALL ACCURACY BY PITCH NUMBER")
    print("="*80)

    overall = results["overall"]
    print(f"\n{'Pitch #':<10} {'Total':>12} {'Accuracy':>12}")
    print("-" * 40)

    for pitch_num in list(sorted(overall.keys()))[:20]:  # First 20
        stats = overall[pitch_num]
        print(f"{pitch_num:<10} {stats['total']:>12,} {stats['accuracy']:>12.4f}")

    if len(overall) > 20:
        print(f"... and {len(overall) - 20} more pitch numbers")

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)

    plot_accuracy_by_pitch_number(
        results,
        os.path.join(args.output_dir, "accuracy_by_pitch_number.png"),
    )

    plot_category_accuracy_by_pitch(
        results,
        os.path.join(args.output_dir, "category_accuracy_by_pitch.png"),
    )

    plot_category_heatmap(
        results,
        os.path.join(args.output_dir, "category_pitch_heatmap.png"),
        max_pitch=min(50, args.max_pitch),
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
