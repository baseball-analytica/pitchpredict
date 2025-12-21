#!/usr/bin/env python3
"""
Compute baseline accuracies for comparison with the xLSTM model.

Baselines:
1. Uniform Random (Valid): Randomly select from grammatically valid next tokens
2. Most Frequent (Valid): Always predict the most common token within the valid category
3. Most Frequent (Global): Always predict the most common token in each category (ignoring grammar)

Usage:
    uv run python scripts/compute_baselines.py \
        --predictions new_val_preds.npz new_test_preds.npz \
        --data_dir /raid/kline/pitchpredict/.pitchpredict_session_data
"""

import argparse
import os
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt

from pitchpredict.backend.algs.deep.types import (
    PitchToken,
    TokenCategory,
    get_category,
    valid_next_tokens,
    get_tokens_in_category,
)
from pitchpredict.backend.algs.deep.nn import TOKEN_DTYPE
from pitchpredict.backend.algs.deep.dataset import SESSION_START_TOKEN


# =============================================================================
# Token Category Info
# =============================================================================

# Build category ranges
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


CATEGORY_DISPLAY_NAMES = {
    TokenCategory.PA_START: "PA Start",
    TokenCategory.PA_END: "PA End",
    TokenCategory.PITCH_TYPE: "Pitch Type",
    TokenCategory.SPEED: "Speed",
    TokenCategory.SPIN_RATE: "Spin Rate",
    TokenCategory.SPIN_AXIS: "Spin Axis",
    TokenCategory.RELEASE_POS_X: "Release Pos X",
    TokenCategory.RELEASE_POS_Z: "Release Pos Z",
    TokenCategory.RELEASE_EXTENSION: "Extension",
    TokenCategory.VX0: "Velocity X",
    TokenCategory.VY0: "Velocity Y",
    TokenCategory.VZ0: "Velocity Z",
    TokenCategory.AX: "Accel X",
    TokenCategory.AY: "Accel Y",
    TokenCategory.AZ: "Accel Z",
    TokenCategory.PLATE_POS_X: "Plate Pos X",
    TokenCategory.PLATE_POS_Z: "Plate Pos Z",
    TokenCategory.RESULT: "Result",
}


def get_target_category(target: int) -> TokenCategory | None:
    """Get the category for a target token."""
    for cat, (start, end) in CATEGORY_RANGES.items():
        if start <= target <= end:
            return cat
    return None


def count_tokens_in_category(cat: TokenCategory) -> int:
    """Count how many tokens are in a category."""
    start, end = CATEGORY_RANGES[cat]
    return end - start + 1


# =============================================================================
# Data Loading
# =============================================================================

def load_tokens(data_dir: str, split: str) -> np.ndarray:
    """Load tokens for a split."""
    # Train is directly in data_dir, val/test are in subdirectories
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
    elif "test" in basename:
        return "test"
    return "test"


# =============================================================================
# Baseline Computations
# =============================================================================

def compute_token_frequencies(data_dir: str, split: str = "train") -> dict[int, int]:
    """Compute token frequencies from training data."""
    tokens = load_tokens(data_dir, split)
    counter = Counter(tokens)
    return dict(counter)


def compute_category_frequencies(token_freqs: dict[int, int]) -> dict[TokenCategory, dict[int, int]]:
    """Compute per-category token frequencies."""
    cat_freqs: dict[TokenCategory, dict[int, int]] = defaultdict(dict)

    for token_id, count in token_freqs.items():
        cat = get_target_category(token_id)
        if cat is not None:
            cat_freqs[cat][token_id] = count

    return dict(cat_freqs)


def get_most_frequent_per_category(cat_freqs: dict[TokenCategory, dict[int, int]]) -> dict[TokenCategory, int]:
    """Get the most frequent token in each category."""
    most_freq = {}
    for cat, freqs in cat_freqs.items():
        if freqs:
            most_freq[cat] = max(freqs.keys(), key=lambda t: freqs[t])
    return most_freq


def compute_uniform_random_accuracy(targets: np.ndarray, prev_tokens: np.ndarray) -> dict[str, float]:
    """
    Compute expected accuracy if we randomly select from valid next tokens.

    For each target, the accuracy is 1/N where N is the number of valid next tokens
    given the previous token.
    """
    category_expected: dict[TokenCategory, list[float]] = defaultdict(list)

    for target, prev in zip(targets, prev_tokens):
        cat = get_target_category(int(target))
        if cat is None:
            continue

        # Get valid next tokens for this previous token
        try:
            prev_token = PitchToken(int(prev))
            valid_tokens = valid_next_tokens(prev_token)
            n_valid = len(valid_tokens)
            expected_acc = 1.0 / n_valid if n_valid > 0 else 0.0
        except (ValueError, KeyError):
            # Fallback: use category size
            expected_acc = 1.0 / count_tokens_in_category(cat)

        category_expected[cat].append(expected_acc)

    # Compute mean expected accuracy per category
    results = {}
    for cat, accs in category_expected.items():
        results[cat.name] = sum(accs) / len(accs) if accs else 0.0

    return results


def compute_most_frequent_accuracy(
    targets: np.ndarray,
    most_freq_per_cat: dict[TokenCategory, int],
) -> dict[str, float]:
    """
    Compute accuracy if we always predict the most frequent token in each category.
    """
    category_correct: dict[TokenCategory, int] = defaultdict(int)
    category_total: dict[TokenCategory, int] = defaultdict(int)

    for target in targets:
        cat = get_target_category(int(target))
        if cat is None:
            continue

        category_total[cat] += 1
        if int(target) == most_freq_per_cat.get(cat):
            category_correct[cat] += 1

    results = {}
    for cat in category_total:
        results[cat.name] = category_correct[cat] / category_total[cat]

    return results


def compute_model_accuracy(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Compute model accuracy per category."""
    category_correct: dict[TokenCategory, int] = defaultdict(int)
    category_total: dict[TokenCategory, int] = defaultdict(int)

    for target, pred in zip(targets, predictions):
        cat = get_target_category(int(target))
        if cat is None:
            continue

        category_total[cat] += 1
        if target == pred:
            category_correct[cat] += 1

    results = {}
    for cat in category_total:
        results[cat.name] = category_correct[cat] / category_total[cat]

    return results


def align_prev_tokens_with_predictions(
    predictions_file: str,
    data_dir: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the previous token for each prediction (needed for valid_next_tokens baseline).

    Returns: (targets, predictions, prev_tokens)
    """
    # Load predictions
    preds = np.load(predictions_file)
    targets = preds["targets"]
    predictions = preds["predictions"]
    session_ids = preds["session_ids"]

    # Load tokens
    tokens = load_tokens(data_dir, split)
    session_starts = get_session_boundaries(tokens)

    # For each prediction, find the previous token
    # The prediction at position i in a session predicts token at position i+1
    # So the "previous token" (context) is at position i

    n_preds = len(targets)
    prev_tokens = np.zeros(n_preds, dtype=TOKEN_DTYPE)

    # Count predictions per session
    unique_sessions, session_counts = np.unique(session_ids, return_counts=True)
    session_pred_counts = dict(zip(unique_sessions, session_counts))

    idx = 0
    for sess_id in range(len(session_starts)):
        if sess_id not in session_pred_counts:
            continue

        n_sess_preds = session_pred_counts[sess_id]
        sess_start = session_starts[sess_id]

        # For each prediction in this session, get the previous token
        for local_pos in range(n_sess_preds):
            abs_pos = sess_start + local_pos
            prev_tokens[idx] = tokens[abs_pos]
            idx += 1

    return targets, predictions, prev_tokens


# =============================================================================
# Plotting
# =============================================================================

def plot_baseline_comparison_bars(
    categories: list[str],
    random_accs: list[float],
    mostfreq_accs: list[float],
    model_accs: list[float],
    output_path: str,
) -> None:
    """Plot grouped bar chart comparing all three methods."""
    fig, ax = plt.subplots(figsize=(16, 7))

    x = np.arange(len(categories))
    width = 0.25

    bars1 = ax.bar(x - width, random_accs, width, label='Uniform Random', color='#d62728', alpha=0.8)
    bars2 = ax.bar(x, mostfreq_accs, width, label='Most Frequent', color='#ff7f0e', alpha=0.8)
    bars3 = ax.bar(x + width, model_accs, width, label='xLSTM Model', color='#2ca02c', alpha=0.8)

    ax.set_xlabel('Token Category', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Baseline Comparison: Random vs Most-Frequent vs xLSTM Model', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on model bars
    for bar, val in zip(bars3, model_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_improvement_over_baselines(
    categories: list[str],
    vs_random: list[float],
    vs_mostfreq: list[float],
    output_path: str,
) -> None:
    """Plot model improvement over baselines."""
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, [v * 100 for v in vs_random], width,
                   label='vs Random', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, [v * 100 for v in vs_mostfreq], width,
                   label='vs Most-Frequent', color='#9467bd', alpha=0.8)

    ax.set_xlabel('Token Category', fontsize=12)
    ax.set_ylabel('Improvement (percentage points)', fontsize=12)
    ax.set_title('xLSTM Model Improvement Over Baselines', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_comparison(
    overall_random: float,
    overall_mostfreq: float,
    overall_model: float,
    output_path: str,
) -> None:
    """Plot overall summary bar chart."""
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = ['Uniform\nRandom', 'Most\nFrequent', 'xLSTM\nModel']
    accuracies = [overall_random, overall_mostfreq, overall_model]
    colors = ['#d62728', '#ff7f0e', '#2ca02c']

    bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Overall Accuracy: Baselines vs Model', fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Add improvement annotations
    ax.annotate('', xy=(2, overall_model), xytext=(1, overall_mostfreq),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    mid_x = 1.5
    mid_y = (overall_model + overall_mostfreq) / 2
    improvement = (overall_model - overall_mostfreq) * 100
    ax.text(mid_x + 0.15, mid_y, f'+{improvement:.1f}pp', fontsize=11, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute baseline accuracies")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("="*80)
    print("BASELINE COMPARISON")
    print("="*80)

    # Load training data for frequency computation
    print("\nComputing token frequencies from training data...")
    token_freqs = compute_token_frequencies(args.data_dir, "train")
    cat_freqs = compute_category_frequencies(token_freqs)
    most_freq_per_cat = get_most_frequent_per_category(cat_freqs)

    print("Most frequent token per category:")
    for cat, token_id in sorted(most_freq_per_cat.items(), key=lambda x: x[0].name):
        if cat in CATEGORY_DISPLAY_NAMES:
            try:
                token_name = PitchToken(token_id).name
            except ValueError:
                token_name = str(token_id)
            cat_total = sum(cat_freqs[cat].values())
            token_freq = cat_freqs[cat][token_id]
            pct = 100 * token_freq / cat_total
            print(f"  {CATEGORY_DISPLAY_NAMES[cat]:<15}: {token_name:<30} ({pct:.1f}%)")

    # Load and process predictions
    print("\nLoading predictions and aligning with previous tokens...")

    all_targets = []
    all_predictions = []
    all_prev_tokens = []

    for pred_path in args.predictions:
        print(f"  Processing: {pred_path}")
        split = infer_split_from_filename(pred_path)
        targets, predictions, prev_tokens = align_prev_tokens_with_predictions(
            pred_path, args.data_dir, split
        )
        all_targets.append(targets)
        all_predictions.append(predictions)
        all_prev_tokens.append(prev_tokens)

    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    prev_tokens = np.concatenate(all_prev_tokens)

    print(f"\nTotal predictions: {len(targets):,}")

    # Compute accuracies
    print("\nComputing accuracies...")

    uniform_random_acc = compute_uniform_random_accuracy(targets, prev_tokens)
    most_freq_acc = compute_most_frequent_accuracy(targets, most_freq_per_cat)
    model_acc = compute_model_accuracy(targets, predictions)

    # Display results
    print("\n" + "="*80)
    print("RESULTS BY CATEGORY")
    print("="*80)

    # Categories to display (excluding structural tokens)
    display_cats = [
        "PITCH_TYPE", "SPEED", "SPIN_RATE", "SPIN_AXIS",
        "RELEASE_POS_X", "RELEASE_POS_Z", "RELEASE_EXTENSION",
        "VX0", "VY0", "VZ0", "AX", "AY", "AZ",
        "PLATE_POS_X", "PLATE_POS_Z", "RESULT",
    ]

    print(f"\n{'Category':<15} {'#Tokens':>8} {'Random':>10} {'MostFreq':>10} {'Model':>10} {'vs Random':>12} {'vs MostFreq':>12}")
    print("-" * 90)

    total_random = 0.0
    total_mostfreq = 0.0
    total_model = 0.0
    total_count = 0

    # Collect data for plotting
    plot_categories = []
    plot_random_accs = []
    plot_mostfreq_accs = []
    plot_model_accs = []
    plot_vs_random = []
    plot_vs_mostfreq = []

    for cat_name in display_cats:
        cat = TokenCategory[cat_name]
        n_tokens = count_tokens_in_category(cat)

        rand_acc = uniform_random_acc.get(cat_name, 0.0)
        freq_acc = most_freq_acc.get(cat_name, 0.0)
        mod_acc = model_acc.get(cat_name, 0.0)

        vs_random = mod_acc - rand_acc
        vs_mostfreq = mod_acc - freq_acc

        display_name = CATEGORY_DISPLAY_NAMES.get(cat, cat_name)
        print(f"{display_name:<15} {n_tokens:>8} {rand_acc:>10.4f} {freq_acc:>10.4f} {mod_acc:>10.4f} {vs_random:>+12.4f} {vs_mostfreq:>+12.4f}")

        # Collect for plotting
        plot_categories.append(display_name)
        plot_random_accs.append(rand_acc)
        plot_mostfreq_accs.append(freq_acc)
        plot_model_accs.append(mod_acc)
        plot_vs_random.append(vs_random)
        plot_vs_mostfreq.append(vs_mostfreq)

        # Weight by count for overall
        cat_start, cat_end = CATEGORY_RANGES[cat]
        cat_count = sum(1 for t in targets if cat_start <= t <= cat_end)
        total_random += rand_acc * cat_count
        total_mostfreq += freq_acc * cat_count
        total_model += mod_acc * cat_count
        total_count += cat_count

    print("-" * 90)

    overall_random = total_random / total_count
    overall_mostfreq = total_mostfreq / total_count
    overall_model = total_model / total_count

    print(f"{'OVERALL':<15} {'':>8} {overall_random:>10.4f} {overall_mostfreq:>10.4f} {overall_model:>10.4f} {overall_model - overall_random:>+12.4f} {overall_model - overall_mostfreq:>+12.4f}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nUniform Random Baseline:  {overall_random:.4f} ({overall_random*100:.2f}%)")
    print(f"Most Frequent Baseline:   {overall_mostfreq:.4f} ({overall_mostfreq*100:.2f}%)")
    print(f"xLSTM Model:              {overall_model:.4f} ({overall_model*100:.2f}%)")
    print(f"\nModel improvement over random:       {(overall_model - overall_random)*100:+.2f} percentage points")
    print(f"Model improvement over most-frequent: {(overall_model - overall_mostfreq)*100:+.2f} percentage points")

    # Relative improvement
    random_headroom = 1.0 - overall_random
    mostfreq_headroom = 1.0 - overall_mostfreq

    model_captures_random = (overall_model - overall_random) / random_headroom if random_headroom > 0 else 0
    model_captures_mostfreq = (overall_model - overall_mostfreq) / mostfreq_headroom if mostfreq_headroom > 0 else 0

    print(f"\nModel captures {model_captures_random*100:.1f}% of the gap between random and perfect")
    print(f"Model captures {model_captures_mostfreq*100:.1f}% of the gap between most-frequent and perfect")

    # Generate plots
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)

    plot_baseline_comparison_bars(
        plot_categories,
        plot_random_accs,
        plot_mostfreq_accs,
        plot_model_accs,
        os.path.join(args.output_dir, "baseline_comparison.png"),
    )

    plot_improvement_over_baselines(
        plot_categories,
        plot_vs_random,
        plot_vs_mostfreq,
        os.path.join(args.output_dir, "baseline_improvement.png"),
    )

    plot_summary_comparison(
        overall_random,
        overall_mostfreq,
        overall_model,
        os.path.join(args.output_dir, "baseline_summary.png"),
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
