#!/usr/bin/env python3
"""
Comprehensive analysis of model predictions by token category and game context.

Computes:
- Per-category accuracy and confusion matrices
- Accuracy by count, inning, outs, base state
- Accuracy vs position in session (does context help?)
- Pitch transition matrix
- Calibration analysis
- Accuracy by handedness matchups

Usage:
    uv run python scripts/analyze_token_categories.py \
        --predictions new_val_preds.npz new_test_preds.npz \
        --data_dir /raid/kline/pitchpredict/.pitchpredict_session_data \
        --output_dir category_analysis/
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from pitchpredict.backend.algs.deep.types import (
    PitchToken,
    TokenCategory,
    _TOKEN_RANGES,
)
from pitchpredict.backend.algs.deep.nn import _CONTEXT_FIELD_SPECS, TOKEN_DTYPE
from pitchpredict.backend.algs.deep.dataset import SESSION_START_TOKEN, SESSION_END_TOKEN


# =============================================================================
# Token Category Lookups
# =============================================================================

def build_category_lookups() -> tuple[dict[int, TokenCategory], dict[TokenCategory, tuple[int, int]]]:
    """Build token_id -> category and category -> (start, end) mappings."""
    token_to_category: dict[int, TokenCategory] = {}
    category_ranges: dict[TokenCategory, tuple[int, int]] = {}

    for start_tok, end_tok, category in _TOKEN_RANGES:
        start_val = start_tok.value
        end_val = end_tok.value
        category_ranges[category] = (start_val, end_val)
        for val in range(start_val, end_val + 1):
            token_to_category[val] = category

    return token_to_category, category_ranges


TOKEN_TO_CATEGORY, CATEGORY_RANGES = build_category_lookups()


CATEGORY_DISPLAY_NAMES = {
    TokenCategory.PAD: "Padding",
    TokenCategory.SESSION_START: "Session Start",
    TokenCategory.SESSION_END: "Session End",
    TokenCategory.PA_START: "PA Start",
    TokenCategory.PA_END: "PA End",
    TokenCategory.PITCH_TYPE: "Pitch Type",
    TokenCategory.SPEED: "Speed (mph)",
    TokenCategory.SPIN_RATE: "Spin Rate",
    TokenCategory.SPIN_AXIS: "Spin Axis",
    TokenCategory.RELEASE_POS_X: "Release Pos X",
    TokenCategory.RELEASE_POS_Z: "Release Pos Z",
    TokenCategory.VX0: "Velocity X",
    TokenCategory.VY0: "Velocity Y",
    TokenCategory.VZ0: "Velocity Z",
    TokenCategory.AX: "Accel X",
    TokenCategory.AY: "Accel Y",
    TokenCategory.AZ: "Accel Z",
    TokenCategory.RELEASE_EXTENSION: "Extension",
    TokenCategory.PLATE_POS_X: "Plate Pos X",
    TokenCategory.PLATE_POS_Z: "Plate Pos Z",
    TokenCategory.RESULT: "Result",
}


def get_token_label(token_id: int) -> str:
    """Get human-readable label for a token."""
    try:
        tok = PitchToken(token_id)
        name = tok.name
        # Clean up the name by removing common prefixes
        prefixes = [
            "IS_", "SPEED_IS_", "SPIN_RATE_IS_", "SPIN_AXIS_IS_",
            "RELEASE_POS_X_IS_", "RELEASE_POS_Z_IS_", "RELEASE_EXTENSION_IS_",
            "VX0_IS_", "VY0_IS_", "VZ0_IS_", "AX_IS_", "AY_IS_", "AZ_IS_",
            "PLATE_POS_X_IS_", "PLATE_POS_Z_IS_", "RESULT_IS_"
        ]
        for prefix in prefixes:
            if name.startswith(prefix):
                return name[len(prefix):]
        return name
    except ValueError:
        return f"T{token_id}"


# =============================================================================
# Data Loading
# =============================================================================

def load_context_data(data_dir: str, split: str) -> dict[str, np.memmap]:
    """Load context memory-mapped files for a split."""
    split_dir = os.path.join(data_dir, split)
    contexts = {}

    for field_name, spec in _CONTEXT_FIELD_SPECS.items():
        path = os.path.join(split_dir, f"pitch_context_{field_name}.bin")
        if os.path.exists(path):
            contexts[field_name] = np.memmap(path, dtype=spec.dtype, mode="r")

    return contexts


def load_tokens(data_dir: str, split: str) -> np.memmap:
    """Load tokens for a split."""
    path = os.path.join(data_dir, split, "pitch_seq.bin")
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def get_session_boundaries(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get session start and end indices."""
    session_starts = np.where(tokens == SESSION_START_TOKEN)[0]
    session_ends = np.zeros(len(session_starts), dtype=np.int64)

    for i in range(len(session_starts) - 1):
        session_ends[i] = session_starts[i + 1]
    session_ends[-1] = len(tokens)

    return session_starts, session_ends


def align_predictions_with_context(
    predictions_file: str,
    data_dir: str,
    split: str,
) -> dict[str, np.ndarray]:
    """
    Align predictions with context data.

    The predictions are saved per-session, where for session s:
    - target[i] corresponds to tokens[i+1] in that session
    - context for predicting target[i] is at position i in the session

    Returns dict with targets, predictions, session_ids, and aligned context fields.
    """
    # Load predictions
    preds = np.load(predictions_file)
    targets = preds["targets"]
    predictions = preds["predictions"]
    session_ids = preds["session_ids"]
    top_k_probs = preds["top_k_probs"] if "top_k_probs" in preds else None

    # Load tokens and context
    tokens = load_tokens(data_dir, split)
    contexts = load_context_data(data_dir, split)
    session_starts, session_ends = get_session_boundaries(tokens)

    # Build alignment: for each prediction, find its absolute position in context
    n_preds = len(targets)
    abs_positions = np.zeros(n_preds, dtype=np.int64)

    # Count predictions per session
    unique_sessions, session_counts = np.unique(session_ids, return_counts=True)
    session_pred_counts = dict(zip(unique_sessions, session_counts))

    # Build position mapping
    idx = 0
    for sess_id in range(len(session_starts)):
        if sess_id not in session_pred_counts:
            continue
        n_sess_preds = session_pred_counts[sess_id]
        sess_start = session_starts[sess_id]

        # Predictions for this session correspond to context positions 0, 1, 2, ...
        # Absolute position = session_start + local_position
        for local_pos in range(n_sess_preds):
            abs_positions[idx] = sess_start + local_pos
            idx += 1

    # Extract aligned context
    result = {
        "targets": targets,
        "predictions": predictions,
        "session_ids": session_ids,
        "position_in_session": np.zeros(n_preds, dtype=np.int32),  # Will fill below
    }

    if top_k_probs is not None:
        result["top_k_probs"] = top_k_probs
        result["top_k_indices"] = preds["top_k_indices"]

    # Extract context fields at aligned positions
    for field_name, ctx_array in contexts.items():
        result[f"ctx_{field_name}"] = ctx_array[abs_positions]

    # Compute position within session
    idx = 0
    for sess_id in range(len(session_starts)):
        if sess_id not in session_pred_counts:
            continue
        n_sess_preds = session_pred_counts[sess_id]
        result["position_in_session"][idx:idx + n_sess_preds] = np.arange(n_sess_preds)
        idx += n_sess_preds

    return result


def infer_split_from_filename(filepath: str) -> str:
    """Infer split name (val/test) from filename."""
    basename = os.path.basename(filepath).lower()
    if "val" in basename:
        return "val"
    elif "test" in basename:
        return "test"
    else:
        return "test"


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_category_accuracy(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, dict]:
    """Compute accuracy metrics for each token category."""
    category_stats: dict[TokenCategory, dict] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    for t, p in zip(targets, predictions):
        cat = TOKEN_TO_CATEGORY.get(int(t))
        if cat is not None:
            category_stats[cat]["total"] += 1
            if t == p:
                category_stats[cat]["correct"] += 1

    results = {}
    for cat, stats in category_stats.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            results[cat.name] = {
                "display_name": CATEGORY_DISPLAY_NAMES.get(cat, cat.name),
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": acc,
            }

    return results


def compute_category_confusion(
    targets: np.ndarray,
    predictions: np.ndarray,
    category: TokenCategory,
) -> tuple[np.ndarray, list[str], dict]:
    """Compute confusion matrix for a specific token category."""
    start_val, end_val = CATEGORY_RANGES[category]

    mask = (targets >= start_val) & (targets <= end_val)
    cat_targets = targets[mask]
    cat_preds = predictions[mask]

    unique_tokens = sorted(set(cat_targets))
    if len(unique_tokens) == 0:
        return np.array([[]]), [], {}

    labels = [get_token_label(t) for t in unique_tokens]
    n_classes = len(unique_tokens)
    token_to_idx = {t: i for i, t in enumerate(unique_tokens)}

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_tok, pred_tok in zip(cat_targets, cat_preds):
        true_idx = token_to_idx.get(true_tok)
        pred_idx = token_to_idx.get(pred_tok)
        if true_idx is not None and pred_idx is not None:
            cm[true_idx, pred_idx] += 1

    stats = {}
    for i, token_id in enumerate(unique_tokens):
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


def compute_accuracy_by_count(
    targets: np.ndarray,
    predictions: np.ndarray,
    count_balls: np.ndarray,
    count_strikes: np.ndarray,
) -> dict[str, dict]:
    """Compute accuracy broken down by ball-strike count."""
    count_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for t, p, balls, strikes in zip(targets, predictions, count_balls, count_strikes):
        count_key = f"{int(balls)}-{int(strikes)}"
        count_stats[count_key]["total"] += 1
        if t == p:
            count_stats[count_key]["correct"] += 1

    results = {}
    for count_key, stats in count_stats.items():
        if stats["total"] > 0:
            results[count_key] = {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / stats["total"],
            }

    return results


def compute_accuracy_by_inning(
    targets: np.ndarray,
    predictions: np.ndarray,
    inning: np.ndarray,
) -> dict[str, dict]:
    """Compute accuracy broken down by inning."""
    inning_stats: dict[int, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for t, p, inn in zip(targets, predictions, inning):
        inn_key = int(inn)
        inning_stats[inn_key]["total"] += 1
        if t == p:
            inning_stats[inn_key]["correct"] += 1

    results = {}
    for inn, stats in sorted(inning_stats.items()):
        if stats["total"] > 0:
            results[str(inn)] = {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / stats["total"],
            }

    return results


def compute_accuracy_by_outs(
    targets: np.ndarray,
    predictions: np.ndarray,
    outs: np.ndarray,
) -> dict[str, dict]:
    """Compute accuracy broken down by outs."""
    outs_stats: dict[int, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for t, p, out in zip(targets, predictions, outs):
        out_key = int(out)
        outs_stats[out_key]["total"] += 1
        if t == p:
            outs_stats[out_key]["correct"] += 1

    results = {}
    for out, stats in sorted(outs_stats.items()):
        if stats["total"] > 0:
            results[str(out)] = {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / stats["total"],
            }

    return results


def compute_accuracy_by_bases(
    targets: np.ndarray,
    predictions: np.ndarray,
    bases_state: np.ndarray,
) -> dict[str, dict]:
    """Compute accuracy broken down by base state."""
    # bases_state is typically encoded as a bitmask: 1=1B, 2=2B, 4=3B
    base_names = {
        0: "Empty",
        1: "1B",
        2: "2B",
        3: "1B+2B",
        4: "3B",
        5: "1B+3B",
        6: "2B+3B",
        7: "Loaded",
    }

    base_stats: dict[int, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for t, p, bases in zip(targets, predictions, bases_state):
        base_key = int(bases) & 7  # Mask to 3 bits
        base_stats[base_key]["total"] += 1
        if t == p:
            base_stats[base_key]["correct"] += 1

    results = {}
    for base, stats in sorted(base_stats.items()):
        if stats["total"] > 0:
            results[base_names.get(base, str(base))] = {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / stats["total"],
            }

    return results


def compute_accuracy_by_position_in_session(
    targets: np.ndarray,
    predictions: np.ndarray,
    position_in_session: np.ndarray,
    bin_size: int = 25,
    min_samples_per_bin: int = 1000,
) -> dict[str, dict]:
    """Compute accuracy vs position in session (does context help?)."""
    # Use actual max position from data
    max_position = int(position_in_session.max()) + 1

    bin_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for t, p, pos in zip(targets, predictions, position_in_session):
        bin_idx = int(pos) // bin_size
        bin_key = f"{bin_idx * bin_size}-{(bin_idx + 1) * bin_size - 1}"
        bin_stats[bin_key]["total"] += 1
        if t == p:
            bin_stats[bin_key]["correct"] += 1

    results = {}
    for bin_key in sorted(bin_stats.keys(), key=lambda x: int(x.split("-")[0])):
        stats = bin_stats[bin_key]
        # Only include bins with enough samples for reliable stats
        if stats["total"] >= min_samples_per_bin:
            results[bin_key] = {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / stats["total"],
            }

    return results


def compute_accuracy_by_handedness(
    targets: np.ndarray,
    predictions: np.ndarray,
    pitcher_throws: np.ndarray,
    batter_hits: np.ndarray,
) -> dict[str, dict]:
    """Compute accuracy by pitcher/batter handedness matchup."""
    # Assuming: -1 = R, 1 = L (or similar encoding)
    def hand_label(val: int) -> str:
        return "L" if val > 0 else "R"

    matchup_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for t, p, p_hand, b_hand in zip(targets, predictions, pitcher_throws, batter_hits):
        matchup = f"{hand_label(p_hand)}HP vs {hand_label(b_hand)}HB"
        matchup_stats[matchup]["total"] += 1
        if t == p:
            matchup_stats[matchup]["correct"] += 1

    results = {}
    for matchup, stats in sorted(matchup_stats.items()):
        if stats["total"] > 0:
            results[matchup] = {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / stats["total"],
            }

    return results


def compute_pitch_transitions(
    targets: np.ndarray,
    session_ids: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Compute pitch type transition matrix (what follows what)."""
    pitch_start, pitch_end = CATEGORY_RANGES[TokenCategory.PITCH_TYPE]

    # Get pitch type tokens
    pitch_mask = (targets >= pitch_start) & (targets <= pitch_end)
    pitch_targets = targets[pitch_mask]
    pitch_sessions = session_ids[pitch_mask]

    unique_pitches = sorted(set(pitch_targets))
    labels = [get_token_label(p) for p in unique_pitches]
    n_pitches = len(unique_pitches)
    pitch_to_idx = {p: i for i, p in enumerate(unique_pitches)}

    # Count transitions (only within same session)
    transitions = np.zeros((n_pitches, n_pitches), dtype=np.int64)

    for i in range(len(pitch_targets) - 1):
        if pitch_sessions[i] == pitch_sessions[i + 1]:
            from_idx = pitch_to_idx[pitch_targets[i]]
            to_idx = pitch_to_idx[pitch_targets[i + 1]]
            transitions[from_idx, to_idx] += 1

    return transitions, labels


def compute_calibration(
    targets: np.ndarray,
    predictions: np.ndarray,
    top_k_probs: np.ndarray,
    top_k_indices: np.ndarray,
    n_bins: int = 10,
    exclude_categories: list[TokenCategory] | None = None,
) -> dict:
    """Compute calibration metrics (ECE and reliability diagram data).

    Args:
        exclude_categories: Token categories to exclude (e.g., PA_START, PA_END
            which are trivially predictable and skew calibration metrics).
    """
    # Filter out excluded categories (e.g., PA_START/PA_END are trivially 100%)
    if exclude_categories:
        keep_mask = np.ones(len(targets), dtype=bool)
        for cat in exclude_categories:
            cat_start, cat_end = CATEGORY_RANGES[cat]
            keep_mask &= ~((targets >= cat_start) & (targets <= cat_end))
        targets = targets[keep_mask]
        predictions = predictions[keep_mask]
        top_k_probs = top_k_probs[keep_mask]

    # Get confidence (top-1 probability) for each prediction
    confidences = top_k_probs[:, 0].astype(np.float32)
    correct = (targets == predictions).astype(np.float32)

    # Bin by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        # Use <= for the last bin to include confidence == 1.0
        if i == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        if mask.sum() > 0:
            bin_acc = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_count = mask.sum()
        else:
            bin_acc = 0.0
            bin_conf = (low + high) / 2
            bin_count = 0
        bin_accs.append(float(bin_acc))
        bin_confs.append(float(bin_conf))
        bin_counts.append(int(bin_count))

    # Expected Calibration Error
    total = len(targets)
    ece = sum(
        (count / total) * abs(acc - conf)
        for acc, conf, count in zip(bin_accs, bin_confs, bin_counts)
        if count > 0
    )

    # Maximum Calibration Error
    mce = max(
        abs(acc - conf)
        for acc, conf, count in zip(bin_accs, bin_confs, bin_counts)
        if count > 0
    ) if any(c > 0 for c in bin_counts) else 0.0

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_boundaries": bin_boundaries.tolist(),
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "mean_confidence": float(confidences.mean()),
        "mean_accuracy": float(correct.mean()),
    }


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    output_path: str,
    title: str,
) -> None:
    """Plot and save confusion matrix heatmap."""
    if len(labels) == 0 or cm.size == 0:
        return

    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)

    n_classes = len(labels)
    if n_classes <= 10:
        figsize = (14, 6)
        fontsize = 8
        annot_fontsize = 7
    elif n_classes <= 20:
        figsize = (18, 8)
        fontsize = 7
        annot_fontsize = 6
    else:
        figsize = (24, 10)
        fontsize = 6
        annot_fontsize = 5

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Raw counts
    ax1 = axes[0]
    im1 = ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks(range(n_classes))
    ax1.set_yticks(range(n_classes))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=fontsize)
    ax1.set_yticklabels(labels, fontsize=fontsize)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title(f"{title} (Counts)")
    plt.colorbar(im1, ax=ax1)

    # Add text annotations for counts
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            if val > 0:
                color = "white" if val > cm.max() / 2 else "black"
                ax1.text(j, i, f"{val}", ha="center", va="center",
                        color=color, fontsize=annot_fontsize)

    # Normalized (recall)
    ax2 = axes[1]
    im2 = ax2.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks(range(n_classes))
    ax2.set_yticks(range(n_classes))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=fontsize)
    ax2.set_yticklabels(labels, fontsize=fontsize)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title(f"{title} (Row-Normalized / Recall)")
    plt.colorbar(im2, ax=ax2)

    # Add text annotations for percentages
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_normalized[i, j]
            if val > 0.005:  # Only show if > 0.5%
                color = "white" if val > 0.5 else "black"
                ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=annot_fontsize)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_category_accuracy_bar(
    category_stats: dict[str, dict],
    output_path: str,
) -> None:
    """Plot bar chart of accuracy by category."""
    pitch_categories = [
        "PITCH_TYPE", "SPEED", "SPIN_RATE", "SPIN_AXIS",
        "RELEASE_POS_X", "RELEASE_POS_Z", "RELEASE_EXTENSION",
        "VX0", "VY0", "VZ0", "AX", "AY", "AZ",
        "PLATE_POS_X", "PLATE_POS_Z", "RESULT"
    ]

    categories = []
    accuracies = []

    for cat_name in pitch_categories:
        if cat_name in category_stats:
            stats = category_stats[cat_name]
            categories.append(stats["display_name"])
            accuracies.append(stats["accuracy"])

    if not categories:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(categories))
    bars = ax.bar(x, accuracies, color="steelblue", edgecolor="black")

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Prediction Accuracy by Token Category")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_accuracy_by_count(
    count_stats: dict[str, dict],
    output_path: str,
) -> None:
    """Plot heatmap of accuracy by ball-strike count."""
    # Create 4x3 grid (0-3 balls x 0-2 strikes)
    acc_grid = np.zeros((4, 3))
    count_grid = np.zeros((4, 3))

    for count_key, stats in count_stats.items():
        parts = count_key.split("-")
        if len(parts) == 2:
            balls, strikes = int(parts[0]), int(parts[1])
            if 0 <= balls <= 3 and 0 <= strikes <= 2:
                acc_grid[balls, strikes] = stats["accuracy"]
                count_grid[balls, strikes] = stats["total"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(acc_grid, cmap="RdYlGn", vmin=0.5, vmax=0.8)

    ax.set_xticks(range(3))
    ax.set_yticks(range(4))
    ax.set_xticklabels(["0 strikes", "1 strike", "2 strikes"])
    ax.set_yticklabels(["0 balls", "1 ball", "2 balls", "3 balls"])
    ax.set_xlabel("Strikes")
    ax.set_ylabel("Balls")
    ax.set_title("Prediction Accuracy by Count")

    # Add text annotations
    for i in range(4):
        for j in range(3):
            if count_grid[i, j] > 0:
                text = f"{acc_grid[i, j]:.1%}\n({int(count_grid[i, j]):,})"
                ax.text(j, i, text, ha="center", va="center", fontsize=9)

    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_accuracy_by_position(
    position_stats: dict[str, dict],
    output_path: str,
) -> None:
    """Plot accuracy vs position in session."""
    positions = []
    accuracies = []
    totals = []

    for pos_key in sorted(position_stats.keys(), key=lambda x: int(x.split("-")[0])):
        stats = position_stats[pos_key]
        mid_pos = (int(pos_key.split("-")[0]) + int(pos_key.split("-")[1])) / 2
        positions.append(mid_pos)
        accuracies.append(stats["accuracy"])
        totals.append(stats["total"])

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Plot accuracy
    color1 = "steelblue"
    ax1.plot(positions, accuracies, 'o-', markersize=6, linewidth=2, color=color1)
    ax1.set_xlabel("Position in Session (tokens)")
    ax1.set_ylabel("Accuracy", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_title("Prediction Accuracy vs Position in Session\n(Does more context help?)")
    ax1.grid(True, alpha=0.3)

    # Dynamic y-axis with some padding
    min_acc = min(accuracies)
    max_acc = max(accuracies)
    padding = (max_acc - min_acc) * 0.2
    ax1.set_ylim(max(0, min_acc - padding), min(1, max_acc + padding))

    # Secondary axis for sample counts
    ax2 = ax1.twinx()
    color2 = "gray"
    ax2.bar(positions, totals, width=20, alpha=0.3, color=color2)
    ax2.set_ylabel("Sample Count", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration(
    calibration: dict,
    output_path: str,
) -> None:
    """Plot reliability diagram."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    ax1 = axes[0]
    bin_confs = calibration["bin_confidences"]
    bin_accs = calibration["bin_accuracies"]
    bin_counts = calibration["bin_counts"]

    # Filter out empty bins
    valid = [(c, a) for c, a, n in zip(bin_confs, bin_accs, bin_counts) if n > 0]
    if valid:
        confs, accs = zip(*valid)
        ax1.bar(confs, accs, width=0.08, alpha=0.7, label="Model")
        ax1.plot([0, 1], [0, 1], 'k--', label="Perfect calibration")

    ax1.set_xlabel("Mean Predicted Confidence")
    ax1.set_ylabel("Fraction Correct")
    ax1.set_title(f"Reliability Diagram\nECE={calibration['ece']:.4f}, MCE={calibration['mce']:.4f}")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Confidence histogram
    ax2 = axes[1]
    bin_boundaries = calibration["bin_boundaries"]
    ax2.bar(
        [(bin_boundaries[i] + bin_boundaries[i+1])/2 for i in range(len(bin_counts))],
        bin_counts,
        width=0.08,
        alpha=0.7,
    )
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Confidence Distribution")
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_transitions(
    transitions: np.ndarray,
    labels: list[str],
    output_path: str,
) -> None:
    """Plot pitch type transition matrix."""
    # Normalize by row
    with np.errstate(divide='ignore', invalid='ignore'):
        trans_norm = transitions.astype(float) / transitions.sum(axis=1, keepdims=True)
        trans_norm = np.nan_to_num(trans_norm)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(trans_norm, cmap="Blues", vmin=0, vmax=0.5)

    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Next Pitch")
    ax.set_ylabel("Current Pitch")
    ax.set_title("Pitch Type Transitions (Row-Normalized)")

    plt.colorbar(im, ax=ax, label="P(next | current)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis of predictions by token category and context"
    )
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
        help="Path to data directory (contains val/, test/ subdirs)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="category_analysis",
        help="Directory to save results",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=["PITCH_TYPE", "SPEED", "SPIN_RATE", "SPIN_AXIS", "PLATE_POS_X", "PLATE_POS_Z", "RESULT"],
        help="Categories to generate confusion matrices for",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and align predictions with context
    print("Loading and aligning predictions with context data...")

    all_data: dict[str, list] = defaultdict(list)
    session_id_offset = 0

    for pred_path in args.predictions:
        print(f"  Processing: {pred_path}")
        split = infer_split_from_filename(pred_path)

        aligned = align_predictions_with_context(pred_path, args.data_dir, split)

        # Offset session_ids to avoid collisions when concatenating multiple files
        # (e.g., val session 0 and test session 0 are different sessions)
        aligned["session_ids"] = aligned["session_ids"] + session_id_offset
        session_id_offset = int(aligned["session_ids"].max()) + 1

        for key, arr in aligned.items():
            all_data[key].append(arr)

    # Concatenate all data
    data = {key: np.concatenate(arrs) for key, arrs in all_data.items()}

    targets = data["targets"]
    predictions = data["predictions"]
    session_ids = data["session_ids"]

    print(f"\nTotal predictions: {len(targets):,}")

    results = {}

    # ==========================================================================
    # 1. Per-category accuracy
    # ==========================================================================
    print("\n" + "="*70)
    print("ACCURACY BY TOKEN CATEGORY")
    print("="*70)

    category_stats = compute_category_accuracy(targets, predictions)
    results["category_accuracy"] = category_stats

    pitch_order = [
        "PITCH_TYPE", "SPEED", "SPIN_RATE", "SPIN_AXIS",
        "RELEASE_POS_X", "RELEASE_POS_Z", "RELEASE_EXTENSION",
        "VX0", "VY0", "VZ0", "AX", "AY", "AZ",
        "PLATE_POS_X", "PLATE_POS_Z", "RESULT",
    ]

    print(f"{'Category':<20} {'Total':>12} {'Correct':>12} {'Accuracy':>12}")
    print("-" * 60)

    for cat_name in pitch_order:
        if cat_name in category_stats:
            stats = category_stats[cat_name]
            print(f"{stats['display_name']:<20} {stats['total']:>12,} {stats['correct']:>12,} {stats['accuracy']:>12.4f}")

    plot_category_accuracy_bar(category_stats, os.path.join(args.output_dir, "category_accuracy.png"))
    print(f"\nSaved: {args.output_dir}/category_accuracy.png")

    # ==========================================================================
    # 2. Confusion matrices per category
    # ==========================================================================
    print("\n" + "="*70)
    print("CONFUSION MATRICES BY CATEGORY")
    print("="*70)

    confusion_results = {}
    for cat_name in args.categories:
        try:
            category = TokenCategory[cat_name]
        except KeyError:
            print(f"  Unknown category: {cat_name}")
            continue

        display_name = CATEGORY_DISPLAY_NAMES.get(category, cat_name)
        cm, labels, stats = compute_category_confusion(targets, predictions, category)

        if len(labels) > 0:
            plot_path = os.path.join(args.output_dir, f"confusion_{cat_name.lower()}.png")
            plot_confusion_matrix(cm, labels, plot_path, display_name)
            print(f"  {display_name}: {len(labels)} classes, saved to {plot_path}")

            confusion_results[cat_name] = {
                "labels": labels,
                "matrix": cm.tolist(),
                "stats": stats,
            }

    results["confusion_matrices"] = confusion_results

    # ==========================================================================
    # 3. Accuracy by count
    # ==========================================================================
    if "ctx_count_balls" in data and "ctx_count_strikes" in data:
        print("\n" + "="*70)
        print("ACCURACY BY COUNT")
        print("="*70)

        count_stats = compute_accuracy_by_count(
            targets, predictions,
            data["ctx_count_balls"], data["ctx_count_strikes"]
        )
        results["accuracy_by_count"] = count_stats

        print(f"{'Count':<10} {'Total':>12} {'Accuracy':>12}")
        print("-" * 40)
        for count in ["0-0", "0-1", "0-2", "1-0", "1-1", "1-2", "2-0", "2-1", "2-2", "3-0", "3-1", "3-2"]:
            if count in count_stats:
                s = count_stats[count]
                print(f"{count:<10} {s['total']:>12,} {s['accuracy']:>12.4f}")

        plot_accuracy_by_count(count_stats, os.path.join(args.output_dir, "accuracy_by_count.png"))
        print(f"\nSaved: {args.output_dir}/accuracy_by_count.png")

    # ==========================================================================
    # 4. Accuracy by inning
    # ==========================================================================
    if "ctx_inning" in data:
        print("\n" + "="*70)
        print("ACCURACY BY INNING")
        print("="*70)

        inning_stats = compute_accuracy_by_inning(targets, predictions, data["ctx_inning"])
        results["accuracy_by_inning"] = inning_stats

        print(f"{'Inning':<10} {'Total':>12} {'Accuracy':>12}")
        print("-" * 40)
        for inn in sorted(inning_stats.keys(), key=lambda x: int(x)):
            s = inning_stats[inn]
            print(f"{inn:<10} {s['total']:>12,} {s['accuracy']:>12.4f}")

    # ==========================================================================
    # 5. Accuracy by outs
    # ==========================================================================
    if "ctx_outs" in data:
        print("\n" + "="*70)
        print("ACCURACY BY OUTS")
        print("="*70)

        outs_stats = compute_accuracy_by_outs(targets, predictions, data["ctx_outs"])
        results["accuracy_by_outs"] = outs_stats

        print(f"{'Outs':<10} {'Total':>12} {'Accuracy':>12}")
        print("-" * 40)
        for outs in sorted(outs_stats.keys(), key=lambda x: int(x)):
            s = outs_stats[outs]
            print(f"{outs:<10} {s['total']:>12,} {s['accuracy']:>12.4f}")

    # ==========================================================================
    # 6. Accuracy by base state
    # ==========================================================================
    if "ctx_bases_state" in data:
        print("\n" + "="*70)
        print("ACCURACY BY BASE STATE")
        print("="*70)

        bases_stats = compute_accuracy_by_bases(targets, predictions, data["ctx_bases_state"])
        results["accuracy_by_bases"] = bases_stats

        print(f"{'Bases':<12} {'Total':>12} {'Accuracy':>12}")
        print("-" * 40)
        for bases, s in bases_stats.items():
            print(f"{bases:<12} {s['total']:>12,} {s['accuracy']:>12.4f}")

    # ==========================================================================
    # 7. Accuracy by position in session
    # ==========================================================================
    print("\n" + "="*70)
    print("ACCURACY BY POSITION IN SESSION")
    print("="*70)

    position_stats = compute_accuracy_by_position_in_session(
        targets, predictions, data["position_in_session"]
    )
    results["accuracy_by_position"] = position_stats

    print(f"{'Position':<12} {'Total':>12} {'Accuracy':>12}")
    print("-" * 40)
    for pos, s in position_stats.items():
        print(f"{pos:<12} {s['total']:>12,} {s['accuracy']:>12.4f}")

    plot_accuracy_by_position(position_stats, os.path.join(args.output_dir, "accuracy_by_position.png"))
    print(f"\nSaved: {args.output_dir}/accuracy_by_position.png")

    # ==========================================================================
    # 8. Accuracy by handedness
    # ==========================================================================
    if "ctx_pitcher_throws" in data and "ctx_batter_hits" in data:
        print("\n" + "="*70)
        print("ACCURACY BY HANDEDNESS MATCHUP")
        print("="*70)

        hand_stats = compute_accuracy_by_handedness(
            targets, predictions,
            data["ctx_pitcher_throws"], data["ctx_batter_hits"]
        )
        results["accuracy_by_handedness"] = hand_stats

        print(f"{'Matchup':<20} {'Total':>12} {'Accuracy':>12}")
        print("-" * 50)
        for matchup, s in hand_stats.items():
            print(f"{matchup:<20} {s['total']:>12,} {s['accuracy']:>12.4f}")

    # ==========================================================================
    # 9. Pitch transitions
    # ==========================================================================
    print("\n" + "="*70)
    print("PITCH TYPE TRANSITIONS")
    print("="*70)

    transitions, trans_labels = compute_pitch_transitions(targets, session_ids)
    results["pitch_transitions"] = {
        "labels": trans_labels,
        "matrix": transitions.tolist(),
    }

    plot_transitions(transitions, trans_labels, os.path.join(args.output_dir, "pitch_transitions.png"))
    print(f"Saved: {args.output_dir}/pitch_transitions.png")

    # ==========================================================================
    # 10. Calibration
    # ==========================================================================
    if "top_k_probs" in data and "top_k_indices" in data:
        print("\n" + "="*70)
        print("CALIBRATION ANALYSIS")
        print("="*70)

        # Exclude PA_START and PA_END which are trivially 100% predictable
        calibration = compute_calibration(
            targets, predictions,
            data["top_k_probs"], data["top_k_indices"],
            exclude_categories=[TokenCategory.PA_START, TokenCategory.PA_END],
        )
        results["calibration"] = calibration

        print("(Excluding PA_START and PA_END tokens)")
        print(f"Expected Calibration Error (ECE): {calibration['ece']:.4f}")
        print(f"Maximum Calibration Error (MCE): {calibration['mce']:.4f}")
        print(f"Mean Confidence: {calibration['mean_confidence']:.4f}")
        print(f"Mean Accuracy: {calibration['mean_accuracy']:.4f}")

        plot_calibration(calibration, os.path.join(args.output_dir, "calibration.png"))
        print(f"\nSaved: {args.output_dir}/calibration.png")

    # ==========================================================================
    # Save all results
    # ==========================================================================
    json_path = os.path.join(args.output_dir, "comprehensive_analysis.json")
    print(f"\n{'='*70}")
    print(f"Saving all results to: {json_path}")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
