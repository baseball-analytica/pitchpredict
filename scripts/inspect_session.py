#!/usr/bin/env python3
"""
Inspect predictions for a specific session.

Shows the actual vs top-k predicted tokens for each token in the session.

Usage:
    uv run python scripts/inspect_session.py --session_id 123 --predictions test_preds.npz
"""

import argparse

import numpy as np

from pitchpredict.backend.algs.deep.types import PitchToken, get_category


def token_id_to_name(token_id: int) -> str:
    """Convert a token ID to its name."""
    try:
        token = PitchToken(token_id)
        return token.name
    except ValueError:
        return f"UNKNOWN({token_id})"


def token_id_to_short_name(token_id: int) -> str:
    """Convert a token ID to a shorter display name."""
    name = token_id_to_name(token_id)
    # Remove common prefixes for readability
    prefixes = [
        "IS_", "SPEED_IS_", "SPIN_RATE_IS_", "SPIN_AXIS_IS_",
        "RELEASE_POS_X_IS_", "RELEASE_POS_Z_IS_", "RELEASE_EXTENSION_IS_",
        "VX0_IS_", "VY0_IS_", "VZ0_IS_", "AX_IS_", "AY_IS_", "AZ_IS_",
        "PLATE_POS_X_IS_", "PLATE_POS_Z_IS_", "RESULT_IS_",
    ]
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def get_category_name(token_id: int) -> str:
    """Get the category name for a token."""
    try:
        token = PitchToken(token_id)
        return get_category(token).name
    except ValueError:
        return "UNKNOWN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect predictions for a specific session"
    )
    parser.add_argument(
        "--session_id",
        type=int,
        default=158,
        help="Session ID to inspect",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="new_test_preds.npz",
        help="Path to predictions.npz file",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use compact output format",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading predictions from: {args.predictions}")
    preds = np.load(args.predictions)
    targets = preds["targets"]
    predictions = preds["predictions"]
    top_k_indices = preds["top_k_indices"]
    top_k_probs = preds["top_k_probs"]
    session_ids = preds["session_ids"]

    # Filter to the specified session
    mask = session_ids == args.session_id
    if not mask.any():
        print(f"Error: Session {args.session_id} not found in predictions file")
        print(f"Session IDs range from {session_ids.min()} to {session_ids.max()}")
        return

    session_targets = targets[mask]
    session_preds = predictions[mask]
    session_top_k = top_k_indices[mask]
    session_top_k_probs = top_k_probs[mask]

    n_tokens = len(session_targets)
    n_correct = (session_targets == session_preds).sum()
    accuracy = n_correct / n_tokens

    print(f"\nSession {args.session_id}: {n_tokens} tokens, {n_correct} correct ({accuracy:.2%})")
    print("=" * 100)

    if args.compact:
        # Compact format: one line per token
        print(f"{'Idx':>4}  {'Category':<18} {'Actual':<20} {'Pred':<20} {'Prob':>6} {'OK':>3}")
        print("-" * 100)

        for i in range(n_tokens):
            actual = session_targets[i]
            pred = session_preds[i]
            prob = session_top_k_probs[i, 0]
            correct = "Y" if actual == pred else ""

            category = get_category_name(actual)
            actual_name = token_id_to_short_name(actual)
            pred_name = token_id_to_short_name(pred)

            print(f"{i:>4}  {category:<18} {actual_name:<20} {pred_name:<20} {prob:>6.2%} {correct:>3}")
    else:
        # Detailed format: show top-k predictions
        for i in range(n_tokens):
            actual = session_targets[i]
            category = get_category_name(actual)
            actual_name = token_id_to_short_name(actual)
            correct = actual == session_preds[i]

            print(f"\n[{i:>3}] {category}: {actual_name}")
            print(f"      {'CORRECT' if correct else 'WRONG'} | Top-{args.top_k} predictions:")

            for k in range(min(args.top_k, session_top_k.shape[1])):
                pred_id = session_top_k[i, k]
                pred_prob = session_top_k_probs[i, k]
                pred_name = token_id_to_short_name(pred_id)
                marker = " <--" if pred_id == actual else ""
                print(f"        {k+1}. {pred_name:<20} {pred_prob:>6.2%}{marker}")


if __name__ == "__main__":
    main()
