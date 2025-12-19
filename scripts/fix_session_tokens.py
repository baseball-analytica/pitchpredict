#!/usr/bin/env python3
"""
Fix dataset by inserting missing SESSION_START and SESSION_END tokens.

Session boundaries are detected when:
- pitcher_id changes, OR
- game_date changes, OR
- total score (score_bat + score_fld) decreases (indicates new game, handles doubleheaders)
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np

# Token and dtype constants (from nn.py)
TOKEN_DTYPE = np.dtype("<u2")  # uint16 little-endian
INT32_DTYPE = np.dtype("<i4")
UINT8_DTYPE = np.dtype("uint8")
FLOAT32_DTYPE = np.dtype("float32")

# Token values from types.py PitchToken enum (auto() starts at 1)
SESSION_START = 1
SESSION_END = 2

# Context field specifications matching nn.py:141-170
CONTEXT_FIELD_SPECS: dict[str, np.dtype] = {
    "pitcher_id": INT32_DTYPE,
    "batter_id": INT32_DTYPE,
    "pitcher_age": FLOAT32_DTYPE,
    "pitcher_throws": UINT8_DTYPE,
    "batter_age": FLOAT32_DTYPE,
    "batter_hits": UINT8_DTYPE,
    "count_balls": INT32_DTYPE,
    "count_strikes": INT32_DTYPE,
    "outs": INT32_DTYPE,
    "bases_state": INT32_DTYPE,
    "score_bat": FLOAT32_DTYPE,
    "score_fld": FLOAT32_DTYPE,
    "inning": INT32_DTYPE,
    "pitch_number": FLOAT32_DTYPE,
    "number_through_order": INT32_DTYPE,
    "game_date": FLOAT32_DTYPE,
    "fielder_2_id": INT32_DTYPE,
    "fielder_3_id": INT32_DTYPE,
    "fielder_4_id": INT32_DTYPE,
    "fielder_5_id": INT32_DTYPE,
    "fielder_6_id": INT32_DTYPE,
    "fielder_7_id": INT32_DTYPE,
    "fielder_8_id": INT32_DTYPE,
    "fielder_9_id": INT32_DTYPE,
    "batter_days_since_prev_game": INT32_DTYPE,
    "pitcher_days_since_prev_game": INT32_DTYPE,
    "strike_zone_top": FLOAT32_DTYPE,
    "strike_zone_bottom": FLOAT32_DTYPE,
}


def context_field_path(prefix: str, field_name: str) -> str:
    """Generate path for a context field file (mirrors nn._context_field_path)."""
    return f"{prefix}_{field_name}.bin"


def load_dataset(data_dir: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Load tokens and context arrays from a dataset directory.

    Returns:
        tokens: uint16 array of token values
        contexts: dict mapping field name to numpy array
    """
    tokens_path = os.path.join(data_dir, "pitch_seq.bin")
    context_prefix = os.path.join(data_dir, "pitch_context")

    # Load tokens
    if not os.path.exists(tokens_path):
        raise FileNotFoundError(f"Token file not found: {tokens_path}")

    tokens = np.fromfile(tokens_path, dtype=TOKEN_DTYPE)
    n_tokens = len(tokens)
    print(f"  Loaded {n_tokens:,} tokens from {tokens_path}")

    # Load context fields
    contexts: dict[str, np.ndarray] = {}
    for field_name, dtype in CONTEXT_FIELD_SPECS.items():
        path = context_field_path(context_prefix, field_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Context file not found: {path}")

        arr = np.fromfile(path, dtype=dtype)
        if len(arr) != n_tokens:
            raise ValueError(
                f"Context field {field_name} has {len(arr)} entries but expected {n_tokens}"
            )
        contexts[field_name] = arr

    print(f"  Loaded {len(CONTEXT_FIELD_SPECS)} context fields")
    return tokens, contexts


def detect_session_boundaries(contexts: dict[str, np.ndarray]) -> list[int]:
    """
    Detect session boundaries based on pitcher_id, game_date, and score changes.

    Returns:
        List of indices where new sessions begin (excluding index 0)
    """
    n = len(contexts["pitcher_id"])
    if n == 0:
        return []

    pitcher_ids = contexts["pitcher_id"]
    game_dates = contexts["game_date"]
    score_bat = contexts["score_bat"]
    score_fld = contexts["score_fld"]

    boundaries: list[int] = []

    for i in range(1, n):
        is_new_session = False

        # Check if pitcher changed
        if pitcher_ids[i] != pitcher_ids[i - 1]:
            is_new_session = True

        # Check if game_date changed
        elif game_dates[i] != game_dates[i - 1]:
            is_new_session = True

        # Check if total score decreased (new game in doubleheader)
        else:
            prev_total = score_bat[i - 1] + score_fld[i - 1]
            curr_total = score_bat[i] + score_fld[i]
            if curr_total < prev_total:
                is_new_session = True

        if is_new_session:
            boundaries.append(i)

    return boundaries


def insert_session_tokens(
    tokens: np.ndarray,
    contexts: dict[str, np.ndarray],
    boundaries: list[int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Insert SESSION_START and SESSION_END tokens at session boundaries.

    For each inserted token, the context is copied from the adjacent pitch:
    - SESSION_START: copy from the first pitch of the session
    - SESSION_END: copy from the last pitch of the session

    Returns:
        new_tokens: token array with session tokens inserted
        new_contexts: dict of context arrays with corresponding entries
    """
    n_original = len(tokens)
    if n_original == 0:
        # Empty dataset - just return empty arrays
        new_tokens = np.array([SESSION_START, SESSION_END], dtype=TOKEN_DTYPE)
        new_contexts = {
            field: np.zeros(2, dtype=dtype)
            for field, dtype in CONTEXT_FIELD_SPECS.items()
        }
        return new_tokens, new_contexts

    # Calculate new size:
    # +1 SESSION_START at beginning
    # +1 SESSION_END at end
    # +2 for each boundary (SESSION_END + SESSION_START)
    n_boundaries = len(boundaries)
    n_new = n_original + 2 + (2 * n_boundaries)

    print(f"  Original tokens: {n_original:,}")
    print(f"  Session boundaries: {n_boundaries:,}")
    print(f"  New tokens: {n_new:,} (+{n_new - n_original:,})")

    # Allocate new arrays
    new_tokens = np.empty(n_new, dtype=TOKEN_DTYPE)
    new_contexts: dict[str, np.ndarray] = {
        field: np.empty(n_new, dtype=dtype)
        for field, dtype in CONTEXT_FIELD_SPECS.items()
    }

    # Build boundary set for O(1) lookup
    boundary_set = set(boundaries)

    # Fill in new arrays
    new_idx = 0

    # Insert SESSION_START at beginning (copy context from first token)
    new_tokens[new_idx] = SESSION_START
    for field in CONTEXT_FIELD_SPECS:
        new_contexts[field][new_idx] = contexts[field][0]
    new_idx += 1

    for orig_idx in range(n_original):
        # Check if this is a session boundary
        if orig_idx in boundary_set:
            # Insert SESSION_END for previous session (copy from previous token)
            new_tokens[new_idx] = SESSION_END
            for field in CONTEXT_FIELD_SPECS:
                new_contexts[field][new_idx] = contexts[field][orig_idx - 1]
            new_idx += 1

            # Insert SESSION_START for new session (copy from current token)
            new_tokens[new_idx] = SESSION_START
            for field in CONTEXT_FIELD_SPECS:
                new_contexts[field][new_idx] = contexts[field][orig_idx]
            new_idx += 1

        # Copy original token and context
        new_tokens[new_idx] = tokens[orig_idx]
        for field in CONTEXT_FIELD_SPECS:
            new_contexts[field][new_idx] = contexts[field][orig_idx]
        new_idx += 1

    # Insert SESSION_END at end (copy context from last token)
    new_tokens[new_idx] = SESSION_END
    for field in CONTEXT_FIELD_SPECS:
        new_contexts[field][new_idx] = contexts[field][n_original - 1]
    new_idx += 1

    assert new_idx == n_new, f"Index mismatch: {new_idx} != {n_new}"

    return new_tokens, new_contexts


def save_dataset(
    output_dir: str,
    tokens: np.ndarray,
    contexts: dict[str, np.ndarray],
) -> None:
    """Save tokens and context arrays to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    tokens_path = os.path.join(output_dir, "pitch_seq.bin")
    context_prefix = os.path.join(output_dir, "pitch_context")

    # Save tokens
    tokens.tofile(tokens_path)
    print(f"  Saved {len(tokens):,} tokens to {tokens_path}")

    # Save context fields
    for field_name, arr in contexts.items():
        path = context_field_path(context_prefix, field_name)
        arr.tofile(path)

    print(f"  Saved {len(contexts)} context fields")


def process_split(input_dir: str, output_dir: str, split_name: str) -> None:
    """Process a single dataset split (train, val, or test)."""
    print(f"\nProcessing {split_name} split...")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")

    # Check if input directory exists
    tokens_path = os.path.join(input_dir, "pitch_seq.bin")
    if not os.path.exists(tokens_path):
        print(f"  Skipping - no pitch_seq.bin found")
        return

    # Load dataset
    tokens, contexts = load_dataset(input_dir)

    if len(tokens) == 0:
        print(f"  Skipping - empty dataset")
        return

    # Detect boundaries
    boundaries = detect_session_boundaries(contexts)
    print(f"  Detected {len(boundaries):,} session boundaries")

    # Insert session tokens
    new_tokens, new_contexts = insert_session_tokens(tokens, contexts, boundaries)

    # Save fixed dataset
    save_dataset(output_dir, new_tokens, new_contexts)

    # Verify counts
    session_starts = np.sum(new_tokens == SESSION_START)
    session_ends = np.sum(new_tokens == SESSION_END)
    print(f"  SESSION_START count: {session_starts:,}")
    print(f"  SESSION_END count: {session_ends:,}")

    if session_starts != session_ends:
        print(f"  WARNING: Mismatched session token counts!")


def main():
    parser = argparse.ArgumentParser(
        description="Fix dataset by inserting missing SESSION_START and SESSION_END tokens"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=".pitchpredict_data",
        help="Input dataset directory (default: .pitchpredict_data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".pitchpredict_data_fixed",
        help="Output dataset directory (default: .pitchpredict_data_fixed)",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Process main (train) split
    process_split(input_dir, output_dir, "train")

    # Process val split if it exists
    val_input = os.path.join(input_dir, "val")
    val_output = os.path.join(output_dir, "val")
    if os.path.isdir(val_input):
        process_split(val_input, val_output, "val")

    # Process test split if it exists
    test_input = os.path.join(input_dir, "test")
    test_output = os.path.join(output_dir, "test")
    if os.path.isdir(test_input):
        process_split(test_input, test_output, "test")

    print("\nDone!")


if __name__ == "__main__":
    main()
