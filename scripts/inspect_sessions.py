#!/usr/bin/env python3
"""Inspect the session structure of a dataset to visualize interleaving."""

import numpy as np
from pathlib import Path
from pitchpredict.backend.algs.deep.nn import TOKEN_DTYPE, INT32_DTYPE, FLOAT32_DTYPE

# Token values
PAD = 0
SESSION_START = 1
SESSION_END = 2
PA_START = 3
PA_END = 4

def load_dataset(data_dir: str):
    """Load tokens and pitcher_id context."""
    tokens_path = Path(data_dir) / "pitch_seq.bin"
    pitcher_id_path = Path(data_dir) / "pitch_context_pitcher_id.bin"
    game_date_path = Path(data_dir) / "pitch_context_game_date.bin"

    tokens = np.fromfile(tokens_path, dtype=TOKEN_DTYPE)
    pitcher_ids = np.fromfile(pitcher_id_path, dtype=INT32_DTYPE)
    game_dates = np.fromfile(game_date_path, dtype=FLOAT32_DTYPE)

    return tokens, pitcher_ids, game_dates


def analyze_sessions(data_dir: str, max_tokens: int = 2000):
    """Analyze and print session structure."""
    tokens, pitcher_ids, game_dates = load_dataset(data_dir)

    print(f"Dataset: {data_dir}")
    print(f"Total tokens: {len(tokens):,}")
    print(f"Unique pitchers: {len(np.unique(pitcher_ids)):,}")
    print()

    # Count session tokens
    n_session_start = np.sum(tokens == SESSION_START)
    n_session_end = np.sum(tokens == SESSION_END)
    n_pa_start = np.sum(tokens == PA_START)
    n_pa_end = np.sum(tokens == PA_END)

    print(f"SESSION_START count: {n_session_start}")
    print(f"SESSION_END count: {n_session_end}")
    print(f"PA_START count: {n_pa_start}")
    print(f"PA_END count: {n_pa_end}")
    print()

    # Show the interleaving pattern
    print("=" * 80)
    print("SESSION STRUCTURE (first portion of dataset)")
    print("=" * 80)

    current_pitcher = None
    session_num = 0
    pa_num = 0
    pitch_in_pa = 0

    # Track pitcher order to show interleaving
    pitcher_sequence = []

    for i in range(min(len(tokens), max_tokens)):
        tok = tokens[i]
        pid = pitcher_ids[i]

        if tok == SESSION_START:
            session_num += 1
            current_pitcher = pid
            pitcher_sequence.append(pid)
            print(f"\n[SESSION {session_num}] Pitcher {pid}")

        elif tok == SESSION_END:
            print(f"  END SESSION (pitcher {pid})")

        elif tok == PA_START:
            pa_num += 1
            pitch_in_pa = 0
            print(f"    PA #{pa_num}", end="")

        elif tok == PA_END:
            # Each pitch is ~16 tokens (type, speed, spin, etc.)
            actual_pitches = pitch_in_pa // 16
            print(f" ({actual_pitches} pitches, {pitch_in_pa} tokens)")

        elif tok > PA_END:  # Actual pitch token
            pitch_in_pa += 1

    print("\n")
    print("=" * 80)
    print("PITCHER SEQUENCE (showing interleaving)")
    print("=" * 80)

    # Show unique pitcher IDs in order they appear
    seen = set()
    unique_order = []
    for pid in pitcher_sequence:
        if pid not in seen:
            seen.add(pid)
            unique_order.append(pid)

    print(f"Pitchers appear in this order: {unique_order[:20]}...")
    print()

    # Count how many times each pitcher appears in sessions
    from collections import Counter
    pitcher_counts = Counter(pitcher_sequence)

    print("Pitcher session counts (top 10):")
    for pid, count in pitcher_counts.most_common(10):
        print(f"  Pitcher {pid}: {count} session(s)")

    print()

    # Check for interleaving - do pitchers appear multiple times non-consecutively?
    interleaved_count = 0
    for i in range(1, len(pitcher_sequence)):
        if pitcher_sequence[i] != pitcher_sequence[i-1]:
            # Check if this pitcher appeared before
            if pitcher_sequence[i] in pitcher_sequence[:i-1]:
                interleaved_count += 1

    print(f"Sessions where pitcher reappears after another pitcher: {interleaved_count}")
    print("(This indicates interleaved sessions - same pitcher multiple sessions in data)")


def inspect_packed_chunks(data_dir: str, seq_len: int = 64, num_chunks: int = 3):
    """Show what actual training chunks look like from PackedPitchDataset."""
    from pitchpredict.backend.algs.deep.dataset import PackedPitchDataset
    from pitchpredict.backend.algs.deep.types import PitchToken

    print("\n")
    print("=" * 80)
    print(f"PACKED DATASET CHUNKS (seq_len={seq_len})")
    print("=" * 80)

    dataset = PackedPitchDataset(data_dir, seq_len=seq_len)
    print(f"Dataset has {len(dataset)} chunks")
    print(f"Chunk boundaries (first 10): {dataset.chunks[:10]}")
    print()

    # Token name lookup
    token_names = {t.value: t.name for t in PitchToken}

    for chunk_idx in range(min(num_chunks, len(dataset))):
        chunk = dataset[chunk_idx]
        x = chunk.x.numpy()
        y = chunk.y.numpy()
        pitcher_ids = chunk.pitcher_id.numpy()

        print(f"--- Chunk {chunk_idx} (positions {dataset.chunks[chunk_idx]}) ---")
        print(f"x shape: {x.shape}, y shape: {y.shape}")

        # Count real vs pad tokens
        pad_count = np.sum(x == 0)
        print(f"Real tokens: {len(x) - pad_count}, PAD tokens: {pad_count}")

        # Find unique pitchers in this chunk (excluding padding zeros)
        non_pad_pitchers = pitcher_ids[pitcher_ids != 0]
        unique_pitchers = np.unique(non_pad_pitchers) if len(non_pad_pitchers) > 0 else []
        print(f"Pitchers in chunk: {list(unique_pitchers)}")

        # Show token sequence with structure markers
        print("Token sequence (first 100 tokens):")
        line = ""
        for i, tok in enumerate(x[:100]):
            name = token_names.get(tok, f"T{tok}")
            if name == "PAD":
                if line:
                    print(f"  {line}")
                print(f"  [{i}] [PAD x {len(x) - i}]")
                break
            elif name == "SESSION_START":
                if line:
                    print(f"  {line}")
                print(f"  [{i}] ** SESSION_START (pitcher {pitcher_ids[i]}) **")
                line = ""
            elif name == "SESSION_END":
                if line:
                    print(f"  {line}")
                print(f"  [{i}] ** SESSION_END **")
                line = ""
            elif name == "PA_START":
                if line:
                    print(f"  {line}")
                line = f"[{i}] PA: "
            elif name == "PA_END":
                line += " -> END"
                print(f"  {line}")
                line = ""
            elif name.startswith("IS_"):
                # Pitch type - start of a pitch
                line += f"{name[3:]},"
            elif name.startswith("RESULT_"):
                # Result - end of a pitch
                line += f"{name[10:]}; "
            # Skip intermediate tokens (speed, spin, etc.)

        if line:
            print(f"  {line}")

        # Show where session boundaries are
        session_starts = np.where(x == SESSION_START)[0]
        session_ends = np.where(x == SESSION_END)[0]
        print(f"Session starts at positions: {session_starts.tolist()}")
        print(f"Session ends at positions: {session_ends.tolist()}")
        print()


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else ".pitchpredict_test"
    analyze_sessions(data_dir)
    inspect_packed_chunks(data_dir, seq_len=64, num_chunks=3)
    inspect_packed_chunks(data_dir, seq_len=256, num_chunks=3)  # Longer chunks span sessions
