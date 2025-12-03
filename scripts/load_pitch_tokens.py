#!/usr/bin/env python3

"""
Utility script that previews chunks from the packed dataset written by
`PitchDataset.save` by reusing `PackedPitchDataset`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure the project sources are importable when the script runs directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from pitchpredict.backend.algs.deep.types import PitchToken  # noqa: E402
from pitchpredict.backend.algs.deep.nn import _CONTEXT_FIELD_SPECS  # noqa: E402
from pitchpredict.backend.algs.deep.dataset import PackedPitchDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a chunk from PackedPitchDataset to verify packed token/context data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="/raid/kline/pitchpredict/.pitchpredict_data/val",
        help="Directory containing the packed token/context files (default: %(default)s).",
    )
    parser.add_argument(
        "--tokens-file",
        type=str,
        default="pitch_seq.bin",
        help="Filename of the token file relative to data-dir (default: %(default)s).",
    )
    parser.add_argument(
        "--context-prefix",
        type=str,
        default="pitch_context",
        help="Context prefix relative to data-dir (default: %(default)s).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length used when building the packed dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=0,
        help="Chunk index to preview (default: %(default)s).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Optional offset to apply before chunking (default: %(default)s).",
    )
    parser.add_argument(
        "-n",
        "--num-tokens",
        type=int,
        default=10,
        help="Number of time steps from the chunk to display (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    context_prefix = os.path.join(data_dir, args.context_prefix)
    tokens_file = os.path.join(data_dir, args.tokens_file)

    dataset = PackedPitchDataset(
        data_dir=str(data_dir),
        seq_len=args.seq_len,
        tokens_file=os.path.basename(tokens_file),
        context_prefix=context_prefix,
    )
    if args.offset:
        dataset.set_offset(args.offset)

    if len(dataset) == 0:
        print("Dataset has zero chunks; verify seq-len and files.")
        return

    if args.chunk_index < 0 or args.chunk_index >= len(dataset):
        raise IndexError(f"chunk_index {args.chunk_index} is out of bounds for dataset of size {len(dataset)}")

    chunk = dataset[args.chunk_index]
    steps = min(args.num_tokens, chunk.x.numel())

    print(
        f"Chunk {args.chunk_index} (offset={args.offset}) "
        f"showing {steps}/{chunk.x.numel()} tokens from {tokens_file}"
    )

    for idx in range(steps):
        token_value = int(chunk.x[idx].item())
        try:
            token = PitchToken(token_value)
            token_str = f"{token.name} (value={token_value})"
        except ValueError:
            token_str = f"<unknown token {token_value}>"

        target_value = int(chunk.y[idx].item())
        try:
            target_token = PitchToken(target_value)
            target_str = target_token.name
        except ValueError:
            target_str = f"{target_value}"

        print(f"{idx+1:>4}: x={token_str} -> y={target_str}")
        for field_name in _CONTEXT_FIELD_SPECS.keys():
            value = getattr(chunk, field_name)[idx].item()
            value = _CONTEXT_FIELD_SPECS[field_name].decode(value)
            print(f"      - {field_name}: {value}")


if __name__ == "__main__":
    main()

