#!/usr/bin/env python3

"""
Split an on-disk PitchPredict dataset into train/val/test partitions by plate
appearance while keeping the binary format produced by PitchDataset.save.

uv run scripts/split_pitch_dataset.py --tokens-path /raid/kline/pitchpredict/.pitchpredict_data/pitch_seq.bin --context-prefix /raid/kline/pitchpredict/.pitchpredict_data/pitch_context --overwrite
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pitchpredict.backend.algs.deep.split import split_saved_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a saved PitchPredict dataset (tokens + per-field contexts) "
            "into train/val/test splits without breaking plate appearances."
        )
    )
    parser.add_argument(
        "--tokens-path",
        type=Path,
        default=REPO_ROOT / ".pitchpredict_data" / "pitch_seq.bin",
        help="Path to the packed token binary produced by PitchDataset.save (default: %(default)s).",
    )
    parser.add_argument(
        "--context-prefix",
        type=Path,
        default=REPO_ROOT / ".pitchpredict_data" / "pitch_context",
        help="Prefix used for the context binaries (default: %(default)s).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.01,
        help="Fraction of tokens to reserve for validation (default: %(default)s).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.01,
        help="Fraction of tokens to reserve for testing (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed used when shuffling plate appearances (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing val/ and test/ directories if they already exist.",
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s: %(message)s",
    )

    split_saved_dataset(
        tokens_path=args.tokens_path,
        context_prefix=args.context_prefix,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
