#!/usr/bin/env python3

"""
Split an on-disk PitchPredict dataset into train/val/test partitions by plate
appearance while keeping the binary format produced by PitchDataset.save.

uv run scripts/split_pitch_dataset.py --tokens-path /raid/kline/pitchpredict/.pitchpredict_data/pitch_seq.bin --context-prefix /raid/kline/pitchpredict/.pitchpredict_data/pitch_context --overwrite
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pitchpredict.backend.algs.deep.types import PitchToken  # noqa: E402
from pitchpredict.backend.algs.deep.nn import (  # noqa: E402
    TOKEN_DTYPE,
    _CONTEXT_FIELD_SPECS,
    _context_field_path,
    _ensure_parent_dir,
)


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


def load_tokens_and_contexts(
    tokens_path: Path, context_prefix: Path
) -> tuple[np.memmap, dict[str, np.memmap]]:
    tokens_path = tokens_path.resolve()
    context_prefix = context_prefix.resolve()

    if not tokens_path.exists():
        raise FileNotFoundError(f"token file not found: {tokens_path}")

    token_file_size = tokens_path.stat().st_size
    if token_file_size % TOKEN_DTYPE.itemsize != 0:
        raise ValueError(
            f"{tokens_path} has {token_file_size} bytes, which is not a multiple of {TOKEN_DTYPE.itemsize}"
        )
    token_count = token_file_size // TOKEN_DTYPE.itemsize

    tokens_memmap = np.memmap(tokens_path, dtype=TOKEN_DTYPE, mode="r", shape=(token_count,))

    context_arrays: dict[str, np.memmap] = {}
    for field_name, spec in _CONTEXT_FIELD_SPECS.items():
        field_path = Path(_context_field_path(str(context_prefix), field_name))
        if not field_path.exists():
            raise FileNotFoundError(f"context field file not found: {field_path}")
        file_size = field_path.stat().st_size
        expected_size = token_count * spec.dtype.itemsize
        if file_size != expected_size:
            raise ValueError(
                f"{field_path} has {file_size} bytes but expected {expected_size} for {token_count} samples"
            )
        context_arrays[field_name] = np.memmap(field_path, dtype=spec.dtype, mode="r", shape=(token_count,))

    return tokens_memmap, context_arrays


def plate_appearance_ranges(tokens: np.memmap) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    pa_end_value = PitchToken.PA_END.value
    for idx, token in enumerate(tokens):
        if int(token) == pa_end_value:
            ranges.append((start, idx + 1))
            start = idx + 1
    if start < len(tokens):
        ranges.append((start, len(tokens)))
    return ranges


def compute_targets(total_tokens: int, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    val_target = max(0, int(round(total_tokens * val_ratio)))
    test_target = max(0, int(round(total_tokens * test_ratio)))

    if total_tokens <= 1:
        allowed = total_tokens
    else:
        allowed = total_tokens - 1

    overflow = max(0, val_target + test_target - allowed)
    while overflow > 0 and (val_target > 0 or test_target > 0):
        if val_target >= test_target and val_target > 0:
            val_target -= 1
        elif test_target > 0:
            test_target -= 1
        overflow -= 1

    return val_target, test_target


def select_indices(
    pa_lengths: Sequence[int],
    indices: list[int],
    target_tokens: int,
) -> tuple[list[int], int]:
    selected: list[int] = []
    token_total = 0
    while indices and token_total < target_tokens:
        idx = indices.pop()
        selected.append(idx)
        token_total += pa_lengths[idx]
    return selected, token_total


def write_split_files(
    split_name: str,
    ranges: Sequence[tuple[int, int]],
    indices: Iterable[int],
    tokens_memmap: np.memmap,
    context_arrays: dict[str, np.memmap],
    tokens_output_path: Path,
    context_prefix_output: Path,
) -> tuple[int, int]:
    ordered_indices = sorted(indices)
    token_count = sum(ranges[idx][1] - ranges[idx][0] for idx in ordered_indices)
    logging.info(
        "Writing %s split: %d tokens across %d plate appearances",
        split_name,
        token_count,
        len(ordered_indices),
    )

    _ensure_parent_dir(str(tokens_output_path))
    with open(tokens_output_path, "wb") as token_file:
        for idx in ordered_indices:
            start, end = ranges[idx]
            token_file.write(tokens_memmap[start:end].tobytes())

    for field_name, array in context_arrays.items():
        field_path = Path(_context_field_path(str(context_prefix_output), field_name))
        _ensure_parent_dir(str(field_path))
        with open(field_path, "wb") as field_file:
            for idx in ordered_indices:
                start, end = ranges[idx]
                field_file.write(array[start:end].tobytes())

    return token_count, len(ordered_indices)


def close_memmaps(tokens_memmap: np.memmap, context_arrays: dict[str, np.memmap]) -> None:
    tokens_memmap._mmap.close()  # type: ignore[attr-defined]
    for array in context_arrays.values():
        array._mmap.close()  # type: ignore[attr-defined]


def finalize_train_files(
    temp_tokens_path: Path,
    final_tokens_path: Path,
    temp_context_prefix: Path,
    final_context_prefix: Path,
) -> None:
    os.replace(temp_tokens_path, final_tokens_path)
    for field_name in _CONTEXT_FIELD_SPECS.keys():
        temp_field = Path(_context_field_path(str(temp_context_prefix), field_name))
        final_field = Path(_context_field_path(str(final_context_prefix), field_name))
        _ensure_parent_dir(str(final_field))
        os.replace(temp_field, final_field)


def cleanup_split_files(tokens_path: Path, context_prefix: Path) -> None:
    if tokens_path.exists():
        tokens_path.unlink()
    for field_name in _CONTEXT_FIELD_SPECS.keys():
        field_path = Path(_context_field_path(str(context_prefix), field_name))
        if field_path.exists():
            field_path.unlink()


def prepare_split_dir(base_dir: Path, name: str, overwrite: bool) -> Path:
    split_dir = base_dir / name
    if split_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{split_dir} already exists. Pass --overwrite to replace the existing split."
            )
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s: %(message)s")

    tokens_path = args.tokens_path.resolve()
    context_prefix = args.context_prefix.resolve()
    base_dir = tokens_path.parent

    logging.info("Loading dataset from %s (context prefix: %s)", tokens_path, context_prefix)
    tokens_memmap, context_arrays = load_tokens_and_contexts(tokens_path, context_prefix)
    total_tokens = len(tokens_memmap)
    if total_tokens == 0:
        logging.warning("Dataset is empty; nothing to split.")
        close_memmaps(tokens_memmap, context_arrays)
        return

    pa_ranges = plate_appearance_ranges(tokens_memmap)
    pa_lengths = [end - start for start, end in pa_ranges]

    val_target, test_target = compute_targets(total_tokens, args.val_ratio, args.test_ratio)
    logging.info(
        "Target token counts -> val: %d (%.2f%%) test: %d (%.2f%%)",
        val_target,
        args.val_ratio * 100,
        test_target,
        args.test_ratio * 100,
    )

    pa_indices = list(range(len(pa_ranges)))
    random.Random(args.seed).shuffle(pa_indices)

    val_indices, _ = select_indices(pa_lengths, pa_indices, val_target)
    test_indices, _ = select_indices(pa_lengths, pa_indices, test_target)
    train_indices = pa_indices  # remaining

    val_dir = prepare_split_dir(base_dir, "val", args.overwrite)
    test_dir = prepare_split_dir(base_dir, "test", args.overwrite)

    train_tokens_tmp = tokens_path.with_name(tokens_path.name + ".train.tmp")
    train_context_prefix_tmp = Path(str(context_prefix) + ".train.tmp")
    cleanup_split_files(train_tokens_tmp, train_context_prefix_tmp)

    train_token_count, train_pa_count = write_split_files(
        "train",
        pa_ranges,
        train_indices,
        tokens_memmap,
        context_arrays,
        train_tokens_tmp,
        train_context_prefix_tmp,
    )
    val_token_count, val_pa_count = write_split_files(
        "val",
        pa_ranges,
        val_indices,
        tokens_memmap,
        context_arrays,
        val_dir / tokens_path.name,
        val_dir / context_prefix.name,
    )
    test_token_count, test_pa_count = write_split_files(
        "test",
        pa_ranges,
        test_indices,
        tokens_memmap,
        context_arrays,
        test_dir / tokens_path.name,
        test_dir / context_prefix.name,
    )

    close_memmaps(tokens_memmap, context_arrays)
    finalize_train_files(train_tokens_tmp, tokens_path, train_context_prefix_tmp, context_prefix)

    logging.info(
        "Actual token counts -> train: %d (%d PA) val: %d (%d PA) test: %d (%d PA)",
        train_token_count,
        train_pa_count,
        val_token_count,
        val_pa_count,
        test_token_count,
        test_pa_count,
    )

    logging.info(
        "Completed split: train=%d tokens, val=%d tokens, test=%d tokens",
        train_token_count,
        val_token_count,
        test_token_count,
    )


if __name__ == "__main__":
    main()

