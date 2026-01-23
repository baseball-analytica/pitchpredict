# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from pitchpredict.backend.algs.deep.types import PitchToken
from pitchpredict.backend.algs.deep.nn import (
    TOKEN_DTYPE,
    _CONTEXT_FIELD_SPECS,
    _context_field_path,
    _ensure_parent_dir,
)

logger = logging.getLogger(__name__)


def _load_tokens_memmap(tokens_path: Path) -> np.memmap:
    if not tokens_path.exists():
        raise FileNotFoundError(f"token file not found: {tokens_path}")

    token_file_size = tokens_path.stat().st_size
    if token_file_size % TOKEN_DTYPE.itemsize != 0:
        raise ValueError(
            f"{tokens_path} has {token_file_size} bytes, which is not a multiple of {TOKEN_DTYPE.itemsize}"
        )
    token_count = token_file_size // TOKEN_DTYPE.itemsize
    return np.memmap(tokens_path, dtype=TOKEN_DTYPE, mode="r", shape=(token_count,))


def _load_context_memmaps(
    context_prefix: Path, sample_count: int
) -> dict[str, np.memmap]:
    arrays: dict[str, np.memmap] = {}
    for field_name, spec in _CONTEXT_FIELD_SPECS.items():
        field_path = Path(_context_field_path(str(context_prefix), field_name))
        if not field_path.exists():
            raise FileNotFoundError(f"context field file not found: {field_path}")
        file_size = field_path.stat().st_size
        expected_size = sample_count * spec.dtype.itemsize
        if file_size != expected_size:
            raise ValueError(
                f"{field_path} has {file_size} bytes but expected {expected_size} for {sample_count} samples"
            )
        arrays[field_name] = np.memmap(
            field_path, dtype=spec.dtype, mode="r", shape=(sample_count,)
        )
    return arrays


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


def compute_targets(
    total_tokens: int, val_ratio: float, test_ratio: float
) -> tuple[int, int]:
    val_target = max(0, int(round(total_tokens * val_ratio)))
    test_target = max(0, int(round(total_tokens * test_ratio)))

    allowed = total_tokens if total_tokens <= 1 else total_tokens - 1
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


def _write_split_files(
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
    logger.info(
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


def _close_memmaps(
    tokens_memmap: np.memmap, context_arrays: dict[str, np.memmap]
) -> None:
    tokens_memmap._mmap.close()  # type: ignore[attr-defined]
    for array in context_arrays.values():
        array._mmap.close()  # type: ignore[attr-defined]


def _finalize_train_files(
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


def _cleanup_split_files(tokens_path: Path, context_prefix: Path) -> None:
    if tokens_path.exists():
        tokens_path.unlink()
    for field_name in _CONTEXT_FIELD_SPECS.keys():
        field_path = Path(_context_field_path(str(context_prefix), field_name))
        if field_path.exists():
            field_path.unlink()


def _prepare_split_dir(base_dir: Path, name: str, overwrite: bool) -> Path:
    split_dir = base_dir / name
    if split_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{split_dir} already exists. Pass overwrite=True to replace the existing split."
            )
        import shutil

        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir


def split_saved_dataset(
    tokens_path: Path | str,
    context_prefix: Path | str,
    *,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
    seed: int = 17,
    overwrite: bool = False,
) -> dict[str, int]:
    """
    Split an already-saved dataset into train/val/test subsets stored in-place.

    Returns a dict containing the token counts for each split.
    """
    val_ratio = max(0.0, float(val_ratio))
    test_ratio = max(0.0, float(test_ratio))
    if val_ratio == 0.0 and test_ratio == 0.0:
        logger.info(
            "Skipping dataset split because both val_ratio and test_ratio are 0."
        )
        return {"train": 0, "val": 0, "test": 0}

    tokens_path = Path(tokens_path).resolve()
    context_prefix = Path(context_prefix).resolve()
    base_dir = tokens_path.parent

    logger.info(
        "Loading dataset from %s (context prefix: %s)", tokens_path, context_prefix
    )
    tokens_memmap = _load_tokens_memmap(tokens_path)
    context_arrays = _load_context_memmaps(context_prefix, len(tokens_memmap))

    if len(tokens_memmap) == 0:
        logger.warning("Dataset is empty; nothing to split.")
        _close_memmaps(tokens_memmap, context_arrays)
        return {"train": 0, "val": 0, "test": 0}

    pa_ranges = plate_appearance_ranges(tokens_memmap)
    pa_lengths = [end - start for start, end in pa_ranges]

    val_target, test_target = compute_targets(len(tokens_memmap), val_ratio, test_ratio)
    logger.info(
        "Target token counts -> val: %d (%.2f%%) test: %d (%.2f%%)",
        val_target,
        val_ratio * 100,
        test_target,
        test_ratio * 100,
    )

    pa_indices = list(range(len(pa_ranges)))
    random.Random(seed).shuffle(pa_indices)

    val_indices, _ = select_indices(pa_lengths, pa_indices, val_target)
    test_indices, _ = select_indices(pa_lengths, pa_indices, test_target)
    train_indices = pa_indices  # remaining

    val_dir = _prepare_split_dir(base_dir, "val", overwrite)
    test_dir = _prepare_split_dir(base_dir, "test", overwrite)

    train_tokens_tmp = tokens_path.with_name(tokens_path.name + ".train.tmp")
    train_context_prefix_tmp = Path(str(context_prefix) + ".train.tmp")
    _cleanup_split_files(train_tokens_tmp, train_context_prefix_tmp)

    try:
        train_token_count, train_pa_count = _write_split_files(
            "train",
            pa_ranges,
            train_indices,
            tokens_memmap,
            context_arrays,
            train_tokens_tmp,
            train_context_prefix_tmp,
        )
        val_token_count, val_pa_count = _write_split_files(
            "val",
            pa_ranges,
            val_indices,
            tokens_memmap,
            context_arrays,
            val_dir / tokens_path.name,
            val_dir / context_prefix.name,
        )
        test_token_count, test_pa_count = _write_split_files(
            "test",
            pa_ranges,
            test_indices,
            tokens_memmap,
            context_arrays,
            test_dir / tokens_path.name,
            test_dir / context_prefix.name,
        )
    except Exception:
        _cleanup_split_files(train_tokens_tmp, train_context_prefix_tmp)
        raise
    finally:
        _close_memmaps(tokens_memmap, context_arrays)

    _finalize_train_files(
        train_tokens_tmp, tokens_path, train_context_prefix_tmp, context_prefix
    )

    logger.info(
        "Actual token counts -> train: %d (%d PA) val: %d (%d PA) test: %d (%d PA)",
        train_token_count,
        train_pa_count,
        val_token_count,
        val_pa_count,
        test_token_count,
        test_pa_count,
    )

    logger.info(
        "Completed split: train=%d tokens, val=%d tokens, test=%d tokens",
        train_token_count,
        val_token_count,
        test_token_count,
    )

    return {
        "train": train_token_count,
        "val": val_token_count,
        "test": test_token_count,
    }
