# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from operator import attrgetter
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from pitchpredict.backend.algs.deep.types import PitchToken, PitchContext

logger = logging.getLogger(__name__)


TOKEN_DTYPE = np.dtype("<u2")
INT32_DTYPE = np.dtype("<i4")
UINT8_DTYPE = np.dtype("uint8")
FLOAT32_DTYPE = np.dtype("float32")


@dataclass(frozen=True)
class _ContextFieldSpec:
    dtype: np.dtype
    getter: Callable[[PitchContext], Any]
    encode: Callable[[Any], Any]
    decode: Callable[[Any], Any]


def _encode_int(value: Any) -> int:
    return int(value)


def _decode_int(value: Any) -> int:
    return int(value)


def _encode_handedness(value: str) -> int:
    if value == "L":
        return 0
    if value == "R":
        return 1
    raise ValueError(f"expected 'L' or 'R', got {value!r}")


def _decode_handedness(value: Any) -> str:
    val = int(value)
    if val == 0:
        return "L"
    if val == 1:
        return "R"
    raise ValueError(f"invalid handedness value: {val}")


def _encode_game_date(value: str) -> float:
    raw = date.fromisoformat(value).toordinal()
    min_date = date.fromisoformat("2023-01-01").toordinal()
    max_date = date.fromisoformat("2025-11-17").toordinal()
    val = (raw - min_date) / (max_date - min_date)
    return max(0.0, min(1.0, val))


def _decode_game_date(value: Any) -> str:
    min_date = date.fromisoformat("2023-01-01").toordinal()
    max_date = date.fromisoformat("2025-11-17").toordinal()
    val = max(0.0, min(1.0, value))
    return date.fromordinal(int(val * (max_date - min_date) + min_date)).isoformat()

def _encode_age(value: int) -> float:
    if not value or value < 15:
        return 0.0

    mean_age = 28.5
    std_dev = 4.0

    return (value - mean_age) / std_dev # z-score

def _decode_age(value: Any) -> float:
    if value == 0.0:
        return 0

    mean_age = 28.5
    std_dev = 4.0
    return value * std_dev + mean_age

def _encode_score(value: int) -> float:
    return float(value) / 10.0 # normalize to 0-1

def _decode_score(value: Any) -> float:
    return value * 10.0

def _encode_pitch_number(value: int) -> float:
    return float(value) / 100.0 # normalize to 0-1

def _decode_pitch_number(value: Any) -> float:
    return value * 100.0

_CONTEXT_FIELD_SPECS: dict[str, _ContextFieldSpec] = {
    "pitcher_id": _ContextFieldSpec(INT32_DTYPE, attrgetter("pitcher_id"), _encode_int, _decode_int),
    "batter_id": _ContextFieldSpec(INT32_DTYPE, attrgetter("batter_id"), _encode_int, _decode_int),
    "pitcher_age": _ContextFieldSpec(FLOAT32_DTYPE, attrgetter("pitcher_age"), _encode_age, _decode_age),
    "pitcher_throws": _ContextFieldSpec(UINT8_DTYPE, attrgetter("pitcher_throws"), _encode_handedness, _decode_handedness),
    "batter_age": _ContextFieldSpec(FLOAT32_DTYPE, attrgetter("batter_age"), _encode_age, _decode_age),
    "batter_hits": _ContextFieldSpec(UINT8_DTYPE, attrgetter("batter_hits"), _encode_handedness, _decode_handedness),
    "count_balls": _ContextFieldSpec(INT32_DTYPE, attrgetter("count_balls"), _encode_int, _decode_int),
    "count_strikes": _ContextFieldSpec(INT32_DTYPE, attrgetter("count_strikes"), _encode_int, _decode_int),
    "outs": _ContextFieldSpec(INT32_DTYPE, attrgetter("outs"), _encode_int, _decode_int),
    "bases_state": _ContextFieldSpec(INT32_DTYPE, attrgetter("bases_state"), _encode_int, _decode_int),
    "score_bat": _ContextFieldSpec(FLOAT32_DTYPE, attrgetter("score_bat"), _encode_score, _decode_score),
    "score_fld": _ContextFieldSpec(FLOAT32_DTYPE, attrgetter("score_fld"), _encode_score, _decode_score),
    "inning": _ContextFieldSpec(INT32_DTYPE, attrgetter("inning"), _encode_int, _decode_int),
    "pitch_number": _ContextFieldSpec(FLOAT32_DTYPE, attrgetter("pitch_number"), _encode_pitch_number, _decode_pitch_number),
    "number_through_order": _ContextFieldSpec(INT32_DTYPE, attrgetter("number_through_order"), _encode_int, _decode_int),
    "game_date": _ContextFieldSpec(FLOAT32_DTYPE, attrgetter("game_date"), _encode_game_date, _decode_game_date),
} # keep parallel with dataset.PackedPitchChunk but without x and y



def _context_field_path(prefix: str, field_name: str) -> str:
    is_directory_hint = prefix.endswith(os.sep)
    normalized = prefix.rstrip(os.sep)
    if is_directory_hint and not normalized:
        normalized = os.sep
    if not normalized:
        normalized = "pitch_context"

    if is_directory_hint:
        directory = normalized
        filename = f"{field_name}.bin"
    else:
        directory, basename = os.path.split(normalized)
        if not basename:
            basename = "pitch_context"
        stem, _ = os.path.splitext(basename)
        stem = stem or basename
        filename = f"{stem}_{field_name}.bin"
    return os.path.join(directory, filename) if directory else filename


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_context_files(contexts: list[PitchContext], prefix: str) -> list[str]:
    saved_paths: list[str] = []
    for field_name, spec in _CONTEXT_FIELD_SPECS.items():
        path = _context_field_path(prefix, field_name)
        _ensure_parent_dir(path)
        array = np.empty(len(contexts), dtype=spec.dtype)
        if field_name == "pitcher_id":
            pitchers: set[int] = set()
        if field_name == "batter_id":
            batters: set[int] = set()
        for idx, context in enumerate(contexts):
            item = spec.getter(context)
            if field_name == "pitcher_id":
                pitchers.add(item)
            if field_name == "batter_id":
                batters.add(item)
            array[idx] = spec.encode(item)
        array.tofile(path)
        saved_paths.append(path)
        if field_name == "pitcher_id":
            logger.info(f"Number of pitchers: {len(pitchers)}")
        if field_name == "batter_id":
            logger.info(f"Number of batters: {len(batters)}")
    return saved_paths


def _read_context_files(sample_count: int, prefix: str) -> list[PitchContext]:
    if sample_count == 0:
        return []

    field_arrays: dict[str, np.memmap] = {}
    contexts: list[PitchContext] = []
    try:
        for field_name, spec in _CONTEXT_FIELD_SPECS.items():
            path = _context_field_path(prefix, field_name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"context field file not found: {path}")
            file_size = os.path.getsize(path)
            expected_size = sample_count * spec.dtype.itemsize
            if file_size != expected_size:
                raise ValueError(
                    f"{path} has {file_size} bytes but expected {expected_size} for {sample_count} samples"
                )
            field_arrays[field_name] = np.memmap(path, dtype=spec.dtype, mode="r", shape=(sample_count,))

        for idx in range(sample_count):
            kwargs = {
                field_name: spec.decode(field_arrays[field_name][idx])
                for field_name, spec in _CONTEXT_FIELD_SPECS.items()
            }
            contexts.append(PitchContext(**kwargs))
    finally:
        for array in field_arrays.values():
            array._mmap.close()  # type: ignore[attr-defined]

    return contexts


class PitchDataset(Dataset):
    """
    PyTorch dataset for PitchPredict data.
    """

    def __init__(
        self,
        pitch_tokens: list[PitchToken],
        pitch_contexts: list[PitchContext],
        seed: int = 0,
        pad_id: int = 0,
        dataset_log_interval: int = 10000,
    ) -> None:
        if len(pitch_tokens) != len(pitch_contexts):
            raise ValueError(f"pitch_tokens and pitch_contexts must have the same length (got {len(pitch_tokens)} tokens vs {len(pitch_contexts)} contexts)")

        self.pad_id = pad_id
        self.seed = seed
        self.pitch_vocab = self._build_vocab(pitch_tokens)
        self.dataset_log_interval = dataset_log_interval
        # keep raw inputs so the dataset can be saved/reloaded cheaply
        self._pitch_tokens = pitch_tokens
        self._pitch_contexts = pitch_contexts

        self.plate_appearances, self.samples = self._make_samples(pitch_tokens, pitch_contexts)
        # if not self.samples:
        #     raise ValueError("no plate appearances with at least two pitches were found")
        # first_seq, _ = self[0]
        # self.feature_dim = first_seq.size(-1)
        self.num_classes = len(self.pitch_vocab)

    @staticmethod
    def load(
        path_tokens: str,
        path_context_prefix: str,
        seed: int = 0,
        pad_id: int = 0,
        dataset_log_interval: int = 10000,
    ) -> "PitchDataset":
        """
        Load the pitch dataset from a token sequence file and the per-field
        context binaries derived from the provided prefix.
        """
        logger.debug("load called")

        token_file_size = os.path.getsize(path_tokens)
        if token_file_size % TOKEN_DTYPE.itemsize != 0:
            raise ValueError(
                f"token file {path_tokens} has {token_file_size} bytes which is not a multiple of {TOKEN_DTYPE.itemsize}"
            )
        token_count = token_file_size // TOKEN_DTYPE.itemsize

        if token_count == 0:
            pitch_tokens: list[PitchToken] = []
        else:
            token_memmap = np.memmap(path_tokens, dtype=TOKEN_DTYPE, mode="r", shape=(token_count,))
            try:
                pitch_tokens = [PitchToken(int(value)) for value in token_memmap]
            finally:
                token_memmap._mmap.close()  # type: ignore[attr-defined]

        pitch_contexts = _read_context_files(len(pitch_tokens), path_context_prefix)

        if len(pitch_contexts) != len(pitch_tokens):
            raise ValueError(
                f"context data length ({len(pitch_contexts)}) "
                f"does not match token length ({len(pitch_tokens)})"
            )

        return PitchDataset(pitch_tokens, pitch_contexts, seed, pad_id, dataset_log_interval)

    def save(
        self,
        path_tokens: str = "./.pitchpredict_data/pitch_seq.bin",
        path_context_prefix: str = "./.pitchpredict_data/pitch_context"
    ) -> None:
        """
        Save the pitch tokens as a binary file and the pitch contexts as one
        binary file per context field derived from the provided prefix.
        """
        logger.debug("save called")

        _ensure_parent_dir(path_tokens)

        token_array = np.empty(len(self._pitch_tokens), dtype=TOKEN_DTYPE)
        for idx, token in enumerate(self._pitch_tokens):
            token_array[idx] = token.value
        token_array.tofile(path_tokens)

        saved_context_paths = _write_context_files(self._pitch_contexts, path_context_prefix)

        total_size = 0.0
        for path in (saved_context_paths + [path_tokens]):
            total_size += os.path.getsize(path)
        logger.info(f"Total size of dataset: {total_size / 1024 / 1024:.2f} MB, {total_size / 1024:.2f} KB")
        logger.info(f"Number of pitch tokens: {len(self._pitch_tokens)}")
        size_per_pitch = total_size / len(self._pitch_tokens)
        logger.info(f"Size per pitch: {size_per_pitch / 1024:.2f} KB")

        logger.info(
            "dataset saved to %s and %d context files derived from %s",
            path_tokens,
            len(saved_context_paths),
            path_context_prefix,
        )

    def _build_vocab(
        self,
        pitch_tokens: list[PitchToken],
    ) -> dict[PitchToken, int]:
        logger.debug("_build_vocab called")

        vocab: dict[PitchToken, int] = {}

        for token in pitch_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

        logger.info("_build_vocab completed successfully")
        return vocab

    def _make_samples(
        self,
        pitch_tokens: list[PitchToken],
        pitch_contexts: list[PitchContext],
    ) -> tuple[list[tuple[list[PitchToken], list[PitchContext]]], list[tuple[int, int]]]:
        """
        Build (sequence, label) pairs from plate appearances while minimizing memory use.
        """
        logger.debug("_make_samples called")
        return [], []

        plate_appearances: list[tuple[list[PitchToken], list[PitchContext]]] = []
        samples: list[tuple[int, int]] = []  # (plate_appearance_index, end_idx)

        pa_tokens: list[PitchToken] = []
        pa_contexts: list[PitchContext] = []

        for idx, (token, context) in enumerate(
            tqdm(
                zip(pitch_tokens, pitch_contexts),
                total=len(pitch_tokens),
                desc="indexing samples",
            )
        ):
            pa_tokens.append(token)
            pa_contexts.append(context)

            if token == PitchToken.PA_END:
                self._finalize_plate_appearance(pa_tokens, pa_contexts, plate_appearances, samples)
                pa_tokens = []
                pa_contexts = []

        # handle trailing tokens if a PA_END was missing at the end of the stream
        if pa_tokens:
            self._finalize_plate_appearance(pa_tokens, pa_contexts, plate_appearances, samples)

        logger.info("_make_samples completed successfully")
        return plate_appearances, samples

    def _finalize_plate_appearance(
        self,
        tokens: list[PitchToken],
        contexts: list[PitchContext],
        plate_appearances: list[tuple[list[PitchToken], list[PitchContext]]],
        samples: list[tuple[int, int]],
    ) -> None:
        """
        Record sample indices for a completed plate appearance.
        """
        # we need at least one input token and one target token
        if len(tokens) < 2:
            return

        pa_idx = len(plate_appearances)
        plate_appearances.append((tokens, contexts))
        for end_idx in range(1, len(tokens)):
            samples.append((pa_idx, end_idx))


    def _token_to_feature(
        self,
        token: PitchToken,
        context: PitchContext,
    ) -> torch.Tensor:
        """
        Convert a token/context pair into a dense feature vector.
        """

        token_one_hot = torch.zeros(len(self.pitch_vocab), dtype=torch.float32)
        token_one_hot[self.pitch_vocab[token]] = 1.0
        context_tensor = context.to_tensor().float()

        feature_tensor = torch.cat([token_one_hot, context_tensor], dim=0)

        return feature_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pa_idx, end_idx = self.samples[index]
        tokens, contexts = self.plate_appearances[pa_idx]

        seq_tokens = tokens[:end_idx]
        seq_contexts = contexts[:end_idx]

        seq_features = torch.stack(
            [self._token_to_feature(tok, ctx) for tok, ctx in zip(seq_tokens, seq_contexts)]
        )
        target_token = tokens[end_idx]
        label = torch.tensor(self.pitch_vocab[target_token], dtype=torch.long)

        return seq_features, label


class DeepPitcherModel(nn.Module):
    """
    A deep learning model for predicting the next pitch of a pitcher.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        pad_idx: int = 0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(out_dim, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: (B, T, input_dim) feature vectors
            lengths: (B,) lengths before padding, sorted descending

        Returns:
            logits: (B, num_classes)
        """
        emb = self.input_proj(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, (h_n, c_n) = self.lstm(packed)

        out_unpacked, out_lengths = pad_packed_sequence(packed_out, batch_first=True)
        idx = (out_lengths - 1).unsqueeze(1).unsqueeze(2).expand(out_unpacked.size(0), 1, out_unpacked.size(2)).to(x.device)
        last_valid = out_unpacked.gather(1, idx).squeeze(1)

        logits = self.classifier(last_valid)
        return logits
