# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import json
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
import os

from pitchpredict.backend.algs.deep.types import PitchToken, PitchContext

logger = logging.getLogger(__name__)


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
        if not self.samples:
            raise ValueError("no plate appearances with at least two pitches were found")
        first_seq, _ = self[0]
        self.feature_dim = first_seq.size(-1)
        self.num_classes = len(self.pitch_vocab)

    @staticmethod
    def load(
        path_tokens: str,
        path_contexts: str,
        seed: int = 0,
        pad_id: int = 0,
        dataset_log_interval: int = 10000,
    ) -> "PitchDataset":
        """
        Load the pitch dataset from the given paths.
        """
        logger.debug("load called")

        with open(path_tokens, "rb") as f:
            token_bytes = f.read()

        if len(token_bytes) % 2 != 0:
            raise ValueError(f"expected an even number of bytes in {path_tokens}, got {len(token_bytes)}")

        pitch_tokens = [
            PitchToken(int.from_bytes(token_bytes[i : i + 2], "big") + 1)
            for i in range(0, len(token_bytes), 2)
        ]

        with open(path_contexts, "r") as f:
            pitch_contexts = [PitchContext.model_validate_json(line) for line in f]

        return PitchDataset(pitch_tokens, pitch_contexts, seed, pad_id, dataset_log_interval)

    def save(
        self,
        path_tokens: str = "./.pitchpredict_data/pitch_data.bin",
        path_contexts: str = "./.pitchpredict_data/pitch_contexts.json"
    ) -> None:
        """
        Save the pitch tokens as a binary file, where each token is an integer index into the pitch vocabulary.
        Save the pitch contexts as a JSON file, where each context is a dictionary.
        """
        logger.debug("save called")

        os.makedirs(os.path.dirname(path_tokens) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(path_contexts) or ".", exist_ok=True)

        with open(path_tokens, "wb") as f:
            for token in self._pitch_tokens:
                f.write(token.value.to_bytes(2, "big"))

        with open(path_contexts, "w") as f:
            for context in self._pitch_contexts:
                f.write(context.model_dump_json() + "\n")

        logger.info(f"dataset saved to {path_tokens} and {path_contexts}")

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
