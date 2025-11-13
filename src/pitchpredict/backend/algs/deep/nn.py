# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import math
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

from pitchpredict.backend.algs.deep.types import PitchToken, PitchContext


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
    ) -> None:
        if len(pitch_tokens) != len(pitch_contexts):
            raise ValueError("pitch_tokens and pitch_contexts must have the same length")
        self.n_pitches = len(pitch_tokens)
        self.n_samples = self.n_pitches - 1

        self.rng = random.Random(seed)
        self.pad_id = pad_id
        self.samples = self._make_samples(pitch_tokens, pitch_contexts)

    def _make_samples(
        self,
        pitch_tokens: list[PitchToken],
        pitch_contexts: list[PitchContext],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Make samples from the given pitch tokens and contexts.
        """
        samples = []

        for i in range(self.n_samples):
            token_tensor = pitch_tokens[i].to_tensor()
            context_tensor = pitch_contexts[i].to_tensor()
            token_tensor_next = pitch_tokens[i + 1].to_tensor()
            context_tensor_next = pitch_contexts[i + 1].to_tensor()

            combined_tensor = torch.cat((token_tensor, context_tensor))
            combined_tensor_next = torch.cat((token_tensor_next, context_tensor_next))

            samples.append((combined_tensor, combined_tensor_next))

        return samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index]


class DeepPitcherModel(nn.Module):
    """
    A deep learning model for predicting the next pitch of a pitcher.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        pad_idx: int = 0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

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
            x: (B, T) token ids
            lengths: (B,) lengths before padding, sorted descending

        Returns:
            logits: (B, num_classes)
        """
        emb = self.embed(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, (h_n, c_n) = self.lstm(packed)

        out_unpacked, out_lengths = pad_packed_sequence(packed_out, batch_first=True)
        idx = (out_lengths - 1).unsqueeze(1).unsqueeze(2).expand(out_unpacked.size(0), 1, out_unpacked.size(2))
        last_valid = out_unpacked.gather(1, idx).squeeze(1)

        logits = self.classifier(last_valid)
        return logits