# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset

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

        self.pad_id = pad_id
        self.seed = seed
        self.pitch_vocab = self._build_vocab(pitch_tokens)

        self.samples = self._make_samples(pitch_tokens)
        if not self.samples:
            raise ValueError("no plate appearances with at least two pitches were found")
        first_seq, _ = self.samples[0]
        self.feature_dim = first_seq.size(-1)
        self.num_classes = len(self.pitch_vocab)

    def _build_vocab(
        self,
        pitch_tokens: list[PitchToken],
    ) -> dict[PitchToken, int]:
        vocab: dict[PitchToken, int] = {}

        for token in pitch_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

        return vocab

    def _make_samples(
        self,
        pitch_tokens: list[PitchToken],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Build (sequence, label) pairs from plate appearances.
        """
        samples: list[tuple[torch.Tensor, torch.Tensor]] = []

        for token in pitch_tokens:
            raise NotImplementedError("not implemented")
            
        
        raise NotImplementedError("not implemented")

    def _add_samples(
        self,
        features: list[torch.Tensor],
        tokens: list[PitchToken],
        samples: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        
        raise NotImplementedError("not implemented")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index]


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
