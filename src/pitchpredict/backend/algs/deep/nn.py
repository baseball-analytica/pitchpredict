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
            raise ValueError(f"pitch_tokens and pitch_contexts must have the same length (got {len(pitch_tokens)} tokens vs {len(pitch_contexts)} contexts)")

        self.pad_id = pad_id
        self.seed = seed
        self.pitch_vocab = self._build_vocab(pitch_tokens)

        self.samples = self._make_samples(pitch_tokens, pitch_contexts)
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
        pitch_contexts: list[PitchContext],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Build (sequence, label) pairs from plate appearances.
        """
        samples: list[tuple[torch.Tensor, torch.Tensor]] = []

        pa_features: list[torch.Tensor] = []
        pa_tokens: list[PitchToken] = []

        for token, context in zip(pitch_tokens, pitch_contexts):
            feature_vec = self._token_to_feature(token, context)
            pa_features.append(feature_vec)
            pa_tokens.append(token)

            if token == PitchToken.PA_END:
                self._add_samples(pa_features, pa_tokens, samples)
                pa_features = []
                pa_tokens = []

        # handle trailing tokens if a PA_END was missing at the end of the stream
        if pa_tokens:
            self._add_samples(pa_features, pa_tokens, samples)

        return samples

    def _add_samples(
        self,
        features: list[torch.Tensor],
        tokens: list[PitchToken],
        samples: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        # we need at least one input token and one target token
        if len(tokens) < 2:
            return

        for end_idx in range(1, len(tokens)):
            seq = torch.stack(features[:end_idx])
            target_token = tokens[end_idx]
            label = torch.tensor(self.pitch_vocab[target_token], dtype=torch.long)
            samples.append((seq, label))

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
        return torch.cat([token_one_hot, context_tensor], dim=0)

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
