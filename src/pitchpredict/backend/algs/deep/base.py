# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import os
from typing import Any

import torch

from pitchpredict.backend.algs.base import PitchPredictAlgorithm
from pitchpredict.backend.algs.deep.building import build_deep_model


class DeepPitchPredictAlgorithm(PitchPredictAlgorithm):
    """
    A deep learning-based pitch prediction algorithm.
    """

    def __init__(
        self,
        name: str = "deep",
        use_existing: bool = True,
        model_path: str = os.getcwd() + "/.pitchpredict_models/deep_pitch.pth",
        # build parameters (for when use_existing is False)
        date_start: str = "2015-04-01",
        date_end: str = "2024-12-31",
        vocab_size: int = 13,
        embed_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name,
            **kwargs,
        )
        self.use_existing = use_existing
        self.model_path = model_path
        self.date_start = date_start
        self.date_end = date_end
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

    async def build_model(
        self,
    ) -> None:
        """
        Build a new model from scratch using the given parameters.
        """
        model = await build_deep_model(
            date_start=self.date_start,
            date_end=self.date_end,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            model_path=self.model_path,
        )
        self.model = model

    def load_model(self) -> None:
        """
        Load a pre-trained model from the given path.
        """
        try:
            self.model = torch.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"error loading model from {self.model_path}: {e}")